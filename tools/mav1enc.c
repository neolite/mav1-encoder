/* Command line front end: y4m (I420) in, IVF out, optional reconstruction
 * dump, per-frame statistics. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/mav1.h"

static void usage(void) {
  fprintf(stderr,
          "usage: mav1enc -i in.y4m -o out.ivf [-r recon.y4m] [-q qindex] [-n frames]\n"
          "               [--cpu|--metal] [--keyint N] [--no-lf] [--no-cdef] [--no-txsearch] [--no-txdepth] [--no-lr] [--lr-test] [--no-switchable] [--fast] [--no-edge] [--tiles LOG2]\n"
          "               [--me-range N] [-v]\n");
  exit(2);
}

static void put_le16(FILE *f, int v) { fputc(v & 255, f); fputc((v >> 8) & 255, f); }
static void put_le32(FILE *f, unsigned v) { put_le16(f, v & 0xFFFF); put_le16(f, v >> 16); }
static void put_le64(FILE *f, unsigned long long v) { put_le32(f, (unsigned)v); put_le32(f, (unsigned)(v >> 32)); }

static double psnr(const uint8_t *a, const uint8_t *b, size_t n) {
  double sse = 0;
  for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; sse += d * d; }
  if (sse == 0) return 99.0;
  return 10 * log10(255.0 * 255.0 * n / sse);
}

int main(int argc, char **argv) {
  const char *in = NULL, *outp = NULL, *recp = NULL;
  int q = 100, nframes = 1 << 30, metal = 1, keyint = 0, lf = 1, cdef = 1, txs = 1, txd = 1, lr = 1, lrt = 0, sw = 1, fast = 0, edge = 1, tiles = -1, verbose = 0, mer = 32;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) outp = argv[++i];
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) recp = argv[++i];
    else if (!strcmp(argv[i], "-q") && i + 1 < argc) q = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-n") && i + 1 < argc) nframes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--keyint") && i + 1 < argc) keyint = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--me-range") && i + 1 < argc) mer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cpu")) metal = 0;
    else if (!strcmp(argv[i], "--metal")) metal = 1;
    else if (!strcmp(argv[i], "--no-lf")) lf = 0;
    else if (!strcmp(argv[i], "--no-cdef")) cdef = 0;
    else if (!strcmp(argv[i], "--no-txsearch")) txs = 0;
    else if (!strcmp(argv[i], "--no-txdepth")) txd = 0;
    else if (!strcmp(argv[i], "--no-lr")) lr = 0;
    else if (!strcmp(argv[i], "--lr-test")) lrt = 1;
    else if (!strcmp(argv[i], "--no-switchable")) sw = 0;
    else if (!strcmp(argv[i], "--fast")) fast = 1;
    else if (!strcmp(argv[i], "--no-edge")) edge = 0;
    else if (!strcmp(argv[i], "--tiles") && i + 1 < argc) tiles = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else usage();
  }
  if (!in || !outp) usage();
  FILE *fi = fopen(in, "rb");
  if (!fi) { perror(in); return 1; }
  char line[512];
  if (!fgets(line, sizeof line, fi) || strncmp(line, "YUV4MPEG2", 9)) { fprintf(stderr, "not y4m\n"); return 1; }
  int w = 0, h = 0, fn = 30, fd = 1;
  for (char *t = strtok(line, " \n"); t; t = strtok(NULL, " \n")) {
    if (t[0] == 'W') w = atoi(t + 1);
    else if (t[0] == 'H') h = atoi(t + 1);
    else if (t[0] == 'F') sscanf(t + 1, "%d:%d", &fn, &fd);
    else if (t[0] == 'C' && strncmp(t + 1, "420", 3)) { fprintf(stderr, "only 4:2:0 input is supported\n"); return 1; }
  }
  Mav1Config cfg;
  mav1_default_config(&cfg, w, h);
  cfg.qindex = q;
  cfg.use_metal = metal;
  cfg.keyint = keyint;
  cfg.loop_filter = lf;
  cfg.cdef = cdef;
  cfg.tx_search = txs;
  cfg.tx_depth = txd;
  cfg.restoration = lr;
  cfg.lr_test = lrt;
  cfg.switchable_interp = sw;
  cfg.me_range = mer;
  cfg.fast = fast;
  cfg.nonref_odd = getenv("MAV1_NREF") ? 1 : 0;
  cfg.intra_edge_filter = edge;
  cfg.tile_cols_log2 = tiles >= 0 ? tiles : fast ? 3 : 0;
  if (fast) cfg.tx_search = cfg.tx_depth = cfg.restoration = cfg.switchable_interp = 0;
  char err[256];
  Mav1Encoder *e = mav1_create(&cfg, err, sizeof err);
  if (!e) { fprintf(stderr, "mav1_create: %s\n", err); return 1; }
  FILE *fo = fopen(outp, "wb");
  FILE *fr = recp ? fopen(recp, "wb") : NULL;
  if (fr) fprintf(fr, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420jpeg\n", w, h, fn, fd);
  /* IVF header; the frame count is patched at the end. */
  fwrite("DKIF", 1, 4, fo); put_le16(fo, 0); put_le16(fo, 32); fwrite("AV01", 1, 4, fo);
  put_le16(fo, w); put_le16(fo, h); put_le32(fo, fn); put_le32(fo, fd); put_le32(fo, 0); put_le32(fo, 0);

  int cw = (w + 1) / 2, ch = (h + 1) / 2;
  size_t ysz = (size_t)w * h, csz = (size_t)cw * ch;
  uint8_t *frame = malloc(ysz + 2 * csz), *rec = malloc(ysz + 2 * csz);
  int count = 0;
  size_t total = 0;
  double psum[3] = {0, 0, 0};
  long tools[16] = {0};
  double tsum[4] = {0, 0, 0, 0};
  while (count < nframes && fgets(line, sizeof line, fi)) {
    if (fread(frame, 1, ysz + 2 * csz, fi) != ysz + 2 * csz) break;
    const uint8_t *planes[3] = {frame, frame + ysz, frame + ysz + csz};
    int strides[3] = {w, cw, cw};
    uint8_t *pkt;
    size_t n;
    if (mav1_encode(e, planes, strides, &pkt, &n)) { fprintf(stderr, "encode failed\n"); return 1; }
    put_le32(fo, (unsigned)n); put_le64(fo, (unsigned long long)count);
    fwrite(pkt, 1, n, fo);
    free(pkt);
    uint8_t *rp[3] = {rec, rec + ysz, rec + ysz + csz};
    mav1_get_recon(e, rp, strides);
    if (fr) { fprintf(fr, "FRAME\n"); fwrite(rec, 1, ysz + 2 * csz, fr); }
    double py = psnr(frame, rec, ysz), pu = psnr(frame + ysz, rec + ysz, csz),
           pv = psnr(frame + ysz + csz, rec + ysz + csz, csz);
    psum[0] += py; psum[1] += pu; psum[2] += pv;
    const Mav1FrameStats *st = mav1_last_stats(e);
    tsum[0] += st->t_analysis; tsum[1] += st->t_recon; tsum[2] += st->t_filter; tsum[3] += st->t_entropy;
    if (verbose)
      fprintf(stderr, "frame %4d %c %8zu bytes  blocks %5d (inter %5d)  PSNR %.2f %.2f %.2f  "
              "ana %.1fms rec %.1fms flt %.1fms ent %.1fms\n",
              count, st->frame_type ? 'P' : 'K', n, st->num_blocks, st->num_inter_blocks, py, pu, pv,
              st->t_analysis * 1e3, st->t_recon * 1e3, st->t_filter * 1e3, st->t_entropy * 1e3);
    for (int k = 0; k < 16; k++) tools[k] += st->tool_use[k];
    total += n;
    count++;
  }
  fseek(fo, 24, SEEK_SET);
  put_le32(fo, count);
  fclose(fo);
  if (fr) fclose(fr);
  fclose(fi);
  if (count)
    fprintf(stderr, "%s: %d frames, %zu bytes (%.1f kbit/s @%d/%d fps), PSNR Y %.3f U %.3f V %.3f, "
            "time/frame: analysis %.2fms recon %.2fms filter %.2fms entropy %.2fms\n",
            mav1_backend_name(e), count, total, total * 8.0 / count * fn / fd / 1000.0, fn, fd,
            psum[0] / count, psum[1] / count, psum[2] / count, tsum[0] / count * 1e3, tsum[1] / count * 1e3,
            tsum[2] / count * 1e3, tsum[3] / count * 1e3);
  fprintf(stderr, "tool usage (blocks):");
  for (int k = 0; k < 16; k++)
    if (mav1_tool_names[k][0]) fprintf(stderr, " %s=%ld", mav1_tool_names[k], tools[k]);
  fprintf(stderr, "\n");
  mav1_destroy(e);
  free(frame);
  free(rec);
  return 0;
}
