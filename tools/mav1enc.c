/* Command line front end: y4m (I420, 8 or 10 bit) in, IVF out, optional
 * reconstruction dump, per-frame statistics. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/mav1.h"

static void usage(void) {
  fprintf(stderr,
          "usage: mav1enc -i in.y4m -o out.ivf [-r recon.y4m] [-q qindex] [-n frames]\n"
          "               [--cpu|--metal] [--keyint N] [--no-lf] [--no-cdef] [--no-txsearch] [--no-txdepth] [--no-lr] [--lr-test] [--no-switchable] [--fast] [--golden] [--altref N] [--no-edge] [--tiles LOG2]\n"
          "               [--me-range N] [--leaf] [--turbo] [-v]\n"
          "               [--bitdepth 8|10 (default: from the y4m C tag)]\n"
          "               [--hdr [NITS]] [--cp N --tc N --mc N --range 0|1] [--cll CLL[,FALL]]\n"
          "               [--rc cqp|vbr|cbr --b:v KBPS [--vbv KBITS]] [--fps NUM:DEN]\n");
  exit(2);
}

static void put_le16(FILE *f, int v) { fputc(v & 255, f); fputc((v >> 8) & 255, f); }
static void put_le32(FILE *f, unsigned v) { put_le16(f, v & 0xFFFF); put_le16(f, v >> 16); }
static void put_le64(FILE *f, unsigned long long v) { put_le32(f, (unsigned)v); put_le32(f, (unsigned)(v >> 32)); }

/* integer sum of squares (exact); per-row partial sums stay in 32 bits at 8
 * bits and 64 bits at 10 */
static double psnr(const uint8_t *a, const uint8_t *b, size_t n, int px) {
  unsigned long long acc = 0;
  if (px == 2) {
    for (size_t i0 = 0; i0 < n; i0 += 2048) {
      size_t e = i0 + 2048 < n ? i0 + 2048 : n;
      unsigned long long s = 0;
      for (size_t i = i0; i < e; i++) {
        int d = (int)*(const uint16_t *)(a + i * 2) - (int)*(const uint16_t *)(b + i * 2);
        s += (unsigned long long)(d * d);
      }
      acc += s;
    }
  } else {
    for (size_t i0 = 0; i0 < n; i0 += 4096) {
      size_t e = i0 + 4096 < n ? i0 + 4096 : n;
      unsigned s = 0;
      for (size_t i = i0; i < e; i++) { int d = (int)a[i] - b[i]; s += (unsigned)(d * d); }
      acc += s;
    }
  }
  double sse = (double)acc, peak = px == 2 ? 1023.0 : 255.0;
  if (sse == 0) return 99.0;
  return 10 * log10(peak * peak * n / sse);
}

int main(int argc, char **argv) {
  const char *in = NULL, *outp = NULL, *recp = NULL;
  int leaf = 0, q = 100, nframes = 1 << 30, metal = 1, keyint = 0, lf = 1, cdef = 1, txs = 1, txd = 1, lr = 1,
      lrt = 0, sw = 1, fast = 0, turbo = 0, gold = 0, altref = 0, edge = 1, tiles = -1, trows = -1, verbose = 0, mer = 32,
      bitdepth = 0, hdr = 0, hdr_nits = 1000, cp = -1, tc = -1, mc = -1, range = 0, cll = -1, fall = -1;
  int rc = MAV1_RC_CQP, bitrate = 0, vbv = 0, fpsn = 30, fpsd = 1;
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
    else if (!strcmp(argv[i], "--turbo")) turbo = fast = 1;
    else if (!strcmp(argv[i], "--golden")) gold = 1;
    else if (!strcmp(argv[i], "--altref") && i + 1 < argc) altref = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--leaf")) leaf = 1;
    else if (!strcmp(argv[i], "--no-edge")) edge = 0;
    else if (!strcmp(argv[i], "--tiles") && i + 1 < argc) tiles = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tile-rows") && i + 1 < argc) trows = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bitdepth") && i + 1 < argc) bitdepth = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hdr")) {
      hdr = 1;
      if (i + 1 < argc && argv[i + 1][0] != '-') hdr_nits = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "--cp") && i + 1 < argc) cp = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tc") && i + 1 < argc) tc = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mc") && i + 1 < argc) mc = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--range") && i + 1 < argc) range = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cll") && i + 1 < argc) {
      cll = atoi(argv[++i]);
      char *c = strchr(argv[i], ',');
      if (c) fall = atoi(c + 1);
    }
    else if (!strcmp(argv[i], "--rc") && i + 1 < argc) {
      i++;
      if (!strcmp(argv[i], "cqp")) rc = MAV1_RC_CQP;
      else if (!strcmp(argv[i], "vbr")) rc = MAV1_RC_VBR;
      else if (!strcmp(argv[i], "cbr")) rc = MAV1_RC_CBR;
      else usage();
    }
    else if (!strcmp(argv[i], "--b:v") && i + 1 < argc) bitrate = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--vbv") && i + 1 < argc) vbv = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--fps") && i + 1 < argc) sscanf(argv[++i], "%d:%d", &fpsn, &fpsd);
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else usage();
  }
  if (!in || !outp) usage();
  if (rc != MAV1_RC_CQP && bitrate <= 0) { fprintf(stderr, "--rc vbr|cbr needs --b:v\n"); return 2; }
  FILE *fi = fopen(in, "rb");
  if (!fi) { perror(in); return 1; }
  char line[512];
  if (!fgets(line, sizeof line, fi) || strncmp(line, "YUV4MPEG2", 9)) { fprintf(stderr, "not y4m\n"); return 1; }
  int w = 0, h = 0, fn = 30, fd = 1, y4m10 = 0;
  for (char *t = strtok(line, " \n"); t; t = strtok(NULL, " \n")) {
    if (t[0] == 'W') w = atoi(t + 1);
    else if (t[0] == 'H') h = atoi(t + 1);
    else if (t[0] == 'F') sscanf(t + 1, "%d:%d", &fn, &fd);
    else if (t[0] == 'C') {
      if (!strncmp(t + 1, "420p10", 6)) y4m10 = 1;
      else if (strncmp(t + 1, "420", 3)) { fprintf(stderr, "only 4:2:0 input is supported\n"); return 1; }
    }
  }
  if (!bitdepth) bitdepth = y4m10 ? 10 : 8;
  if (bitdepth == 10 && !y4m10) { fprintf(stderr, "--bitdepth 10 needs a C420p10 y4m\n"); return 2; }
  if (bitdepth != 8 && bitdepth != 10) { fprintf(stderr, "bitdepth must be 8 or 10\n"); return 2; }
  Mav1Config cfg;
  mav1_default_config(&cfg, w, h);
  cfg.bitdepth = bitdepth;
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
  cfg.turbo = turbo;
  cfg.multi_ref = gold;
  cfg.altref = altref;
  cfg.leaf = leaf;
  cfg.nonref_odd = getenv("MAV1_NREF") ? 1 : 0;
  cfg.intra_edge_filter = edge;
  /* color description: explicit flags, else the HDR defaults
   * (BT.2020 primaries, PQ, BT.2020 matrix) */
  if (hdr) {
    cfg.color_primaries = cp >= 0 ? cp : 9;   /* CP_BT_2020 */
    cfg.transfer_characteristics = tc >= 0 ? tc : 16; /* TC_SMPTE_2084 (PQ) */
    cfg.matrix_coefficients = mc >= 0 ? mc : 9; /* MC_BT_2020_NCL */
    cfg.color_range = range;
    cfg.hdr = 1;
    /* BT.2020 mastering display: r(0.708,0.292) g(0.170,0.797) b(0.131,0.046) w(0.3127,0.3290) */
    cfg.mastering_display[0] = 46423; cfg.mastering_display[1] = 19137;
    cfg.mastering_display[2] = 11140; cfg.mastering_display[3] = 52225;
    cfg.mastering_display[4] = 8586;  cfg.mastering_display[5] = 3015;
    cfg.mastering_display[6] = 20493; cfg.mastering_display[7] = 21568;
    cfg.max_luma = hdr_nits * 256;    /* 24.8 fixed */
    cfg.min_luma = 1;                 /* 18.14 fixed (1 centinit) */
    if (cll >= 0) { cfg.max_cll = cll; cfg.max_fall = fall; }
  } else if (cp >= 0 || tc >= 0 || mc >= 0) {
    cfg.color_primaries = cp;
    cfg.transfer_characteristics = tc;
    cfg.matrix_coefficients = mc;
    cfg.color_range = range;
  }
  cfg.rc = rc;
  cfg.bitrate = bitrate;
  cfg.vbv_size = vbv;
  cfg.fps_num = fn;
  cfg.fps_den = fd;
  /* turbo: 2 tile columns (BD -2..-3% vs 8 at the same speed) */
  cfg.tile_cols_log2 = tiles >= 0 ? tiles : turbo ? 2 : fast ? 3 : 0;
  cfg.tile_rows_log2 = trows >= 0 ? trows : turbo ? 1 : 0; /* 8 tiles: +20% for +0.3% BD */
  if (fast) cfg.tx_search = cfg.tx_depth = cfg.restoration = cfg.switchable_interp = 0;
  if (getenv("MAV1_TXS")) cfg.tx_search = atoi(getenv("MAV1_TXS"));
  if (getenv("MAV1_TXD")) cfg.tx_depth = atoi(getenv("MAV1_TXD"));
  if (getenv("MAV1_LR")) cfg.restoration = atoi(getenv("MAV1_LR"));
  if (getenv("MAV1_SW")) cfg.switchable_interp = atoi(getenv("MAV1_SW"));
  char err[256];
  Mav1Encoder *e = mav1_create(&cfg, err, sizeof err);
  if (!e) { fprintf(stderr, "mav1_create: %s\n", err); return 1; }
  FILE *fo = fopen(outp, "wb");
  FILE *fr = recp ? fopen(recp, "wb") : NULL;
  if (fr)
    fprintf(fr, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420%s\n", w, h, fn, fd, bitdepth == 10 ? "p10" : "jpeg");
  /* IVF header; the frame count is patched at the end. */
  fwrite("DKIF", 1, 4, fo); put_le16(fo, 0); put_le16(fo, 32); fwrite("AV01", 1, 4, fo);
  put_le16(fo, w); put_le16(fo, h); put_le32(fo, fn); put_le32(fo, fd); put_le32(fo, 0); put_le32(fo, 0);

  int cw = (w + 1) / 2, ch = (h + 1) / 2;
  size_t px = bitdepth == 10 ? 2 : 1;
  size_t ysz = (size_t)w * h * px, csz = (size_t)cw * ch * px;
  /* input frames are kept until their temporal unit comes out (reordering) */
  int ring = (turbo && altref < 2 ? 2 : altref) + 8; /* + frames whose packets wait for the writer queue */
  size_t fsz = ysz + 2 * csz;
  uint8_t *frames = malloc(fsz * ring), *rec = malloc(fsz);
  int count = 0, sent = 0, eos = 0;
  size_t total = 0;
  double psum[3] = {0, 0, 0};
  long tools[16] = {0};
  double tsum[4] = {0, 0, 0, 0};
  int strides[3] = {(int)(w * px), (int)(cw * px), (int)(cw * px)}; /* bytes */
  while (!eos) {
    if (sent < nframes && fgets(line, sizeof line, fi) && fread(frames + fsz * (sent % ring), 1, fsz, fi) == fsz) {
      uint8_t *frame = frames + fsz * (sent % ring);
      const uint8_t *planes[3] = {frame, frame + ysz, frame + ysz + csz};
      if (mav1_send_frame(e, planes, strides)) { fprintf(stderr, "encode failed\n"); return 1; }
      sent++;
    } else {
      mav1_send_frame(e, NULL, NULL);
      eos = 1;
    }
    uint8_t *pkt;
    size_t n;
    while (!mav1_receive_packet(e, &pkt, &n)) {
    uint8_t *frame = frames + fsz * (count % ring);
    put_le32(fo, (unsigned)n); put_le64(fo, (unsigned long long)count);
    fwrite(pkt, 1, n, fo);
    free(pkt);
    uint8_t *rp[3] = {rec, rec + ysz, rec + ysz + csz};
    mav1_get_recon(e, rp, strides);
    if (fr) { fprintf(fr, "FRAME\n"); fwrite(rec, 1, fsz, fr); }
    double py = psnr(frame, rec, ysz / px, (int)px), pu = psnr(frame + ysz, rec + ysz, csz / px, (int)px),
           pv = psnr(frame + ysz + csz, rec + ysz + csz, csz / px, (int)px);
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
  free(frames);
  free(rec);
  return 0;
}
