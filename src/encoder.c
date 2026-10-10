/* Frame-level driver: buffer management, stage sequencing and bitstream
 * assembly. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "backend.h"
#include "enc_internal.h"
#include "kernels/k_lr.h"
#include "fwd_matrices.h"
#include "kernels/k_analysis.h"
#include "kernels/k_me.h"
#include "kernels/k_fwd.h"
#include "kernels/k_recon.h"

#define PAGE 16384

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int buf_alloc(Buf *b, size_t size) {
  size = (size + PAGE - 1) / PAGE * PAGE;
  if (size == 0) size = PAGE;
  void *p = NULL;
  if (posix_memalign(&p, PAGE, size)) return -1;
  memset(p, 0, size);
  b->base = p;
  b->size = size;
  return 0;
}

static void buf_free(Buf *b) {
  free(b->base);
  b->base = NULL;
}

void mav1_default_config(Mav1Config *cfg, int width, int height) {
  memset(cfg, 0, sizeof(*cfg));
  cfg->width = width;
  cfg->height = height;
  cfg->qindex = 100;
  cfg->use_metal = 1;
  cfg->keyint = 0;
  cfg->tx_search = 1;
  cfg->tx_depth = 1;
  cfg->restoration = 1;
  cfg->switchable_interp = 1;
  cfg->intra_edge_filter = 1;
  cfg->loop_filter = 1;
  cfg->cdef = 1;
  cfg->me_range = 32;
}

Mav1Encoder *mav1_create(const Mav1Config *cfg, char *err, size_t errlen) {
  if (cfg->width < 16 || cfg->height < 16 || cfg->width > 65536 || cfg->height > 65536) {
    snprintf(err, errlen, "unsupported frame size %dx%d", cfg->width, cfg->height);
    return NULL;
  }
  Mav1Encoder *e = calloc(1, sizeof(*e));
  e->cfg = *cfg;
  if (e->cfg.qindex < 0) e->cfg.qindex = 0;
  if (e->cfg.qindex > 255) e->cfg.qindex = 255;
  e->w = cfg->width;
  e->h = cfg->height;
  e->mi_cols = 2 * ((e->w + 7) >> 3);
  e->mi_rows = 2 * ((e->h + 7) >> 3);
  e->sb_cols = (e->mi_cols + 15) >> 4;
  e->sb_rows = (e->mi_rows + 15) >> 4;
  e->num_sbs = e->sb_cols * e->sb_rows;
  /* tile_info: uniform spacing, tile columns only (one tile row) */
  {
    int k = 0;
    while (((MAX_TILE_WIDTH >> 6) << k) < e->sb_cols) k++;
    e->min_log2_tile_cols = k;
    int maxCols = e->sb_cols < MAX_TILE_COLS ? e->sb_cols : MAX_TILE_COLS;
    for (k = 0; (1 << k) < maxCols; k++) {}
    e->max_log2_tile_cols = k;
    int l = e->cfg.tile_cols_log2;
    if (l < e->min_log2_tile_cols) l = e->min_log2_tile_cols;
    if (l > e->max_log2_tile_cols) l = e->max_log2_tile_cols;
    e->tile_cols_log2 = l;
    e->tile_w_sb = (e->sb_cols + (1 << l) - 1) >> l;
    e->tile_cols = (e->sb_cols + e->tile_w_sb - 1) / e->tile_w_sb;
  }
  e->stride[0] = e->sb_cols * 64;
  e->stride[1] = e->stride[2] = e->sb_cols * 32;
  e->plane_h[0] = e->sb_rows * 64;
  e->plane_h[1] = e->plane_h[2] = e->sb_rows * 32;
  e->offset[0] = 0;
  e->offset[1] = e->stride[0] * e->plane_h[0];
  e->offset[2] = e->offset[1] + e->stride[1] * e->plane_h[1];
  e->frame_bytes = (size_t)e->offset[2] + (size_t)e->stride[2] * e->plane_h[2];

  int ok = 0;
  ok |= buf_alloc(&e->src, e->frame_bytes);
  ok |= buf_alloc(&e->rec, e->frame_bytes);
  ok |= buf_alloc(&e->ref, e->frame_bytes);
  if (cfg->multi_ref) ok |= buf_alloc(&e->gold, e->frame_bytes);
  ok |= buf_alloc(&e->params, sizeof(FrameParams));
  ok |= buf_alloc(&e->blocks, sizeof(BlockInfo) * MAX_BLOCKS_PER_SB * e->num_sbs);
  ok |= buf_alloc(&e->sbs, sizeof(SbInfo) * e->num_sbs);
  ok |= buf_alloc(&e->coefs, sizeof(int32_t) * COEFS_PER_SB * e->num_sbs);
  ok |= buf_alloc(&e->mats, sizeof(int32_t) * FWD_SLOT_SIZE * FWD_NUM_SLOTS);
  ok |= buf_alloc(&e->mvfield, sizeof(int32_t) * 2 * NUM_NODES * e->num_sbs);
  ok |= buf_alloc(&e->mvgold, cfg->multi_ref ? sizeof(int32_t) * 2 * NUM_NODES * e->num_sbs : 0);
  ok |= buf_alloc(&e->acosts, sizeof(long) * A_ITEMS * (size_t)e->num_sbs);
  ok |= buf_alloc(&e->ainfos, sizeof(BlockInfo) * A_ITEMS * (size_t)e->num_sbs);
  {
    /* pass-1 GPU scratch: the largest level dispatch (threads x slot size) */
    size_t mx = 0;
    for (int l = 0; l < 4; l++) {
      size_t t = (size_t)e->num_sbs * (1u << (2 * l)) * (l == 3 ? 1 : l == 2 ? 5 : A_SLOTS) * analysis_slot_ints(l);
      if (t > mx) mx = t;
    }
    ok |= buf_alloc(&e->ascratch, sizeof(int32_t) * mx);
  }
  ok |= buf_alloc(&e->mescratch, sizeof(int32_t) * ME_SB_INTS * (size_t)e->num_sbs);
  ok |= buf_alloc(&e->scratch, sizeof(int32_t) * SCRATCH_INTS * (size_t)e->num_sbs);
  ok |= buf_alloc(&e->lfmask, (size_t)e->mi_rows * e->mi_cols);
  ok |= buf_alloc(&e->dbk, e->frame_bytes);
  ok |= buf_alloc(&e->lrout, e->frame_bytes);
  ok |= buf_alloc(&e->lrunits, sizeof(LrUnit) * 3 * (size_t)(e->sb_rows + 1) * (e->sb_cols + 1));
  e->num_cdef_presets = e->cfg.cdef ? 4 : 0;
  ok |= buf_alloc(&e->cdefbuf, e->frame_bytes * (e->num_cdef_presets ? e->num_cdef_presets : 1));
  if (ok) {
    snprintf(err, errlen, "out of memory");
    mav1_destroy(e);
    return NULL;
  }
  fwd_build_matrices((int32_t *)e->mats.base);
  e->mi = calloc((size_t)e->mi_rows * e->mi_cols, sizeof(MiInfo));
  e->cdef_idx = calloc((size_t)e->num_sbs, 1);
  e->cdef_choice = calloc((size_t)e->num_sbs, 1);

  if (cfg->use_metal) {
    e->be = backend_metal_create(e, err, errlen);
    if (!e->be) {
      mav1_destroy(e);
      return NULL;
    }
  } else {
    e->be = backend_cpu_create(e);
  }
  return e;
}

void mav1_destroy(Mav1Encoder *e) {
  if (!e) return;
  if (e->be) e->be->destroy(e->be);
  buf_free(&e->src);
  buf_free(&e->rec);
  buf_free(&e->ref);
  buf_free(&e->gold);
  buf_free(&e->mvgold);
  buf_free(&e->params);
  buf_free(&e->blocks);
  buf_free(&e->sbs);
  buf_free(&e->coefs);
  buf_free(&e->mats);
  buf_free(&e->mvfield);
  buf_free(&e->scratch);
  buf_free(&e->mescratch);
  buf_free(&e->acosts);
  buf_free(&e->ainfos);
  buf_free(&e->ascratch);
  free(e->mi);
  free(e->cdef_idx);
  free(e->cdef_choice);
  buf_free(&e->lfmask);
  buf_free(&e->cdefbuf);
  buf_free(&e->dbk);
  buf_free(&e->lrout);
  buf_free(&e->lrunits);
  free(e);
}

const char *mav1_backend_name(const Mav1Encoder *e) { return e->be->name; }
const Mav1FrameStats *mav1_last_stats(const Mav1Encoder *e) { return &e->last; }

size_t mav1_sequence_header(const Mav1Encoder *e, uint8_t *out) { return write_sequence_header(e, out); }

/* Copies the input into the padded source buffer, replicating the right
 * and bottom edges up to the superblock-aligned size. */
static void load_source(Mav1Encoder *e, const uint8_t *const planes[3], const int strides[3]) {
  for (int p = 0; p < 3; p++) {
    int pw = p ? (e->w + 1) >> 1 : e->w;
    int ph = p ? (e->h + 1) >> 1 : e->h;
    int aw = e->stride[p], ah = e->plane_h[p];
    uint8_t *dst = e->src.base + e->offset[p];
    for (int y = 0; y < ah; y++) {
      const uint8_t *s = planes[p] + (size_t)(y < ph ? y : ph - 1) * strides[p];
      uint8_t *d = dst + (size_t)y * aw;
      memcpy(d, s, pw);
      memset(d + pw, s[pw - 1], aw - pw);
    }
  }
}

/* get_filter_type( plane ) (7.11.2.8) for every intra block, stored in
 * BlockInfo.interp_filter bits 4 (luma) and 5 (chroma). Blocks are >= 8x8,
 * so AvailUChroma / AvailLChroma equal AvailU / AvailL. */
static int smooth_mode(int m) { return m == SMOOTH_PRED || m == SMOOTH_V_PRED || m == SMOOTH_H_PRED; }

static void edge_filter_types(Mav1Encoder *e) {
  const FrameParams *fp = ENC_PARAMS(e);
  int cols = e->mi_cols, rows = e->mi_rows;
  uint8_t *ys = calloc((size_t)rows * cols, 1), *uvs = calloc((size_t)rows * cols, 1);
  BlockInfo *blocks = ENC_BLOCKS(e);
  const SbInfo *sbs = ENC_SBS(e);
  for (int s = 0; s < e->num_sbs; s++)
    for (int i = 0; i < sbs[s].num_blocks; i++) {
      const BlockInfo *b = &blocks[sbs[s].first_block + i];
      int ysm = smooth_mode(b->y_mode) && !b->is_inter, uvsm = !b->is_inter && smooth_mode(b->uv_mode);
      for (int r = b->mi_row; r < b->mi_row + Num_4x4_Blocks_High[b->bsize] && r < rows; r++)
        for (int c = b->mi_col; c < b->mi_col + Num_4x4_Blocks_Wide[b->bsize] && c < cols; c++) {
          ys[r * cols + c] = (uint8_t)ysm;
          uvs[r * cols + c] = (uint8_t)uvsm;
        }
    }
  for (int s = 0; s < e->num_sbs; s++)
    for (int i = 0; i < sbs[s].num_blocks; i++) {
      BlockInfo *b = &blocks[sbs[s].first_block + i];
      int r = b->mi_row, c = b->mi_col;
      int availU = r > 0, availL = c > tile_col_start(fp, c);
      int fy = (availU && ys[(r - 1) * cols + c]) || (availL && ys[r * cols + c - 1]);
      /* chroma: above (r - 1, c + 1) as MiCol is even; left (r + 1, c - 1) as MiRow is even */
      int fuv = (availU && uvs[(r - 1) * cols + k_min(c + 1, cols - 1)]) ||
                (availL && uvs[k_min(r + 1, rows - 1) * cols + c - 1]);
      b->interp_filter = (b->interp_filter & 15) | fy << 4 | fuv << 5;
    }
  free(ys);
  free(uvs);
}

static void setup_params(Mav1Encoder *e, int isKey) {
  FrameParams *fp = ENC_PARAMS(e);
  memset(fp, 0, sizeof(*fp));
  fp->mi_cols = e->mi_cols;
  fp->mi_rows = e->mi_rows;
  fp->sb_cols = e->sb_cols;
  fp->sb_rows = e->sb_rows;
  for (int p = 0; p < 3; p++) {
    fp->plane_w[p] = e->stride[p];
    fp->plane_h[p] = e->plane_h[p];
    fp->stride[p] = e->stride[p];
    fp->offset[p] = e->offset[p];
  }
  int q = e->hdr.base_q_idx;
  fp->base_q_idx = q;
  for (int p = 0; p < 3; p++) {
    fp->dc_q[p] = Dc_Qlookup[0][q];
    fp->ac_q[p] = Ac_Qlookup[0][q];
  }
  fp->frame_is_intra = isKey;
  fp->reduced_tx_set = 0;
  fp->enable_intra_edge_filter = e->cfg.intra_edge_filter;
  fp->tx_search = e->cfg.tx_search;
  fp->lossless = q == 0;
  fp->tx_mode_select = q != 0 && e->cfg.tx_depth;
  fp->switchable_interp = !isKey && e->cfg.switchable_interp;
  /* GOLDEN only when it holds another frame than LAST */
  fp->golden = !isKey && e->cfg.multi_ref && e->gold_id != e->last_id;
  fp->fast = e->cfg.fast;
  fp->leaf = e->nonref && !getenv("MAV1_LEAFINTRA");
  fp->depth_thr = getenv("MAV1_DT") ? atoi(getenv("MAV1_DT")) : 0;
  fp->tile_w_sb = e->tile_w_sb;
  long acq = fp->ac_q[0];
  fp->lambda = (int32_t)(acq * acq * 256 / 430);
  fp->frame_w = e->w;
  fp->frame_h = e->h;
  fp->me_range = e->cfg.me_range;

  /* Loop filter level from q (libaom's LPF_PICK_FROM_Q rule for 8-bit). */
  FrameHdr *h = &e->hdr;
  int lvl = 0;
  if (e->cfg.loop_filter && q > 0 && !(e->nonref && !getenv("MAV1_LEAFLF"))) {
    lvl = (int)((acq * 20723 + 1015158 + (1 << 17)) >> 18);
    if (!isKey) lvl -= 4;
    lvl = lvl < 0 ? 0 : (lvl > MAX_LOOP_FILTER ? MAX_LOOP_FILTER : lvl);
  }
  for (int i = 0; i < 4; i++) h->lf_level[i] = fp->lf_level[i] = lvl;
  h->lf_sharpness = fp->lf_sharpness = 0;
  /* CDEF parameter sets, selected per superblock by SSE. */
  static const int presets[4][4] = {{0, 0, 0, 0}, {2, 1, 1, 0}, {4, 2, 2, 1}, {8, 4, 4, 2}};
  h->cdef_damping = fp->cdef_damping = 3 + (q >> 6);
  h->cdef_bits = fp->cdef_bits = 2;
  for (int i = 0; i < 4; i++) {
    h->cdef_y_pri[i] = fp->cdef_y_pri[i] = presets[i][0];
    h->cdef_y_sec[i] = fp->cdef_y_sec[i] = presets[i][1];
    h->cdef_uv_pri[i] = fp->cdef_uv_pri[i] = presets[i][2];
    h->cdef_uv_sec[i] = fp->cdef_uv_sec[i] = presets[i][3];
  }
  /* Loop restoration units: 64x64 (lr_unit_shift = 0, lr_uv_shift = 0). */
  int off = 0;
  for (int p = 0; p < 3; p++) {
    int sub = p > 0, size = RESTORATION_TILESIZE_MAX >> 2;
    int pw = (e->w + sub) >> sub, ph = (e->h + sub) >> sub;
    fp->lr_type[p] = h->lr_type[p] = RESTORE_NONE;
    fp->lr_unit_size[p] = size;
    fp->lr_unit_rows[p] = k_max((ph + (size >> 1)) / size, 1); /* count_units_in_frame */
    fp->lr_unit_cols[p] = k_max((pw + (size >> 1)) / size, 1);
    fp->lr_unit_off[p] = off;
    off += fp->lr_unit_rows[p] * fp->lr_unit_cols[p];
  }
}

/* Per-mi block size and skip flag for the loop filters. */
static void build_filter_info(Mav1Encoder *e) {
  uint8_t *info = e->lfmask.base;
  const SbInfo *sbs = ENC_SBS(e);
  const BlockInfo *blocks = ENC_BLOCKS(e);
  for (int i = 0; i < e->num_sbs; i++)
    for (int k = 0; k < sbs[i].num_blocks; k++) {
      const BlockInfo *b = &blocks[sbs[i].first_block + k];
      int skip = b->eob[0] == 0 && b->eob[1] == 0 && b->eob[2] == 0;
      int bw4 = Num_4x4_Blocks_Wide[b->bsize], bh4 = Num_4x4_Blocks_High[b->bsize];
      for (int y = 0; y < bh4 && b->mi_row + y < e->mi_rows; y++)
        for (int x = 0; x < bw4 && b->mi_col + x < e->mi_cols; x++)
          info[(b->mi_row + y) * e->mi_cols + b->mi_col + x] = (uint8_t)(b->bsize | (skip << 5) | (b->tx_depth << 6));
    }
}

/* Chooses the CDEF parameter set of every superblock by SSE against the
 * source and assembles the final frame into `rec`. */
static void cdef_select(Mav1Encoder *e) {
  for (int sb = 0; sb < e->num_sbs; sb++) {
    int sr = sb / e->sb_cols, sc = sb % e->sb_cols;
    long best = -1;
    int bestK = 0;
    for (int k = 0; k < e->num_cdef_presets; k++) {
      const uint8_t *out = e->cdefbuf.base + (size_t)k * e->frame_bytes;
      long sse = 0;
      for (int p = 0; p < 3; p++) {
        int sub = p > 0, n = 64 >> sub;
        int pw = p ? (e->w + 1) >> 1 : e->w, ph = p ? (e->h + 1) >> 1 : e->h;
        for (int y = sr * n; y < (sr + 1) * n && y < ph; y++)
          for (int x = sc * n; x < (sc + 1) * n && x < pw; x++) {
            size_t o = e->offset[p] + (size_t)y * e->stride[p] + x;
            int d = (int)out[o] - (int)e->src.base[o];
            sse += d * d;
          }
      }
      if (best < 0 || sse < best) { best = sse; bestK = k; }
    }
    e->cdef_choice[sb] = (int8_t)bestK;
    const uint8_t *out = e->cdefbuf.base + (size_t)bestK * e->frame_bytes;
    for (int p = 0; p < 3; p++) {
      int sub = p > 0, n = 64 >> sub;
      int mw = (e->mi_cols * 4) >> sub, mh = (e->mi_rows * 4) >> sub;
      for (int y = sr * n; y < (sr + 1) * n && y < mh; y++) {
        int x0 = sc * n, x1 = (sc + 1) * n < mw ? (sc + 1) * n : mw;
        size_t o = e->offset[p] + (size_t)y * e->stride[p];
        memcpy(e->rec.base + o + x0, out + o + x0, x1 - x0);
      }
    }
  }
}

const char *const mav1_tool_names[16] = {"intra", "inter", "cfl", "angle_delta", "tx_non_dct", "skip",
                                          "rect_part", "compound", "obmc", "lr_wiener_units", "lr_sgr_units", "golden", "", "", "", ""};

static void count_tools(Mav1FrameStats *st, const BlockInfo *b) {
  st->tool_use[0] += !b->is_inter;
  st->tool_use[1] += b->is_inter;
  st->tool_use[2] += !b->is_inter && b->uv_mode == UV_CFL_PRED;
  st->tool_use[3] += !b->is_inter && (b->angle_y != 0 || b->angle_uv != 0);
  st->tool_use[4] += b->eob[0] > 0 && b->tx_type != DCT_DCT;
  st->tool_use[5] += b->eob[0] == 0 && b->eob[1] == 0 && b->eob[2] == 0;
  st->tool_use[6] += Block_Width(b->bsize) != Block_Height(b->bsize);
  st->tool_use[11] += b->is_inter && b->ref_frame == GOLDEN_FRAME;
}

/* Frame-level quantizer cascade (low delay): the key frame and every 4th
 * frame are coded finer, as they are the references most of the GOP
 * predicts from. MAV1_QK / MAV1_Q4 / MAV1_QP override the offsets. */
static int env_or(const char *n, int d) { const char *v = getenv(n); return v ? atoi(v) : d; }
static int frame_qindex(const Mav1Encoder *e, int isKey) {
  int q = e->cfg.qindex;
  if (q == 0) return 0;
  int off = isKey ? env_or("MAV1_QK", -24) : (e->order_hint % 4 == 0 ? env_or("MAV1_Q4", -8) : env_or("MAV1_QP", 4));
  if (!isKey && e->nonref) off = env_or("MAV1_QN", 12);
  q += off;
  return q < 1 ? 1 : q > 255 ? 255 : q;
}

int mav1_encode(Mav1Encoder *e, const uint8_t *const planes[3], const int strides[3], uint8_t **out,
                size_t *outSize) {
  int isKey = e->frame_num == 0 || (e->cfg.keyint > 0 && e->frame_num % e->cfg.keyint == 0);
  memset(&e->last, 0, sizeof(e->last));
  e->last.frame_type = isKey ? 0 : 1;
  load_source(e, planes, strides);

  FrameHdr *h = &e->hdr;
  memset(h, 0, sizeof(*h));
  /* two-layer temporal hierarchy (single reference): odd frames are not
   * kept as references, so they are the cheap "leaf" frames */
  e->nonref = !isKey && e->cfg.nonref_odd && (e->order_hint & 1);
  h->frame_type = isKey ? KEY_FRAME : INTER_FRAME;
  h->base_q_idx = frame_qindex(e, isKey);
  if (isKey) e->order_hint = 0;
  h->order_hint = e->order_hint;
  /* GOLDEN (slot 1): key frames and every 4th frame (the finer-q ones) */
  e->golden_refresh = !isKey && !e->nonref && e->cfg.multi_ref && e->order_hint % 4 == 0;
  h->refresh_frame_flags = isKey ? 0xFF : (e->nonref ? 0x00 : 0x01 | (e->golden_refresh ? 0x02 : 0));
  setup_params(e, isKey);

  double t0 = now();
  if (!isKey) e->be->motion_search(e->be, e);
  if (getenv("MAV1_TIMING")) fprintf(stderr, "me %.1fms\n", (now() - t0) * 1000);
  e->be->analyze(e->be, e);
  if (e->cfg.intra_edge_filter) edge_filter_types(e);
  double t1 = now();
  e->be->recon(e->be, e);
  double t2 = now();
  build_filter_info(e);
  e->be->filter(e->be, e);
  int useLr = e->cfg.restoration && e->cfg.qindex > 0; /* AllLossless disables it */
  if (useLr) memcpy(e->dbk.base, e->rec.base, e->frame_bytes); /* UpscaledCurrFrame */
  if (e->cfg.cdef && e->cfg.qindex > 0) cdef_select(e);
  if (useLr) {
    lr_select(e);
    memcpy(e->lrout.base, e->rec.base, e->frame_bytes);
    e->be->restore(e->be, e);
    Buf t = e->rec;
    e->rec = e->lrout;
    e->lrout = t;
  }
  double t3 = now();

  uint8_t *tile = NULL;
  size_t tileSize = write_tile(e, &tile);
  size_t cap = tileSize + 1024;
  uint8_t *buf = malloc(cap);
  size_t pos = write_temporal_delimiter(buf);
  if (isKey) pos += write_sequence_header(e, buf + pos);
  pos += write_frame_obu(e, tile, tileSize, buf + pos);
  free(tile);
  double t4 = now();

  const SbInfo *sbs = ENC_SBS(e);
  const BlockInfo *blocks = ENC_BLOCKS(e);
  for (int i = 0; i < e->num_sbs; i++) {
    e->last.num_blocks += sbs[i].num_blocks;
    for (int k = 0; k < sbs[i].num_blocks; k++) {
      const BlockInfo *b = &blocks[sbs[i].first_block + k];
      e->last.num_inter_blocks += b->is_inter;
      count_tools(&e->last, b);
    }
  }
  e->last.bytes = pos;
  e->last.t_analysis = t1 - t0;
  e->last.t_recon = t2 - t1;
  e->last.t_filter = t3 - t2;
  e->last.t_entropy = t4 - t3;

  /* The reconstruction becomes the reference of the next frame. */
  if (!e->nonref) {
    Buf tmp = e->ref;
    e->ref = e->rec;
    e->rec = tmp;
    e->last_id = e->frame_num;
    if (e->cfg.multi_ref && (isKey || e->golden_refresh)) {
      memcpy(e->gold.base, e->ref.base, e->frame_bytes);
      e->gold_id = e->frame_num;
    }
  }
  e->have_ref = 1;
  e->frame_num++;
  e->order_hint++;
  *out = buf;
  *outSize = pos;
  return 0;
}

void mav1_get_recon(const Mav1Encoder *e, uint8_t *const planes[3], const int strides[3]) {
  /* After mav1_encode the last reconstruction lives in `ref`. */
  for (int p = 0; p < 3; p++) {
    int pw = p ? (e->w + 1) >> 1 : e->w;
    int ph = p ? (e->h + 1) >> 1 : e->h;
    for (int y = 0; y < ph; y++)
      memcpy(planes[p] + (size_t)y * strides[p], (e->nonref ? e->rec.base : e->ref.base) + e->offset[p] + (size_t)y * e->stride[p], pw);
  }
}
