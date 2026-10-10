/* Frame-level driver: buffer management, stage sequencing and bitstream
 * assembly. */
#include <sys/mman.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>
#include <time.h>
#include <pthread.h>

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
  /* zero-filled pages, faulted in on first use */
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED) return -1;
  b->base = p;
  b->size = size;
  return 0;
}

static void buf_free(Buf *b) {
  if (b->base) munmap(b->base, b->size);
  b->base = NULL;
}

static int wctx_init(Mav1Encoder *e);
static Mav1Encoder *create_enc(const Mav1Config *cfg, int isEngine, char *err, size_t errlen);
static void wctx_free(Mav1Encoder *e);

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

static Mav1Encoder *create_enc(const Mav1Config *cfg, int isEngine, char *err, size_t errlen) {
  if (cfg->width < 16 || cfg->height < 16 || cfg->width > 65536 || cfg->height > 65536) {
    snprintf(err, errlen, "unsupported frame size %dx%d", cfg->width, cfg->height);
    return NULL;
  }
  Mav1Encoder *e = calloc(1, sizeof(*e));
  e->cfg = *cfg;
  if (e->cfg.turbo) {
    e->cfg.fast = 1;
    if (e->cfg.altref < 2) e->cfg.altref = 2;
    e->cfg.leaf = 1;
  }
  if (e->cfg.altref == 1) e->cfg.altref = 0;
  if (e->cfg.altref > 16) e->cfg.altref = 16;
  if (e->cfg.altref) e->cfg.multi_ref = e->cfg.nonref_odd = 0; /* the second reference is ALTREF */
  if (!e->cfg.altref) e->cfg.leaf = 0;
  /* savings on leaf frames (testing knob MAV1_LEAFOPT, default all): 1 inter
   * only, 2 no deblocking, 4 no CDEF, 8 no LR, 16 no tx type search, 32 ME
   * range 16 */
  e->leafopt_cfg = getenv("MAV1_LEAFOPT") ? (int)strtol(getenv("MAV1_LEAFOPT"), NULL, 0) : 0x3F;
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

  /* turbo driver: the frames are coded by the engines; the driver keeps
   * only the reference slots (and no backend) */
  int drv = !isEngine && e->cfg.turbo && e->cfg.altref && !getenv("MAV1_NOASYNC") && !getenv("MAV1_NOPAR");
#define SZ(x) (drv ? 0 : (x))
  int ok = 0;
  ok |= buf_alloc(&e->src, SZ(e->frame_bytes));
  ok |= buf_alloc(&e->rec, SZ(e->frame_bytes));
  ok |= buf_alloc(&e->ref, SZ(e->frame_bytes));
  int ref2 = e->cfg.multi_ref || e->cfg.altref;
  if (ref2) ok |= buf_alloc(&e->gold, SZ(e->frame_bytes));
  if (e->cfg.altref && !isEngine) {
    ok |= buf_alloc(&e->slot[0], e->frame_bytes);
    ok |= buf_alloc(&e->slot[2], e->frame_bytes);
    ok |= buf_alloc(&e->slot[3], e->frame_bytes);
    if (e->cfg.leaf)
      for (int i = 4; i < 8; i++) ok |= buf_alloc(&e->slot[i], e->frame_bytes);
    e->inq = calloc((size_t)e->cfg.altref + 1, sizeof *e->inq);
  }
  ok |= buf_alloc(&e->params, sizeof(FrameParams));
  ok |= buf_alloc(&e->blocks, SZ(sizeof(BlockInfo) * MAX_BLOCKS_PER_SB * e->num_sbs));
  ok |= buf_alloc(&e->sbs, SZ(sizeof(SbInfo) * e->num_sbs));
  ok |= buf_alloc(&e->coefs, SZ(sizeof(int32_t) * COEFS_PER_SB * e->num_sbs));
  ok |= buf_alloc(&e->mats, sizeof(int32_t) * FWD_SLOT_SIZE * FWD_NUM_SLOTS);
  ok |= buf_alloc(&e->mvfield, SZ(sizeof(int32_t) * 2 * NUM_NODES * e->num_sbs));
  ok |= buf_alloc(&e->mvgold, SZ(ref2 ? sizeof(int32_t) * 2 * NUM_NODES * e->num_sbs : 0));
  ok |= buf_alloc(&e->acosts, SZ(sizeof(long) * A_ITEMS * (size_t)e->num_sbs));
  ok |= buf_alloc(&e->ainfos, SZ(sizeof(BlockInfo) * A_ITEMS * (size_t)e->num_sbs));
  {
    /* pass-1 GPU scratch: the largest level dispatch (threads x slot size) */
    size_t mx = 0;
    for (int l = 0; l < 4; l++) {
      size_t t = (size_t)e->num_sbs * (1u << (2 * l)) * (l == 3 ? 1 : l == 2 ? 5 : A_SLOTS) * analysis_slot_ints(l);
      if (t > mx) mx = t;
    }
    ok |= buf_alloc(&e->ascratch, SZ(sizeof(int32_t) * mx));
  }
  ok |= buf_alloc(&e->hpel, SZ(e->cfg.turbo ? (size_t)(ME_QP_ENTRIES * 16 + 1) * e->w * e->h + (size_t)(ME_QP_ENTRIES + 1) * ((e->w + 3) >> 2) * ((e->h + 3) >> 2) : 0));
  ok |= buf_alloc(&e->mescratch, SZ(sizeof(int32_t) * ME_SB_INTS * (size_t)e->num_sbs));
  ok |= buf_alloc(&e->scratch, SZ(sizeof(int32_t) * SCRATCH_INTS * (size_t)e->num_sbs));
  ok |= buf_alloc(&e->lfmask, SZ((size_t)e->mi_rows * e->mi_cols));
  ok |= buf_alloc(&e->dbk, SZ(e->frame_bytes));
  ok |= buf_alloc(&e->lrout, SZ(e->frame_bytes));
  ok |= buf_alloc(&e->lrunits, SZ(sizeof(LrUnit) * 3 * (size_t)(e->sb_rows + 1) * (e->sb_cols + 1)));
  e->num_cdef_presets = e->cfg.cdef ? 4 : 0;
  ok |= buf_alloc(&e->cdefbuf, SZ(e->frame_bytes * (e->num_cdef_presets ? e->num_cdef_presets : 1)));
#undef SZ
  if (ok) {
    snprintf(err, errlen, "out of memory");
    mav1_destroy(e);
    return NULL;
  }
  e->is_engine = isEngine;
  if (e->cfg.turbo && e->cfg.altref && !getenv("MAV1_NOASYNC") && wctx_init(e)) {
    snprintf(err, errlen, "out of memory");
    mav1_destroy(e);
    return NULL;
  }
  fwd_build_matrices((int32_t *)e->mats.base);
  e->mi = calloc((size_t)e->mi_rows * e->mi_cols, sizeof(MiInfo));
  e->cdef_idx = calloc((size_t)e->num_sbs, 1);
  e->cdef_choice = calloc((size_t)e->num_sbs, 1);

  if (drv) {
  } else if (cfg->use_metal) {
    e->be = backend_metal_create(e, err, errlen);
    if (!e->be) {
      mav1_destroy(e);
      return NULL;
    }
  } else {
    e->be = backend_cpu_create(e);
  }
  if (drv) {
    e->eng_next = 1;
    for (int i = 0; i < NUM_ENG; i++) {
      e->eng[i] = create_enc(cfg, 1, err, errlen);
      if (!e->eng[i]) {
        mav1_destroy(e);
        return NULL;
      }
      e->eng_q[i] = (void *)dispatch_queue_create("mav1.engine", DISPATCH_QUEUE_SERIAL);
      e->eng_busy[i] = (void *)dispatch_group_create();
    }
    e->inflight = (void *)dispatch_semaphore_create(6);
  }
  return e;
}

Mav1Encoder *mav1_create(const Mav1Config *cfg, char *err, size_t errlen) { return create_enc(cfg, 0, err, errlen); }

void mav1_destroy(Mav1Encoder *e) {
  if (!e) return;
  wctx_free(e);
  for (int i = 0; i < NUM_ENG; i++)
    if (e->eng[i]) {
      dispatch_group_wait((dispatch_group_t)e->eng_busy[i], DISPATCH_TIME_FOREVER);
      mav1_destroy(e->eng[i]);
      dispatch_release((dispatch_queue_t)e->eng_q[i]);
      dispatch_release((dispatch_group_t)e->eng_busy[i]);
    }
  for (int k = 0; k < 8; k++) {
    if (e->slot_ver[k]) dispatch_release((dispatch_group_t)e->slot_ver[k]);
    if (e->slot_rd[k]) dispatch_release((dispatch_group_t)e->slot_rd[k]);
  }
  if (e->inflight) dispatch_release((dispatch_semaphore_t)e->inflight);
  if (e->be) e->be->destroy(e->be);
  buf_free(&e->src);
  buf_free(&e->rec);
  buf_free(&e->ref);
  buf_free(&e->gold);
  for (int i = 0; i < 8; i++) buf_free(&e->slot[i]);
  for (int i = 0; i < e->inq_n; i++) free(e->inq[i]);
  free(e->inq);
  for (int i = e->out_pos; i < e->out_n; i++) { free(e->outq[i].data); free(e->outq[i].recon); }
  free(e->outq);
  free(e->shown_recon);
  if (e->pending) free(e->pend_data);
  buf_free(&e->mvgold);
  buf_free(&e->params);
  buf_free(&e->blocks);
  buf_free(&e->sbs);
  buf_free(&e->coefs);
  buf_free(&e->mats);
  buf_free(&e->mvfield);
  buf_free(&e->scratch);
  buf_free(&e->mescratch);
  buf_free(&e->hpel);
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

const char *mav1_backend_name(const Mav1Encoder *e) { return e->be ? e->be->name : e->eng[0]->be->name; }
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

/* Edge filter types of the blocks of SB rows [row0, row1) (the rows above
 * done before); ys / uvs map every mi to its luma / chroma smooth flag. */
void edge_types_rows(Mav1Encoder *e, uint8_t *ys, uint8_t *uvs, int row0, int row1) {
  const FrameParams *fp = ENC_PARAMS(e);
  int cols = e->mi_cols, rows = e->mi_rows;
  BlockInfo *blocks = ENC_BLOCKS(e);
  const SbInfo *sbs = ENC_SBS(e);
  int s0 = row0 * e->sb_cols, s1 = row1 * e->sb_cols;
  for (int s = s0; s < s1; s++)
    for (int i = 0; i < sbs[s].num_blocks; i++) {
      const BlockInfo *b = &blocks[sbs[s].first_block + i];
      int ysm = smooth_mode(b->y_mode) && !b->is_inter, uvsm = !b->is_inter && smooth_mode(b->uv_mode);
      for (int r = b->mi_row; r < b->mi_row + Num_4x4_Blocks_High[b->bsize] && r < rows; r++)
        for (int c = b->mi_col; c < b->mi_col + Num_4x4_Blocks_Wide[b->bsize] && c < cols; c++) {
          ys[r * cols + c] = (uint8_t)ysm;
          uvs[r * cols + c] = (uint8_t)uvsm;
        }
    }
  for (int s = s0; s < s1; s++)
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
}

static void edge_filter_types(Mav1Encoder *e) {
  uint8_t *ys = calloc((size_t)e->mi_rows * e->mi_cols, 1), *uvs = calloc((size_t)e->mi_rows * e->mi_cols, 1);
  edge_types_rows(e, ys, uvs, 0, e->sb_rows);
  free(ys);
  free(uvs);
}

static int env_or(const char *n, int d);
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
  fp->ref2_frame = GOLDEN_FRAME;
  if (e->cfg.altref) { /* frames inside a mini-GOP: ALTREF */
    fp->golden = !isKey && e->frame_kind >= 2;
    fp->ref2_frame = ALTREF_FRAME;
    fp->compound = e->hdr.reference_select; /* LAST + ALTREF */
  }
  fp->fast = e->cfg.fast;
  /* turbo savings (testing knob MAV1_TURBO): 1 square partitions only on
   * inter frames; motion search bits T_ME_* of k_me.h (2, 4, 8 cost
   * quality, off by default; 16 quarter-pel planes, 32 decimated coarse grid,
   * 64 children +-2, 512 64x64 refinement +-2); 128 reduced intra search in
   * inter frames; 256 square partitions on key frames too; 1024 no 8x8 in leaf frames (off by
   * default); 2048 two CDEF sets (off and MAV1_CP, default preset 2); 4096, 8192, 16384 (off),
   * 32768 half-pel only for 64x64 / 32x32 search nodes (k_me.h) */
  fp->turbo = e->cfg.turbo ? (getenv("MAV1_TURBO") ? (int)strtol(getenv("MAV1_TURBO"), NULL, 0) : 0x8BF1) : 0;
  fp->leaf = e->cfg.leaf ? (e->leafopt & 1) : e->nonref && !getenv("MAV1_LEAFINTRA");
  if (e->leafopt & 16) fp->tx_search = 0;
  /* turbo depth pruning (SVT-style): a node is not split when an ancestor's
   * NONE cost per sample is below depth_thr / 16, the threshold growing with
   * the quantizer (MAV1_DT: the value at qindex 128) */
  {
    int dt = getenv("MAV1_DT") ? atoi(getenv("MAV1_DT")) : e->sb_rows >= 16 ? 288 : 352;
    int bias = getenv("MAV1_DTB") ? atoi(getenv("MAV1_DTB")) : 0;
    fp->depth_thr = e->cfg.turbo ? dt * (q + bias) / (128 + bias) : 0;
  }
  fp->tile_w_sb = e->tile_w_sb;
  long acq = fp->ac_q[0];
  fp->lambda = (int32_t)(acq * acq * 256 / 430);
  fp->frame_w = e->w;
  fp->frame_h = e->h;
  fp->me_range = e->cfg.me_range;
  if ((e->leafopt & 32) && fp->me_range > 16) fp->me_range = 16;

  /* Loop filter level from q (libaom's LPF_PICK_FROM_Q rule for 8-bit). */
  FrameHdr *h = &e->hdr;
  int lvl = 0;
  if (e->cfg.loop_filter && q > 0 && !(e->cfg.leaf ? (e->leafopt & 2) : e->nonref && !getenv("MAV1_LEAFLF"))) {
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
  if (e->leafopt & 4) { /* CDEF off for this frame: one all-zero strength */
    h->cdef_damping = fp->cdef_damping = 3;
    h->cdef_bits = fp->cdef_bits = 0;
  }
  int pi[4] = {0, 1, 2, 3};
  if ((fp->turbo & 2048) && h->cdef_bits == 2) { /* turbo: two sets, off and one strength */
    h->cdef_bits = fp->cdef_bits = 1;
    pi[1] = env_or("MAV1_CP", 2);
  }
  for (int i = 0; i < 4; i++) {
    const int *ps = presets[pi[i]];
    h->cdef_y_pri[i] = fp->cdef_y_pri[i] = ps[0];
    h->cdef_y_sec[i] = fp->cdef_y_sec[i] = ps[1];
    h->cdef_uv_pri[i] = fp->cdef_uv_pri[i] = ps[2];
    h->cdef_uv_sec[i] = fp->cdef_uv_sec[i] = ps[3];
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
  dispatch_apply((size_t)e->num_sbs, DISPATCH_APPLY_AUTO, ^(size_t sbi) {
    int sb = (int)sbi;
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
  });
}

const char *const mav1_tool_names[16] = {"intra", "inter", "cfl", "angle_delta", "tx_non_dct", "skip",
                                          "rect_part", "compound", "obmc", "lr_wiener_units", "lr_sgr_units", "golden", "altref", "", "", ""};

static void count_tools(Mav1FrameStats *st, const BlockInfo *b) {
  st->tool_use[0] += !b->is_inter;
  st->tool_use[1] += b->is_inter;
  st->tool_use[2] += !b->is_inter && b->uv_mode == UV_CFL_PRED;
  st->tool_use[3] += !b->is_inter && (b->angle_y != 0 || b->angle_uv != 0);
  st->tool_use[4] += b->eob[0] > 0 && b->tx_type != DCT_DCT;
  st->tool_use[5] += b->eob[0] == 0 && b->eob[1] == 0 && b->eob[2] == 0;
  st->tool_use[6] += Block_Width(b->bsize) != Block_Height(b->bsize);
  st->tool_use[7] += b->is_inter && b->ref_frame1 > INTRA_FRAME;
  st->tool_use[11] += b->is_inter && b->ref_frame == GOLDEN_FRAME;
  st->tool_use[12] += b->is_inter && b->ref_frame == ALTREF_FRAME;
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
  if (!isKey && e->cfg.altref) off = e->frame_kind == 1 ? env_or("MAV1_QA", -8) : env_or("MAV1_QP", 4);
  if (!isKey && e->cfg.leaf && e->frame_kind == 3) off = env_or("MAV1_QM", 0); /* pyramid middle frame */
  if (!isKey && e->cfg.leaf && e->nonref) off = env_or("MAV1_QL", 8);
  q += off;
  return q < 1 ? 1 : q > 255 ? 255 : q;
}

/* Runs the stages of one frame up to the entropy coding (header fields in
 * e->hdr, source loaded). The final reconstruction is left in `rec`. */
static void run_frame_core(Mav1Encoder *e, int isKey) {
  setup_params(e, isKey);

  double t0 = now(), tme, tan, t1, t2;
  if (e->cfg.turbo && e->be->frame_core) {
    e->be->frame_core(e->be, e); /* stages overlapped (timed as recon) */
    tme = tan = t1 = t0;
    t2 = now();
  } else {
    if (!isKey) e->be->motion_search(e->be, e);
    tme = now();
    e->be->analyze(e->be, e);
    tan = now();
    if (e->cfg.intra_edge_filter) edge_filter_types(e);
    t1 = now();
    e->be->recon(e->be, e);
    t2 = now();
  }
  build_filter_info(e);
  int cdefOff = (e->leafopt & 4) != 0, nPresets = e->num_cdef_presets;
  if (nPresets) e->num_cdef_presets = 1 << e->hdr.cdef_bits;
  if (cdefOff) { /* the filter stage skips CDEF; every superblock codes index 0 */
    e->num_cdef_presets = 0;
    memset(e->cdef_choice, 0, (size_t)e->num_sbs);
  }
  e->be->filter(e->be, e);
  e->num_cdef_presets = nPresets;
  int useLr = e->cfg.restoration && e->cfg.qindex > 0 && !(e->leafopt & 8); /* AllLossless disables it */
  if (useLr) memcpy(e->dbk.base, e->rec.base, e->frame_bytes); /* UpscaledCurrFrame */
  if (e->cfg.cdef && e->cfg.qindex > 0 && !cdefOff) cdef_select(e);
  if (useLr) {
    lr_select(e);
    memcpy(e->lrout.base, e->rec.base, e->frame_bytes);
    e->be->restore(e->be, e);
    Buf t = e->rec;
    e->rec = e->lrout;
    e->lrout = t;
  }
  if (e->gate_out) {
    dispatch_semaphore_signal((dispatch_semaphore_t)e->gate_out);
    e->gate_out = NULL;
  }
  double t3 = now();
  e->last.t_analysis = t1 - t0;
  e->last.t_recon = t2 - t1;
  e->last.t_filter = t3 - t2;
  e->last.t_entropy = 0;
  if (getenv("MAV1_TIMING"))
    fprintf(stderr, "kind %d nonref %d: me %.1f ana %.1f edge %.1f rec %.1f flt %.1f ms\n", e->frame_kind, e->nonref,
            (tme - t0) * 1e3, (tan - tme) * 1e3, (t1 - tan) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3);
}

/* Entropy codes the frame described by s (the encoder or a writer
 * snapshot) and adds its block statistics to *st. */
static size_t frame_tail(Mav1Encoder *s, uint8_t **tile, Mav1FrameStats *st) {
  double t3 = now();
  size_t tileSize = write_tile(s, tile);
  const SbInfo *sbs = ENC_SBS(s);
  const BlockInfo *blocks = ENC_BLOCKS(s);
  for (int i = 0; i < s->num_sbs; i++) {
    st->num_blocks += sbs[i].num_blocks;
    for (int k = 0; k < sbs[i].num_blocks; k++) {
      const BlockInfo *b = &blocks[sbs[i].first_block + k];
      st->num_inter_blocks += b->is_inter;
      count_tools(st, b);
    }
  }
  st->t_entropy += now() - t3;
  if (getenv("MAV1_TIMING")) fprintf(stderr, "  ent %.1f ms\n", (now() - t3) * 1e3);
  return tileSize;
}

/* Runs every stage of one frame; returns the tile data. */
static size_t run_frame(Mav1Encoder *e, int isKey, uint8_t **tile) {
  run_frame_core(e, isKey);
  return frame_tail(e, tile, &e->last);
}

/* Hands the coded frame's writer inputs to a writer context: its buffers
 * are swapped with the encoder's (the next frame's core writes the others)
 * and a snapshot of the encoder state is taken. */
static Mav1Encoder *snap_take(Mav1Encoder *e, WCtx **out) {
  WCtx *c = &e->wctx[e->wnext];
  e->wnext = (e->wnext + 1) % NUM_WCTX;
  dispatch_semaphore_wait((dispatch_semaphore_t)c->free, DISPATCH_TIME_FOREVER);
  Buf t;
#define SWAPB(f) t = e->f; e->f = c->f; c->f = t;
  SWAPB(blocks) SWAPB(sbs) SWAPB(coefs) SWAPB(params) SWAPB(lrunits)
#undef SWAPB
  memcpy(c->cdef_choice, e->cdef_choice, (size_t)e->num_sbs);
  *c->snap = *e;
  Mav1Encoder *s = c->snap;
  s->blocks = c->blocks;
  s->sbs = c->sbs;
  s->coefs = c->coefs;
  s->params = c->params;
  s->lrunits = c->lrunits;
  s->mi = c->mi;
  s->cdef_idx = c->cdef_idx;
  s->cdef_choice = c->cdef_choice;
  *out = c;
  return s;
}

static int wctx_init(Mav1Encoder *e) {
  int ok = 0;
  for (int i = 0; i < NUM_WCTX; i++) {
    WCtx *c = &e->wctx[i];
    ok |= buf_alloc(&c->params, e->params.size);
    ok |= buf_alloc(&c->blocks, e->blocks.size);
    ok |= buf_alloc(&c->sbs, e->sbs.size);
    ok |= buf_alloc(&c->coefs, e->coefs.size);
    ok |= buf_alloc(&c->lrunits, e->lrunits.size);
    c->mi = calloc((size_t)e->mi_rows * e->mi_cols, sizeof(MiInfo));
    c->cdef_idx = calloc((size_t)e->num_sbs, 1);
    c->cdef_choice = calloc((size_t)e->num_sbs, 1);
    c->snap = calloc(1, sizeof(Mav1Encoder));
    c->free = (void *)dispatch_semaphore_create(1);
  }
  e->wq = (void *)dispatch_queue_create("mav1.writer", DISPATCH_QUEUE_SERIAL);
  pthread_mutex_t *mu = malloc(sizeof *mu);
  pthread_mutex_init(mu, NULL);
  e->outmu = mu;
  return ok;
}

static void wctx_free(Mav1Encoder *e) {
  if (!e->wq) return;
  dispatch_sync((dispatch_queue_t)e->wq, ^{});
  for (int i = 0; i < NUM_WCTX; i++) {
    WCtx *c = &e->wctx[i];
    buf_free(&c->params);
    buf_free(&c->blocks);
    buf_free(&c->sbs);
    buf_free(&c->coefs);
    buf_free(&c->lrunits);
    free(c->mi);
    free(c->cdef_idx);
    free(c->cdef_choice);
    free(c->snap);
    if (c->free) dispatch_release((dispatch_semaphore_t)c->free);
  }
  dispatch_release((dispatch_queue_t)e->wq);
  e->wq = NULL;
  pthread_mutex_destroy(e->outmu);
  free(e->outmu);
}

/* Runs op on the writer queue (in order with the frames' entropy coding),
 * or inline. */
static void q_run(Mav1Encoder *e, void (^op)(void)) {
  if (e->wq) dispatch_async((dispatch_queue_t)e->wq, op);
  else op();
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
  h->show_frame = 1;
  /* ref_frame_idx: LAST -> slot 0, GOLDEN -> slot 1 (multi_ref), others slot 0 */
  if (e->cfg.multi_ref) h->ref_idx[GOLDEN_FRAME - LAST_FRAME] = 1;
  /* GOLDEN (slot 1): key frames and every 4th frame (the finer-q ones) */
  e->golden_refresh = !isKey && !e->nonref && e->cfg.multi_ref && e->order_hint % 4 == 0;
  h->refresh_frame_flags = isKey ? 0xFF : (e->nonref ? 0x00 : 0x01 | (e->golden_refresh ? 0x02 : 0));

  uint8_t *tile = NULL;
  size_t tileSize = run_frame(e, isKey, &tile);
  double t3 = now();
  size_t cap = tileSize + 1024;
  uint8_t *buf = malloc(cap);
  size_t pos = write_temporal_delimiter(buf);
  if (isKey) pos += write_sequence_header(e, buf + pos);
  pos += write_frame_obu(e, tile, tileSize, buf + pos);
  free(tile);
  e->last.t_entropy += now() - t3;
  e->last.bytes = pos;

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

static void copy_out(const Mav1Encoder *e, const uint8_t *frame, uint8_t *const planes[3], const int strides[3]) {
  for (int p = 0; p < 3; p++) {
    int pw = p ? (e->w + 1) >> 1 : e->w;
    int ph = p ? (e->h + 1) >> 1 : e->h;
    for (int y = 0; y < ph; y++)
      memcpy(planes[p] + (size_t)y * strides[p], frame + e->offset[p] + (size_t)y * e->stride[p], pw);
  }
}

static size_t packed_size(const Mav1Encoder *e) {
  size_t cw = (size_t)(e->w + 1) >> 1, ch = (size_t)(e->h + 1) >> 1;
  return (size_t)e->w * e->h + 2 * cw * ch;
}

/* Packed I420 copy of a padded frame buffer. */
static uint8_t *pack_frame(const Mav1Encoder *e, const uint8_t *frame) {
  uint8_t *o = malloc(packed_size(e));
  int cw = (e->w + 1) >> 1, ch = (e->h + 1) >> 1;
  uint8_t *pl[3] = {o, o + (size_t)e->w * e->h, o + (size_t)e->w * e->h + (size_t)cw * ch};
  int st[3] = {e->w, cw, cw};
  copy_out(e, frame, pl, st);
  return o;
}

void mav1_get_recon(const Mav1Encoder *e, uint8_t *const planes[3], const int strides[3]) {
  if (e->cfg.altref > 0) {
    int cw = (e->w + 1) >> 1, ch = (e->h + 1) >> 1;
    const uint8_t *s = e->shown_recon;
    if (!s) return;
    const uint8_t *src[3] = {s, s + (size_t)e->w * e->h, s + (size_t)e->w * e->h + (size_t)cw * ch};
    for (int p = 0; p < 3; p++) {
      int pw = p ? cw : e->w, ph = p ? ch : e->h;
      for (int y = 0; y < ph; y++) memcpy(planes[p] + (size_t)y * strides[p], src[p] + (size_t)y * pw, pw);
    }
    return;
  }
  /* After mav1_encode the last reconstruction lives in `ref`. */
  copy_out(e, e->nonref ? e->rec.base : e->ref.base, planes, strides);
}

/* ------------------------------------------------------------------------ */
/* ALTREF mini-GOPs: for display frames d+1 .. d+L (d the last shown         */
/* reference) the frame d+L is coded first as a hidden ALTREF, then d+1 ..  */
/* d+L-1 are coded as shown frames predicting from LAST (past) and ALTREF   */
/* (future); a show_existing_frame header finally displays the ALTREF.      */
/* Slots: 0 holds the frames inside the GOP, 2 / 3 alternate as the ALTREF  */
/* slot (the previous ALTREF is the next GOP's anchor).                     */

static int rel_dist(int a, int b) {
  int d = (a - b) & ((1 << 7) - 1); /* OrderHintBits = 7 */
  return d >= 64 ? d - 128 : d;
}

/* Temporal unit being assembled by the ALTREF driver. Hidden frames are
 * appended to it; a shown frame (or show_existing_frame header) completes
 * it. Statistics of all frames coded into it are summed. With the writer
 * queue every append runs on the queue, in coding order. */
typedef struct { uint8_t *buf; size_t pos, cap; Mav1FrameStats st; } Tu;

static void tu_reserve(Tu *t, size_t n) {
  if (t->pos + n > t->cap) {
    t->cap = t->pos + n;
    t->buf = realloc(t->buf, t->cap);
  }
}

/* A new temporal unit: with the temporal delimiter unless omitTd (a key
 * frame writes TD and sequence header itself). */
static Tu *tu_new(int frameType, int omitTd) {
  Tu *t = calloc(1, sizeof *t);
  t->cap = 1024;
  t->buf = malloc(t->cap);
  if (!omitTd) t->pos = write_temporal_delimiter(t->buf);
  t->st.frame_type = frameType;
  return t;
}

static void alt_push(Mav1Encoder *e, uint8_t *data, size_t size, uint8_t *recon, const Mav1FrameStats *st) {
  if (e->outmu) pthread_mutex_lock(e->outmu);
  if (e->out_n == e->out_cap) {
    e->out_cap = e->out_cap ? 2 * e->out_cap : 8;
    e->outq = realloc(e->outq, sizeof(*e->outq) * e->out_cap);
  }
  struct AltPacket *p = &e->outq[e->out_n++];
  p->data = data;
  p->size = size;
  p->recon = recon;
  p->st = *st;
  p->st.bytes = size;
  if (e->outmu) pthread_mutex_unlock(e->outmu);
}

/* Completes the temporal unit (queued after its frames). */
static void tu_close(Mav1Encoder *e, Tu *t, uint8_t *recon) {
  q_run(e, ^{
    alt_push(e, t->buf, t->pos, recon, &t->st);
    free(t);
  });
}

static void tu_show_existing(Mav1Encoder *e, Tu *t, int slot, uint8_t *recon) {
  q_run(e, ^{
    tu_reserve(t, 64);
    t->pos += write_show_existing(slot, t->buf + t->pos);
  });
  tu_close(e, t, recon);
}

static void stats_add(Mav1FrameStats *a, const Mav1FrameStats *b) {
  a->num_blocks += b->num_blocks;
  a->num_inter_blocks += b->num_inter_blocks;
  for (int k = 0; k < 16; k++) a->tool_use[k] += b->tool_use[k];
  a->t_analysis += b->t_analysis;
  a->t_recon += b->t_recon;
  a->t_filter += b->t_filter;
  a->t_entropy += b->t_entropy;
}

/* Codes one frame of the ALTREF driver: frame (packed I420, display index
 * disp), with LAST from lastSlot and ALTREF from arfSlot (-1: none).
 * Appends the frame OBU (and, for key frames, the TD and sequence header)
 * to the temporal unit t; with the writer queue the entropy coding and the
 * append run there, after the frame's core. With frame engines the frame is
 * coded by an engine, concurrently with the driver and the other engine.
 * Returns the packed reconstruction (complete once the frame's OBU is
 * appended). */
static uint8_t *alt_code(Mav1Encoder *e, uint8_t **framep, int disp, int isKey, int kind, int lastSlot,
                         int arfSlot, int refresh, int hidden, Tu *t) {
  if (isKey) e->key_disp = disp;
  e->nonref = e->cfg.leaf && !isKey && refresh == 0;
  e->leafopt = e->nonref ? e->leafopt_cfg : 0;
  e->frame_kind = isKey ? 0 : kind;
  e->order_hint = disp - e->key_disp;
  FrameHdr *h = &e->hdr;
  memset(h, 0, sizeof(*h));
  h->frame_type = isKey ? KEY_FRAME : INTER_FRAME;
  h->base_q_idx = frame_qindex(e, isKey);
  h->order_hint = e->order_hint;
  h->show_frame = !hidden;
  h->showable_frame = hidden;
  h->refresh_frame_flags = isKey ? 0xFF : refresh;
  memset(e->sign_bias, 0, sizeof e->sign_bias);
  if (!isKey) {
    for (int i = 0; i < REFS_PER_FRAME; i++) h->ref_idx[i] = lastSlot;
    if (arfSlot >= 0) h->ref_idx[ALTREF_FRAME - LAST_FRAME] = arfSlot;
    for (int i = 0; i < REFS_PER_FRAME; i++)
      e->sign_bias[LAST_FRAME + i] = rel_dist(e->slot_hint[h->ref_idx[i]], h->order_hint) > 0;
    /* compound LAST + ALTREF (BIDIR, COMPOUND_AVERAGE) inside the mini-GOP;
     * skip_mode_params( ): LAST is a forward and ALTREF a backward
     * reference, so skip_mode_present is coded */
    h->reference_select = kind >= 2 && arfSlot >= 0 && !getenv("MAV1_NOCOMP");
    h->skip_mode_allowed = h->reference_select;
  }
  /* the engine coding the frame (the driver itself without engines) */
  static double tl0;
  int tl = getenv("MAV1_TL") != NULL;
  if (tl && !tl0) tl0 = now();
  if (tl) fprintf(stderr, "%7.1f drv  d%d k%d\n", (now() - tl0) * 1e3, disp, kind);
  /* reference frames on engine 0 (they form a chain), the others
   * alternately on engines 1 and 2 */
  int ei = h->refresh_frame_flags ? 0 : e->eng_next;
  Mav1Encoder *E = e->eng[0] ? e->eng[ei] : e;
  if (e->eng[0] && ei) e->eng_next = 3 - ei;
  int rf = h->refresh_frame_flags;
  FrameHdr hdr = *h;
  struct { int v[8]; } sb;
  memcpy(sb.v, e->sign_bias, sizeof sb.v);
  int keyDisp = e->key_disp, nonref = e->nonref, leafopt = e->leafopt, frameKind = e->frame_kind,
      orderHint = e->order_hint;
  unsigned refGen = isKey ? 0 : e->slot_gen[lastSlot], goldGen = !isKey && arfSlot >= 0 ? e->slot_gen[arfSlot] : 0;
  if (rf) e->gen++;
  for (int k = 0; k < 8; k++)
    if ((rf >> k) & 1) {
      e->slot_gen[k] = e->gen;
      e->slot_hint[k] = h->order_hint;
    }
  uint8_t *frame = *framep;
  *framep = NULL;
  int cw = (e->w + 1) >> 1, ch = (e->h + 1) >> 1;
  uint8_t *recon = malloc(packed_size(e));
  typedef struct { Mav1Encoder *s; WCtx *c; Mav1FrameStats st; } Job;
  Job *job = calloc(1, sizeof *job);
  /* slot dependencies (engines): the versions read and their reader
   * groups; for each refreshed slot the previous version, its readers and
   * the new version */
  int nr = isKey ? 0 : arfSlot >= 0 ? 2 : 1, rs_[2] = {lastSlot, arfSlot};
  struct { void *ver[2], *rd[2], *pver[8], *prd[8], *nver[8]; } dep;
  memset(&dep, 0, sizeof dep);
  if (e->eng[0]) {
    for (int i = 0; i < nr; i++) {
      int k = rs_[i];
      if (e->slot_ver[k]) dispatch_retain((dispatch_group_t)(dep.ver[i] = e->slot_ver[k]));
      if (!e->slot_rd[k]) e->slot_rd[k] = (void *)dispatch_group_create();
      dispatch_group_enter((dispatch_group_t)e->slot_rd[k]);
      dispatch_retain((dispatch_group_t)(dep.rd[i] = e->slot_rd[k]));
    }
    for (int k = 0; k < 8; k++)
      if ((rf >> k) & 1) {
        dep.pver[k] = e->slot_ver[k]; /* the references move to dep */
        dep.prd[k] = e->slot_rd[k];
        dispatch_group_t g = dispatch_group_create();
        dispatch_group_enter(g);
        dispatch_retain(g);
        dep.nver[k] = (void *)g;
        e->slot_ver[k] = (void *)g;
        e->slot_rd[k] = NULL;
      }
  }
  void (^setup)(void) = ^{
    for (int i = 0; i < nr; i++) /* the references are written */
      if (dep.ver[i]) {
        dispatch_group_wait((dispatch_group_t)dep.ver[i], DISPATCH_TIME_FOREVER);
        dispatch_release((dispatch_group_t)dep.ver[i]);
      }
    E->hdr = hdr;
    memcpy(E->sign_bias, sb.v, sizeof sb.v);
    E->key_disp = keyDisp;
    E->nonref = nonref;
    E->leafopt = leafopt;
    E->frame_kind = frameKind;
    E->order_hint = orderHint;
    const uint8_t *pl[3] = {frame, frame + (size_t)e->w * e->h, frame + (size_t)e->w * e->h + (size_t)cw * ch};
    int st[3] = {e->w, cw, cw};
    load_source(E, pl, st);
    free(frame);
    if (!isKey) {
      memcpy(E->ref.base, e->slot[lastSlot].base, e->frame_bytes);
      if (arfSlot >= 0) memcpy(E->gold.base, e->slot[arfSlot].base, e->frame_bytes);
      E->ref_gen = refGen;
      E->gold_gen = goldGen;
    }
    for (int i = 0; i < nr; i++)
      if (dep.rd[i]) {
        dispatch_group_leave((dispatch_group_t)dep.rd[i]);
        dispatch_release((dispatch_group_t)dep.rd[i]);
      }
  };
  /* the frame's core; then the slot writes and the writer job's inputs */
  dispatch_semaphore_t coreDone = e->eng[0] ? dispatch_semaphore_create(0) : NULL;
  void (^core)(void) = ^{
    if (tl) fprintf(stderr, "%7.1f e%d   d%d core start\n", (now() - tl0) * 1e3, ei, disp);
    memset(&E->last, 0, sizeof E->last);
    run_frame_core(E, isKey);
    if (tl) fprintf(stderr, "%7.1f e%d   d%d core end\n", (now() - tl0) * 1e3, ei, disp);
    job->st = E->last;
    job->s = E->wq ? snap_take(E, &job->c) : E;
    for (int k = 0; k < 8; k++)
      if ((rf >> k) & 1) {
        /* the previous picture's write and its readers are done */
        if (dep.pver[k]) {
          dispatch_group_wait((dispatch_group_t)dep.pver[k], DISPATCH_TIME_FOREVER);
          dispatch_release((dispatch_group_t)dep.pver[k]);
        }
        if (dep.prd[k]) {
          dispatch_group_wait((dispatch_group_t)dep.prd[k], DISPATCH_TIME_FOREVER);
          dispatch_release((dispatch_group_t)dep.prd[k]);
        }
        if (e->slot[k].base) memcpy(e->slot[k].base, E->rec.base, e->frame_bytes);
      }
    for (int k = 0; k < 8; k++)
      if (dep.nver[k]) {
        dispatch_group_leave((dispatch_group_t)dep.nver[k]);
        dispatch_release((dispatch_group_t)dep.nver[k]);
      }
    uint8_t *rp[3] = {recon, recon + (size_t)e->w * e->h, recon + (size_t)e->w * e->h + (size_t)cw * ch};
    int rs[3] = {e->w, cw, cw};
    copy_out(E, E->rec.base, rp, rs);
  };
  if (e->eng[0]) {
    /* GPU priority: a reference frame releases the pending non-reference
     * frame when its GPU stages are done; a non-reference frame waits */
    dispatch_semaphore_t gateIn = NULL;
    if (!getenv("MAV1_NOGATE")) {
      if (rf) {
        void *g = e->gate_pending;
        e->gate_pending = NULL;
        /* handed over inside the job (E->gate_out belongs to the engine) */
        dispatch_group_async((dispatch_group_t)e->eng_busy[ei], (dispatch_queue_t)e->eng_q[ei], ^{ E->gate_out = g; });
      } else {
        if (e->gate_pending) dispatch_semaphore_signal((dispatch_semaphore_t)e->gate_pending);
        gateIn = dispatch_semaphore_create(0);
        e->gate_pending = (void *)gateIn;
      }
    }
    dispatch_semaphore_wait((dispatch_semaphore_t)e->inflight, DISPATCH_TIME_FOREVER);
    dispatch_group_async((dispatch_group_t)e->eng_busy[ei], (dispatch_queue_t)e->eng_q[ei], ^{
      setup();
      if (gateIn) {
        dispatch_semaphore_wait(gateIn, DISPATCH_TIME_FOREVER);
        dispatch_release(gateIn);
      }
      core();
      dispatch_semaphore_signal(coreDone);
    });
  } else {
    setup();
    core();
  }
  q_run(e, ^{
    if (coreDone) {
      dispatch_semaphore_wait(coreDone, DISPATCH_TIME_FOREVER);
      dispatch_release(coreDone);
    }
    Mav1Encoder *s = job->s;
    Mav1FrameStats st2 = job->st;
    uint8_t *tile = NULL;
    size_t tileSize = frame_tail(s, &tile, &st2);
    double t3 = now();
    tu_reserve(t, tileSize + 1024);
    if (isKey) {
      t->pos += write_temporal_delimiter(t->buf + t->pos);
      t->pos += write_sequence_header(s, t->buf + t->pos);
    }
    t->pos += write_frame_obu(s, tile, tileSize, t->buf + t->pos);
    free(tile);
    st2.t_entropy += now() - t3;
    stats_add(&t->st, &st2);
    if (job->c) dispatch_semaphore_signal((dispatch_semaphore_t)job->c->free);
    free(job);
    if (e->inflight) dispatch_semaphore_signal((dispatch_semaphore_t)e->inflight);
    if (tl) fprintf(stderr, "%7.1f wq   d%d written\n", (now() - tl0) * 1e3, disp);
  });
  return recon;
}

/* Codes the display frames strictly between a (slot as) and b (slot bs),
 * both already coded. With 3 or more frames the middle one is coded first
 * as a hidden reference (slot 4 + depth, predicting from a and b), then each
 * half recursively; otherwise the frames are non-reference leaf frames. */
static Tu *pyr_code(Mav1Encoder *e, Tu *t, int a, int as, int b, int bs, int depth) {
  int d0 = e->in_disp - e->inq_n;
  if (b - a - 1 >= 3 && depth < 4) {
    int m = (a + b) / 2, ms = 4 + depth;
    if (!t) t = tu_new(1, 0);
    uint8_t *rec = alt_code(e, &e->inq[m - d0], m, 0, 3, as, bs, 1 << ms, 1, t);
    t = pyr_code(e, t, a, as, m, ms, depth + 1);
    if (!t) t = tu_new(1, 0);
    tu_show_existing(e, t, ms, rec);
    return pyr_code(e, NULL, m, ms, b, bs, depth + 1);
  }
  for (int d = a + 1; d < b; d++) {
    if (!t) t = tu_new(1, 0);
    tu_close(e, t, alt_code(e, &e->inq[d - d0], d, 0, 2, as, bs, 0, 0, t));
    t = NULL;
  }
  return t;
}

/* Codes the queued frames that can be coded (all of them when flushing). */
static void alt_drain(Mav1Encoder *e, int flush) {
  int N = e->cfg.altref;
  while (e->inq_n > 0) {
    int d0 = e->in_disp - e->inq_n;
    int isKey = d0 == 0 || (e->cfg.keyint > 0 && d0 % e->cfg.keyint == 0);
    int L = isKey ? 1 : N;
    if (!isKey && e->cfg.keyint > 0) L = k_min(L, (d0 / e->cfg.keyint + 1) * e->cfg.keyint - d0);
    if (e->inq_n < L) {
      if (!flush) return;
      L = e->inq_n;
    }
    if (isKey || L == 1) {
      /* key frame, or a regular frame predicting from the anchor */
      Tu *t = tu_new(!isKey, isKey);
      uint8_t *rec = alt_code(e, &e->inq[0], d0, isKey, 0, e->anchor_slot, -1, 0x01, 0, t);
      e->anchor_slot = 0;
      tu_close(e, t, rec);
    } else if (e->cfg.leaf) {
      int arfSlot = e->anchor_slot == 2 ? 3 : 2;
      Tu *t = tu_new(1, 0);
      uint8_t *arfRecon = alt_code(e, &e->inq[L - 1], d0 + L - 1, 0, 1, e->anchor_slot, -1, 1 << arfSlot, 1, t);
      t = pyr_code(e, t, d0 - 1, e->anchor_slot, d0 + L - 1, arfSlot, 0);
      if (!t) t = tu_new(1, 0);
      tu_show_existing(e, t, arfSlot, arfRecon);
      e->anchor_slot = arfSlot;
    } else {
      int arfSlot = e->anchor_slot == 2 ? 3 : 2;
      Tu *t = tu_new(1, 0);
      uint8_t *arfRecon = alt_code(e, &e->inq[L - 1], d0 + L - 1, 0, 1, e->anchor_slot, -1, 1 << arfSlot, 1, t);
      for (int i = 0; i < L - 1; i++) {
        if (i > 0) t = tu_new(1, 0);
        tu_close(e, t, alt_code(e, &e->inq[i], d0 + i, 0, 2, i == 0 ? e->anchor_slot : 0, arfSlot, 0x01, 0, t));
      }
      t = tu_new(1, 0);
      tu_show_existing(e, t, arfSlot, arfRecon);
      e->anchor_slot = arfSlot;
    }
    for (int i = 0; i < L; i++) free(e->inq[i]);
    memmove(e->inq, e->inq + L, sizeof(*e->inq) * (e->inq_n - L));
    e->inq_n -= L;
  }
}

int mav1_send_frame(Mav1Encoder *e, const uint8_t *const planes[3], const int strides[3]) {
  if (e->cfg.altref <= 0) {
    if (!planes) return 0;
    if (e->pending) free(e->pend_data);
    e->pending = 1;
    return mav1_encode(e, planes, strides, &e->pend_data, &e->pend_size);
  }
  if (planes) {
    uint8_t *f = malloc(packed_size(e));
    int cw = (e->w + 1) >> 1, ch = (e->h + 1) >> 1;
    for (int p = 0; p < 3; p++) {
      int pw = p ? cw : e->w, ph = p ? ch : e->h;
      uint8_t *d = f + (p ? (size_t)e->w * e->h + (p - 1) * (size_t)cw * ch : 0);
      for (int y = 0; y < ph; y++) memcpy(d + (size_t)y * pw, planes[p] + (size_t)y * strides[p], pw);
    }
    e->inq[e->inq_n++] = f;
    e->in_disp++;
  }
  if (!planes) e->flushing = 1;
  alt_drain(e, planes == NULL);
  if (e->flushing && e->gate_pending) { /* no reference frame follows */
    dispatch_semaphore_signal((dispatch_semaphore_t)e->gate_pending);
    e->gate_pending = NULL;
  }
  return 0;
}

int mav1_receive_packet(Mav1Encoder *e, uint8_t **out, size_t *outSize) {
  if (e->cfg.altref <= 0) {
    if (!e->pending) return 1;
    e->pending = 0;
    *out = e->pend_data;
    *outSize = e->pend_size;
    return 0;
  }
  /* with the writer queue, packets come out as their frames complete;
   * when flushing, every queued one is waited for */
  if (e->wq && e->flushing) dispatch_sync((dispatch_queue_t)e->wq, ^{});
  if (e->outmu) pthread_mutex_lock(e->outmu);
  if (e->out_pos == e->out_n) {
    e->out_pos = e->out_n = 0;
    if (e->outmu) pthread_mutex_unlock(e->outmu);
    return 1;
  }
  struct AltPacket p = e->outq[e->out_pos++];
  if (e->outmu) pthread_mutex_unlock(e->outmu);
  *out = p.data;
  *outSize = p.size;
  free(e->shown_recon);
  e->shown_recon = p.recon;
  e->last = p.st;
  return 0;
}
