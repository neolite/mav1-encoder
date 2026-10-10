/* Metal entry points. Each wraps a kernel shared with the CPU backend. */
#include "k_analysis.h"
#include "k_recon.h"
#include "k_me.h"
#include "k_lf.h"
#include "k_cdef.h"
#include "k_lr.h"

kernel void mav1_lf(device const FrameParams *fp [[buffer(0)]],
                    device uint8_t *rec [[buffer(1)]],
                    device const uint8_t *info [[buffer(2)]],
                    constant int &pass [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  int lines = pass == 0 ? fp->mi_rows : fp->mi_cols;
  if ((int)gid >= 3 * lines) return;
  int plane = (int)gid / lines, line = (int)gid % lines;
  if (plane > 0 && (line & 1)) return;
  lf_line(fp, rec, info, plane, pass, line);
}

kernel void mav1_lr(device const FrameParams *fp [[buffer(0)]],
                    device const uint8_t *cdef [[buffer(1)]],
                    device const uint8_t *dbk [[buffer(2)]],
                    device uint8_t *out [[buffer(3)]],
                    device const LrUnit *units [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
  int rows4 = (fp->frame_h + 3) >> 2, cols4 = (fp->frame_w + 3) >> 2;
  int n = rows4 * cols4;
  if ((int)gid >= 3 * n) return;
  int plane = (int)gid / n, i = (int)gid % n;
  lr_block(fp, plane, i / cols4, i % cols4, cdef, dbk, out, units);
}

kernel void mav1_cdef(device const FrameParams *fp [[buffer(0)]],
                      device const uint8_t *cur [[buffer(1)]],
                      device uint8_t *out [[buffer(2)]],
                      device const uint8_t *info [[buffer(3)]],
                      constant int &idx [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  int bc = (fp->mi_cols + 1) >> 1, br = (fp->mi_rows + 1) >> 1;
  if ((int)gid >= bc * br) return;
  cdef_block(fp, cur, out, info, ((int)gid / bc) * 2, ((int)gid % bc) * 2, idx);
}

/* SATD of 4x4 unit u of the node against the EIGHTTAP luma prediction at
 * (mvr, mvc); the prediction is evaluated per sample exactly as
 * predict_inter_block( ) computes it (same intermediate rounding). */
INLINE int me_unit_satd(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                        THR const MeNode *m, int mvr, int mvc, int u) {
  int stride = fp->stride[0];
  DEV const uint8_t *rp = ref + fp->offset[0];
  int nu = m->w >> 2;
  int ux = (u % nu) * 4, uy = (u / nu) * 4;
  int startX = (m->x << SUBPEL_BITS) + 2 * mvc;
  int startY = (m->y << SUBPEL_BITS) + 2 * mvr;
  /* unscaled reference: the spec's startX = (((x << 4) + 2 * mv) << 6) + 32 */
  startX = (startX << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) + 32;
  startY = (startY << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) + 32;
  int lastX = fp->frame_w - 1, lastY = fp->frame_h - 1;
  KCONST int32_t *fh = Subpel_Filters[EIGHTTAP][(startX >> 6) & SUBPEL_MASK];
  KCONST int32_t *fv = Subpel_Filters[EIGHTTAP][(startY >> 6) & SUBPEL_MASK];
  int hrow[11][4];
  for (int rr = 0; rr < 11; rr++) {
    int ry = Clip3(0, lastY, (startY >> 10) + uy + rr - 3);
    DEV const uint8_t *row = rp + ry * stride;
    for (int c = 0; c < 4; c++) {
      int sum = 0;
      int bx = (startX >> 10) + ux + c - 3;
      for (int t = 0; t < 8; t++) sum += fh[t] * row[Clip3(0, lastX, bx + t)];
      hrow[rr][c] = Round2(sum, 3);
    }
  }
  int32_t d[16];
  DEV const uint8_t *sp = src + fp->offset[0] + (m->y + uy) * stride + m->x + ux;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) {
      int sum = 0;
      for (int t = 0; t < 8; t++) sum += fv[t] * hrow[r + t][c];
      d[r * 4 + c] = (int)sp[r * stride + c] - Clip1(Round2(sum, 11));
    }
  return satd4x4(d);
}

/* One SIMD group (32 lanes) per search node of one level. */
kernel void mav1_me(device const FrameParams *fp [[buffer(0)]],
                    device const uint8_t *src [[buffer(1)]],
                    device const uint8_t *ref [[buffer(2)]],
                    device int32_t *mvField [[buffer(3)]],
                    device int32_t *scratch [[buffer(4)]],
                    constant int &level [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
  int nn = 1 << (2 * level);
  int sbIdx = (int)tg / nn, i = (int)tg % nn;
  if (sbIdx >= fp->sb_cols * fp->sb_rows) return;
  MeNode m = me_node(fp, sbIdx, level, i, mvField);
  device int32_t *out = mvField + (sbIdx * NUM_NODES + NODE_BASE(level) + i) * 2;
  if (!m.valid) {
    if (lane == 0) out[0] = out[1] = 0;
    return;
  }
  int dy = 0, dx = 0;
  for (int stage = 0; stage < m.stages; stage++) {
    if (stage > 0) me_refine_stage(&m, dy, dx);
    int best = INT_MAX, bestK = INT_MAX;
    for (int k = (int)lane; k < m.count; k += 32) {
      int cy, cx;
      me_candidate(&m, k, &cy, &cx);
      int c = me_cost(fp, src, ref, &m, cy, cx);
      if (c < best) {
        best = c;
        bestK = k;
      }
    }
    /* lexicographic (cost, k) minimum = the serial first minimum */
    int gmin = simd_min(best);
    int kmin = simd_min(best == gmin ? bestK : INT_MAX);
    me_candidate(&m, kmin, &dy, &dx);
  }
  /* sub-pel: the 9 candidates (centre + 8 neighbours) x 4x4 units are
   * spread over the lanes; per-candidate sums by simd_sum. The (cost, index)
   * minimum equals the serial first minimum of me_finish. */
  int mvr = dy * 8, mvc = dx * 8, bestS = 0;
  int nUnits = (m.w >> 2) * (m.w >> 2);
  for (int step = 4; step >= 2; step >>= 1) {
    int sums[ME_LANES];
    for (int k = 0; k < ME_LANES; k++) sums[k] = 0;
    int first = step == 4 ? 0 : 1; /* the quarter step reuses the centre cost */
    for (int w = (int)lane; w < (ME_LANES - first) * nUnits; w += 32) {
      int k = first + w / nUnits, u = w % nUnits;
      int dr = 0, dc = 0;
      if (k > 0) me_neighbour(k - 1, step, &dr, &dc);
      sums[k] += me_unit_satd(fp, src, ref, &m, mvr + dr, mvc + dc, u);
    }
    int bc = INT_MAX, bk = 0;
    for (int k = 0; k < ME_LANES; k++) {
      int c = (k == 0 && step == 2) ? bestS : simd_sum(sums[k]);
      if (k >= first || step == 2) {
        if (c < bc) { bc = c; bk = k; }
      }
    }
    bestS = bc;
    if (bk > 0) {
      int dr, dc;
      me_neighbour(bk - 1, step, &dr, &dc);
      mvr += dr;
      mvc += dc;
    }
  }
  if (lane == 0) {
    out[0] = mvr;
    out[1] = mvc;
  }
}

kernel void mav1_analyze(device const FrameParams *fp [[buffer(0)]],
                         device const uint8_t *src [[buffer(1)]],
                         device const uint8_t *ref [[buffer(2)]],
                         device const int32_t *mvField [[buffer(3)]],
                         device long *costs [[buffer(4)]],
                         device BlockInfo *infos [[buffer(5)]],
                         device int32_t *scratch [[buffer(6)]],
                         constant int &level [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
  /* pass 1: one thread per (superblock, node of the level, slot) */
  int ns = level_slots(fp, level), nn = 1 << (2 * level);
  int total = fp->sb_cols * fp->sb_rows * nn * ns;
  if ((int)gid >= total) return;
  int sb = (int)gid / (nn * ns), rem = (int)gid % (nn * ns);
  analyze_item(fp, sb, level, rem / ns, rem % ns, src, ref, mvField, costs, infos,
               scratch + (long)gid * analysis_slot_ints(level));
}

kernel void mav1_recon(device const FrameParams *fp [[buffer(0)]],
                       device const uint8_t *src [[buffer(1)]],
                       device uint8_t *rec [[buffer(2)]],
                       device const uint8_t *ref [[buffer(3)]],
                       device BlockInfo *blocks [[buffer(4)]],
                       device const SbInfo *sbs [[buffer(5)]],
                       device int32_t *coefs [[buffer(6)]],
                       device const int32_t *mats [[buffer(7)]],
                       device int32_t *scratch [[buffer(8)]],
                       constant int2 &wave [[buffer(9)]],   /* x: wave index t, y: first row */
                       uint gid [[thread_position_in_grid]]) {
  int r = wave.y + (int)gid, c = wave.x - 2 * r;
  if (r >= fp->sb_rows || c < 0 || c >= fp->sb_cols) return;
  int idx = r * fp->sb_cols + c;
  recon_sb(fp, sbs[idx], blocks, src, rec, ref, coefs, mats, scratch + (long)idx * SCRATCH_INTS);
}
