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
  /* one thread per (plane, line, 16-mi segment) */
  int lines = pass == 0 ? fp->mi_rows : fp->mi_cols, segs = ((pass == 0 ? fp->mi_cols : fp->mi_rows) + 15) >> 4;
  if ((int)gid >= 3 * lines * segs) return;
  int seg = (int)gid % segs, pl = (int)gid / segs;
  int plane = pl / lines, line = pl % lines;
  if (plane > 0 && (line & 1)) return;
  lf_line_seg(fp, rec, info, plane, pass, line, seg * 16, seg * 16 + 16);
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
                      constant int &n [[buffer(4)]],
                      constant long &frameBytes [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
  int bc = (fp->mi_cols + 1) >> 1, br = (fp->mi_rows + 1) >> 1;
  if ((int)gid >= bc * br) return;
  cdef_block_all(fp, cur, out, frameBytes, info, ((int)gid / bc) * 2, ((int)gid % bc) * 2, n);
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

/* Quarter-pel planes of LAST (and the second reference): one thread per sample. */
kernel void mav1_hpel(device const FrameParams *fp [[buffer(0)]],
                      device const uint8_t *ref [[buffer(1)]],
                      device const uint8_t *gold [[buffer(2)]],
                      device uint8_t *hp [[buffer(3)]],
                      device const uint8_t *src [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  int n = fp->frame_w * fp->frame_h;
  int r = (int)gid / n, o = (int)gid % n;
  if (r > 1 || (r == 1 && !fp->golden)) return;
  int y = o / fp->frame_w, x = o % fp->frame_w;
  qpel_build(fp, r ? gold : ref, hp + (long)r * 16 * n, y, x);
  if ((fp->turbo & T_ME_DEC) && y < dec_h(fp) && x < dec_w(fp)) {
    dec_build(fp, r ? gold : ref, (device uint8_t *)dec_plane(fp, hp, 1 + r), y, x);
    if (r == 0) dec_build(fp, src, (device uint8_t *)dec_plane(fp, hp, 0), y, x);
  }
}

/* One SIMD group (32 lanes) per search node of one level. */
kernel void mav1_me(device const FrameParams *fp [[buffer(0)]],
                    device const uint8_t *src [[buffer(1)]],
                    device const uint8_t *ref [[buffer(2)]],
                    device int32_t *mvField [[buffer(3)]],
                    device int32_t *scratch [[buffer(4)]],
                    constant int &level [[buffer(5)]],
                    device const uint8_t *gold [[buffer(6)]],
                    device int32_t *mvFieldGold [[buffer(7)]],
                    device const uint8_t *hp [[buffer(8)]],
                    constant int2 &band [[buffer(9)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
  /* groups [0, nodes): LAST, [nodes, 2 * nodes): GOLDEN */
  /* superblocks [band.x, band.x + band.y) */
  int nn = 1 << (2 * level), nodes = band.y * nn;
  int g = (int)tg;
  DEV const uint8_t *ds = dec_plane(fp, hp, 0), *dr = dec_plane(fp, hp, 1);
  if (g >= nodes) {
    dr = dec_plane(fp, hp, 2);
    if (!fp->golden || g >= 2 * nodes) return;
    g -= nodes;
    ref = gold;
    mvField = mvFieldGold;
    hp += 16 * (long)fp->frame_w * fp->frame_h;
  }
  int sbIdx = band.x + g / nn, i = g % nn;
  MeNode m = me_node(fp, sbIdx, level, i, mvField);
  device int32_t *out = mvField + (sbIdx * NUM_NODES + NODE_BASE(level) + i) * 2;
  if (!m.valid) {
    if (lane == 0) out[0] = out[1] = 0;
    return;
  }
  if (m.copy) {
    if (lane == 0) { out[0] = m.pr; out[1] = m.pc; }
    return;
  }
  int dy = 0, dx = 0;
  for (int stage = 0; stage < m.stages; stage++) {
    if (stage > 0) me_refine_stage(&m, dy, dx);
    int best = INT_MAX, bestK = INT_MAX;
    for (int k = (int)lane; k < m.count; k += 32) {
      int cy, cx;
      me_candidate(&m, k, &cy, &cx);
      int c = me_cost(fp, src, ref, ds, dr, &m, cy, cx);
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
  if (me_take_parent(fp, level, &m, dy, dx)) {
    if (lane == 0) { out[0] = m.pr; out[1] = m.pc; }
    return;
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
      sums[k] += (fp->turbo & T_ME_HPEL) ? me_unit_satd_hp(fp, src, ref, hp, &m, mvr + dr, mvc + dc, u)
                                         : me_unit_satd(fp, src, ref, &m, mvr + dr, mvc + dc, u);
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
                         device const uint8_t *gold [[buffer(8)]],
                         device const int32_t *mvFieldGold [[buffer(9)]],
                         constant int2 &band [[buffer(10)]],
                         uint gid [[thread_position_in_grid]]) {
  /* pass 1: one thread per (superblock, node of the level, slot) */
  int ns = level_slots(fp, level), nn = 1 << (2 * level);
  int total = band.y * nn * ns;
  if ((int)gid >= total) return;
  int sb = band.x + (int)gid / (nn * ns), rem = (int)gid % (nn * ns);
  analyze_item(fp, sb, level, rem / ns, rem % ns, src, ref, gold, mvField, mvFieldGold, costs, infos,
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
                       device const uint8_t *gold [[buffer(10)]],
                       uint gid [[thread_position_in_grid]]) {
  int r = wave.y + (int)gid, c = wave.x - 2 * r;
  if (r >= fp->sb_rows || c < 0 || c >= fp->sb_cols) return;
  int idx = r * fp->sb_cols + c;
  recon_sb(fp, sbs[idx], blocks, src, rec, ref, gold, coefs, mats, scratch + (long)idx * SCRATCH_INTS);
}

/* ------------------------------------------------------------------------ */
/* Cooperative analysis: one SIMD group per candidate block. Every cost is  */
/* the same integer eval_block( ) computes (same predictions, SATD summed   */
/* per 4x4 unit in any order), so the table equals the CPU one.             */

/* SATD of 4x4 unit u of a bw x bh block at (bx, by) of plane p against its
 * EIGHTTAP inter prediction (predict_inter_block( ) evaluated per sample). */
/* EIGHTTAP prediction of 4x4 unit u (Clip1 of the single-reference output). */
INLINE void coop_pred_unit(DEV const FrameParams *fp, DEV const uint8_t *ref, int p, int bx, int by, int bw, int bh,
                           int mvr, int mvc, int u, THR int *out) {
  int sub = p > 0;
  int stride = fp->stride[p];
  DEV const uint8_t *rp = ref + fp->offset[p];
  int nu = bw >> 2;
  int ux = (u % nu) * 4, uy = (u / nu) * 4;
  int startX = (((bx << SUBPEL_BITS) + ((2 * mvc) >> sub)) << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) + 32;
  int startY = (((by << SUBPEL_BITS) + ((2 * mvr) >> sub)) << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) + 32;
  int lastX = ((fp->frame_w + sub) >> sub) - 1, lastY = ((fp->frame_h + sub) >> sub) - 1;
  int fhi = bw <= 4 ? 4 : EIGHTTAP, fvi = bh <= 4 ? 4 : EIGHTTAP;
  KCONST int32_t *fh = Subpel_Filters[fhi][(startX >> 6) & SUBPEL_MASK];
  KCONST int32_t *fv = Subpel_Filters[fvi][(startY >> 6) & SUBPEL_MASK];
  int hrow[11][4];
  for (int rr = 0; rr < 11; rr++) {
    int ry = Clip3(0, lastY, (startY >> 10) + uy + rr - 3);
    DEV const uint8_t *row = rp + ry * stride;
    for (int c = 0; c < 4; c++) {
      int sum = 0;
      int x0 = (startX >> 10) + ux + c - 3;
      for (int t = 0; t < 8; t++) sum += fh[t] * row[Clip3(0, lastX, x0 + t)];
      hrow[rr][c] = Round2(sum, 3);
    }
  }
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) {
      int sum = 0;
      for (int t = 0; t < 8; t++) sum += fv[t] * hrow[r + t][c];
      out[r * 4 + c] = Clip1(Round2(sum, 11));
    }
}

/* SATD of 4x4 unit u of a bw x bh block at (bx, by) of plane p against its
 * EIGHTTAP inter prediction (predict_inter_block( ) evaluated per sample);
 * with ref2, against the rounded average of both references' predictions
 * (compound_cost( )). */
INLINE int coop_inter_unit(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, int p,
                           int bx, int by, int bw, int bh, int mvr, int mvc, int u) {
  int pr[16];
  coop_pred_unit(fp, ref, p, bx, by, bw, bh, mvr, mvc, u, pr);
  int stride = fp->stride[p], nu = bw >> 2;
  DEV const uint8_t *sp = src + fp->offset[p] + (by + (u / nu) * 4) * stride + bx + (u % nu) * 4;
  int32_t d[16];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) d[r * 4 + c] = (int)sp[r * stride + c] - pr[r * 4 + c];
  return satd4x4(d);
}

INLINE int coop_comp_unit(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                          DEV const uint8_t *ref2, int p, int bx, int by, int bw, int bh, int mvr, int mvc, int mvr2,
                          int mvc2, int u) {
  int pr[16], pr2[16];
  coop_pred_unit(fp, ref, p, bx, by, bw, bh, mvr, mvc, u, pr);
  coop_pred_unit(fp, ref2, p, bx, by, bw, bh, mvr2, mvc2, u, pr2);
  int stride = fp->stride[p], nu = bw >> 2;
  DEV const uint8_t *sp = src + fp->offset[p] + (by + (u / nu) * 4) * stride + bx + (u % nu) * 4;
  int32_t d[16];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) d[r * 4 + c] = (int)sp[r * stride + c] - ((pr[r * 4 + c] + pr2[r * 4 + c] + 1) >> 1);
  return satd4x4(d);
}

INLINE long coop_compound_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                               DEV const uint8_t *ref2, int x, int y, int w, int h, int mvr, int mvc, int mvr2,
                               int mvc2, uint lane) {
  int lam = fp->ac_q[0] / 16;
  long c = 0;
  for (int p = 0; p < 3; p++) {
    int sub = p > 0, bw = w >> sub, bh = h >> sub, n = (bw >> 2) * (bh >> 2), s = 0;
    for (int u = (int)lane; u < n; u += 32)
      s += coop_comp_unit(fp, src, ref, ref2, p, x >> sub, y >> sub, bw, bh, mvr, mvc, mvr2, mvc2, u);
    c += simd_sum(s);
  }
  return c + (long)lam * (8 + mv_bits8(mvr, mvc) + mv_bits8(mvr2, mvc2) + COMPOUND_BITS8) / 8;
}

/* LAST, second reference and compound SATDs of the whole block (all planes)
 * from one evaluation of both predictions per 4x4 unit: s[0..2]. */
INLINE void coop_inter3(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                        DEV const uint8_t *ref2, int x, int y, int w, int h, int mvr, int mvc, int mvr2, int mvc2,
                        uint lane, THR long *s) {
  int a0 = 0, a1 = 0, a2 = 0;
  for (int p = 0; p < 3; p++) {
    int sub = p > 0, bw = w >> sub, bh = h >> sub, bx = x >> sub, by = y >> sub, nu = bw >> 2, n = nu * (bh >> 2);
    int stride = fp->stride[p];
    for (int u = (int)lane; u < n; u += 32) {
      int pr[16], pr2[16];
      coop_pred_unit(fp, ref, p, bx, by, bw, bh, mvr, mvc, u, pr);
      coop_pred_unit(fp, ref2, p, bx, by, bw, bh, mvr2, mvc2, u, pr2);
      DEV const uint8_t *sp = src + fp->offset[p] + (by + (u / nu) * 4) * stride + bx + (u % nu) * 4;
      int32_t d0[16], d1[16], d2[16];
      for (int rr = 0; rr < 4; rr++)
        for (int c = 0; c < 4; c++) {
          int sv = sp[rr * stride + c], k = rr * 4 + c;
          d0[k] = sv - pr[k];
          d1[k] = sv - pr2[k];
          d2[k] = sv - ((pr[k] + pr2[k] + 1) >> 1);
        }
      a0 += satd4x4(d0);
      a1 += satd4x4(d1);
      a2 += satd4x4(d2);
    }
  }
  s[0] = simd_sum(a0);
  s[1] = simd_sum(a1);
  s[2] = simd_sum(a2);
}

INLINE int coop_inter_satd(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, int p,
                           int bx, int by, int bw, int bh, int mvr, int mvc, uint lane) {
  int n = (bw >> 2) * (bh >> 2), s = 0;
  for (int u = (int)lane; u < n; u += 32) s += coop_inter_unit(fp, src, ref, p, bx, by, bw, bh, mvr, mvc, u);
  return simd_sum(s);
}

/* Edges of predict_intra( ) as the analysis calls it (haveLeft = x > 0,
 * haveAbove = y > 0, no above-right / below-left, no edge filter). */
INLINE void coop_edges(DEV const FrameParams *fp, DEV const uint8_t *px, int stride, int planeIdx, int x, int y,
                       int w, int h, threadgroup int *A, threadgroup int *L, uint lane) {
  int sub = planeIdx > 0;
  int maxX = ((fp->mi_cols * 4) >> sub) - 1, maxY = ((fp->mi_rows * 4) >> sub) - 1;
  int haveLeft = x > 0, haveAbove = y > 0;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = (int)lane; i < w + h; i += 32) {
    if (!haveAbove && haveLeft) A[i] = px[y * stride + x - 1];
    else if (!haveAbove && !haveLeft) A[i] = (1 << 7) - 1;
    else A[i] = px[(y - 1) * stride + k_min(k_min(maxX, x + w - 1), x + i)];
    if (!haveLeft && haveAbove) L[i] = px[(y - 1) * stride + x];
    else if (!haveLeft && !haveAbove) L[i] = (1 << 7) + 1;
    else L[i] = px[k_min(k_min(maxY, y + h - 1), y + i) * stride + x - 1];
  }
  if (lane == 0) {
    int c;
    if (haveAbove && haveLeft) c = px[(y - 1) * stride + x - 1];
    else if (haveAbove) c = px[(y - 1) * stride + x];
    else if (haveLeft) c = px[y * stride + x - 1];
    else c = 1 << 7;
    A[-1] = c;
    L[-1] = c;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

/* One predicted sample (i, j) of predict_intra( ) without edge upsampling. */
INLINE int coop_intra_px(int mode, int pAngle, int dx, int dy, int dcAvg, int i, int j, int w, int h, int log2W,
                         int log2H, threadgroup const int *A, threadgroup const int *L) {
  if (intra_is_directional(mode)) {
    if (pAngle < 90) {
      int idx = (i + 1) * dx;
      int base = (idx >> 6) + j;
      int shift = (idx >> 1) & 0x1F;
      int maxBaseX = w + h - 1;
      return base < maxBaseX ? Round2(A[base] * (32 - shift) + A[base + 1] * shift, 5) : A[maxBaseX];
    } else if (pAngle > 90 && pAngle < 180) {
      int idx = (j << 6) - (i + 1) * dx;
      int base = idx >> 6;
      if (base >= -1) {
        int shift = (idx >> 1) & 0x1F;
        return Round2(A[base] * (32 - shift) + A[base + 1] * shift, 5);
      }
      idx = (i << 6) - (j + 1) * dy;
      base = idx >> 6;
      int shift = (idx >> 1) & 0x1F;
      return Round2(L[base] * (32 - shift) + L[base + 1] * shift, 5);
    } else if (pAngle > 180) {
      int idx = (j + 1) * dy;
      int base = (idx >> 6) + i;
      int shift = (idx >> 1) & 0x1F;
      return Round2(L[base] * (32 - shift) + L[base + 1] * shift, 5);
    } else if (pAngle == 90) {
      return A[j];
    }
    return L[i];
  } else if (mode == SMOOTH_PRED) {
    KCONST int32_t *wX = intra_sm_weights(log2W);
    KCONST int32_t *wY = intra_sm_weights(log2H);
    return Round2(wY[i] * A[j] + (256 - wY[i]) * L[h - 1] + wX[j] * L[i] + (256 - wX[j]) * A[w - 1], 9);
  } else if (mode == SMOOTH_V_PRED) {
    KCONST int32_t *wY = intra_sm_weights(log2H);
    return Round2(wY[i] * A[j] + (256 - wY[i]) * L[h - 1], 8);
  } else if (mode == SMOOTH_H_PRED) {
    KCONST int32_t *wX = intra_sm_weights(log2W);
    return Round2(wX[j] * L[i] + (256 - wX[j]) * A[w - 1], 8);
  } else if (mode == DC_PRED) {
    return dcAvg;
  }
  int base = A[j] + L[i] - A[-1];
  int pLeft = k_abs(base - L[i]), pTop = k_abs(base - A[j]), pTopLeft = k_abs(base - A[-1]);
  if (pLeft <= pTop && pLeft <= pTopLeft) return L[i];
  return pTop <= pTopLeft ? A[j] : A[-1];
}

/* intra_satd( ) with the work spread over the lanes (edges prebuilt). */
INLINE int coop_intra_satd(DEV const uint8_t *px, int stride, int x, int y, int log2W, int log2H, int mode,
                           int delta, threadgroup const int *A, threadgroup const int *L, uint lane) {
  int w = 1 << log2W, h = 1 << log2H;
  int haveLeft = x > 0, haveAbove = y > 0;
  int pAngle = 0, dx = 0, dy = 0, dcAvg = 0;
  if (intra_is_directional(mode)) {
    pAngle = Mode_To_Angle[mode] + delta * ANGLE_STEP;
    if (pAngle < 90) dx = Dr_Intra_Derivative[pAngle];
    else if (pAngle > 90 && pAngle < 180) dx = Dr_Intra_Derivative[180 - pAngle];
    if (pAngle > 90 && pAngle < 180) dy = Dr_Intra_Derivative[pAngle - 90];
    else if (pAngle > 180) dy = Dr_Intra_Derivative[270 - pAngle];
  } else if (mode == DC_PRED) {
    int sum = 0;
    if (haveLeft && haveAbove) {
      for (int k = 0; k < h; k++) sum += L[k];
      for (int k = 0; k < w; k++) sum += A[k];
      dcAvg = (sum + ((w + h) >> 1)) / (w + h);
    } else if (haveLeft) {
      for (int k = 0; k < h; k++) sum += L[k];
      dcAvg = Clip1((sum + (h >> 1)) >> log2H);
    } else if (haveAbove) {
      for (int k = 0; k < w; k++) sum += A[k];
      dcAvg = Clip1((sum + (w >> 1)) >> log2W);
    } else {
      dcAvg = 1 << 7;
    }
  }
  int nu = w >> 2, n = nu * (h >> 2), s = 0;
  for (int u = (int)lane; u < n; u += 32) {
    int ux = (u % nu) * 4, uy = (u / nu) * 4;
    int32_t d[16];
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
        d[r * 4 + c] = (int)px[(y + uy + r) * stride + x + ux + c] -
                       coop_intra_px(mode, pAngle, dx, dy, dcAvg, uy + r, ux + c, w, h, log2W, log2H, A, L);
    s += satd4x4(d);
  }
  return simd_sum(s);
}

/* CfL cost of chroma plane p (the eval_block CfL step, lanes over samples):
 * DC prediction (a constant), best alpha by SSE over -16..16 (first
 * minimum), SATD of that prediction. */
INLINE int coop_cfl_satd(DEV const FrameParams *fp, DEV const uint8_t *src, int p, int cx, int cy, int clw, int clh,
                         int maxLumaW, int maxLumaH, threadgroup int *A, threadgroup int *L, uint lane) {
  int cw = 1 << clw, ch = 1 << clh;
  DEV const uint8_t *pp = src + fp->offset[p];
  int stride = fp->stride[p], ls = fp->stride[0];
  DEV const uint8_t *luma = src + fp->offset[0];
  coop_edges(fp, pp, stride, p, cx, cy, cw, ch, A, L, lane);
  int haveLeft = cx > 0, haveAbove = cy > 0, dcAvg, sum = 0;
  if (haveLeft && haveAbove) {
    for (int k = 0; k < ch; k++) sum += L[k];
    for (int k = 0; k < cw; k++) sum += A[k];
    dcAvg = (sum + ((cw + ch) >> 1)) / (cw + ch);
  } else if (haveLeft) {
    for (int k = 0; k < ch; k++) sum += L[k];
    dcAvg = Clip1((sum + (ch >> 1)) >> clh);
  } else if (haveAbove) {
    for (int k = 0; k < cw; k++) sum += A[k];
    dcAvg = Clip1((sum + (cw >> 1)) >> clw);
  } else {
    dcAvg = 1 << 7;
  }
  int n = cw * ch, lsum = 0;
  for (int t = (int)lane; t < n; t += 32) {
    int i = t >> clw, j = t & (cw - 1);
    DEV const uint8_t *l = luma + k_min((cy + i) << 1, maxLumaH - 2) * ls + k_min((cx + j) << 1, maxLumaW - 2);
    lsum += ((int)l[0] + l[1] + l[ls] + l[ls + 1]) << 1;
  }
  int avg = Round2(simd_sum(lsum), clw + clh);
  int sse[33];
  for (int a = 0; a < 33; a++) sse[a] = 0;
  for (int t = (int)lane; t < n; t += 32) {
    int i = t >> clw, j = t & (cw - 1);
    DEV const uint8_t *l = luma + k_min((cy + i) << 1, maxLumaH - 2) * ls + k_min((cx + j) << 1, maxLumaW - 2);
    int v = (((int)l[0] + l[1] + l[ls] + l[ls + 1]) << 1) - avg;
    int s0 = pp[(cy + i) * stride + cx + j];
    for (int a = 0; a < 33; a++) {
      int d = Clip3(0, 255, dcAvg + Round2Signed((a - 16) * v, 6)) - s0;
      sse[a] += d * d;
    }
  }
  int bestA = 0;
  long best = -1;
  for (int a = 0; a < 33; a++) {
    long e2 = simd_sum(sse[a]);
    if (best < 0 || e2 < best) { best = e2; bestA = a - 16; }
  }
  int nu = cw >> 2, nUnits = nu * (ch >> 2), sat = 0;
  for (int u = (int)lane; u < nUnits; u += 32) {
    int ux = (u % nu) * 4, uy = (u / nu) * 4;
    int32_t d[16];
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++) {
        int i = uy + r, j = ux + c;
        DEV const uint8_t *l = luma + k_min((cy + i) << 1, maxLumaH - 2) * ls + k_min((cx + j) << 1, maxLumaW - 2);
        int v = (((int)l[0] + l[1] + l[ls] + l[ls + 1]) << 1) - avg;
        d[r * 4 + c] = (int)pp[(cy + i) * stride + cx + j] - Clip3(0, 255, dcAvg + Round2Signed(bestA * v, 6));
      }
    sat += satd4x4(d);
  }
  return simd_sum(sat);
}

/* eval_block( ) for one SIMD group. */
INLINE long coop_inter_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                            DEV const int32_t *mvField, int mvNode, int x, int y, int w, int h, THR int *mvr,
                            THR int *mvc, uint lane) {
  int lam = fp->ac_q[0] / 16;
  int cx = x >> 1, cy = y >> 1;
  int sbIdx = (y >> 6) * fp->sb_cols + (x >> 6);
  DEV const int32_t *mvp = mvField + (sbIdx * NUM_NODES + mvNode) * 2;
  *mvr = mvp[0];
  *mvc = mvp[1];
  long c = coop_inter_satd(fp, src, ref, 0, x, y, w, h, *mvr, *mvc, lane);
  for (int p = 1; p < 3; p++) c += coop_inter_satd(fp, src, ref, p, cx, cy, w >> 1, h >> 1, *mvr, *mvc, lane);
  return c + (long)lam * (4 + mv_bits8(*mvr, *mvc)) / 8;
}

INLINE long coop_eval_block(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                            DEV const uint8_t *gold, DEV const int32_t *mvField, DEV const int32_t *mvFieldGold,
                            int mvNode, int miRow, int miCol, int bsize,
                            DEV int32_t *scratch, int ps, THR BlockInfo *out, threadgroup int *A,
                            threadgroup int *L, uint lane) {
  DEV uint8_t *pred = (DEV uint8_t *)scratch;
  DEV int32_t *tmpArea = scratch + (ps * ps + 3) / 4;
  int lw = Mi_Width_Log2[bsize] + 2, lh = Mi_Height_Log2[bsize] + 2;
  int w = 1 << lw, h = 1 << lh;
  int x = miCol * 4, y = miRow * 4;
  int lam = fp->ac_q[0] / 16;
  int cx = x >> 1, cy = y >> 1, clw = lw - 1, clh = lh - 1;
  long interCost = -1;
  int mvr = 0, mvc = 0, refFrame = LAST_FRAME, refFrame1 = 0, mvr1 = 0, mvc1 = 0;
  if (!fp->frame_is_intra && fp->golden && fp->compound) {
    /* same costs as below, both predictions evaluated once */
    int sbIdx = (y >> 6) * fp->sb_cols + (x >> 6);
    DEV const int32_t *m0 = mvField + (sbIdx * NUM_NODES + mvNode) * 2;
    DEV const int32_t *m1 = mvFieldGold + (sbIdx * NUM_NODES + mvNode) * 2;
    int lr = m0[0], lc = m0[1], gr = m1[0], gc = m1[1];
    long s[3];
    coop_inter3(fp, src, ref, gold, x, y, w, h, lr, lc, gr, gc, lane, s);
    interCost = s[0] + (long)lam * (4 + mv_bits8(lr, lc)) / 8;
    mvr = lr;
    mvc = lc;
    long g = s[1] + (long)lam * (4 + mv_bits8(gr, gc)) / 8 + (long)lam * GOLDEN_REF_BITS8 / 8;
    if (g < interCost) {
      interCost = g;
      mvr = gr;
      mvc = gc;
      refFrame = fp->ref2_frame;
    }
    long cc = s[2] + (long)lam * (8 + mv_bits8(lr, lc) + mv_bits8(gr, gc) + COMPOUND_BITS8) / 8;
    if (cc < interCost) {
      interCost = cc;
      mvr = lr;
      mvc = lc;
      mvr1 = gr;
      mvc1 = gc;
      refFrame = LAST_FRAME;
      refFrame1 = fp->ref2_frame;
    }
  } else if (!fp->frame_is_intra) {
    interCost = coop_inter_cost(fp, src, ref, mvField, mvNode, x, y, w, h, &mvr, &mvc, lane);
    if (fp->golden) {
      int lr = mvr, lc = mvc, gr, gc;
      long g = coop_inter_cost(fp, src, gold, mvFieldGold, mvNode, x, y, w, h, &gr, &gc, lane) +
               (long)lam * GOLDEN_REF_BITS8 / 8;
      if (g < interCost) { /* tie: LAST */
        interCost = g;
        mvr = gr;
        mvc = gc;
        refFrame = fp->ref2_frame;
      }
      if (fp->compound) {
        long cc = coop_compound_cost(fp, src, ref, gold, x, y, w, h, lr, lc, gr, gc, lane);
        if (cc < interCost) {
          interCost = cc;
          mvr = lr;
          mvc = lc;
          mvr1 = gr;
          mvc1 = gc;
          refFrame = LAST_FRAME;
          refFrame1 = fp->ref2_frame;
        }
      }
    }
  }
  long best = -1;
  int bestMode = DC_PRED, bestDelta = 0;
  int skipIntra = interCost >= 0 && (fp->leaf || (fp->fast && interCost * 16 < (long)w * h * lam * 16));
  if (!skipIntra) {
    DEV const uint8_t *py = src + fp->offset[0];
    coop_edges(fp, py, fp->stride[0], 0, x, y, w, h, A, L, lane);
    for (int mode = DC_PRED; mode <= PAETH_PRED; mode++) {
      if (turbo_mode_off(fp, mode)) continue;
      long c = coop_intra_satd(py, fp->stride[0], x, y, lw, lh, mode, 0, A, L, lane) + (long)lam * mode_bits8(mode) / 8;
      if (best < 0 || c < best) { best = c; bestMode = mode; }
    }
    if (intra_is_directional(bestMode) && !fp->fast) {
      int base = bestMode;
      for (int d = -MAX_ANGLE_DELTA; d <= MAX_ANGLE_DELTA; d++) {
        if (d == 0) continue;
        long c = coop_intra_satd(py, fp->stride[0], x, y, lw, lh, base, d, A, L, lane) +
                 (long)lam * (mode_bits8(base) + 16) / 8;
        if (c < best) { best = c; bestDelta = d; }
      }
    }
  } else {
    best = 1L << 40;
  }
  int skipUv = fp->fast && interCost >= 0 && best + (long)lam * 16 / 8 >= interCost;
  long bestUv = skipUv ? 0 : -1;
  int bestUvMode = DC_PRED;
  if (!skipUv) {
    long cu[PAETH_PRED + 1];
    for (int m = DC_PRED; m <= PAETH_PRED; m++) cu[m] = 0;
    for (int p = 1; p < 3; p++) {
      DEV const uint8_t *pp = src + fp->offset[p];
      coop_edges(fp, pp, fp->stride[p], p, cx, cy, w >> 1, h >> 1, A, L, lane);
      for (int m = DC_PRED; m <= PAETH_PRED; m++) {
        if (intra_is_directional(m) && m != V_PRED && m != H_PRED && m != bestMode) continue;
        if (turbo_mode_off(fp, m)) continue;
        int delta = (m == bestMode) ? bestDelta : 0;
        cu[m] += coop_intra_satd(pp, fp->stride[p], cx, cy, clw, clh, m, delta, A, L, lane);
      }
    }
    for (int m = DC_PRED; m <= PAETH_PRED; m++) {
      if (intra_is_directional(m) && m != V_PRED && m != H_PRED && m != bestMode) continue;
      if (turbo_mode_off(fp, m)) continue;
      long c = cu[m] + (long)lam * (m == DC_PRED ? 8 : 24) / 8;
      if (bestUv < 0 || c < bestUv) { bestUv = c; bestUvMode = m; }
    }
    if (cfl_allowed(bsize, fp->lossless) && !turbo_no_cfl(fp)) {
      long c = (long)lam * 40 / 8;
      for (int p = 1; p < 3; p++) c += coop_cfl_satd(fp, src, p, cx, cy, clw, clh, x + w, y + h, A, L, lane);
      if (c < bestUv) { bestUv = c; bestUvMode = UV_CFL_PRED; }
    }
  }
  out->mi_row = miRow;
  out->mi_col = miCol;
  out->bsize = bsize;
  out->is_inter = 0;
  out->y_mode = bestMode;
  out->angle_y = bestDelta;
  out->uv_mode = bestUvMode;
  out->angle_uv = (bestUvMode == bestMode) ? bestDelta : 0;
  out->cfl_alpha_u = out->cfl_alpha_v = 0;
  out->tx_depth = 0;
  out->interp_filter = 0;
  out->tx_types_lo = out->tx_types_hi = 0;
  out->mv_row = 0;
  out->mv_col = 0;
  out->ref_frame = INTRA_FRAME;
  out->ref_frame1 = 0;
  out->mv1_row = out->mv1_col = 0;
  out->tx_type = DCT_DCT;
  long cost = best + bestUv + (long)lam * (fp->frame_is_intra ? 0 : 16) / 8;
  if (interCost >= 0 && interCost < cost) {
    cost = interCost;
    out->is_inter = 1;
    out->y_mode = NEWMV;
    out->uv_mode = DC_PRED;
    out->angle_y = 0;
    out->angle_uv = 0;
    out->mv_row = mvr;
    out->mv_col = mvc;
    out->ref_frame = refFrame;
    out->ref_frame1 = refFrame1;
    out->mv1_row = mvr1;
    out->mv1_col = mvc1;
  }
  return cost;
}

/* Pass 1, cooperative: one SIMD group (threadgroup of 32) per item. */
kernel void mav1_analyze_coop(device const FrameParams *fp [[buffer(0)]],
                              device const uint8_t *src [[buffer(1)]],
                              device const uint8_t *ref [[buffer(2)]],
                              device const int32_t *mvField [[buffer(3)]],
                              device long *costs [[buffer(4)]],
                              device BlockInfo *infos [[buffer(5)]],
                              device int32_t *scratch [[buffer(6)]],
                              constant int &level [[buffer(7)]],
                              device const uint8_t *gold [[buffer(8)]],
                              device const int32_t *mvFieldGold [[buffer(9)]],
                              constant int2 &band [[buffer(10)]],
                              uint gid [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int eA[EDGE_LEN], eL[EDGE_LEN];
  int ns = level_slots(fp, level), nn = 1 << (2 * level);
  if ((int)gid >= band.y * nn * ns) return;
  int sbIdx = band.x + (int)gid / (nn * ns), rem = (int)gid % (nn * ns);
  int i = rem / ns, slot = rem % ns;
  int sbRow = sbIdx / fp->sb_cols, sbCol = sbIdx % fp->sb_cols;
  int n = 1 << level, size4 = 16 >> level, halfSz = size4 >> 1;
  int node = NODE_BASE(level) + i;
  int r = sbRow * 16 + (i / n) * size4, c = sbCol * 16 + (i % n) * size4;
  long idx = (long)sbIdx * A_ITEMS + node * A_SLOTS + slot;
  if (r >= fp->mi_rows || c >= fp->mi_cols || depth_pruned(fp, costs, sbIdx, level, i)) { if (lane == 0) costs[idx] = -1; return; }
  int hasRows = (r + halfSz) < fp->mi_rows, hasCols = (c + halfSz) < fp->mi_cols;
  int p, k;
  slot_part(slot, &p, &k);
  if (!part_allowed(fp, level, p, hasRows, hasCols) || k >= part_count(fp, p, r, c, halfSz)) {
    if (lane == 0) costs[idx] = -1;
    return;
  }
  int dr, dc;
  int bs = part_block(p, level_bsize(level), k, halfSz, &dr, &dc);
  BlockInfo b;
  long cost = coop_eval_block(fp, src, ref, gold, mvField, mvFieldGold, node, r + dr, c + dc, bs,
                              scratch + (long)gid * analysis_slot_ints(level), 64 >> level, &b, eA + EDGE_OFF,
                              eL + EDGE_OFF, lane);
  if (lane == 0) {
    costs[idx] = cost;
    infos[idx] = b;
  }
}
