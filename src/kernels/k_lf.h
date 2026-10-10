/* Deblocking loop filter, spec 7.14. With TX_MODE_LARGEST every block has a
 * single transform per plane, so LoopfilterTxSizes follow from MiSizes.
 * loop_filter_delta_enabled = 0 and no segmentation: the level depends only
 * on plane and pass. One work item filters all edges of one 4-sample row
 * (pass 0, vertical edges) or column (pass 1, horizontal edges); the spec
 * guarantees order independence within a pass. */
#ifndef K_LF_H
#define K_LF_H

#include "k_scan.h"

/* info = MiSize | skip << 5 | tx_depth << 6 (LoopfilterTxSizes of the block). */
INLINE int lf_tx_size(int info, int plane) {
  return plane == 0 ? depth_tx_size(info & 31, (info >> 6) & 3) : uv_tx_size(info & 31, 1, 1);
}

INLINE void lf_sample(DEV uint8_t *px, int pitch /* step across the edge */, int plane, int lvl, int sharp,
                      int filterSize) {
  int shift = sharp > 4 ? 2 : (sharp > 0 ? 1 : 0);
  int limit = sharp > 0 ? Clip3(1, 9 - sharp, lvl >> shift) : k_max(1, lvl >> shift);
  int blimit = 2 * (lvl + 2) + limit;
  int thresh = lvl >> 4;
#define S(k) ((int)px[(k) * pitch])
  int q0 = S(0), q1 = S(1), q2 = S(2), q3 = S(3);
  int p0 = S(-1), p1 = S(-2), p2 = S(-3), p3 = S(-4);
  int hevMask = (k_abs(p1 - p0) > thresh) || (k_abs(q1 - q0) > thresh);
  int filterLen = filterSize == 4 ? 4 : (plane != 0 ? 6 : (filterSize == 8 ? 8 : 16));
  int mask = 0;
  mask |= k_abs(p1 - p0) > limit;
  mask |= k_abs(q1 - q0) > limit;
  mask |= k_abs(p0 - q0) * 2 + k_abs(p1 - q1) / 2 > blimit;
  if (filterLen >= 6) {
    mask |= k_abs(p2 - p1) > limit;
    mask |= k_abs(q2 - q1) > limit;
  }
  if (filterLen >= 8) {
    mask |= k_abs(p3 - p2) > limit;
    mask |= k_abs(q3 - q2) > limit;
  }
  if (mask) return;
  int flatMask = 0, flatMask2 = 0;
  if (filterSize >= 8) {
    int m = 0;
    m |= k_abs(p1 - p0) > 1;
    m |= k_abs(q1 - q0) > 1;
    m |= k_abs(p2 - p0) > 1;
    m |= k_abs(q2 - q0) > 1;
    if (filterLen >= 8) {
      m |= k_abs(p3 - p0) > 1;
      m |= k_abs(q3 - q0) > 1;
    }
    flatMask = m == 0;
  }
  if (filterSize >= 16) {
    int q4 = S(4), q5 = S(5), q6 = S(6), p4 = S(-5), p5 = S(-6), p6 = S(-7);
    int m = 0;
    m |= k_abs(p6 - p0) > 1;
    m |= k_abs(q6 - q0) > 1;
    m |= k_abs(p5 - p0) > 1;
    m |= k_abs(q5 - q0) > 1;
    m |= k_abs(p4 - p0) > 1;
    m |= k_abs(q4 - q0) > 1;
    flatMask2 = m == 0;
  }
  if (filterSize == 4 || !flatMask) {
    /* narrow filter */
    int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
    int filter = hevMask ? Clip3(-128, 127, ps1 - qs1) : 0;
    filter = Clip3(-128, 127, filter + 3 * (qs0 - ps0));
    int filter1 = Clip3(-128, 127, filter + 4) >> 3;
    int filter2 = Clip3(-128, 127, filter + 3) >> 3;
    px[0] = (uint8_t)(Clip3(-128, 127, qs0 - filter1) + 128);
    px[-pitch] = (uint8_t)(Clip3(-128, 127, ps0 + filter2) + 128);
    if (!hevMask) {
      filter = Round2(filter1, 1);
      px[pitch] = (uint8_t)(Clip3(-128, 127, qs1 - filter) + 128);
      px[-2 * pitch] = (uint8_t)(Clip3(-128, 127, ps1 + filter) + 128);
    }
  } else {
    int log2Size = (filterSize == 8 || !flatMask2) ? 3 : 4;
    int n = log2Size == 4 ? 6 : (plane == 0 ? 3 : 2);
    int n2 = (log2Size == 3 && plane == 0) ? 0 : 1;
    int F[12];
    for (int i = -n; i < n; i++) {
      int t = 0;
      for (int j = -n; j <= n; j++) {
        int p = Clip3(-(n + 1), n, i + j);
        int tap = k_abs(j) <= n2 ? 2 : 1;
        t += S(p) * tap;
      }
      F[i + n] = Round2(t, log2Size);
    }
    for (int i = -n; i < n; i++) px[i * pitch] = (uint8_t)F[i + n];
  }
#undef S
}

/* Filters every edge along one line of 4x4 units. pass 0: `line` is a luma
 * mi row (step 1 << subY), edges at each mi col; pass 1: `line` is an mi col. */
/* Edges of mi positions [k0, k1) of a line. The edges of one pass are
 * independent (a filter is no longer than the transforms on either side,
 * so neighbouring edges never touch the same samples), so lines may be
 * split into segments. */
INLINE void lf_line_seg(DEV const FrameParams *fp, DEV uint8_t *rec, DEV const uint8_t *miSizes, int plane, int pass,
                        int line, int k0, int k1) {
  int sub = plane > 0;
  int lvl = Clip3(0, MAX_LOOP_FILTER, fp->lf_level[plane == 0 ? pass : plane + 1]);
  if (lvl == 0) return;
  int stride = fp->stride[plane];
  DEV uint8_t *base = rec + fp->offset[plane];
  int dx = pass == 0, dy = pass == 1;
  int n = k_min(k1, pass == 0 ? fp->mi_cols : fp->mi_rows);
  int step = 1 << sub;
  for (int k = k0; k < n; k += step) {
    int row0 = pass == 0 ? line : k, col0 = pass == 0 ? k : line;
    int x = col0 * MI_SIZE, y = row0 * MI_SIZE;
    int row = row0 | sub, col = col0 | sub;
    if (x >= fp->frame_w || y >= fp->frame_h) continue;
    if (pass == 0 && x == 0) continue;
    if (pass == 1 && y == 0) continue;
    int xP = x >> sub, yP = y >> sub;
    int prevRow = row - (dy << sub), prevCol = col - (dx << sub);
    int txSz = lf_tx_size(miSizes[row * fp->mi_cols + col], plane);
    int prevTxSz = lf_tx_size(miSizes[prevRow * fp->mi_cols + prevCol], plane);
    /* Inner transform edges only occur in intra blocks, which are always
     * filtered, so applyFilter = isTxEdge. */
    int isTxEdge = pass == 0 ? (xP % Tx_Width[txSz]) == 0 : (yP % Tx_Height[txSz]) == 0;
    if (!isTxEdge) continue;
    int baseSize = pass == 0 ? k_min(Tx_Width[prevTxSz], Tx_Width[txSz]) : k_min(Tx_Height[prevTxSz], Tx_Height[txSz]);
    int filterSize = plane == 0 ? k_min(16, baseSize) : k_min(8, baseSize);
    for (int i = 0; i < MI_SIZE; i++) {
      int sx = xP + dy * i, sy = yP + dx * i;
      lf_sample(base + sy * stride + sx, pass == 0 ? 1 : stride, plane, lvl, fp->lf_sharpness, filterSize);
    }
  }
}

INLINE void lf_line(DEV const FrameParams *fp, DEV uint8_t *rec, DEV const uint8_t *miSizes, int plane, int pass,
                    int line) {
  lf_line_seg(fp, rec, miSizes, plane, pass, line, 0, 1 << 30);
}

#endif
