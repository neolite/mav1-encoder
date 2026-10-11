/* CDEF, spec 7.15. Filters one 8x8 luma block (and its 4x4 chroma blocks)
 * of the deblocked frame `cur` into `out` with explicit strengths.
 * info[mi] bit 5 holds Skips (blocks are at least 8x8, so one mi suffices). */

#include "k_common.h"

INLINE int KNAME(cdef_constrain)(int diff, int threshold, int damping) {
  if (!threshold) return 0;
  int dampingAdj = k_max(0, damping - FloorLog2(threshold));
  int sign = diff < 0 ? -1 : 1;
  return sign * Clip3(0, k_abs(diff), threshold - (k_abs(diff) >> dampingAdj));
}

INLINE void KNAME(cdef_direction)(DEV const KPIX *px, int stride, int bdMid, THR int *yDir, THR int *var) {
  int cost[8];
  int partial[8][15];
  for (int i = 0; i < 8; i++) {
    cost[i] = 0;
    for (int j = 0; j < 15; j++) partial[i][j] = 0;
  }
  int bestCost = 0;
  *yDir = 0;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      int x = (int)px[i * stride + j] - bdMid;
      partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      partial[4][7 + i - j] += x;
      partial[5][3 - i / 2 + j] += x;
      partial[6][j] += x;
      partial[7][i / 2 + j] += x;
    }
  for (int i = 0; i < 8; i++) {
    cost[2] += partial[2][i] * partial[2][i];
    cost[6] += partial[6][i] * partial[6][i];
  }
  cost[2] *= Div_Table[8];
  cost[6] *= Div_Table[8];
  for (int i = 0; i < 7; i++) {
    cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * Div_Table[i + 1];
    cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * Div_Table[i + 1];
  }
  cost[0] += partial[0][7] * partial[0][7] * Div_Table[8];
  cost[4] += partial[4][7] * partial[4][7] * Div_Table[8];
  for (int i = 1; i < 8; i += 2) {
    for (int j = 0; j < 4 + 1; j++) cost[i] += partial[i][3 + j] * partial[i][3 + j];
    cost[i] *= Div_Table[8];
    for (int j = 0; j < 4 - 1; j++)
      cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j]) * Div_Table[2 * j + 2];
  }
  for (int i = 0; i < 8; i++)
    if (cost[i] > bestCost) {
      bestCost = cost[i];
      *yDir = i;
    }
  *var = (bestCost - cost[(*yDir + 4) & 7]) >> 10;
}

INLINE void KNAME(cdef_filter)(DEV const FrameParams *fp, DEV const KPIX *cur, DEV KPIX *out, int plane, int r, int c,
                               int priStr, int secStr, int damping, int dir) {
  int sub = plane > 0;
  int stride = fp->stride[plane];
  DEV const KPIX *src = cur + fp->offset[plane];
  DEV KPIX *dst = out + fp->offset[plane];
  int x0 = (c * MI_SIZE) >> sub, y0 = (r * MI_SIZE) >> sub;
  int w = 8 >> sub, h = 8 >> sub;
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) {
      int sum = 0;
      int x = src[(y0 + i) * stride + x0 + j];
      int mx = x, mn = x;
      for (int k = 0; k < 2; k++)
        for (int sign = -1; sign <= 1; sign += 2)
          for (int t = 0; t < 3; t++) {
            int d = t == 0 ? dir : ((dir + (t == 1 ? -2 : 2)) & 7);
            int yy = y0 + i + sign * Cdef_Directions[d][k][0];
            int xx = x0 + j + sign * Cdef_Directions[d][k][1];
            int candR = (yy * (1 << sub)) >> MI_SIZE_LOG2, candC = (xx * (1 << sub)) >> MI_SIZE_LOG2;
            if (!(candC >= 0 && candC < fp->mi_cols && candR >= 0 && candR < fp->mi_rows)) continue;
            int p = src[yy * stride + xx];
            if (t == 0)
              sum += Cdef_Pri_Taps[priStr & 1][k] * KNAME(cdef_constrain)(p - x, priStr, damping);
            else
              sum += Cdef_Sec_Taps[priStr & 1][k] * KNAME(cdef_constrain)(p - x, secStr, damping);
            mx = k_max(p, mx);
            mn = k_min(p, mn);
          }
      dst[(y0 + i) * stride + x0 + j] = (KPIX)Clip3(mn, mx, x + ((8 + sum - (sum < 0)) >> 4));
    }
}

INLINE void KNAME(cdef_copy)(DEV const FrameParams *fp, DEV const KPIX *cur, DEV KPIX *out, int r, int c) {
  for (int plane = 0; plane < 3; plane++) {
    int sub = plane > 0, stride = fp->stride[plane];
    int x0 = (c * MI_SIZE) >> sub, y0 = (r * MI_SIZE) >> sub, n = 8 >> sub;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        out[fp->offset[plane] + (y0 + i) * stride + x0 + j] = cur[fp->offset[plane] + (y0 + i) * stride + x0 + j];
  }
}

/* 7.15.1 for the 8x8 block at mi (r, c) with parameter set `idx` (-1: copy). */
INLINE void KNAME(cdef_block)(DEV const FrameParams *fp, DEV const KPIX *cur, DEV KPIX *out, DEV const uint8_t *info,
                              int r, int c, int idx) {
  KNAME(cdef_copy)(fp, cur, out, r, c);
  if (idx < 0) return;
  int skip = (info[r * fp->mi_cols + c] >> 5) & 1;
  if (skip) return;
  int yDir, var;
  KNAME(cdef_direction)(cur + fp->offset[0] + r * MI_SIZE * fp->stride[0] + c * MI_SIZE, fp->stride[0], fp->bd_mid,
                        &yDir, &var);
  /* 7.15: priStr/secStr and damping are scaled by coeffShift = bd - 8 */
  int sh = fp->coeff_shift;
  int damping = fp->cdef_damping + sh;
  int priStr = fp->cdef_y_pri[idx] << sh, secStr = fp->cdef_y_sec[idx] << sh;
  int dir = priStr == 0 ? 0 : yDir;
  int varStr = (var >> 6) ? k_min(FloorLog2(var >> 6), 12) : 0;
  priStr = var ? (priStr * (4 + varStr) + 8) >> 4 : 0;
  KNAME(cdef_filter)(fp, cur, out, 0, r, c, priStr, secStr, damping, dir);
  priStr = fp->cdef_uv_pri[idx] << sh;
  secStr = fp->cdef_uv_sec[idx] << sh;
  dir = priStr == 0 ? 0 : Cdef_Uv_Dir[1][1][yDir];
  KNAME(cdef_filter)(fp, cur, out, 1, r, c, priStr, secStr, damping - 1, dir);
  KNAME(cdef_filter)(fp, cur, out, 2, r, c, priStr, secStr, damping - 1, dir);
}

/* KNAME(cdef_block)( ) of the 8x8 block at mi (r, c) for parameter sets
 * 0 .. n-1, set k written to out + k * frameBytes; the direction is
 * computed once and an all-zero set is a copy (as filtering returns x). */
INLINE void KNAME(cdef_block_all)(DEV const FrameParams *fp, DEV const KPIX *cur, DEV KPIX *out, long frameBytes,
                                  DEV const uint8_t *info, int r, int c, int n) {
  int skip = (info[r * fp->mi_cols + c] >> 5) & 1;
  int yDir = 0, var = 0;
  if (!skip)
    KNAME(cdef_direction)(cur + fp->offset[0] + r * MI_SIZE * fp->stride[0] + c * MI_SIZE, fp->stride[0], fp->bd_mid,
                          &yDir, &var);
  int sh = fp->coeff_shift;
  int damping = fp->cdef_damping + sh;
  for (int idx = 0; idx < n; idx++) {
    DEV KPIX *o = out + (long)idx * frameBytes;
    KNAME(cdef_copy)(fp, cur, o, r, c);
    if (skip) continue;
    int priStr = fp->cdef_y_pri[idx] << sh, secStr = fp->cdef_y_sec[idx] << sh;
    int uvPri = fp->cdef_uv_pri[idx] << sh, uvSec = fp->cdef_uv_sec[idx] << sh;
    if (priStr == 0 && secStr == 0 && uvPri == 0 && uvSec == 0) continue;
    int dir = priStr == 0 ? 0 : yDir;
    int varStr = (var >> 6) ? k_min(FloorLog2(var >> 6), 12) : 0;
    priStr = var ? (priStr * (4 + varStr) + 8) >> 4 : 0;
    KNAME(cdef_filter)(fp, cur, o, 0, r, c, priStr, secStr, damping, dir);
    dir = uvPri == 0 ? 0 : Cdef_Uv_Dir[1][1][yDir];
    KNAME(cdef_filter)(fp, cur, o, 1, r, c, uvPri, uvSec, damping - 1, dir);
    KNAME(cdef_filter)(fp, cur, o, 2, r, c, uvPri, uvSec, damping - 1, dir);
  }
}
