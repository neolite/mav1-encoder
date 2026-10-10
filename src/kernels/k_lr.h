/*
 * Loop restoration (spec 7.17): Wiener and self-guided filters applied per
 * 4x4 luma block (2x2 in chroma), exactly as the decoding process describes.
 * `cdef` is UpscaledCdefFrame, `dbk` is UpscaledCurrFrame (deblocked, before
 * CDEF), the result goes to `out` (LrFrame).
 */
#ifndef K_LR_H
#define K_LR_H

#include "k_common.h"

typedef struct {
  int32_t type;         /* LrType: RESTORE_NONE / RESTORE_WIENER / RESTORE_SGRPROJ */
  int32_t set;          /* LrSgrSet */
  int32_t xqd[2];       /* LrSgrXqd */
  int32_t wiener[2][3]; /* LrWiener[pass][j]: pass 0 vertical, pass 1 horizontal */
  int32_t pad[2];
} LrUnit;

typedef struct {
  int x, y, w, h;          /* block in plane samples */
  int stripeStart, stripeEnd;
  int planeEndX, planeEndY;
  int unitRow, unitCol;
} LrBlock;

/* Geometry of loop_restore_block( plane, row, col ); returns 0 when empty. */
INLINE int lr_block_setup(DEV const FrameParams *fp, int plane, int row, int col, THR LrBlock *lb) {
  int sub = plane > 0;
  int lumaY = row * MI_SIZE;
  int stripeNum = (lumaY + 8) / 64;
  lb->stripeStart = (-8 + stripeNum * 64) >> sub;
  lb->stripeEnd = lb->stripeStart + (64 >> sub) - 1;
  int unitSize = fp->lr_unit_size[plane];
  lb->unitRow = k_min(fp->lr_unit_rows[plane] - 1, ((row * MI_SIZE + 8) >> sub) / unitSize);
  lb->unitCol = k_min(fp->lr_unit_cols[plane] - 1, ((col * MI_SIZE) >> sub) / unitSize);
  lb->planeEndX = ((fp->frame_w + sub) >> sub) - 1;
  lb->planeEndY = ((fp->frame_h + sub) >> sub) - 1;
  lb->x = (col * MI_SIZE) >> sub;
  lb->y = (row * MI_SIZE) >> sub;
  lb->w = k_min(MI_SIZE >> sub, lb->planeEndX - lb->x + 1);
  lb->h = k_min(MI_SIZE >> sub, lb->planeEndY - lb->y + 1);
  return lb->w > 0 && lb->h > 0;
}

INLINE int lr_unit_index(DEV const FrameParams *fp, int plane, THR const LrBlock *lb) {
  return fp->lr_unit_off[plane] + lb->unitRow * fp->lr_unit_cols[plane] + lb->unitCol;
}

/* get_source_sample( ) (7.17.6) */
INLINE int lr_src(DEV const FrameParams *fp, int plane, THR const LrBlock *lb, DEV const uint8_t *cdef,
                  DEV const uint8_t *dbk, int x, int y) {
  x = k_max(0, k_min(lb->planeEndX, x));
  y = k_max(0, k_min(lb->planeEndY, y));
  int stride = fp->stride[plane];
  if (y < lb->stripeStart) {
    y = k_max(lb->stripeStart - 2, y);
    return dbk[fp->offset[plane] + y * stride + x];
  } else if (y > lb->stripeEnd) {
    y = k_min(lb->stripeEnd + 2, y);
    return dbk[fp->offset[plane] + y * stride + x];
  }
  return cdef[fp->offset[plane] + y * stride + x];
}

/* Wiener filter process (7.17.4) for 8-bit: InterRound0 = 3, InterRound1 = 11. */
INLINE void lr_wiener(DEV const FrameParams *fp, int plane, THR const LrBlock *lb, DEV const uint8_t *cdef,
                      DEV const uint8_t *dbk, DEV uint8_t *out, THR const LrUnit *u) {
  int vf[7], hf[7];
  vf[3] = 128;
  hf[3] = 128;
  for (int i = 0; i < 3; i++) {
    vf[i] = vf[6 - i] = u->wiener[0][i];
    vf[3] -= 2 * u->wiener[0][i];
    hf[i] = hf[6 - i] = u->wiener[1][i];
    hf[3] -= 2 * u->wiener[1][i];
  }
  const int round0 = 3, round1 = 11;
  int offset = 1 << (8 + FILTER_BITS - round0 - 1);
  int limit = (1 << (8 + 1 + FILTER_BITS - round0)) - 1;
  int inter[10][4];
  for (int r = 0; r < lb->h + 6; r++)
    for (int c = 0; c < lb->w; c++) {
      int s = 0;
      for (int t = 0; t < 7; t++) s += hf[t] * lr_src(fp, plane, lb, cdef, dbk, lb->x + c + t - 3, lb->y + r - 3);
      int v = Round2(s, round0);
      inter[r][c] = Clip3(-offset, limit - offset, v);
    }
  int stride = fp->stride[plane];
  for (int r = 0; r < lb->h; r++)
    for (int c = 0; c < lb->w; c++) {
      int s = 0;
      for (int t = 0; t < 7; t++) s += vf[t] * inter[r + t][c];
      int v = Round2(s, round1);
      out[fp->offset[plane] + (lb->y + r) * stride + lb->x + c] = (uint8_t)Clip3(0, 255, v);
    }
}

/* Box filter process (7.17.3); F gets w x h values (row stride 4). */
INLINE void lr_box(DEV const FrameParams *fp, int plane, THR const LrBlock *lb, DEV const uint8_t *cdef,
                   DEV const uint8_t *dbk, int set, int pass, THR int *F) {
  int r = Sgr_Params[set][pass * 2 + 0];
  if (r == 0) return;
  int eps = Sgr_Params[set][pass * 2 + 1];
  int A[6][6], B[6][6];
  int n = (2 * r + 1) * (2 * r + 1);
  int n2e = n * n * eps;
  int s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
  int oneOverN = ((1 << SGRPROJ_RECIP_BITS) + (n / 2)) / n;
  for (int i = -1; i < lb->h + 1; i++)
    for (int j = -1; j < lb->w + 1; j++) {
      long a = 0, b = 0;
      for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++) {
          long c = lr_src(fp, plane, lb, cdef, dbk, lb->x + j + dx, lb->y + i + dy);
          a += c * c;
          b += c;
        }
      /* BitDepth 8: the Round2 by 0 are no-ops */
      long d = b;
      long p = a * n - d * d;
      if (p < 0) p = 0;
      long z = (p * s + (1L << (SGRPROJ_MTABLE_BITS - 1))) >> SGRPROJ_MTABLE_BITS;
      long a2;
      if (z >= 255) a2 = 256;
      else if (z == 0) a2 = 1;
      else a2 = ((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1);
      long b2 = ((1L << SGRPROJ_SGR_BITS) - a2) * b * oneOverN;
      A[i + 1][j + 1] = (int)a2;
      B[i + 1][j + 1] = (int)((b2 + (1L << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS);
    }
  int stride = fp->stride[plane];
  for (int i = 0; i < lb->h; i++) {
    int shift = 5;
    if (pass == 0 && (i & 1)) shift = 4;
    for (int j = 0; j < lb->w; j++) {
      long a = 0, b = 0;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          int weight;
          if (pass == 0) weight = ((i + dy) & 1) ? (dx == 0 ? 6 : 5) : 0;
          else weight = (dx == 0 || dy == 0) ? 4 : 3;
          a += weight * A[i + dy + 1][j + dx + 1];
          b += weight * B[i + dy + 1][j + dx + 1];
        }
      long v = a * cdef[fp->offset[plane] + (lb->y + i) * stride + lb->x + j] + b;
      int sh = SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS;
      F[i * 4 + j] = (int)((v + (1L << (sh - 1))) >> sh);
    }
  }
}

/* Self guided filter process (7.17.2) */
INLINE void lr_sgr(DEV const FrameParams *fp, int plane, THR const LrBlock *lb, DEV const uint8_t *cdef,
                   DEV const uint8_t *dbk, DEV uint8_t *out, THR const LrUnit *u) {
  int flt0[16], flt1[16];
  lr_box(fp, plane, lb, cdef, dbk, u->set, 0, flt0);
  lr_box(fp, plane, lb, cdef, dbk, u->set, 1, flt1);
  int w0 = u->xqd[0], w1 = u->xqd[1];
  int w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
  int r0 = Sgr_Params[u->set][0], r1 = Sgr_Params[u->set][2];
  int stride = fp->stride[plane];
  for (int i = 0; i < lb->h; i++)
    for (int j = 0; j < lb->w; j++) {
      int o = fp->offset[plane] + (lb->y + i) * stride + lb->x + j;
      int uu = cdef[o] << SGRPROJ_RST_BITS;
      int v = w1 * uu;
      v += r0 ? w0 * flt0[i * 4 + j] : w0 * uu;
      v += r1 ? w2 * flt1[i * 4 + j] : w2 * uu;
      int s = Round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
      out[o] = (uint8_t)Clip3(0, 255, s);
    }
}

/* loop_restore_block( ) with an explicit unit (used by the parameter search
 * as well); RESTORE_NONE copies the CDEF output. */
INLINE void lr_apply(DEV const FrameParams *fp, int plane, THR const LrBlock *lb, DEV const uint8_t *cdef,
                     DEV const uint8_t *dbk, DEV uint8_t *out, THR const LrUnit *u) {
  if (u->type == RESTORE_WIENER) {
    lr_wiener(fp, plane, lb, cdef, dbk, out, u);
  } else if (u->type == RESTORE_SGRPROJ) {
    lr_sgr(fp, plane, lb, cdef, dbk, out, u);
  } else {
    int stride = fp->stride[plane];
    for (int i = 0; i < lb->h; i++)
      for (int j = 0; j < lb->w; j++) {
        int o = fp->offset[plane] + (lb->y + i) * stride + lb->x + j;
        out[o] = cdef[o];
      }
  }
}

/* One work item: plane and 4x4 luma block (row, col) of the frame. */
INLINE void lr_block(DEV const FrameParams *fp, int plane, int row, int col, DEV const uint8_t *cdef,
                     DEV const uint8_t *dbk, DEV uint8_t *out, DEV const LrUnit *units) {
  LrBlock lb;
  if (!lr_block_setup(fp, plane, row, col, &lb)) return;
  LrUnit u;
  if (fp->lr_type[plane] == RESTORE_NONE) u.type = RESTORE_NONE;
  else u = units[lr_unit_index(fp, plane, &lb)];
  lr_apply(fp, plane, &lb, cdef, dbk, out, &u);
}

#endif
