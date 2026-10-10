/*
 * Single-reference translational inter prediction, transcribed from AV1 spec
 * sections 7.11.3.2 (rounding), 7.11.3.3 (motion vector scaling, unscaled
 * reference) and 7.11.3.4 (block inter prediction).
 */
#ifndef K_INTER_H
#define K_INTER_H

#include "k_common.h"

/*
 * Predicts a w x h block of one plane at sample position (x, y) and writes
 * Clip1( pred ) to dst (top-left of the block, row stride dstStride). `refW`/`refH` are the reference frame's luma
 * RefUpscaledWidth / RefFrameHeight (the real frame size, not the 8-aligned
 * allocation). `tmp` must hold (h + 7) * w ints.
 */
INLINE void predict_inter_block(DEV const uint8_t *ref, int refStride, int refW, int refH,
                                DEV uint8_t *dst, int dstStride, int x, int y, int w, int h,
                                int mvRow, int mvCol, int subX, int subY,
                                int filter, /* InterpFilters: [0] (vertical) | [1] (horizontal) << 2 */
                                DEV int32_t *tmp) {
  const int InterRound0 = 3, InterRound1 = 11;
  /* Motion vector scaling with xScale == yScale == 1 << REF_SCALE_SHIFT. */
  int halfSample = 1 << (SUBPEL_BITS - 1);
  int origX = (x << SUBPEL_BITS) + ((2 * mvCol) >> subX) + halfSample;
  int origY = (y << SUBPEL_BITS) + ((2 * mvRow) >> subY) + halfSample;
  long baseX = (long)origX * (1 << REF_SCALE_SHIFT) - ((long)halfSample << REF_SCALE_SHIFT);
  long baseY = (long)origY * (1 << REF_SCALE_SHIFT) - ((long)halfSample << REF_SCALE_SHIFT);
  int off = (1 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
  int sh = REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS;
  int startX = (int)(baseX >= 0 ? Round2L(baseX, sh) : -Round2L(-baseX, sh)) + off;
  int startY = (int)(baseY >= 0 ? Round2L(baseY, sh) : -Round2L(-baseY, sh)) + off;
  int xStep = 1 << SCALE_SUBPEL_BITS, yStep = 1 << SCALE_SUBPEL_BITS;

  int lastX = ((refW + subX) >> subX) - 1;
  int lastY = ((refH + subY) >> subY) - 1;
  int intermediateHeight = (((h - 1) * yStep + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS) + 8;

  int fv = filter & 3, fh = (filter >> 2) & 3;
  if (w <= 4) { if (fh == EIGHTTAP || fh == EIGHTTAP_SHARP) fh = 4; else if (fh == EIGHTTAP_SMOOTH) fh = 5; }
  if (h <= 4) { if (fv == EIGHTTAP || fv == EIGHTTAP_SHARP) fv = 4; else if (fv == EIGHTTAP_SMOOTH) fv = 5; }

  for (int r = 0; r < intermediateHeight; r++) {
    int ry = Clip3(0, lastY, (startY >> 10) + r - 3);
    for (int c = 0; c < w; c++) {
      int s = 0;
      int p = startX + xStep * c;
      for (int t = 0; t < 8; t++)
        s += Subpel_Filters[fh][(p >> 6) & SUBPEL_MASK][t] *
             ref[ry * refStride + Clip3(0, lastX, (p >> 10) + t - 3)];
      tmp[r * w + c] = Round2(s, InterRound0);
    }
  }
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      int s = 0;
      int p = (startY & 1023) + yStep * r;
      for (int t = 0; t < 8; t++)
        s += Subpel_Filters[fv][(p >> 6) & SUBPEL_MASK][t] * tmp[((p >> 10) + t) * w + c];
      dst[r * dstStride + c] = (uint8_t)Clip1(Round2(s, InterRound1));
    }
  }
}

/* Compound prediction of one reference (7.11.3.1 with isCompound = 1): the
 * vertical filter output rounded by InterRound1 = 7, not clipped.
 * The final sample is Clip1( Round2( p0 + p1, 1 + InterPostRound ) ) with
 * InterPostRound = 4. */
INLINE void predict_inter_raw(DEV const uint8_t *ref, int refStride, int refW, int refH,
                              DEV int32_t *dst, int dstStride, int x, int y, int w, int h,
                                int mvRow, int mvCol, int subX, int subY,
                                int filter, /* InterpFilters: [0] (vertical) | [1] (horizontal) << 2 */
                                DEV int32_t *tmp) {
  const int InterRound0 = 3, InterRound1 = 7; /* isCompound */
  /* Motion vector scaling with xScale == yScale == 1 << REF_SCALE_SHIFT. */
  int halfSample = 1 << (SUBPEL_BITS - 1);
  int origX = (x << SUBPEL_BITS) + ((2 * mvCol) >> subX) + halfSample;
  int origY = (y << SUBPEL_BITS) + ((2 * mvRow) >> subY) + halfSample;
  long baseX = (long)origX * (1 << REF_SCALE_SHIFT) - ((long)halfSample << REF_SCALE_SHIFT);
  long baseY = (long)origY * (1 << REF_SCALE_SHIFT) - ((long)halfSample << REF_SCALE_SHIFT);
  int off = (1 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
  int sh = REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS;
  int startX = (int)(baseX >= 0 ? Round2L(baseX, sh) : -Round2L(-baseX, sh)) + off;
  int startY = (int)(baseY >= 0 ? Round2L(baseY, sh) : -Round2L(-baseY, sh)) + off;
  int xStep = 1 << SCALE_SUBPEL_BITS, yStep = 1 << SCALE_SUBPEL_BITS;

  int lastX = ((refW + subX) >> subX) - 1;
  int lastY = ((refH + subY) >> subY) - 1;
  int intermediateHeight = (((h - 1) * yStep + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS) + 8;

  int fv = filter & 3, fh = (filter >> 2) & 3;
  if (w <= 4) { if (fh == EIGHTTAP || fh == EIGHTTAP_SHARP) fh = 4; else if (fh == EIGHTTAP_SMOOTH) fh = 5; }
  if (h <= 4) { if (fv == EIGHTTAP || fv == EIGHTTAP_SHARP) fv = 4; else if (fv == EIGHTTAP_SMOOTH) fv = 5; }

  for (int r = 0; r < intermediateHeight; r++) {
    int ry = Clip3(0, lastY, (startY >> 10) + r - 3);
    for (int c = 0; c < w; c++) {
      int s = 0;
      int p = startX + xStep * c;
      for (int t = 0; t < 8; t++)
        s += Subpel_Filters[fh][(p >> 6) & SUBPEL_MASK][t] *
             ref[ry * refStride + Clip3(0, lastX, (p >> 10) + t - 3)];
      tmp[r * w + c] = Round2(s, InterRound0);
    }
  }
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      int s = 0;
      int p = (startY & 1023) + yStep * r;
      for (int t = 0; t < 8; t++)
        s += Subpel_Filters[fv][(p >> 6) & SUBPEL_MASK][t] * tmp[((p >> 10) + t) * w + c];
      dst[r * dstStride + c] = Round2(s, InterRound1);
    }
  }
}

#endif
