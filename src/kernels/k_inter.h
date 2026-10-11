/*
 * Single-reference translational inter prediction, transcribed from AV1 spec
 * sections 7.11.3.2 (rounding), 7.11.3.3 (motion vector scaling, unscaled
 * reference) and 7.11.3.4 (block inter prediction).
 */

#include "k_common.h"


#ifndef __METAL_VERSION__
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
/* CPU fast path of both predictors (unscaled references: the filter phases
 * are constant over the block). Same arithmetic as the spec loops: rows
 * that only meet zero vertical taps are skipped, interior rows read
 * without clamping. Writes Clip1( Round2( ., round1 ) ) pixels to dst8, or
 * Round2( ., round1 ) ints to dst32. The NEON loads are 8-bit only; 10-bit
 * runs the identical scalar loops. */
static inline void KNAME(inter_fast)(const KPIX *ref, int refStride, int lastX, int lastY, int startX, int startY, int w,
                                     int h, int fh, int fv, int round1, int pxMax, KPIX *dst8, int32_t *dst32, int dstStride,
                                     int32_t *tmp) {
  int x0 = (startX >> 10) - 3, y0 = (startY >> 10) - 3;
  int px = (startX >> 6) & SUBPEL_MASK, py = (startY >> 6) & SUBPEL_MASK;
  (void)pxMax;
  int fH[8], fV[8];
  for (int t = 0; t < 8; t++) { fH[t] = Subpel_Filters[fh][px][t]; fV[t] = Subpel_Filters[fv][py][t]; }
  int rlo = py ? 0 : 3, rhi = py ? h + 7 : h + 3;
  int interior = x0 >= 0 && x0 + w + 7 <= lastX + 1;
  int clipX[64 + 8];
  if (!interior)
    for (int c = 0; c < w + 7; c++) clipX[c] = Clip3(0, lastX, x0 + c);
  for (int r = rlo; r < rhi; r++) {
    const KPIX *row = ref + Clip3(0, lastY, y0 + r) * refStride;
    int32_t *t = tmp + r * w;
    if (!px) {
      int c = 0;
#if defined(__aarch64__) && MAV1_BD == 8
      if (interior)
        for (; c + 8 <= w; c += 8) {
          uint16x8_t u = vshll_n_u8(vld1_u8(row + x0 + 3 + c), 4);
          vst1q_s32(t + c, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u))));
          vst1q_s32(t + c + 4, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u))));
        }
#endif
      for (; c < w; c++) t[c] = (int32_t)(interior ? row[x0 + 3 + c] : row[clipX[c + 3]]) << 4;
    } else if (interior) {
      const KPIX *q = row + x0;
      int c = 0;
#if defined(__aarch64__) && MAV1_BD == 8
      /* eight taps of eight outputs; each tap's window is one u8x8 load */
      for (; c + 8 <= w; c += 8) {
        int32x4_t alo = vdupq_n_s32(0), ahi = alo;
        for (int k = 0; k < 8; k++) {
          int16x8_t win = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(q + c + k)));
          int16x4_t fk = vdup_n_s16(fH[k]);
          alo = vmlal_s16(alo, vget_low_s16(win), fk);
          ahi = vmlal_s16(ahi, vget_high_s16(win), fk);
        }
        int32x4_t four = vdupq_n_s32(4);
        vst1q_s32(t + c, vshrq_n_s32(vaddq_s32(alo, four), 3));
        vst1q_s32(t + c + 4, vshrq_n_s32(vaddq_s32(ahi, four), 3));
      }
#endif
      for (; c < w; c++) {
        int a = 0;
        for (int k = 0; k < 8; k++) a += fH[k] * q[c + k];
        t[c] = (a + 4) >> 3;
      }
    } else {
      for (int c = 0; c < w; c++) {
        int a = 0;
        for (int k = 0; k < 8; k++) a += fH[k] * row[clipX[c + k]];
        t[c] = (a + 4) >> 3;
      }
    }
  }
  int rr = 1 << (round1 - 1);
  for (int r = 0; r < h; r++) {
    const int32_t *t = tmp + r * w;
    int c = 0;
#if defined(__aarch64__) && MAV1_BD == 8
    for (; c + 4 <= w; c += 4) {
      int64x2_t lo, hi;
      if (!py) {
        int32x4_t tv = vld1q_s32(t + 3 * w + c);
        lo = vmull_s32(vget_low_s32(tv), vdup_n_s32(128));
        hi = vmull_s32(vget_high_s32(tv), vdup_n_s32(128));
      } else {
        lo = hi = vdupq_n_s64(0);
        for (int k = 0; k < 8; k++) {
          int32x4_t tv = vld1q_s32(t + k * w + c);
          int32x2_t fk = vdup_n_s32(fV[k]);
          lo = vmlal_s32(lo, vget_low_s32(tv), fk);
          hi = vmlal_s32(hi, vget_high_s32(tv), fk);
        }
      }
      int64x2_t rrv = vdupq_n_s64(rr), sh = vdupq_n_s64(-round1);
      lo = vshlq_s64(vaddq_s64(lo, rrv), sh);
      hi = vshlq_s64(vaddq_s64(hi, rrv), sh);
      int32x4_t v = vuzp1q_s32(vreinterpretq_s32_s64(lo), vreinterpretq_s32_s64(hi));
      if (dst8 && MAV1_BD == 8) {
        /* Clip1: saturate s32 -> s16 -> u8 (8-bit pixels only) */
        uint8x8_t b = vqmovun_s16(vcombine_s16(vqmovn_s32(v), vdup_n_s16(0)));
        vst1_lane_u32((uint32_t *)(dst8 + r * dstStride + c), vreinterpret_u32_u8(b), 0);
      } else {
        vst1q_s32(dst32 + r * dstStride + c, v);
      }
    }
#endif
    if (!py) {
      for (; c < w; c++) {
        int v = (t[3 * w + c] * 128 + rr) >> round1;
        if (dst8) dst8[r * dstStride + c] = (KPIX)Clip3(0, pxMax, v);
        else dst32[r * dstStride + c] = v;
      }
    } else {
      for (; c < w; c++) {
        int a = 0;
        for (int k = 0; k < 8; k++) a += fV[k] * t[k * w + c];
        int v = (a + rr) >> round1;
        if (dst8) dst8[r * dstStride + c] = (KPIX)Clip3(0, pxMax, v);
        else dst32[r * dstStride + c] = v;
      }
    }
  }
}
#endif

/*
 * Predicts a w x h block of one plane at sample position (x, y) and writes
 * Clip1( pred ) to dst (top-left of the block, row stride dstStride). `refW`/`refH` are the reference frame's luma
 * RefUpscaledWidth / RefFrameHeight (the real frame size, not the 8-aligned
 * allocation). `tmp` must hold (h + 7) * w ints.
 */
INLINE void KNAME(predict_inter_block)(DEV const KPIX *ref, int refStride, int refW, int refH,
                                       DEV KPIX *dst, int dstStride, int x, int y, int w, int h,
                                       int mvRow, int mvCol, int subX, int subY,
                                       int filter, /* InterpFilters: [0] (vertical) | [1] (horizontal) << 2 */
                                       DEV int32_t *tmp, int pxMax) {
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

#ifndef __METAL_VERSION__
  if (w <= 64) {
    (void)intermediateHeight;
    KNAME(inter_fast)(ref, refStride, lastX, lastY, startX, startY, w, h, fh, fv, InterRound1, pxMax, dst, NULL,
                      dstStride, tmp);
    return;
  }
#endif
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
      dst[r * dstStride + c] = (KPIX)Clip3(0, pxMax, Round2(s, InterRound1));
    }
  }
}

/* Compound prediction of one reference (7.11.3.1 with isCompound = 1): the
 * vertical filter output rounded by InterRound1 = 7, not clipped.
 * The final sample is Clip1( Round2( p0 + p1, 1 + InterPostRound ) ) with
 * InterPostRound = 4. */
INLINE void KNAME(predict_inter_raw)(DEV const KPIX *ref, int refStride, int refW, int refH,
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

#ifndef __METAL_VERSION__
  if (w <= 64) {
    (void)intermediateHeight;
    KNAME(inter_fast)(ref, refStride, lastX, lastY, startX, startY, w, h, fh, fv, InterRound1, 0, NULL, dst,
                      dstStride, tmp);
    return;
  }
#endif
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
