/*
 * Intra prediction, transcribed from AV1 spec section 7.11.2.
 * Prediction is written straight into the reconstruction plane, exactly as
 * the decoder writes CurrFrame.
 */
#ifndef K_INTRA_H
#define K_INTRA_H

#include "k_common.h"

#define EDGE_OFF 16   /* index offset so that AboveRow[-2] etc. are addressable */
#define EDGE_LEN (EDGE_OFF + 129 * 2 + 16)

INLINE int intra_is_directional(int mode) { return mode >= V_PRED && mode <= D67_PRED; }

INLINE KCONST int32_t *intra_sm_weights(int log2) {
  if (log2 == 2) return Sm_Weights_Tx_4x4;
  if (log2 == 3) return Sm_Weights_Tx_8x8;
  if (log2 == 4) return Sm_Weights_Tx_16x16;
  if (log2 == 5) return Sm_Weights_Tx_32x32;
  return Sm_Weights_Tx_64x64;
}

/* Intra edge filter strength selection (7.11.2.9). */
INLINE int intra_edge_strength(int w, int h, int filterType, int delta) {
  int d = k_abs(delta), blkWh = w + h, strength = 0;
  if (filterType == 0) {
    if (blkWh <= 8) { if (d >= 56) strength = 1; }
    else if (blkWh <= 12) { if (d >= 40) strength = 1; }
    else if (blkWh <= 16) { if (d >= 40) strength = 1; }
    else if (blkWh <= 24) { if (d >= 8) strength = 1; if (d >= 16) strength = 2; if (d >= 32) strength = 3; }
    else if (blkWh <= 32) { strength = 1; if (d >= 4) strength = 2; if (d >= 32) strength = 3; }
    else strength = 3;
  } else {
    if (blkWh <= 8) { if (d >= 40) strength = 1; if (d >= 64) strength = 2; }
    else if (blkWh <= 16) { if (d >= 20) strength = 1; if (d >= 48) strength = 2; }
    else if (blkWh <= 24) { if (d >= 4) strength = 3; }
    else strength = 3;
  }
  return strength;
}

INLINE int intra_use_upsample(int w, int h, int filterType, int delta) {
  int d = k_abs(delta), blkWh = w + h;
  if (d <= 0 || d >= 40) return 0;
  return filterType == 0 ? (blkWh <= 16) : (blkWh <= 8);
}

/* Intra edge filter (7.11.2.12); buf points at element 0 (index -1 valid). */
INLINE void intra_edge_filter(THR int32_t *buf, int sz, int strength) {
  if (strength == 0) return;
  int32_t edge[129 + 2];
  for (int i = 0; i < sz; i++) edge[i] = buf[i - 1];
  for (int i = 1; i < sz; i++) {
    int s = 0;
    for (int j = 0; j < INTRA_EDGE_TAPS; j++) {
      int k = Clip3(0, sz - 1, i - 2 + j);
      s += Intra_Edge_Kernel[strength - 1][j] * edge[k];
    }
    buf[i - 1] = (s + 8) >> 4;
  }
}

/* Intra edge upsample (7.11.2.11). */
INLINE void intra_edge_upsample(THR int32_t *buf, int numPx) {
  int32_t dup[64 + 3];
  dup[0] = buf[-1];
  for (int i = -1; i < numPx; i++) dup[i + 2] = buf[i];
  dup[numPx + 2] = buf[numPx - 1];
  buf[-2] = dup[0];
  for (int i = 0; i < numPx; i++) {
    int s = -dup[i] + (9 * dup[i + 1]) + (9 * dup[i + 2]) - dup[i + 3];
    s = Clip1(Round2(s, 4));
    buf[2 * i - 1] = s;
    buf[2 * i] = dup[i + 2];
  }
}

typedef struct {
  int32_t haveLeft, haveAbove, haveAboveRight, haveBelowLeft;
  int32_t mode, angleDelta, log2W, log2H;
  int32_t maxX, maxY;
  int32_t edgeFilter;      /* enable_intra_edge_filter */
  int32_t filterType;      /* result of get_filter_type( plane ) */
} IntraArgs;

/*
 * predict_intra( ) for one transform block at (x, y) of a plane whose
 * samples live in `px` with row stride `stride`. The prediction is written to
 * `dst` (top-left of the block, row stride `dstStride`), which may alias px.
 */
INLINE void predict_intra(DEV const uint8_t *px, int stride, int x, int y, IntraArgs a,
                          DEV uint8_t *dst, int dstStride) {
  int w = 1 << a.log2W, h = 1 << a.log2H;
  int32_t aboveBuf[EDGE_LEN], leftBuf[EDGE_LEN];
  THR int32_t *AboveRow = aboveBuf + EDGE_OFF;
  THR int32_t *LeftCol = leftBuf + EDGE_OFF;
  int haveLeft = a.haveLeft, haveAbove = a.haveAbove;

  for (int i = 0; i < w + h; i++) {
    if (!haveAbove && haveLeft) AboveRow[i] = px[y * stride + x - 1];
    else if (!haveAbove && !haveLeft) AboveRow[i] = (1 << 7) - 1;
    else {
      int aboveLimit = k_min(a.maxX, x + (a.haveAboveRight ? 2 * w : w) - 1);
      AboveRow[i] = px[(y - 1) * stride + k_min(aboveLimit, x + i)];
    }
  }
  for (int i = 0; i < w + h; i++) {
    if (!haveLeft && haveAbove) LeftCol[i] = px[(y - 1) * stride + x];
    else if (!haveLeft && !haveAbove) LeftCol[i] = (1 << 7) + 1;
    else {
      int leftLimit = k_min(a.maxY, y + (a.haveBelowLeft ? 2 * h : h) - 1);
      LeftCol[i] = px[k_min(leftLimit, y + i) * stride + x - 1];
    }
  }
  if (haveAbove && haveLeft) AboveRow[-1] = px[(y - 1) * stride + x - 1];
  else if (haveAbove) AboveRow[-1] = px[(y - 1) * stride + x];
  else if (haveLeft) AboveRow[-1] = px[y * stride + x - 1];
  else AboveRow[-1] = 1 << 7;
  LeftCol[-1] = AboveRow[-1];

  int mode = a.mode;
  if (intra_is_directional(mode)) {
    int pAngle = Mode_To_Angle[mode] + a.angleDelta * ANGLE_STEP;
    int upsampleAbove = 0, upsampleLeft = 0;
    if (a.edgeFilter) {
      if (pAngle != 90 && pAngle != 180) {
        if (pAngle > 90 && pAngle < 180 && (w + h) >= 24) {
          int s = LeftCol[0] * 5 + AboveRow[-1] * 6 + AboveRow[0] * 5;
          LeftCol[-1] = AboveRow[-1] = Round2(s, 4);
        }
        if (haveAbove) {
          int strength = intra_edge_strength(w, h, a.filterType, pAngle - 90);
          int numPx = k_min(w, (a.maxX - x + 1)) + (pAngle < 90 ? h : 0) + 1;
          intra_edge_filter(AboveRow, numPx, strength);
        }
        if (haveLeft) {
          int strength = intra_edge_strength(w, h, a.filterType, pAngle - 180);
          int numPx = k_min(h, (a.maxY - y + 1)) + (pAngle > 180 ? w : 0) + 1;
          intra_edge_filter(LeftCol, numPx, strength);
        }
      }
      upsampleAbove = intra_use_upsample(w, h, a.filterType, pAngle - 90);
      if (upsampleAbove) intra_edge_upsample(AboveRow, w + (pAngle < 90 ? h : 0));
      upsampleLeft = intra_use_upsample(w, h, a.filterType, pAngle - 180);
      if (upsampleLeft) intra_edge_upsample(LeftCol, h + (pAngle > 180 ? w : 0));
    }
    int dx = 0, dy = 0;
    if (pAngle < 90) dx = Dr_Intra_Derivative[pAngle];
    else if (pAngle > 90 && pAngle < 180) dx = Dr_Intra_Derivative[180 - pAngle];
    if (pAngle > 90 && pAngle < 180) dy = Dr_Intra_Derivative[pAngle - 90];
    else if (pAngle > 180) dy = Dr_Intra_Derivative[270 - pAngle];
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        int pred;
        if (pAngle < 90) {
          int idx = (i + 1) * dx;
          int base = (idx >> (6 - upsampleAbove)) + (j << upsampleAbove);
          int shift = ((idx * (1 << upsampleAbove)) >> 1) & 0x1F;
          int maxBaseX = (w + h - 1) << upsampleAbove;
          if (base < maxBaseX)
            pred = Round2(AboveRow[base] * (32 - shift) + AboveRow[base + 1] * shift, 5);
          else
            pred = AboveRow[maxBaseX];
        } else if (pAngle > 90 && pAngle < 180) {
          int idx = (j << 6) - (i + 1) * dx;
          int base = idx >> (6 - upsampleAbove);
          if (base >= -(1 << upsampleAbove)) {
            int shift = ((idx * (1 << upsampleAbove)) >> 1) & 0x1F;
            pred = Round2(AboveRow[base] * (32 - shift) + AboveRow[base + 1] * shift, 5);
          } else {
            idx = (i << 6) - (j + 1) * dy;
            base = idx >> (6 - upsampleLeft);
            int shift = ((idx * (1 << upsampleLeft)) >> 1) & 0x1F;
            pred = Round2(LeftCol[base] * (32 - shift) + LeftCol[base + 1] * shift, 5);
          }
        } else if (pAngle > 180) {
          int idx = (j + 1) * dy;
          int base = (idx >> (6 - upsampleLeft)) + (i << upsampleLeft);
          int shift = ((idx * (1 << upsampleLeft)) >> 1) & 0x1F;
          pred = Round2(LeftCol[base] * (32 - shift) + LeftCol[base + 1] * shift, 5);
        } else if (pAngle == 90) {
          pred = AboveRow[j];
        } else {
          pred = LeftCol[i];
        }
        dst[i * dstStride + j] = (uint8_t)pred;
      }
    }
  } else if (mode == SMOOTH_PRED) {
    KCONST int32_t *wX = intra_sm_weights(a.log2W);
    KCONST int32_t *wY = intra_sm_weights(a.log2H);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int s = wY[i] * AboveRow[j] + (256 - wY[i]) * LeftCol[h - 1] +
                wX[j] * LeftCol[i] + (256 - wX[j]) * AboveRow[w - 1];
        dst[i * dstStride + j] = (uint8_t)Round2(s, 9);
      }
  } else if (mode == SMOOTH_V_PRED) {
    KCONST int32_t *wY = intra_sm_weights(a.log2H);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int s = wY[i] * AboveRow[j] + (256 - wY[i]) * LeftCol[h - 1];
        dst[i * dstStride + j] = (uint8_t)Round2(s, 8);
      }
  } else if (mode == SMOOTH_H_PRED) {
    KCONST int32_t *wX = intra_sm_weights(a.log2W);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int s = wX[j] * LeftCol[i] + (256 - wX[j]) * AboveRow[w - 1];
        dst[i * dstStride + j] = (uint8_t)Round2(s, 8);
      }
  } else if (mode == DC_PRED) {
    int avg;
    if (haveLeft && haveAbove) {
      int sum = 0;
      for (int k = 0; k < h; k++) sum += LeftCol[k];
      for (int k = 0; k < w; k++) sum += AboveRow[k];
      sum += (w + h) >> 1;
      avg = sum / (w + h);
    } else if (haveLeft) {
      int sum = 0;
      for (int k = 0; k < h; k++) sum += LeftCol[k];
      avg = Clip1((sum + (h >> 1)) >> a.log2H);
    } else if (haveAbove) {
      int sum = 0;
      for (int k = 0; k < w; k++) sum += AboveRow[k];
      avg = Clip1((sum + (w >> 1)) >> a.log2W);
    } else {
      avg = 1 << 7;
    }
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) dst[i * dstStride + j] = (uint8_t)avg;
  } else { /* PAETH_PRED */
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int base = AboveRow[j] + LeftCol[i] - AboveRow[-1];
        int pLeft = k_abs(base - LeftCol[i]);
        int pTop = k_abs(base - AboveRow[j]);
        int pTopLeft = k_abs(base - AboveRow[-1]);
        int pred;
        if (pLeft <= pTop && pLeft <= pTopLeft) pred = LeftCol[i];
        else if (pTop <= pTopLeft) pred = AboveRow[j];
        else pred = AboveRow[-1];
        dst[i * dstStride + j] = (uint8_t)pred;
      }
  }
}

/* Chroma-from-luma, spec 7.11.5. `luma` is the luma plane origin, `dst`
 * the chroma tx block holding the DC prediction (overwritten). */
INLINE void cfl_predict(DEV const uint8_t *luma, int lumaStride, int startX, int startY, int maxLumaW,
                        int maxLumaH, int log2W, int log2H, int alpha, DEV const int32_t *dcPred,
                        DEV uint8_t *dst, int dstStride) {
  int w = 1 << log2W, h = 1 << log2H;
  int lumaAvg = 0;
  for (int pass = 0; pass < 2; pass++) {
    for (int i = 0; i < h; i++) {
      int lumaY = k_min((startY + i) << 1, maxLumaH - 2);
      for (int j = 0; j < w; j++) {
        int lumaX = k_min((startX + j) << 1, maxLumaW - 2);
        DEV const uint8_t *l = luma + lumaY * lumaStride + lumaX;
        int v = ((int)l[0] + l[1] + l[lumaStride] + l[lumaStride + 1]) << 1;
        if (pass == 0) {
          lumaAvg += v;
        } else {
          int scaled = Round2Signed(alpha * (v - lumaAvg), 6);
          dst[i * dstStride + j] = (uint8_t)Clip3(0, 255, dcPred[i * w + j] + scaled);
        }
      }
    }
    if (pass == 0) lumaAvg = Round2(lumaAvg, log2W + log2H);
  }
}

/* Picks the alpha in [-16, 16] minimising the SSE against `src`. */
INLINE int cfl_best_alpha(DEV const uint8_t *luma, int lumaStride, int startX, int startY, int maxLumaW,
                          int maxLumaH, int log2W, int log2H, DEV const int32_t *dcPred, DEV const uint8_t *src,
                          int srcStride, DEV uint8_t *dst, int dstStride, int excludeZero) {
  /* SSE of every alpha in one pass over the samples (the same values as
   * predicting the block 33 times); first minimum over alpha = -16..16. */
  int w = 1 << log2W, h = 1 << log2H;
  int lumaAvg = 0;
  for (int i = 0; i < h; i++) {
    DEV const uint8_t *row = luma + k_min((startY + i) << 1, maxLumaH - 2) * lumaStride;
    for (int j = 0; j < w; j++) {
      DEV const uint8_t *l = row + k_min((startX + j) << 1, maxLumaW - 2);
      lumaAvg += ((int)l[0] + l[1] + l[lumaStride] + l[lumaStride + 1]) << 1;
    }
  }
  lumaAvg = Round2(lumaAvg, log2W + log2H);
  long sse[33];
  for (int a = 0; a < 33; a++) sse[a] = 0;
  for (int i = 0; i < h; i++) {
    DEV const uint8_t *row = luma + k_min((startY + i) << 1, maxLumaH - 2) * lumaStride;
    for (int j = 0; j < w; j++) {
      DEV const uint8_t *l = row + k_min((startX + j) << 1, maxLumaW - 2);
      int v = (((int)l[0] + l[1] + l[lumaStride] + l[lumaStride + 1]) << 1) - lumaAvg;
      int dp = dcPred[i * w + j], s0 = src[i * srcStride + j];
      for (int a = 0; a < 33; a++) {
        int d = Clip3(0, 255, dp + Round2Signed((a - 16) * v, 6)) - s0;
        sse[a] += d * d;
      }
    }
  }
  long best = -1;
  int bestA = 0;
  for (int a = -16; a <= 16; a++) {
    if (a == 0 && excludeZero) continue;
    if (best < 0 || sse[a + 16] < best) { best = sse[a + 16]; bestA = a; }
  }
  return bestA;
}

#endif
