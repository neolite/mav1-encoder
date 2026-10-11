/*
 * Inverse transforms, transcribed from AV1 spec section 7.13.
 * The encoder must reproduce the decoder's reconstruction bit-exactly, so
 * this file follows the specification step by step rather than optimising.
 */
#ifndef K_ITX_H
#define K_ITX_H

#include "k_common.h"

INLINE int itx_brev(int numBits, int x) {
  int t = 0;
  for (int i = 0; i < numBits; i++) t += ((x >> i) & 1) << (numBits - 1 - i);
  return t;
}

INLINE int itx_cos128(int angle) {
  int angle2 = angle & 255;
  if (angle2 <= 64) return Cos128_Lookup[angle2];
  if (angle2 <= 128) return -Cos128_Lookup[128 - angle2];
  if (angle2 <= 192) return -Cos128_Lookup[angle2 - 128];
  return Cos128_Lookup[256 - angle2];
}

INLINE int itx_sin128(int angle) { return itx_cos128(angle - 64); }

/* Butterfly rotation B( a, b, angle, flip ). */
INLINE void itx_B(THR int32_t *T, int a, int b, int angle, int flip) {
  long x = (long)T[a] * itx_cos128(angle) - (long)T[b] * itx_sin128(angle);
  long y = (long)T[a] * itx_sin128(angle) + (long)T[b] * itx_cos128(angle);
  T[a] = (int32_t)Round2L(x, 12);
  T[b] = (int32_t)Round2L(y, 12);
  if (flip) { int32_t t = T[a]; T[a] = T[b]; T[b] = t; }
}

/* Hadamard rotation H( a, b, flip, r ). */
INLINE void itx_H(THR int32_t *T, int a, int b, int flip, int r) {
  if (flip) { int t = a; a = b; b = t; }
  int32_t x = T[a], y = T[b];
  int lo = -(1 << (r - 1)), hi = (1 << (r - 1)) - 1;
  T[a] = Clip3(lo, hi, x + y);
  T[b] = Clip3(lo, hi, x - y);
}

INLINE void itx_dct(THR int32_t *T, int n, int r) {
  int32_t copyT[64];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) T[i] = copyT[itx_brev(n, i)];

  if (n == 6) for (int i = 0; i < 16; i++) itx_B(T, 32 + i, 63 - i, 63 - 4 * itx_brev(4, i), 0);
  if (n >= 5) for (int i = 0; i < 8; i++) itx_B(T, 16 + i, 31 - i, 6 + (itx_brev(3, 7 - i) << 3), 0);
  if (n == 6) for (int i = 0; i < 16; i++) itx_H(T, 32 + i * 2, 33 + i * 2, i & 1, r);
  if (n >= 4) for (int i = 0; i < 4; i++) itx_B(T, 8 + i, 15 - i, 12 + (itx_brev(2, 3 - i) << 4), 0);
  if (n >= 5) for (int i = 0; i < 8; i++) itx_H(T, 16 + 2 * i, 17 + 2 * i, i & 1, r);
  if (n == 6)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        itx_B(T, 62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * itx_brev(2, i) + 64 * j, 1);
  if (n >= 3) for (int i = 0; i < 2; i++) itx_B(T, 4 + i, 7 - i, 56 - 32 * i, 0);
  if (n >= 4) for (int i = 0; i < 4; i++) itx_H(T, 8 + 2 * i, 9 + 2 * i, i & 1, r);
  if (n >= 5)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        itx_B(T, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), 1);
  if (n == 6)
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 2; j++)
        itx_H(T, 32 + i * 4 + j, 35 + i * 4 - j, i & 1, r);
  for (int i = 0; i < 2; i++) itx_B(T, 2 * i, 2 * i + 1, 32 + 16 * i, 1 - i);
  if (n >= 3) for (int i = 0; i < 2; i++) itx_H(T, 4 + 2 * i, 5 + 2 * i, i, r);
  if (n >= 4) for (int i = 0; i < 2; i++) itx_B(T, 14 - i, 9 + i, 48 + 64 * i, 1);
  if (n >= 5)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        itx_H(T, 16 + 4 * i + j, 19 + 4 * i - j, i & 1, r);
  if (n == 6)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++)
        itx_B(T, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, 1);
  for (int i = 0; i < 2; i++) itx_H(T, i, 3 - i, 0, r);
  if (n >= 3) itx_B(T, 6, 5, 32, 1);
  if (n >= 4)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        itx_H(T, 8 + 4 * i + j, 11 + 4 * i - j, i, r);
  if (n >= 5) for (int i = 0; i < 4; i++) itx_B(T, 29 - i, 18 + i, 48 + (i >> 1) * 64, 1);
  if (n == 6)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        itx_H(T, 32 + 8 * i + j, 39 + 8 * i - j, i & 1, r);
  if (n >= 3) for (int i = 0; i < 4; i++) itx_H(T, i, 7 - i, 0, r);
  if (n >= 4) for (int i = 0; i < 2; i++) itx_B(T, 13 - i, 10 + i, 32, 1);
  if (n >= 5)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++)
        itx_H(T, 16 + i * 8 + j, 23 + i * 8 - j, i, r);
  if (n == 6) for (int i = 0; i < 8; i++) itx_B(T, 59 - i, 36 + i, i < 4 ? 48 : 112, 1);
  if (n >= 4) for (int i = 0; i < 8; i++) itx_H(T, i, 15 - i, 0, r);
  if (n >= 5) for (int i = 0; i < 4; i++) itx_B(T, 27 - i, 20 + i, 32, 1);
  if (n == 6)
    for (int i = 0; i < 8; i++) {
      itx_H(T, 32 + i, 47 - i, 0, r);
      itx_H(T, 48 + i, 63 - i, 1, r);
    }
  if (n >= 5) for (int i = 0; i < 16; i++) itx_H(T, i, 31 - i, 0, r);
  if (n == 6) for (int i = 0; i < 8; i++) itx_B(T, 55 - i, 40 + i, 32, 1);
  if (n == 6) for (int i = 0; i < 32; i++) itx_H(T, i, 63 - i, 0, r);
}

INLINE void itx_adst_in_perm(THR int32_t *T, int n) {
  int32_t copyT[16];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) {
    int idx = (i & 1) ? (i - 1) : (n0 - i - 1);
    T[i] = copyT[idx];
  }
}

INLINE void itx_adst_out_perm(THR int32_t *T, int n) {
  int32_t copyT[16];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) {
    int a = (i >> 3) & 1;
    int b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
    int c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
    int d = (i & 1) ^ ((i >> 1) & 1);
    int idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
    T[i] = (i & 1) ? -copyT[idx] : copyT[idx];
  }
}

#define SINPI_1_9 1321
#define SINPI_2_9 2482
#define SINPI_3_9 3344
#define SINPI_4_9 3803

INLINE void itx_adst4(THR int32_t *T) {
  long s0 = (long)SINPI_1_9 * T[0];
  long s1 = (long)SINPI_2_9 * T[0];
  long s2 = (long)SINPI_3_9 * T[1];
  long s3 = (long)SINPI_4_9 * T[2];
  long s4 = (long)SINPI_1_9 * T[2];
  long s5 = (long)SINPI_2_9 * T[3];
  long s6 = (long)SINPI_4_9 * T[3];
  long a7 = (long)T[0] - T[2];
  long b7 = a7 + T[3];
  s0 = s0 + s3;
  s1 = s1 - s4;
  s3 = s2;
  s2 = SINPI_3_9 * b7;
  s0 = s0 + s5;
  s1 = s1 - s6;
  long x0 = s0 + s3;
  long x1 = s1 + s3;
  long x2 = s2;
  long x3 = s0 + s1;
  x3 = x3 - s3;
  T[0] = (int32_t)Round2L(x0, 12);
  T[1] = (int32_t)Round2L(x1, 12);
  T[2] = (int32_t)Round2L(x2, 12);
  T[3] = (int32_t)Round2L(x3, 12);
}

INLINE void itx_adst8(THR int32_t *T, int r) {
  itx_adst_in_perm(T, 3);
  for (int i = 0; i < 4; i++) itx_B(T, 2 * i, 2 * i + 1, 60 - 16 * i, 1);
  for (int i = 0; i < 4; i++) itx_H(T, i, 4 + i, 0, r);
  for (int i = 0; i < 2; i++) itx_B(T, 4 + 3 * i, 5 + i, 48 - 32 * i, 1);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) itx_H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (int i = 0; i < 2; i++) itx_B(T, 2 + 4 * i, 3 + 4 * i, 32, 1);
  itx_adst_out_perm(T, 3);
}

INLINE void itx_adst16(THR int32_t *T, int r) {
  itx_adst_in_perm(T, 4);
  for (int i = 0; i < 8; i++) itx_B(T, 2 * i, 2 * i + 1, 62 - 8 * i, 1);
  for (int i = 0; i < 8; i++) itx_H(T, i, 8 + i, 0, r);
  for (int i = 0; i < 2; i++) {
    itx_B(T, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, 1);
    itx_B(T, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, 1);
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) itx_H(T, 8 * j + i, 4 + 8 * j + i, 0, r);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) itx_B(T, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, 1);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) itx_H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (int i = 0; i < 4; i++) itx_B(T, 2 + 4 * i, 3 + 4 * i, 32, 1);
  itx_adst_out_perm(T, 4);
}

INLINE void itx_adst(THR int32_t *T, int n, int r) {
  if (n == 2) itx_adst4(T);
  else if (n == 3) itx_adst8(T, r);
  else itx_adst16(T, r);
}

INLINE void itx_identity(THR int32_t *T, int n) {
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) {
    if (n == 2) T[i] = (int32_t)Round2L((long)T[i] * 5793, 12);
    else if (n == 3) T[i] = T[i] * 2;
    else if (n == 4) T[i] = (int32_t)Round2L((long)T[i] * 11586, 12);
    else T[i] = T[i] * 4;
  }
}

INLINE void itx_wht(THR int32_t *T, int shift) {
  int32_t a = T[0] >> shift, c = T[1] >> shift, d = T[2] >> shift, b = T[3] >> shift;
  a += c;
  d -= b;
  int32_t e = (a - d) >> 1;
  b = e - b;
  c = e - c;
  a -= b;
  d += c;
  T[0] = a; T[1] = b; T[2] = c; T[3] = d;
}

/* 1D kernel classes used by the 2D process. */
enum { ITX_DCT = 0, ITX_ADST = 1, ITX_IDTX = 2 };

INLINE int itx_row_kind(int txType) {
  if (txType == DCT_DCT || txType == ADST_DCT || txType == FLIPADST_DCT || txType == H_DCT) return ITX_DCT;
  if (txType == DCT_ADST || txType == ADST_ADST || txType == DCT_FLIPADST || txType == FLIPADST_FLIPADST ||
      txType == ADST_FLIPADST || txType == FLIPADST_ADST || txType == H_ADST || txType == H_FLIPADST)
    return ITX_ADST;
  return ITX_IDTX;
}

INLINE int itx_col_kind(int txType) {
  if (txType == DCT_DCT || txType == DCT_ADST || txType == DCT_FLIPADST || txType == V_DCT) return ITX_DCT;
  if (txType == ADST_DCT || txType == ADST_ADST || txType == FLIPADST_DCT || txType == FLIPADST_FLIPADST ||
      txType == ADST_FLIPADST || txType == FLIPADST_ADST || txType == V_ADST || txType == V_FLIPADST)
    return ITX_ADST;
  return ITX_IDTX;
}

INLINE void itx_1d(THR int32_t *T, int kind, int n, int r) {
  if (kind == ITX_DCT) itx_dct(T, n, r);
  else if (kind == ITX_ADST) itx_adst(T, n, r);
  else itx_identity(T, n);
}

INLINE int itx_flip_ud(int t) {
  return t == FLIPADST_DCT || t == FLIPADST_ADST || t == V_FLIPADST || t == FLIPADST_FLIPADST;
}

INLINE int itx_flip_lr(int t) {
  return t == DCT_FLIPADST || t == ADST_FLIPADST || t == H_FLIPADST || t == FLIPADST_FLIPADST;
}

/*
 * 2D inverse transform (spec 7.13.3). `dequant` holds the (Min(32,w) x
 * Min(32,h)) dequantised coefficients with row stride 64; `residual` receives
 * h rows of w values with row stride 64.
 */
INLINE void itx_2d_ref(DEV int32_t *dequant, DEV int32_t *residual, int txSz, int txType, int lossless) {
  int log2W = Tx_Width_Log2[txSz], log2H = Tx_Height_Log2[txSz];
  int w = 1 << log2W, h = 1 << log2H;
  int rowShift = lossless ? 0 : Transform_Row_Shift[txSz];
  int colShift = lossless ? 0 : 4;
  int rowClampRange = 8 + 8;
  int colClampRange = k_max(8 + 6, 16);
  int rowKind = itx_row_kind(txType), colKind = itx_col_kind(txType);
  int32_t T[64];
  for (int i = 0; i < h; i++) {
    int nz = 0;
    for (int j = 0; j < w; j++) {
      T[j] = (i < 32 && j < 32) ? dequant[i * 64 + j] : 0;
      nz |= T[j];
    }
    if (!nz && !lossless) { /* every row kernel maps zeros to zeros */
      for (int j = 0; j < w; j++) residual[i * 64 + j] = 0;
      continue;
    }
    if (k_abs(log2W - log2H) == 1)
      for (int j = 0; j < w; j++) T[j] = (int32_t)Round2L((long)T[j] * 2896, 12);
    if (lossless) itx_wht(T, 2);
    else itx_1d(T, rowKind, log2W, rowClampRange);
    for (int j = 0; j < w; j++) residual[i * 64 + j] = Round2(T[j], rowShift);
  }
  int lo = -(1 << (colClampRange - 1)), hi = (1 << (colClampRange - 1)) - 1;
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) residual[i * 64 + j] = Clip3(lo, hi, residual[i * 64 + j]);
  for (int j = 0; j < w; j++) {
    for (int i = 0; i < h; i++) T[i] = residual[i * 64 + j];
    if (lossless) itx_wht(T, 0);
    else itx_1d(T, colKind, log2H, colClampRange);
    for (int i = 0; i < h; i++) residual[i * 64 + j] = Round2(T[i], colShift);
  }
}

#if !defined(__METAL_VERSION__) && defined(__aarch64__)
/*
 * NEON inverse transform: the same butterflies as above with four lanes of
 * the row/column dimension packed into each register, so every operation is
 * bit-identical to the scalar kernel. The row pass works on four rows at a
 * time (lane = row, 4x4 transposes at the edges), the column pass on four
 * columns (lane = column, contiguous loads and stores).
 */
#include <arm_neon.h>

/* Four int64 lanes (one per row/column) for the 32x32 -> 64 products. */
typedef struct { int64x2_t lo, hi; } v64x4;
static inline v64x4 v64_mul(int32x4_t a, int c) {
  int32x2_t vc = vdup_n_s32(c);
  v64x4 r;
  r.lo = vmull_s32(vget_low_s32(a), vc);
  r.hi = vmull_s32(vget_high_s32(a), vc);
  return r;
}
static inline v64x4 v64_cvt(int32x4_t a) {
  v64x4 r;
  r.lo = vmovl_s32(vget_low_s32(a));
  r.hi = vmovl_s32(vget_high_s32(a));
  return r;
}
static inline v64x4 v64_add(v64x4 a, v64x4 b) {
  v64x4 r = { vaddq_s64(a.lo, b.lo), vaddq_s64(a.hi, b.hi) };
  return r;
}
static inline v64x4 v64_sub(v64x4 a, v64x4 b) {
  v64x4 r = { vsubq_s64(a.lo, b.lo), vsubq_s64(a.hi, b.hi) };
  return r;
}
/* Low 32 bits of each lane, i.e. (int32_t)x like the scalar casts. */
static inline int32x4_t v64_to32(v64x4 a) {
  return vuzp1q_s32(vreinterpretq_s32_s64(a.lo), vreinterpretq_s32_s64(a.hi));
}
static inline int32x4_t v64_rnd12(v64x4 a) {
  int64x2_t half = vdupq_n_s64(2048);
  a.lo = vshrq_n_s64(vaddq_s64(a.lo, half), 12);
  a.hi = vshrq_n_s64(vaddq_s64(a.hi, half), 12);
  return v64_to32(a);
}
static inline int32x4_t vround2(int32x4_t v, int n) {
  if (!n) return v;
  return vshlq_s32(vaddq_s32(v, vdupq_n_s32(1 << (n - 1))), vdupq_n_s32(-n));
}

static inline void vitx_B(int32x4_t *T, int a, int b, int angle, int flip) {
  v64x4 x = v64_sub(v64_mul(T[a], itx_cos128(angle)), v64_mul(T[b], itx_sin128(angle)));
  v64x4 y = v64_add(v64_mul(T[a], itx_sin128(angle)), v64_mul(T[b], itx_cos128(angle)));
  T[a] = v64_rnd12(x);
  T[b] = v64_rnd12(y);
  if (flip) {
    int32x4_t t = T[a];
    T[a] = T[b];
    T[b] = t;
  }
}

static inline void vitx_H(int32x4_t *T, int a, int b, int flip, int r) {
  if (flip) {
    int t = a;
    a = b;
    b = t;
  }
  int32x4_t x = T[a], y = T[b];
  int32x4_t lo = vdupq_n_s32(-(1 << (r - 1))), hi = vdupq_n_s32((1 << (r - 1)) - 1);
  T[a] = vminq_s32(vmaxq_s32(vaddq_s32(x, y), lo), hi);
  T[b] = vminq_s32(vmaxq_s32(vsubq_s32(x, y), lo), hi);
}

static void vitx_dct(int32x4_t *T, int n, int r) {
  int32x4_t copyT[64];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) T[i] = copyT[itx_brev(n, i)];

  if (n == 6) for (int i = 0; i < 16; i++) vitx_B(T, 32 + i, 63 - i, 63 - 4 * itx_brev(4, i), 0);
  if (n >= 5) for (int i = 0; i < 8; i++) vitx_B(T, 16 + i, 31 - i, 6 + (itx_brev(3, 7 - i) << 3), 0);
  if (n == 6) for (int i = 0; i < 16; i++) vitx_H(T, 32 + i * 2, 33 + i * 2, i & 1, r);
  if (n >= 4) for (int i = 0; i < 4; i++) vitx_B(T, 8 + i, 15 - i, 12 + (itx_brev(2, 3 - i) << 4), 0);
  if (n >= 5) for (int i = 0; i < 8; i++) vitx_H(T, 16 + 2 * i, 17 + 2 * i, i & 1, r);
  if (n == 6)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        vitx_B(T, 62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * itx_brev(2, i) + 64 * j, 1);
  if (n >= 3) for (int i = 0; i < 2; i++) vitx_B(T, 4 + i, 7 - i, 56 - 32 * i, 0);
  if (n >= 4) for (int i = 0; i < 4; i++) vitx_H(T, 8 + 2 * i, 9 + 2 * i, i & 1, r);
  if (n >= 5)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        vitx_B(T, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), 1);
  if (n == 6)
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 2; j++)
        vitx_H(T, 32 + i * 4 + j, 35 + i * 4 - j, i & 1, r);
  for (int i = 0; i < 2; i++) vitx_B(T, 2 * i, 2 * i + 1, 32 + 16 * i, 1 - i);
  if (n >= 3) for (int i = 0; i < 2; i++) vitx_H(T, 4 + 2 * i, 5 + 2 * i, i, r);
  if (n >= 4) for (int i = 0; i < 2; i++) vitx_B(T, 14 - i, 9 + i, 48 + 64 * i, 1);
  if (n >= 5)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        vitx_H(T, 16 + 4 * i + j, 19 + 4 * i - j, i & 1, r);
  if (n == 6)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++)
        vitx_B(T, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, 1);
  for (int i = 0; i < 2; i++) vitx_H(T, i, 3 - i, 0, r);
  if (n >= 3) vitx_B(T, 6, 5, 32, 1);
  if (n >= 4)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        vitx_H(T, 8 + 4 * i + j, 11 + 4 * i - j, i, r);
  if (n >= 5) for (int i = 0; i < 4; i++) vitx_B(T, 29 - i, 18 + i, 48 + (i >> 1) * 64, 1);
  if (n == 6)
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        vitx_H(T, 32 + 8 * i + j, 39 + 8 * i - j, i & 1, r);
  if (n >= 3) for (int i = 0; i < 4; i++) vitx_H(T, i, 7 - i, 0, r);
  if (n >= 4) for (int i = 0; i < 2; i++) vitx_B(T, 13 - i, 10 + i, 32, 1);
  if (n >= 5)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++)
        vitx_H(T, 16 + i * 8 + j, 23 + i * 8 - j, i, r);
  if (n == 6) for (int i = 0; i < 8; i++) vitx_B(T, 59 - i, 36 + i, i < 4 ? 48 : 112, 1);
  if (n >= 4) for (int i = 0; i < 8; i++) vitx_H(T, i, 15 - i, 0, r);
  if (n >= 5) for (int i = 0; i < 4; i++) vitx_B(T, 27 - i, 20 + i, 32, 1);
  if (n == 6)
    for (int i = 0; i < 8; i++) {
      vitx_H(T, 32 + i, 47 - i, 0, r);
      vitx_H(T, 48 + i, 63 - i, 1, r);
    }
  if (n >= 5) for (int i = 0; i < 16; i++) vitx_H(T, i, 31 - i, 0, r);
  if (n == 6) for (int i = 0; i < 8; i++) vitx_B(T, 55 - i, 40 + i, 32, 1);
  if (n == 6) for (int i = 0; i < 32; i++) vitx_H(T, i, 63 - i, 0, r);
}

static void vitx_adst_in_perm(int32x4_t *T, int n) {
  int32x4_t copyT[16];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) {
    int idx = (i & 1) ? (i - 1) : (n0 - i - 1);
    T[i] = copyT[idx];
  }
}

static void vitx_adst_out_perm(int32x4_t *T, int n) {
  int32x4_t copyT[16];
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) copyT[i] = T[i];
  for (int i = 0; i < n0; i++) {
    int a = (i >> 3) & 1;
    int b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
    int c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
    int d = (i & 1) ^ ((i >> 1) & 1);
    int idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
    T[i] = (i & 1) ? vnegq_s32(copyT[idx]) : copyT[idx];
  }
}

static void vitx_adst4(int32x4_t *T) {
  v64x4 s0 = v64_mul(T[0], SINPI_1_9), s1 = v64_mul(T[0], SINPI_2_9);
  v64x4 s2 = v64_mul(T[1], SINPI_3_9), s3 = v64_mul(T[2], SINPI_4_9);
  v64x4 s4 = v64_mul(T[2], SINPI_1_9);
  v64x4 s5 = v64_mul(T[3], SINPI_2_9), s6 = v64_mul(T[3], SINPI_4_9);
  v64x4 a7 = v64_sub(v64_cvt(T[0]), v64_cvt(T[2]));
  v64x4 b7 = v64_add(a7, v64_cvt(T[3]));
  s0 = v64_add(s0, s3);
  s1 = v64_sub(s1, s4);
  s3 = s2;
  s2 = v64_mul(v64_to32(b7), SINPI_3_9);
  s0 = v64_add(s0, s5);
  s1 = v64_sub(s1, s6);
  v64x4 x0 = v64_add(s0, s3);
  v64x4 x1 = v64_add(s1, s3);
  v64x4 x2 = s2;
  v64x4 x3 = v64_sub(v64_add(s0, s1), s3);
  T[0] = v64_rnd12(x0);
  T[1] = v64_rnd12(x1);
  T[2] = v64_rnd12(x2);
  T[3] = v64_rnd12(x3);
}

static void vitx_adst8(int32x4_t *T, int r) {
  vitx_adst_in_perm(T, 3);
  for (int i = 0; i < 4; i++) vitx_B(T, 2 * i, 2 * i + 1, 60 - 16 * i, 1);
  for (int i = 0; i < 4; i++) vitx_H(T, i, 4 + i, 0, r);
  for (int i = 0; i < 2; i++) vitx_B(T, 4 + 3 * i, 5 + i, 48 - 32 * i, 1);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) vitx_H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (int i = 0; i < 2; i++) vitx_B(T, 2 + 4 * i, 3 + 4 * i, 32, 1);
  vitx_adst_out_perm(T, 3);
}

static void vitx_adst16(int32x4_t *T, int r) {
  vitx_adst_in_perm(T, 4);
  for (int i = 0; i < 8; i++) vitx_B(T, 2 * i, 2 * i + 1, 62 - 8 * i, 1);
  for (int i = 0; i < 8; i++) vitx_H(T, i, 8 + i, 0, r);
  for (int i = 0; i < 2; i++) {
    vitx_B(T, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, 1);
    vitx_B(T, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, 1);
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) vitx_H(T, 8 * j + i, 4 + 8 * j + i, 0, r);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) vitx_B(T, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, 1);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++) vitx_H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
  for (int i = 0; i < 4; i++) vitx_B(T, 2 + 4 * i, 3 + 4 * i, 32, 1);
  vitx_adst_out_perm(T, 4);
}

static void vitx_adst(int32x4_t *T, int n, int r) {
  if (n == 2) vitx_adst4(T);
  else if (n == 3) vitx_adst8(T, r);
  else vitx_adst16(T, r);
}

static void vitx_identity(int32x4_t *T, int n) {
  int n0 = 1 << n;
  for (int i = 0; i < n0; i++) {
    if (n == 2) T[i] = v64_rnd12(v64_mul(T[i], 5793));
    else if (n == 3) T[i] = vaddq_s32(T[i], T[i]);
    else if (n == 4) T[i] = v64_rnd12(v64_mul(T[i], 11586));
    else T[i] = vshlq_n_s32(T[i], 2);
  }
}

static void vitx_wht(int32x4_t *T, int shift) {
  int32x4_t sh = vdupq_n_s32(-shift);
  int32x4_t a = vshlq_s32(T[0], sh), c = vshlq_s32(T[1], sh), d = vshlq_s32(T[2], sh), b = vshlq_s32(T[3], sh);
  a = vaddq_s32(a, c);
  d = vsubq_s32(d, b);
  int32x4_t e = vshrq_n_s32(vsubq_s32(a, d), 1);
  b = vsubq_s32(e, b);
  c = vsubq_s32(e, c);
  a = vsubq_s32(a, b);
  d = vaddq_s32(d, c);
  T[0] = a;
  T[1] = b;
  T[2] = c;
  T[3] = d;
}

static void vitx_1d(int32x4_t *T, int kind, int n, int r) {
  if (kind == ITX_DCT) vitx_dct(T, n, r);
  else if (kind == ITX_ADST) vitx_adst(T, n, r);
  else vitx_identity(T, n);
}

/* In-place 4x4 transpose of int32x4 quads. */
static inline void itx_trn4(int32x4_t *r0, int32x4_t *r1, int32x4_t *r2, int32x4_t *r3) {
  int32x4x2_t t0 = vtrnq_s32(*r0, *r1);
  int32x4x2_t t1 = vtrnq_s32(*r2, *r3);
  *r0 = vcombine_s32(vget_low_s32(t0.val[0]), vget_low_s32(t1.val[0]));
  *r1 = vcombine_s32(vget_low_s32(t0.val[1]), vget_low_s32(t1.val[1]));
  *r2 = vcombine_s32(vget_high_s32(t0.val[0]), vget_high_s32(t1.val[0]));
  *r3 = vcombine_s32(vget_high_s32(t0.val[1]), vget_high_s32(t1.val[1]));
}

static void itx_2d_neon(int32_t *dequant, int32_t *residual, int txSz, int txType, int lossless) {
  int log2W = Tx_Width_Log2[txSz], log2H = Tx_Height_Log2[txSz];
  int w = 1 << log2W, h = 1 << log2H;
  int rowShift = lossless ? 0 : Transform_Row_Shift[txSz];
  int colShift = lossless ? 0 : 4;
  int colClampRange = k_max(8 + 6, 16);
  int rowKind = itx_row_kind(txType), colKind = itx_col_kind(txType);
  int32x4_t T[64];
  /* row pass, four rows at a time (lane = row index) */
  for (int i = 0; i < h; i += 4) {
    uint32x4_t nzv = vdupq_n_u32(0);
    for (int j0 = 0; j0 < w; j0 += 4) {
      int32x4_t r0, r1, r2, r3;
      if (i < 32 && j0 + 3 < 32) {
        r0 = vld1q_s32(dequant + (i + 0) * 64 + j0);
        r1 = vld1q_s32(dequant + (i + 1) * 64 + j0);
        r2 = vld1q_s32(dequant + (i + 2) * 64 + j0);
        r3 = vld1q_s32(dequant + (i + 3) * 64 + j0);
        itx_trn4(&r0, &r1, &r2, &r3);
      } else {
        r0 = r1 = r2 = r3 = vdupq_n_s32(0);
      }
      T[j0] = r0;
      T[j0 + 1] = r1;
      T[j0 + 2] = r2;
      T[j0 + 3] = r3;
      nzv = vorrq_u32(nzv, vreinterpretq_u32_s32(vorrq_s32(vorrq_s32(r0, r1), vorrq_s32(r2, r3))));
    }
    if (vmaxvq_u32(nzv) == 0 && !lossless) { /* every row kernel maps zeros to zeros */
      int32x4_t z = vdupq_n_s32(0);
      for (int j0 = 0; j0 < w; j0 += 4)
        for (int r = 0; r < 4; r++) vst1q_s32(residual + (i + r) * 64 + j0, z);
      continue;
    }
    if (k_abs(log2W - log2H) == 1)
      for (int j = 0; j < w; j++) T[j] = v64_rnd12(v64_mul(T[j], 2896));
    if (lossless) vitx_wht(T, 2);
    else vitx_1d(T, rowKind, log2W, 16);
    for (int j = 0; j < w; j++) T[j] = vround2(T[j], rowShift);
    for (int j0 = 0; j0 < w; j0 += 4) {
      int32x4_t r0 = T[j0], r1 = T[j0 + 1], r2 = T[j0 + 2], r3 = T[j0 + 3];
      itx_trn4(&r0, &r1, &r2, &r3);
      vst1q_s32(residual + (i + 0) * 64 + j0, r0);
      vst1q_s32(residual + (i + 1) * 64 + j0, r1);
      vst1q_s32(residual + (i + 2) * 64 + j0, r2);
      vst1q_s32(residual + (i + 3) * 64 + j0, r3);
    }
  }
  int32x4_t lo = vdupq_n_s32(-(1 << (colClampRange - 1))), hi = vdupq_n_s32((1 << (colClampRange - 1)) - 1);
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j += 4) {
      int32x4_t v = vld1q_s32(residual + i * 64 + j);
      vst1q_s32(residual + i * 64 + j, vminq_s32(vmaxq_s32(v, lo), hi));
    }
  /* column pass, four columns at a time (lane = column index) */
  for (int j0 = 0; j0 < w; j0 += 4) {
    for (int i = 0; i < h; i++) T[i] = vld1q_s32(residual + i * 64 + j0);
    if (lossless) vitx_wht(T, 0);
    else vitx_1d(T, colKind, log2H, colClampRange);
    for (int i = 0; i < h; i++) vst1q_s32(residual + i * 64 + j0, vround2(T[i], colShift));
  }
}
#endif

INLINE void itx_2d(DEV int32_t *dequant, DEV int32_t *residual, int txSz, int txType, int lossless) {
#if !defined(__METAL_VERSION__) && defined(__aarch64__)
  itx_2d_neon(dequant, residual, txSz, txType, lossless);
#else
  itx_2d_ref(dequant, residual, txSz, txType, lossless);
#endif
}

#endif
