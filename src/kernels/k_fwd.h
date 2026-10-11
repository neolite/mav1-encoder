/*
 * Forward transform and quantisation.
 *
 * The forward transform is not normative. To stay consistent with the
 * normative inverse transform (k_itx.h), the 1D forward kernels are the
 * (pseudo-)inverses of the integer inverse kernels; they are derived once on
 * the CPU (fwd_matrices.c) and stored as Q20 integer matrices so that the CPU
 * and GPU backends produce bit-identical coefficients.
 */
#ifndef K_FWD_H
#define K_FWD_H

#include "k_common.h"
#include "k_itx.h"

#define FWD_BITS 20
#define FWD_SLOT_SIZE 4096
/* Matrix for 1D kind (ITX_DCT/ADST/IDTX) and length 1 << log2n:
 * Min(32, n) rows of n Q20 entries. */
#define FWD_SLOT(kind, log2n) (((kind) * 5 + (log2n) - 2) * FWD_SLOT_SIZE)
#define FWD_NUM_SLOTS 15

INLINE int dq_denom(int txSz) {
  if (txSz == TX_32X32 || txSz == TX_16X32 || txSz == TX_32X16 || txSz == TX_16X64 || txSz == TX_64X16) return 2;
  if (txSz == TX_64X64 || txSz == TX_32X64 || txSz == TX_64X32) return 4;
  return 1;
}

/*
 * Forward 2D transform of `resid` (h rows of w samples, row stride 64, already
 * flipped for FLIPADST types). Writes Min(32,h) x Min(32,w) coefficients in
 * Q4 (4 fractional bits) to `out` (row stride 64), in the same domain as the
 * decoder's Dequant array.
 */
INLINE void fwd_2d_ref(DEV const int32_t *resid, DEV int32_t *out, int txSz, int txType,
                       DEV const int32_t *mats, DEV int32_t *tmp) {
  int log2W = Tx_Width_Log2[txSz], log2H = Tx_Height_Log2[txSz];
  int w = 1 << log2W, h = 1 << log2H;
  int tw = k_min(32, w), th = k_min(32, h);
  DEV const int32_t *Mh = mats + FWD_SLOT(itx_row_kind(txType), log2W);
  DEV const int32_t *Mv = mats + FWD_SLOT(itx_col_kind(txType), log2H);
  for (int i = 0; i < h; i++)
    for (int k = 0; k < tw; k++) {
      long s = 0;
      for (int j = 0; j < w; j++) s += (long)resid[i * 64 + j] * Mh[k * w + j];
      tmp[i * 64 + k] = (int32_t)Round2L(s, FWD_BITS - 4);
    }
  int shifts = Transform_Row_Shift[txSz] + 4;
  int rect = k_abs(log2W - log2H) == 1;
  for (int k2 = 0; k2 < th; k2++)
    for (int k = 0; k < tw; k++) {
      long s = 0;
      for (int i = 0; i < h; i++) s += (long)Mv[k2 * h + i] * tmp[i * 64 + k];
      s = s * (1 << shifts);
      if (rect) s = Round2L(s, 12) * 5793; /* undo the 2896/4096 row pre-scale */
      long v = s >= 0 ? Round2L(s, FWD_BITS) : -Round2L(-s, FWD_BITS);
      out[k2 * 64 + k] = (int32_t)v;
    }
}

/*
 * Dead-zone scalar quantiser. `coef` holds Q4 coefficients (row stride 64);
 * the quantised levels are written to `quant` in the decoder's layout
 * (row stride tw). Returns the number of non-zero levels.
 */
INLINE int quantize_block(DEV const int32_t *coef, DEV int32_t *quant, int txSz, int dcQ, int acQ,
                          int dcRound, int acRound) {
  int tw = k_min(32, Tx_Width[txSz]), th = k_min(32, Tx_Height[txSz]);
  int denom = dq_denom(txSz);
  int nz = 0;
  /* level = (a + bias) / (16 q): (a + bias) >> 4 is exact for /16, and the
   * remaining /q uses a reciprocal exact for every t < 2^19 (larger values
   * fall back to a plain division). */
  unsigned long invDc = ((unsigned long)1 << 44) / dcQ + 1;
  unsigned long invAc = ((unsigned long)1 << 44) / acQ + 1;
  for (int i = 0; i < th; i++)
    for (int j = 0; j < tw; j++) {
      int q = (i == 0 && j == 0) ? dcQ : acQ;
      int rnd = (i == 0 && j == 0) ? dcRound : acRound;
      unsigned long inv = (i == 0 && j == 0) ? invDc : invAc;
      int v = coef[i * 64 + j];
      long a = (long)k_abs(v) * denom;
      long t = (a + ((long)q * rnd >> 2)) >> 4;
      int level = t < (1L << 19) ? (int)(((unsigned long)t * inv) >> 44) : (int)(t / q);
      if (level > 0xFFFF) level = 0xFFFF;
      quant[i * tw + j] = v < 0 ? -level : level;
      nz += level != 0;
    }
  return nz;
}

/* Dequantisation, spec 7.12.3 step 1 (no quantizer matrices). Output row
 * stride 64, as consumed by itx_2d (which reads only the Min(32, h) x
 * Min(32, w) region written here). */
INLINE void dequant_block(DEV const int32_t *quant, DEV int32_t *dq, int txSz, int dcQ, int acQ) {
  int tw = k_min(32, Tx_Width[txSz]), th = k_min(32, Tx_Height[txSz]);
  int denom = dq_denom(txSz);
  for (int i = 0; i < th; i++)
    for (int j = 0; j < tw; j++) {
      int q = (i == 0 && j == 0) ? dcQ : acQ;
      long d = (long)quant[i * tw + j] * q;
      long ad = d < 0 ? -d : d;
      long d2 = (ad & 0xFFFFFF) >> (denom >> 1);
      if (d < 0) d2 = -d2;
      dq[i * 64 + j] = Clip3(-(1 << 15), (1 << 15) - 1, (int)d2);
    }
}

/* Forward Walsh-Hadamard transform for lossless blocks: the exact inverse
 * of the spec's inverse WHT (7.13.2.10) with shift 2. resid stride 64,
 * out[i * 4 + j] holds Dequant[i][j] / 4, i.e. the coefficient level. */
INLINE void fwht4x4(DEV const int32_t *resid, DEV int32_t *out) {
  int t[16];
  for (int j = 0; j < 4; j++) { /* columns */
    int a1 = resid[0 * 64 + j], b1 = resid[1 * 64 + j], c1 = resid[2 * 64 + j], d1 = resid[3 * 64 + j];
    a1 += b1;
    d1 = d1 - c1;
    int e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= c1;
    d1 += b1;
    t[0 * 4 + j] = a1; t[1 * 4 + j] = c1; t[2 * 4 + j] = d1; t[3 * 4 + j] = b1;
  }
  for (int i = 0; i < 4; i++) { /* rows */
    int a1 = t[i * 4 + 0], b1 = t[i * 4 + 1], c1 = t[i * 4 + 2], d1 = t[i * 4 + 3];
    a1 += b1;
    d1 -= c1;
    int e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= c1;
    d1 += b1;
    out[i * 4 + 0] = a1; out[i * 4 + 1] = c1; out[i * 4 + 2] = d1; out[i * 4 + 3] = b1;
  }
}

#if !defined(__METAL_VERSION__) && defined(__aarch64__)
/*
 * NEON forward transform: identical integer arithmetic to fwd_2d_ref (each
 * product is a widening 32->64 multiply, so the summation order change keeps
 * every result bit-exact). Pass 1 takes four output columns at a time with
 * a dot product over j; pass 2 takes four output columns per accumulator.
 */
#include <arm_neon.h>

static inline void fwd_2d_neon(DEV const int32_t *resid, DEV int32_t *out, int txSz, int txType,
                               DEV const int32_t *mats, DEV int32_t *tmp) {
  int log2W = Tx_Width_Log2[txSz], log2H = Tx_Height_Log2[txSz];
  int w = 1 << log2W, h = 1 << log2H;
  int tw = k_min(32, w), th = k_min(32, h);
  DEV const int32_t *Mh = mats + FWD_SLOT(itx_row_kind(txType), log2W);
  DEV const int32_t *Mv = mats + FWD_SLOT(itx_col_kind(txType), log2H);
  for (int i = 0; i < h; i++) {
    DEV const int32_t *r = resid + i * 64;
    for (int k0 = 0; k0 < tw; k0 += 4) {
      int64x2_t a0l = vdupq_n_s64(0), a0h = a0l, a1l = a0l, a1h = a0l;
      int64x2_t a2l = a0l, a2h = a0l, a3l = a0l, a3h = a0l;
      const int32_t *m0 = Mh + (k0 + 0) * w, *m1 = Mh + (k0 + 1) * w;
      const int32_t *m2 = Mh + (k0 + 2) * w, *m3 = Mh + (k0 + 3) * w;
      for (int j = 0; j < w; j += 4) {
        int32x4_t rv = vld1q_s32(r + j);
        int32x2_t rl = vget_low_s32(rv), rh = vget_high_s32(rv);
        int32x4_t m;
        m = vld1q_s32(m0 + j);
        a0l = vmlal_s32(a0l, rl, vget_low_s32(m));
        a0h = vmlal_s32(a0h, rh, vget_high_s32(m));
        m = vld1q_s32(m1 + j);
        a1l = vmlal_s32(a1l, rl, vget_low_s32(m));
        a1h = vmlal_s32(a1h, rh, vget_high_s32(m));
        m = vld1q_s32(m2 + j);
        a2l = vmlal_s32(a2l, rl, vget_low_s32(m));
        a2h = vmlal_s32(a2h, rh, vget_high_s32(m));
        m = vld1q_s32(m3 + j);
        a3l = vmlal_s32(a3l, rl, vget_low_s32(m));
        a3h = vmlal_s32(a3h, rh, vget_high_s32(m));
      }
      tmp[i * 64 + k0 + 0] = (int32_t)Round2L(vaddvq_s64(a0l) + vaddvq_s64(a0h), FWD_BITS - 4);
      tmp[i * 64 + k0 + 1] = (int32_t)Round2L(vaddvq_s64(a1l) + vaddvq_s64(a1h), FWD_BITS - 4);
      tmp[i * 64 + k0 + 2] = (int32_t)Round2L(vaddvq_s64(a2l) + vaddvq_s64(a2h), FWD_BITS - 4);
      tmp[i * 64 + k0 + 3] = (int32_t)Round2L(vaddvq_s64(a3l) + vaddvq_s64(a3h), FWD_BITS - 4);
    }
  }
  int shifts = Transform_Row_Shift[txSz] + 4;
  int rect = k_abs(log2W - log2H) == 1;
  int64x2_t shv = vdupq_n_s64(shifts), rnd = vdupq_n_s64(1 << (FWD_BITS - 1));
  for (int k2 = 0; k2 < th; k2++) {
    DEV const int32_t *mv = Mv + k2 * h;
    for (int k0 = 0; k0 < tw; k0 += 4) {
      int64x2_t sl = vdupq_n_s64(0), sh = sl;
      for (int i = 0; i < h; i++) {
        int32x4_t tv = vld1q_s32(tmp + i * 64 + k0);
        int32x2_t m = vdup_n_s32(mv[i]);
        sl = vmlal_s32(sl, vget_low_s32(tv), m);
        sh = vmlal_s32(sh, vget_high_s32(tv), m);
      }
      sl = vshlq_s64(sl, shv);
      sh = vshlq_s64(sh, shv);
      if (rect) {
        /* s = Round2L(s, 12) * 5793 (undoes the 2896/4096 row pre-scale) */
        int64x2_t h2 = vdupq_n_s64(2048);
        int32x4_t s32 = vuzp1q_s32(
            vreinterpretq_s32_s64(vshrq_n_s64(vaddq_s64(sl, h2), 12)),
            vreinterpretq_s32_s64(vshrq_n_s64(vaddq_s64(sh, h2), 12)));
        int32x2_t c = vdup_n_s32(5793);
        sl = vmull_s32(vget_low_s32(s32), c);
        sh = vmull_s32(vget_high_s32(s32), c);
      }
      /* v = s >= 0 ? Round2L(s, FWD_BITS) : -Round2L(-s, FWD_BITS) */
      int64x2_t negl = vshrq_n_s64(sl, 63), negh = vshrq_n_s64(sh, 63);
      sl = vshrq_n_s64(vaddq_s64(vsubq_s64(veorq_s64(sl, negl), negl), rnd), FWD_BITS);
      sh = vshrq_n_s64(vaddq_s64(vsubq_s64(veorq_s64(sh, negh), negh), rnd), FWD_BITS);
      sl = vsubq_s64(veorq_s64(sl, negl), negl);
      sh = vsubq_s64(veorq_s64(sh, negh), negh);
      vst1q_s32(out + k2 * 64 + k0,
                vuzp1q_s32(vreinterpretq_s32_s64(sl), vreinterpretq_s32_s64(sh)));
    }
  }
}
#endif

INLINE void fwd_2d(DEV const int32_t *resid, DEV int32_t *out, int txSz, int txType,
                   DEV const int32_t *mats, DEV int32_t *tmp) {
#if !defined(__METAL_VERSION__) && defined(__aarch64__)
  fwd_2d_neon(resid, out, txSz, txType, mats, tmp);
#else
  fwd_2d_ref(resid, out, txSz, txType, mats, tmp);
#endif
}

#endif
