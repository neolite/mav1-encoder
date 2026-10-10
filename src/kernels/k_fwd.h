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
INLINE void fwd_2d(DEV const int32_t *resid, DEV int32_t *out, int txSz, int txType,
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
  for (int i = 0; i < th; i++)
    for (int j = 0; j < tw; j++) {
      int q = (i == 0 && j == 0) ? dcQ : acQ;
      int rnd = (i == 0 && j == 0) ? dcRound : acRound;
      int v = coef[i * 64 + j];
      long a = (long)k_abs(v) * denom;
      long step = (long)q * 16;
      int level = (int)((a + (step * rnd >> 6)) / step);
      if (level > 0xFFFF) level = 0xFFFF;
      quant[i * tw + j] = v < 0 ? -level : level;
      nz += level != 0;
    }
  return nz;
}

/* Dequantisation, spec 7.12.3 step 1 (no quantizer matrices). Output row
 * stride 64, as consumed by itx_2d. */
INLINE void dequant_block(DEV const int32_t *quant, DEV int32_t *dq, int txSz, int dcQ, int acQ) {
  int tw = k_min(32, Tx_Width[txSz]), th = k_min(32, Tx_Height[txSz]);
  int denom = dq_denom(txSz);
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) dq[i * 64 + j] = 0;
  for (int i = 0; i < th; i++)
    for (int j = 0; j < tw; j++) {
      int q = (i == 0 && j == 0) ? dcQ : acQ;
      long d = (long)quant[i * tw + j] * q;
      long ad = d < 0 ? -d : d;
      long d2 = (ad & 0xFFFFFF) / denom;
      if (d < 0) d2 = -d2;
      dq[i * 64 + j] = Clip3(-(1 << 15), (1 << 15) - 1, (int)d2);
    }
}

#endif
