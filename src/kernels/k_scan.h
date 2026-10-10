/*
 * Scan orders and transform-set helpers (spec 5.11.47 get_scan,
 * 5.11.48 get_tx_set, 5.11.40 compute_tx_type).
 */
#ifndef K_SCAN_H
#define K_SCAN_H

#include "k_common.h"

INLINE KCONST int32_t *scan_mrow(int txSz) {
  if (txSz == TX_4X4) return Mrow_Scan_4x4;
  if (txSz == TX_4X8) return Mrow_Scan_4x8;
  if (txSz == TX_8X4) return Mrow_Scan_8x4;
  if (txSz == TX_8X8) return Mrow_Scan_8x8;
  if (txSz == TX_8X16) return Mrow_Scan_8x16;
  if (txSz == TX_16X8) return Mrow_Scan_16x8;
  if (txSz == TX_16X16) return Mrow_Scan_16x16;
  if (txSz == TX_4X16) return Mrow_Scan_4x16;
  return Mrow_Scan_16x4;
}

INLINE KCONST int32_t *scan_mcol(int txSz) {
  if (txSz == TX_4X4) return Mcol_Scan_4x4;
  if (txSz == TX_4X8) return Mcol_Scan_4x8;
  if (txSz == TX_8X4) return Mcol_Scan_8x4;
  if (txSz == TX_8X8) return Mcol_Scan_8x8;
  if (txSz == TX_8X16) return Mcol_Scan_8x16;
  if (txSz == TX_16X8) return Mcol_Scan_16x8;
  if (txSz == TX_16X16) return Mcol_Scan_16x16;
  if (txSz == TX_4X16) return Mcol_Scan_4x16;
  return Mcol_Scan_16x4;
}

INLINE KCONST int32_t *scan_default(int txSz) {
  if (txSz == TX_4X4) return Default_Scan_4x4;
  if (txSz == TX_4X8) return Default_Scan_4x8;
  if (txSz == TX_8X4) return Default_Scan_8x4;
  if (txSz == TX_8X8) return Default_Scan_8x8;
  if (txSz == TX_8X16) return Default_Scan_8x16;
  if (txSz == TX_16X8) return Default_Scan_16x8;
  if (txSz == TX_16X16) return Default_Scan_16x16;
  if (txSz == TX_16X32) return Default_Scan_16x32;
  if (txSz == TX_32X16) return Default_Scan_32x16;
  if (txSz == TX_4X16) return Default_Scan_4x16;
  if (txSz == TX_16X4) return Default_Scan_16x4;
  if (txSz == TX_8X32) return Default_Scan_8x32;
  if (txSz == TX_32X8) return Default_Scan_32x8;
  return Default_Scan_32x32;
}

INLINE KCONST int32_t *get_scan(int txSz, int planeTxType) {
  if (txSz == TX_16X64) return Default_Scan_16x32;
  if (txSz == TX_64X16) return Default_Scan_32x16;
  if (Tx_Size_Sqr_Up[txSz] == TX_64X64) return Default_Scan_32x32;
  if (planeTxType == IDTX) return scan_default(txSz);
  int preferRow = planeTxType == V_DCT || planeTxType == V_ADST || planeTxType == V_FLIPADST;
  int preferCol = planeTxType == H_DCT || planeTxType == H_ADST || planeTxType == H_FLIPADST;
  if (preferRow) return scan_mrow(txSz);
  if (preferCol) return scan_mcol(txSz);
  return scan_default(txSz);
}

INLINE int get_tx_set(int txSz, int isInter, int reducedTxSet) {
  int txSzSqr = Tx_Size_Sqr[txSz];
  int txSzSqrUp = Tx_Size_Sqr_Up[txSz];
  if (txSzSqrUp > TX_32X32) return TX_SET_DCTONLY;
  if (isInter) {
    if (reducedTxSet || txSzSqrUp == TX_32X32) return TX_SET_INTER_3;
    if (txSzSqr == TX_16X16) return TX_SET_INTER_2;
    return TX_SET_INTER_1;
  }
  if (txSzSqrUp == TX_32X32) return TX_SET_DCTONLY;
  if (reducedTxSet) return TX_SET_INTRA_2;
  if (txSzSqr == TX_16X16) return TX_SET_INTRA_2;
  return TX_SET_INTRA_1;
}

INLINE int is_tx_type_in_set(int txSet, int txType, int isInter) {
  return isInter ? Tx_Type_In_Set_Inter[txSet][txType] : Tx_Type_In_Set_Intra[txSet][txType];
}

/* compute_tx_type( ) for a chroma plane. `lumaTxType` is TxTypes[ ][ ] at the
 * co-located luma position (only used for inter blocks). */
INLINE int chroma_tx_type(int txSz, int isInter, int uvMode, int lumaTxType, int reducedTxSet) {
  if (Tx_Size_Sqr_Up[txSz] > TX_32X32) return DCT_DCT;
  int txSet = get_tx_set(txSz, isInter, reducedTxSet);
  int t = isInter ? lumaTxType : Mode_To_Txfm[uvMode];
  if (!is_tx_type_in_set(txSet, t, isInter)) return DCT_DCT;
  return t;
}

/* Size of the coefficient array the decoder scans for this transform size. */
INLINE int seg_eob(int txSz) {
  if (txSz == TX_16X64 || txSz == TX_64X16) return 512;
  return k_min(1024, Tx_Width[txSz] * Tx_Height[txSz]);
}

/* get_tx_size( plane, txSz ) for chroma. */
INLINE int uv_tx_size(int miSize, int subX, int subY) {
  int uvTx = Max_Tx_Size_Rect[Subsampled_Size[miSize][subX][subY]];
  if (Tx_Width[uvTx] == 64 || Tx_Height[uvTx] == 64) {
    if (Tx_Width[uvTx] == 16) return TX_16X32;
    if (Tx_Height[uvTx] == 16) return TX_32X16;
    return TX_32X32;
  }
  return uvTx;
}

/* uv_mode cdf selection (spec 8.3.2): whether UV_CFL_PRED may be coded. */
INLINE int cfl_allowed(int miSize, int lossless) {
  if (lossless) return Subsampled_Size[miSize][1][1] == BLOCK_4X4;
  return k_max(Block_Width(miSize), Block_Height(miSize)) <= 32;
}

#endif
