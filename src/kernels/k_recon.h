/*
 * Closed-loop reconstruction of one 64x64 superblock.
 *
 * Mode decisions (partition, prediction modes, motion vectors) are made
 * beforehand; this kernel performs prediction, transform type selection,
 * forward transform, quantisation and the normative reconstruction, in
 * decode order, so that its output matches what a decoder produces. It
 * reproduces BlockDecoded (spec 5.11.3 / 5.11.35) to derive the
 * haveAboveRight / haveBelowLeft edge availability exactly.
 *
 * Superblocks may run in parallel as long as SB (r, c) starts after
 * (r, c - 1) and (r - 1, c + 1) have finished (wavefront c + 2 * r).
 */
#ifndef K_RECON_H
#define K_RECON_H

#include "k_common.h"
#include "k_itx.h"
#include "k_fwd.h"
#include "k_scan.h"
#include "k_intra.h"
#include "k_inter.h"

#define SCRATCH_INTS 24576
#define S_RESID 0
#define S_COEF 4096
#define S_DQ 8192
#define S_RES 12288
#define S_TMP 16384
#define S_QTRY 21504
#define S_QBEST 22528

#define BD_DIM 18
#define BD(p, y, x) bd[(p) * BD_DIM * BD_DIM + ((y) + 1) * BD_DIM + ((x) + 1)]

/* Rough bit estimate (in 1/8 bit) of a coefficient block, used only for
 * the rate term of the transform type decision. */
INLINE int est_coef_bits(DEV const int32_t *quant, KCONST int32_t *scan, int eob) {
  if (eob == 0) return 8;
  int bits = 8 * 6 + 8 * 2 * FloorLog2(eob + 1);
  for (int c = 0; c < eob; c++) {
    int l = k_abs(quant[scan[c]]);
    if (l == 0) bits += 6;
    else bits += 8 * (2 + 2 * FloorLog2(l));
  }
  return bits;
}

INLINE int find_eob(DEV const int32_t *quant, KCONST int32_t *scan, int n) {
  for (int c = n - 1; c >= 0; c--)
    if (quant[scan[c]] != 0) return c + 1;
  return 0;
}

/*
 * Transform, quantise and reconstruct one transform block whose prediction
 * is already in `rec`. Returns the chosen tx type in *outType and the eob.
 */
INLINE int code_tx_block(DEV const FrameParams *fp, DEV const uint8_t *src, DEV uint8_t *rec, int stride,
                         int x, int y, int txSz, int plane, THR const int *cands, int numCands,
                         DEV int32_t *coefOut, DEV const int32_t *mats, DEV int32_t *scratch,
                         THR int *outType) {
  int w = Tx_Width[txSz], h = Tx_Height[txSz];
  int tw = k_min(32, w), th = k_min(32, h);
  DEV int32_t *resid = scratch + S_RESID;
  DEV int32_t *coef = scratch + S_COEF;
  DEV int32_t *dq = scratch + S_DQ;
  DEV int32_t *res = scratch + S_RES;
  DEV int32_t *tmp = scratch + S_TMP;
  DEV int32_t *qtry = scratch + S_QTRY;
  DEV int32_t *qbest = scratch + S_QBEST;
  int dcQ = fp->dc_q[plane], acQ = fp->ac_q[plane];
  long bestCost = -1;
  int bestType = DCT_DCT, bestEob = 0;

  for (int ci = 0; ci < numCands; ci++) {
    int t = cands[ci];
    int fud = itx_flip_ud(t), flr = itx_flip_lr(t);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int yy = fud ? (h - 1 - i) : i, xx = flr ? (w - 1 - j) : j;
        int o = (y + yy) * stride + x + xx;
        resid[i * 64 + j] = (int)src[o] - (int)rec[o];
      }
    fwd_2d(resid, coef, txSz, t, mats, tmp);
    quantize_block(coef, qtry, txSz, dcQ, acQ, 26, 22);
    KCONST int32_t *scan = get_scan(txSz, t);
    int eob = find_eob(qtry, scan, seg_eob(txSz));
    long sse = 0;
    if (eob > 0) {
      dequant_block(qtry, dq, txSz, dcQ, acQ);
      itx_2d(dq, res, txSz, t, 0);
      for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
          int yy = fud ? (h - 1 - i) : i, xx = flr ? (w - 1 - j) : j;
          int o = (y + yy) * stride + x + xx;
          int d = (int)src[o] - Clip1((int)rec[o] + res[i * 64 + j]);
          sse += d * d;
        }
    } else {
      for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
          int d = resid[i * 64 + j];
          sse += d * d;
        }
    }
    long cost = sse * 16 + (long)fp->lambda * est_coef_bits(qtry, scan, eob) / 8;
    if (bestCost < 0 || cost < bestCost) {
      bestCost = cost;
      bestType = eob > 0 ? t : DCT_DCT;
      bestEob = eob;
      for (int k = 0; k < tw * th; k++) qbest[k] = qtry[k];
    }
  }

  for (int k = 0; k < tw * th; k++) coefOut[k] = qbest[k];
  if (bestEob > 0) {
    int fud = itx_flip_ud(bestType), flr = itx_flip_lr(bestType);
    dequant_block(qbest, dq, txSz, dcQ, acQ);
    itx_2d(dq, res, txSz, bestType, 0);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int yy = fud ? (h - 1 - i) : i, xx = flr ? (w - 1 - j) : j;
        int o = (y + yy) * stride + x + xx;
        rec[o] = (uint8_t)Clip1((int)rec[o] + res[i * 64 + j]);
      }
  }
  *outType = bestType;
  return bestEob;
}

INLINE void recon_block(DEV const FrameParams *fp, DEV BlockInfo *b, DEV const uint8_t *src, DEV uint8_t *rec,
                        DEV const uint8_t *ref, DEV int32_t *coefs, DEV const int32_t *mats,
                        DEV int32_t *scratch, THR uint8_t *bd) {
  int bs = b->bsize;
  int MiRow = b->mi_row, MiCol = b->mi_col;
  int AvailU = MiRow > 0, AvailL = MiCol > 0;
  int lumaTxType = DCT_DCT;
  for (int plane = 0; plane < 3; plane++) {
    int subX = plane > 0, subY = plane > 0;
    int txSz = plane == 0 ? Max_Tx_Size_Rect[bs] : uv_tx_size(bs, subX, subY);
    int planeSz = Subsampled_Size[bs][subX][subY];
    int num4x4W = Num_4x4_Blocks_Wide[planeSz], num4x4H = Num_4x4_Blocks_High[planeSz];
    int baseX = (MiCol >> subX) * 4, baseY = (MiRow >> subY) * 4;
    int stride = fp->stride[plane];
    DEV uint8_t *px = rec + fp->offset[plane];
    DEV const uint8_t *spx = src + fp->offset[plane];
    if (b->is_inter) {
      predict_inter_block(ref + fp->offset[plane], stride, fp->frame_w, fp->frame_h,
                          px + baseY * stride + baseX, stride, baseX, baseY, Block_Width(bs) >> subX, Block_Height(bs) >> subY,
                          b->mv_row, b->mv_col, subX, subY, EIGHTTAP, scratch + S_TMP);
    }
    int stepX = Tx_Width[txSz] >> 2, stepY = Tx_Height[txSz] >> 2;
    int cands[16];
    int numCands = 0;
    int isInter = b->is_inter;
    if (plane == 0) {
      int set = get_tx_set(txSz, isInter, fp->reduced_tx_set);
      if (!fp->tx_search || set == TX_SET_DCTONLY) {
        cands[numCands++] = DCT_DCT;
      } else {
        for (int t = 0; t < 16; t++)
          if (is_tx_type_in_set(set, t, isInter)) cands[numCands++] = t;
      }
    } else {
      cands[numCands++] = chroma_tx_type(txSz, isInter, b->uv_mode, lumaTxType, fp->reduced_tx_set);
    }
    for (int y = 0; y < num4x4H; y += stepY) {
      for (int x = 0; x < num4x4W; x += stepX) {
        int startX = baseX + 4 * x, startY = baseY + 4 * y;
        int row = (startY << subY) >> 2, col = (startX << subX) >> 2;
        int sbr = row & 15, sbc = col & 15;
        if (!isInter) {
          IntraArgs a;
          a.haveLeft = AvailL || x > 0;
          a.haveAbove = AvailU || y > 0;
          a.haveAboveRight = BD(plane, (sbr >> subY) - 1, (sbc >> subX) + stepX);
          a.haveBelowLeft = BD(plane, (sbr >> subY) + stepY, (sbc >> subX) - 1);
          a.mode = plane == 0 ? b->y_mode : b->uv_mode;
          a.angleDelta = plane == 0 ? b->angle_y : b->angle_uv;
          a.log2W = Tx_Width_Log2[txSz];
          a.log2H = Tx_Height_Log2[txSz];
          a.maxX = ((fp->mi_cols * 4) >> subX) - 1;
          a.maxY = ((fp->mi_rows * 4) >> subY) - 1;
          a.edgeFilter = 0;
          a.filterType = 0;
          predict_intra(px, stride, startX, startY, a, px + startY * stride + startX, stride);
        }
        int t = DCT_DCT;
        int eob = code_tx_block(fp, spx, px, stride, startX, startY, txSz, plane, cands, numCands,
                                coefs + b->coef_off[plane], mats, scratch, &t);
        b->eob[plane] = eob;
        if (plane == 0) { lumaTxType = t; b->tx_type = t; }
        for (int i = 0; i < stepY; i++)
          for (int j = 0; j < stepX; j++) BD(plane, (sbr >> subY) + i, (sbc >> subX) + j) = 1;
      }
    }
  }
}

INLINE void recon_sb(DEV const FrameParams *fp, SbInfo sb, DEV BlockInfo *blocks, DEV const uint8_t *src,
                     DEV uint8_t *rec, DEV const uint8_t *ref, DEV int32_t *coefs, DEV const int32_t *mats,
                     DEV int32_t *scratch) {
  uint8_t bd[3 * BD_DIM * BD_DIM];
  int r = sb.sb_row * 16, c = sb.sb_col * 16;
  int sbSize4 = 16;
  /* clear_block_decoded_flags( r, c, sbSize4 ) with a single tile. */
  for (int plane = 0; plane < 3; plane++) {
    int subX = plane > 0, subY = plane > 0;
    int sbWidth4 = (fp->mi_cols - c) >> subX;
    int sbHeight4 = (fp->mi_rows - r) >> subY;
    for (int y = -1; y <= (sbSize4 >> subY); y++)
      for (int x = -1; x <= (sbSize4 >> subX); x++) {
        if (y < 0 && x < sbWidth4) BD(plane, y, x) = 1;
        else if (x < 0 && y < sbHeight4) BD(plane, y, x) = 1;
        else BD(plane, y, x) = 0;
      }
    BD(plane, sbSize4 >> subY, -1) = 0;
  }
  for (int i = 0; i < sb.num_blocks; i++)
    recon_block(fp, blocks + sb.first_block + i, src, rec, ref, coefs, mats, scratch, bd);
}

#endif
