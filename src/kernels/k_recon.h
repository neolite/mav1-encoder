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
#define S_SAVE 23552  /* 64x64 uint8 pixel backup for the tx_depth search */

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
                         THR int *outType, THR long *outCost) {
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

  if (fp->lossless) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        int o = (y + i) * stride + x + j;
        resid[i * 64 + j] = (int)src[o] - (int)rec[o];
      }
    fwht4x4(resid, qbest);
    KCONST int32_t *scan = get_scan(TX_4X4, DCT_DCT);
    bestEob = find_eob(qbest, scan, 16);
    for (int k = 0; k < 16; k++) coefOut[k] = qbest[k];
    if (bestEob > 0) {
      dequant_block(qbest, dq, TX_4X4, dcQ, acQ);
      itx_2d(dq, res, TX_4X4, DCT_DCT, 1);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
          int o = (y + i) * stride + x + j;
          rec[o] = (uint8_t)Clip1((int)rec[o] + res[i * 64 + j]);
        }
    }
    *outType = DCT_DCT;
    *outCost = 0;
    return bestEob;
  }

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
    if (numCands > 1) { /* a single candidate's residual is still in res */
      dequant_block(qbest, dq, txSz, dcQ, acQ);
      itx_2d(dq, res, txSz, bestType, 0);
    }
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        int yy = fud ? (h - 1 - i) : i, xx = flr ? (w - 1 - j) : j;
        int o = (y + yy) * stride + x + xx;
        rec[o] = (uint8_t)Clip1((int)rec[o] + res[i * 64 + j]);
      }
  }
  *outType = bestType;
  *outCost = bestCost;
  return bestEob;
}

INLINE int tx_type_get(DEV const BlockInfo *b, int idx) {
  if (idx >= 16) return DCT_DCT;
  return (int)(((uint32_t)(idx < 8 ? b->tx_types_lo : b->tx_types_hi) >> (4 * (idx & 7))) & 15u);
}

INLINE void tx_type_set(DEV BlockInfo *b, int idx, int t) {
  if (idx >= 16) return;
  int sh = 4 * (idx & 7);
  if (idx < 8) b->tx_types_lo = (int32_t)(((uint32_t)b->tx_types_lo & ~(15u << sh)) | ((uint32_t)t << sh));
  else b->tx_types_hi = (int32_t)(((uint32_t)b->tx_types_hi & ~(15u << sh)) | ((uint32_t)t << sh));
}

/* Predicts (intra), transforms and reconstructs every transform block of
 * one plane (spec residual( ) / transform_block( ) order). Returns the RD
 * cost; MaxLumaW/H follow the spec's definition. */
INLINE long code_plane(DEV const FrameParams *fp, DEV BlockInfo *b, DEV const uint8_t *src, DEV uint8_t *rec,
                       DEV int32_t *coefs, DEV const int32_t *mats, DEV int32_t *scratch, THR uint8_t *bd,
                       int plane, int txSz, THR int *maxLumaW, THR int *maxLumaH) {
  int bs = b->bsize;
  int MiRow = b->mi_row, MiCol = b->mi_col;
  int AvailU = MiRow > 0, AvailL = MiCol > tile_col_start(fp, MiCol);
  int subX = plane > 0, subY = plane > 0;
  int planeSz = Subsampled_Size[bs][subX][subY];
  int num4x4W = Num_4x4_Blocks_Wide[planeSz], num4x4H = Num_4x4_Blocks_High[planeSz];
  int baseX = (MiCol >> subX) * 4, baseY = (MiRow >> subY) * 4;
  int stride = fp->stride[plane];
  DEV uint8_t *px = rec + fp->offset[plane];
  DEV const uint8_t *spx = src + fp->offset[plane];
  int stepX = Tx_Width[txSz] >> 2, stepY = Tx_Height[txSz] >> 2;
  int cands[16];
  int numCands = 0;
  int isInter = b->is_inter;
  if (plane == 0) {
    int set = get_tx_set(txSz, isInter, fp->reduced_tx_set);
    if (!fp->tx_search || set == TX_SET_DCTONLY || fp->lossless) {
      cands[numCands++] = DCT_DCT;
    } else {
      for (int t = 0; t < 16; t++)
        if (is_tx_type_in_set(set, t, isInter)) cands[numCands++] = t;
    }
  } else {
    cands[numCands++] = chroma_tx_type(txSz, isInter, b->uv_mode, b->tx_type, fp->reduced_tx_set);
  }
  long total = 0;
  int txIdx = 0, anyNz = 0;
  for (int y = 0; y < num4x4H; y += stepY) {
    for (int x = 0; x < num4x4W; x += stepX) {
      int startX = baseX + 4 * x, startY = baseY + 4 * y;
      /* transform_block( ) returns early for blocks outside the frame */
      if (startX >= ((fp->mi_cols * 4) >> subX) || startY >= ((fp->mi_rows * 4) >> subY)) continue;
      int row = (startY << subY) >> 2, col = (startX << subX) >> 2;
      int sbr = row & 15, sbc = col & 15;
      if (!isInter) {
        IntraArgs a;
        a.haveLeft = AvailL || x > 0;
        a.haveAbove = AvailU || y > 0;
        a.haveAboveRight = BD(plane, (sbr >> subY) - 1, (sbc >> subX) + stepX);
        a.haveBelowLeft = BD(plane, (sbr >> subY) + stepY, (sbc >> subX) - 1);
        int isCfl = plane > 0 && b->uv_mode == UV_CFL_PRED;
        a.mode = plane == 0 ? b->y_mode : (isCfl ? DC_PRED : b->uv_mode);
        a.angleDelta = plane == 0 ? b->angle_y : b->angle_uv;
        a.log2W = Tx_Width_Log2[txSz];
        a.log2H = Tx_Height_Log2[txSz];
        a.maxX = ((fp->mi_cols * 4) >> subX) - 1;
        a.maxY = ((fp->mi_rows * 4) >> subY) - 1;
        a.edgeFilter = fp->enable_intra_edge_filter;
        a.filterType = (b->interp_filter >> (plane == 0 ? 4 : 5)) & 1;
        predict_intra(px, stride, startX, startY, a, px + startY * stride + startX, stride);
        if (isCfl) {
          DEV int32_t *dc = scratch + S_TMP;
          DEV uint8_t *d = px + startY * stride + startX;
          int tw = Tx_Width[txSz], th = Tx_Height[txSz];
          for (int i = 0; i < th; i++)
            for (int j = 0; j < tw; j++) dc[i * tw + j] = d[i * stride + j];
          DEV const uint8_t *luma = rec + fp->offset[0];
          int alpha;
          if (txIdx > 0) {
            alpha = plane == 1 ? b->cfl_alpha_u : b->cfl_alpha_v; /* one alpha per block */
          } else if (plane == 1) {
            alpha = cfl_best_alpha(luma, fp->stride[0], startX, startY, *maxLumaW, *maxLumaH, a.log2W, a.log2H, dc,
                                   spx + startY * stride + startX, stride, d, stride, 0);
            b->cfl_alpha_u = alpha;
          } else {
            /* (0, 0) is not codable: V must be non-zero when U is zero. */
            alpha = cfl_best_alpha(luma, fp->stride[0], startX, startY, *maxLumaW, *maxLumaH, a.log2W, a.log2H, dc,
                                   spx + startY * stride + startX, stride, d, stride, b->cfl_alpha_u == 0);
            b->cfl_alpha_v = alpha;
          }
          cfl_predict(luma, fp->stride[0], startX, startY, *maxLumaW, *maxLumaH, a.log2W, a.log2H, alpha, dc, d,
                      stride);
        }
      }
      if (plane == 0) {
        *maxLumaW = startX + stepX * 4;
        *maxLumaH = startY + stepY * 4;
      }
      int t = DCT_DCT;
      long c = 0;
      int eob = code_tx_block(fp, spx, px, stride, startX, startY, txSz, plane, cands, numCands,
                              coefs + b->coef_off[plane] + txIdx * Tx_Width[txSz] * Tx_Height[txSz], mats,
                              scratch, &t, &c);
      total += c;
      if (plane == 0) {
        tx_type_set(b, txIdx, t);
        if (txIdx == 0) b->tx_type = t;
      }
      txIdx++;
      anyNz |= eob > 0;
      for (int i = 0; i < stepY; i++)
        for (int j = 0; j < stepX; j++) BD(plane, (sbr >> subY) + i, (sbc >> subX) + j) = 1;
    }
  }
  /* eob[] is used as a "has coefficients" flag; the writer derives the
   * eob of each transform from the coefficients. */
  b->eob[plane] = anyNz;
  return total;
}

INLINE void recon_block(DEV const FrameParams *fp, DEV BlockInfo *b, DEV const uint8_t *src, DEV uint8_t *rec,
                        DEV const uint8_t *ref, DEV const uint8_t *gold, DEV int32_t *coefs,
                        DEV const int32_t *mats, DEV int32_t *scratch, THR uint8_t *bd) {
  int bs = b->bsize;
  int comp = b->is_inter && b->ref_frame1 > INTRA_FRAME; /* LAST + second reference */
  DEV const uint8_t *ref1 = gold;
  if (b->is_inter && b->ref_frame != LAST_FRAME) ref = gold; /* second reference */
  int MiRow = b->mi_row, MiCol = b->mi_col;
  int maxLumaW = 0, maxLumaH = 0;
  b->tx_depth = 0;
  b->interp_filter &= 0x30; /* keep the intra edge filter types */
  if (b->is_inter && fp->switchable_interp && !comp) {
    /* dual filter search on the luma prediction: 3 vertical x 3 horizontal */
    int w = Block_Width(bs), h = Block_Height(bs);
    int x0 = MiCol * 4, y0 = MiRow * 4, stride = fp->stride[0];
    DEV uint8_t *px = rec + fp->offset[0] + y0 * stride + x0;
    DEV const uint8_t *sp = src + fp->offset[0] + y0 * stride + x0;
    int vw = k_min(w, fp->frame_w - x0), vh = k_min(h, fp->frame_h - y0);
    long best = -1;
    for (int f = 0; f < 9; f++) {
      int packed = (f % 3) | ((f / 3) << 2);
      predict_inter_block(ref + fp->offset[0], stride, fp->frame_w, fp->frame_h, px, stride, x0, y0, w, h,
                          b->mv_row, b->mv_col, 0, 0, packed, scratch + S_TMP);
      long sse = 0;
      for (int i = 0; i < vh; i++)
        for (int j = 0; j < vw; j++) {
          int d = (int)px[i * stride + j] - (int)sp[i * stride + j];
          sse += d * d;
        }
      long c = sse * 16 + (long)fp->lambda * ((f % 3 != 0) + (f / 3 != 0));
      if (best < 0 || c < best) {
        best = c;
        b->interp_filter = packed;
      }
    }
  }
  for (int plane = 0; plane < 3; plane++) {
    int subX = plane > 0, subY = plane > 0;
    int stride = fp->stride[plane];
    int baseX = (MiCol >> subX) * 4, baseY = (MiRow >> subY) * 4;
    if (comp) {
      /* COMPOUND_AVERAGE; the raw predictions use the residual / coefficient
       * scratch, which is free until code_plane( ) */
      int w = Block_Width(bs) >> subX, h = Block_Height(bs) >> subY;
      DEV int32_t *p0 = scratch + S_RESID, *p1 = scratch + S_COEF;
      predict_inter_raw(ref + fp->offset[plane], stride, fp->frame_w, fp->frame_h, p0, w, baseX, baseY, w, h,
                        b->mv_row, b->mv_col, subX, subY, b->interp_filter, scratch + S_TMP);
      predict_inter_raw(ref1 + fp->offset[plane], stride, fp->frame_w, fp->frame_h, p1, w, baseX, baseY, w, h,
                        b->mv1_row, b->mv1_col, subX, subY, b->interp_filter, scratch + S_TMP);
      DEV uint8_t *d = rec + fp->offset[plane] + baseY * stride + baseX;
      for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) d[i * stride + j] = (uint8_t)Clip1(Round2(p0[i * w + j] + p1[i * w + j], 5));
    } else if (b->is_inter) {
      predict_inter_block(ref + fp->offset[plane], stride, fp->frame_w, fp->frame_h,
                          rec + fp->offset[plane] + baseY * stride + baseX, stride, baseX, baseY,
                          Block_Width(bs) >> subX, Block_Height(bs) >> subY, b->mv_row, b->mv_col, subX, subY,
                          b->interp_filter, scratch + S_TMP);
    }
    if (fp->lossless) {
      code_plane(fp, b, src, rec, coefs, mats, scratch, bd, plane, TX_4X4, &maxLumaW, &maxLumaH);
      continue;
    }
    if (plane > 0) {
      code_plane(fp, b, src, rec, coefs, mats, scratch, bd, plane, uv_tx_size(bs, subX, subY), &maxLumaW,
                 &maxLumaH);
      continue;
    }
    int maxDepth = (!b->is_inter && fp->tx_mode_select) ? k_min(MAX_TX_DEPTH, Max_Tx_Depth[bs]) : 0;
    if (maxDepth == 0) {
      code_plane(fp, b, src, rec, coefs, mats, scratch, bd, 0, Max_Tx_Size_Rect[bs], &maxLumaW, &maxLumaH);
      continue;
    }
    /* tx_depth RD search: code the luma plane at each depth from the same
     * starting state (pixels and BlockDecoded flags), keep the best. */
    int w = Block_Width(bs), h = Block_Height(bs);
    DEV uint8_t *save = (DEV uint8_t *)(scratch + S_SAVE);
    DEV uint8_t *px = rec + fp->offset[0] + baseY * stride + baseX;
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) save[i * w + j] = px[i * stride + j];
    uint8_t bdSave[BD_DIM * BD_DIM];
    for (int i = 0; i < BD_DIM * BD_DIM; i++) bdSave[i] = bd[i];
    long best = -1;
    int bestD = 0;
    for (int d = 0; d <= maxDepth + 1; d++) {
      int depth = d <= maxDepth ? d : bestD; /* final pass re-codes the winner */
      if (d == maxDepth + 1 && bestD == maxDepth) break; /* state already holds the winner */
      if (d > 0) {
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++) px[i * stride + j] = save[i * w + j];
        for (int i = 0; i < BD_DIM * BD_DIM; i++) bd[i] = bdSave[i];
      }
      long c = code_plane(fp, b, src, rec, coefs, mats, scratch, bd, 0, depth_tx_size(bs, depth), &maxLumaW,
                          &maxLumaH) + (long)fp->lambda * 8 * depth;
      if (d <= maxDepth && (best < 0 || c < best)) {
        best = c;
        bestD = depth;
      }
      if (d == maxDepth + 1) break;
    }
    b->tx_depth = bestD;
  }
}

INLINE void recon_sb(DEV const FrameParams *fp, SbInfo sb, DEV BlockInfo *blocks, DEV const uint8_t *src,
                     DEV uint8_t *rec, DEV const uint8_t *ref, DEV const uint8_t *gold, DEV int32_t *coefs, DEV const int32_t *mats,
                     DEV int32_t *scratch) {
  uint8_t bd[3 * BD_DIM * BD_DIM];
  int r = sb.sb_row * 16, c = sb.sb_col * 16;
  int sbSize4 = 16;
  /* clear_block_decoded_flags( r, c, sbSize4 ) */
  for (int plane = 0; plane < 3; plane++) {
    int subX = plane > 0, subY = plane > 0;
    int sbWidth4 = (tile_col_end(fp, c) - c) >> subX;
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
    recon_block(fp, blocks + sb.first_block + i, src, rec, ref, gold, coefs, mats, scratch, bd);
}

#endif
