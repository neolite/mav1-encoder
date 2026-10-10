/*
 * Open-loop mode and partition analysis for one 64x64 superblock.
 *
 * Every square node (64x64 down to 8x8) is evaluated on source pixels:
 * intra modes are predicted from the (source) neighbours and scored by a
 * 4x4 Hadamard SATD plus a rate estimate; inter blocks use the motion
 * vector found for that node by the motion search kernel (k_me.h). The
 * cheapest partition tree is then selected bottom-up (no recursion, so the
 * code is valid MSL) and the resulting blocks are written in decode order.
 *
 * All metrics are integers so that the CPU and GPU produce identical
 * decisions.
 */
#ifndef K_ANALYSIS_H
#define K_ANALYSIS_H

#include "k_common.h"
#include "k_intra.h"
#include "k_inter.h"

#define MAX_BLOCKS_PER_SB 64
#define COEFS_PER_SB 6144
/* Analysis scratch layout (int32 units). */
#define A_PRED 0        /* 64x64 uint8 prediction (1024 ints) */
#define A_TMP 1024      /* inter filter intermediate ((64 + 7) * 64 ints) */
#define A_SCRATCH_INTS (1024 + 4608)

/* Node indexing: level 0 = 64x64 (1 node), 1 = 32x32 (4), 2 = 16x16 (16),
 * 3 = 8x8 (64); nodes are in raster order within each level. */
#define NODE_BASE(l) ((l) == 0 ? 0 : (l) == 1 ? 1 : (l) == 2 ? 5 : 21)
#define NUM_NODES 85

INLINE int level_bsize(int l) {
  return l == 0 ? BLOCK_64X64 : l == 1 ? BLOCK_32X32 : l == 2 ? BLOCK_16X16 : BLOCK_8X8;
}

/* Approximate cost (1/8 bit) of each luma intra mode. */
INLINE int mode_bits8(int mode) {
  if (mode == DC_PRED) return 12;
  if (mode == V_PRED || mode == H_PRED || mode == SMOOTH_PRED || mode == PAETH_PRED) return 26;
  if (mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED) return 34;
  return 40;
}

INLINE int satd4x4(THR const int32_t *d) {
  int32_t t[16];
  for (int i = 0; i < 4; i++) {
    int a0 = d[i * 4 + 0] + d[i * 4 + 1], a1 = d[i * 4 + 0] - d[i * 4 + 1];
    int a2 = d[i * 4 + 2] + d[i * 4 + 3], a3 = d[i * 4 + 2] - d[i * 4 + 3];
    t[i * 4 + 0] = a0 + a2; t[i * 4 + 1] = a1 + a3;
    t[i * 4 + 2] = a0 - a2; t[i * 4 + 3] = a1 - a3;
  }
  int s = 0;
  for (int j = 0; j < 4; j++) {
    int a0 = t[0 * 4 + j] + t[1 * 4 + j], a1 = t[0 * 4 + j] - t[1 * 4 + j];
    int a2 = t[2 * 4 + j] + t[3 * 4 + j], a3 = t[2 * 4 + j] - t[3 * 4 + j];
    s += k_abs(a0 + a2) + k_abs(a1 + a3) + k_abs(a0 - a2) + k_abs(a1 - a3);
  }
  return s;
}

/* SATD between a source block and a prediction stored with stride 64. */
INLINE int block_satd(DEV const uint8_t *src, int stride, DEV const uint8_t *pred, int w, int h) {
  int total = 0;
  int32_t d[16];
  for (int by = 0; by < h; by += 4)
    for (int bx = 0; bx < w; bx += 4) {
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          d[i * 4 + j] = (int)src[(by + i) * stride + bx + j] - (int)pred[(by + i) * 64 + bx + j];
      total += satd4x4(d);
    }
  return total;
}

INLINE int intra_satd(DEV const FrameParams *fp, DEV const uint8_t *plane, int stride, int planeIdx,
                      int x, int y, int log2, int mode, int delta, DEV uint8_t *pred) {
  int sub = planeIdx > 0;
  IntraArgs a;
  a.haveLeft = x > 0;
  a.haveAbove = y > 0;
  a.haveAboveRight = 0;
  a.haveBelowLeft = 0;
  a.mode = mode;
  a.angleDelta = delta;
  a.log2W = log2;
  a.log2H = log2;
  a.maxX = ((fp->mi_cols * 4) >> sub) - 1;
  a.maxY = ((fp->mi_rows * 4) >> sub) - 1;
  a.edgeFilter = 0;
  a.filterType = 0;
  predict_intra(plane, stride, x, y, a, pred, 64);
  return block_satd(plane + y * stride + x, stride, pred, 1 << log2, 1 << log2);
}

INLINE int mv_bits8(int mvr, int mvc) {
  return 8 * (2 + 2 * FloorLog2(k_abs(mvr) + 1) + 2 * FloorLog2(k_abs(mvc) + 1));
}

/* Evaluates one node; returns its cost and fills the decision in *out. */
INLINE long eval_node(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                      DEV const int32_t *mvField, int level, int nodeIdx, int miRow, int miCol,
                      DEV int32_t *scratch, THR BlockInfo *out) {
  DEV uint8_t *pred = (DEV uint8_t *)(scratch + A_PRED);
  int bsize = level_bsize(level);
  int log2 = 6 - level;
  int x = miCol * 4, y = miRow * 4;
  int lam = fp->ac_q[0];

  /* luma intra */
  long best = -1;
  int bestMode = DC_PRED, bestDelta = 0;
  for (int mode = DC_PRED; mode <= PAETH_PRED; mode++) {
    long c = intra_satd(fp, src + fp->offset[0], fp->stride[0], 0, x, y, log2, mode, 0, pred) +
             (long)lam * mode_bits8(mode) / 8;
    if (best < 0 || c < best) { best = c; bestMode = mode; }
  }
  if (intra_is_directional(bestMode)) {
    int base = bestMode;
    for (int d = -MAX_ANGLE_DELTA; d <= MAX_ANGLE_DELTA; d++) {
      if (d == 0) continue;
      long c = intra_satd(fp, src + fp->offset[0], fp->stride[0], 0, x, y, log2, base, d, pred) +
               (long)lam * (mode_bits8(base) + 16) / 8;
      if (c < best) { best = c; bestDelta = d; }
    }
  }
  /* chroma intra */
  int cx = x >> 1, cy = y >> 1, clog2 = log2 - 1;
  long bestUv = -1;
  int bestUvMode = DC_PRED;
  for (int m = DC_PRED; m <= PAETH_PRED; m++) {
    if (intra_is_directional(m) && m != V_PRED && m != H_PRED && m != bestMode) continue;
    int delta = (m == bestMode) ? bestDelta : 0;
    long c = intra_satd(fp, src + fp->offset[1], fp->stride[1], 1, cx, cy, clog2, m, delta, pred) +
             intra_satd(fp, src + fp->offset[2], fp->stride[2], 2, cx, cy, clog2, m, delta, pred) +
             (long)lam * (m == DC_PRED ? 8 : 24) / 8;
    if (bestUv < 0 || c < bestUv) { bestUv = c; bestUvMode = m; }
  }
  if (log2 <= 5) {
    /* CfL estimated from source luma (recon picks the final alphas). */
    DEV int32_t *dc = scratch + A_TMP;
    int cw = 1 << clog2;
    long c = (long)lam * 40 / 8;
    for (int p = 1; p < 3; p++) {
      intra_satd(fp, src + fp->offset[p], fp->stride[p], p, cx, cy, clog2, DC_PRED, 0, pred);
      for (int i = 0; i < cw; i++)
        for (int j = 0; j < cw; j++) dc[i * cw + j] = pred[i * 64 + j];
      DEV const uint8_t *sp = src + fp->offset[p] + cy * fp->stride[p] + cx;
      int al = cfl_best_alpha(src + fp->offset[0], fp->stride[0], cx, cy, x + (1 << log2), y + (1 << log2), clog2,
                              clog2, dc, sp, fp->stride[p], pred, 64, 0);
      cfl_predict(src + fp->offset[0], fp->stride[0], cx, cy, x + (1 << log2), y + (1 << log2), clog2, clog2, al, dc,
                  pred, 64);
      c += block_satd(sp, fp->stride[p], pred, cw, cw);
    }
    if (c < bestUv) { bestUv = c; bestUvMode = UV_CFL_PRED; }
  }

  out->mi_row = miRow;
  out->mi_col = miCol;
  out->bsize = bsize;
  out->is_inter = 0;
  out->y_mode = bestMode;
  out->angle_y = bestDelta;
  out->uv_mode = bestUvMode;
  out->angle_uv = (bestUvMode == bestMode) ? bestDelta : 0;
  out->cfl_alpha_u = out->cfl_alpha_v = 0;
  out->mv_row = 0;
  out->mv_col = 0;
  out->tx_type = DCT_DCT;
  long cost = best + bestUv + (long)lam * (fp->frame_is_intra ? 0 : 16) / 8;

  if (!fp->frame_is_intra) {
    int sbIdx = (miRow >> 4) * fp->sb_cols + (miCol >> 4);
    DEV const int32_t *mvp = mvField + (sbIdx * NUM_NODES + NODE_BASE(level) + nodeIdx) * 2;
    int mvr = mvp[0], mvc = mvp[1];
    DEV int32_t *tmp = scratch + A_TMP;
    int w = 1 << log2;
    predict_inter_block(ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, pred, 64, x, y, w, w,
                        mvr, mvc, 0, 0, EIGHTTAP, tmp);
    long interCost = block_satd(src + fp->offset[0] + y * fp->stride[0] + x, fp->stride[0], pred, w, w);
    for (int p = 1; p < 3; p++) {
      predict_inter_block(ref + fp->offset[p], fp->stride[p], fp->frame_w, fp->frame_h, pred, 64, cx, cy,
                          w >> 1, w >> 1, mvr, mvc, 1, 1, EIGHTTAP, tmp);
      interCost += block_satd(src + fp->offset[p] + cy * fp->stride[p] + cx, fp->stride[p], pred, w >> 1, w >> 1);
    }
    interCost += (long)lam * (4 + mv_bits8(mvr, mvc)) / 8;
    if (interCost < cost) {
      cost = interCost;
      out->is_inter = 1;
      out->y_mode = NEWMV;
      out->uv_mode = DC_PRED;
      out->angle_y = 0;
      out->angle_uv = 0;
      out->mv_row = mvr;
      out->mv_col = mvc;
    }
  }
  return cost;
}

/*
 * Analyses superblock `sbIdx` and writes its blocks (decode order) to
 * blocks[sbIdx * MAX_BLOCKS_PER_SB ...]; returns the block count via sb.
 */
INLINE void analyze_sb(DEV const FrameParams *fp, int sbIdx, DEV const uint8_t *src, DEV const uint8_t *ref,
                       DEV const int32_t *mvField, DEV BlockInfo *blocks, DEV SbInfo *sbs,
                       DEV int32_t *scratch) {
  int sbRow = sbIdx / fp->sb_cols, sbCol = sbIdx % fp->sb_cols;
  long cost[NUM_NODES];
  uint8_t mode[NUM_NODES];   /* 0: absent, 1: NONE, 2: SPLIT */
  BlockInfo dec[NUM_NODES];
  int lam = fp->ac_q[0];

  for (int level = 3; level >= 0; level--) {
    int n = 1 << level;
    int size4 = 16 >> level;
    for (int i = 0; i < n * n; i++) {
      int node = NODE_BASE(level) + i;
      int r = sbRow * 16 + (i / n) * size4, c = sbCol * 16 + (i % n) * size4;
      if (r >= fp->mi_rows || c >= fp->mi_cols) {
        cost[node] = 0;
        mode[node] = 0;
        continue;
      }
      int halfSz = size4 >> 1;
      int hasRows = (r + halfSz) < fp->mi_rows, hasCols = (c + halfSz) < fp->mi_cols;
      long splitCost = -1;
      if (level < 3) {
        int cn = n * 2;
        int ci = (i / n) * 2 * cn + (i % n) * 2;
        int cb = NODE_BASE(level + 1);
        splitCost = cost[cb + ci] + cost[cb + ci + 1] + cost[cb + ci + cn] + cost[cb + ci + cn + 1] +
                    (long)lam * 16 / 8;
      }
      long noneCost = -1;
      if (level == 3 || (hasRows && hasCols)) {
        BlockInfo b;
        noneCost = eval_node(fp, src, ref, mvField, level, i, r, c, scratch, &b) + (long)lam * 8 / 8;
        dec[node] = b;
      }
      if (noneCost >= 0 && (splitCost < 0 || noneCost <= splitCost)) {
        cost[node] = noneCost;
        mode[node] = 1;
      } else {
        cost[node] = splitCost;
        mode[node] = 2;
      }
    }
  }

  /* Emit blocks in decode (Z) order: for each 8x8 position in Z order, find
   * the chosen NONE ancestor; emit it when this position is its top-left. */
  int count = 0;
  int coefOff = sbIdx * COEFS_PER_SB;
  for (int z = 0; z < 64; z++) {
    int zx = 0, zy = 0;
    for (int b = 0; b < 3; b++) {
      zx |= ((z >> (2 * b)) & 1) << b;
      zy |= ((z >> (2 * b + 1)) & 1) << b;
    }
    for (int level = 0; level <= 3; level++) {
      int shift = 3 - level;
      int n = 1 << level;
      int node = NODE_BASE(level) + (zy >> shift) * n + (zx >> shift);
      if (mode[node] == 0) break;
      if (mode[node] == 1) {
        if ((zx & ((1 << shift) - 1)) == 0 && (zy & ((1 << shift) - 1)) == 0) {
          BlockInfo b = dec[node];
          int w = 64 >> level;
          b.coef_off[0] = coefOff;
          coefOff += k_min(32, w) * k_min(32, w);
          b.coef_off[1] = coefOff;
          coefOff += (w >> 1) * (w >> 1);
          b.coef_off[2] = coefOff;
          coefOff += (w >> 1) * (w >> 1);
          b.eob[0] = b.eob[1] = b.eob[2] = 0;
          blocks[sbIdx * MAX_BLOCKS_PER_SB + count] = b;
          count++;
        }
        break;
      }
    }
  }
  SbInfo s;
  s.first_block = sbIdx * MAX_BLOCKS_PER_SB;
  s.num_blocks = count;
  s.sb_row = sbRow;
  s.sb_col = sbCol;
  sbs[sbIdx] = s;
}

#endif
