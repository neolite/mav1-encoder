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
                      int x, int y, int log2W, int log2H, int mode, int delta, DEV uint8_t *pred) {
  int sub = planeIdx > 0;
  IntraArgs a;
  a.haveLeft = x > 0;
  a.haveAbove = y > 0;
  a.haveAboveRight = 0;
  a.haveBelowLeft = 0;
  a.mode = mode;
  a.angleDelta = delta;
  a.log2W = log2W;
  a.log2H = log2H;
  a.maxX = ((fp->mi_cols * 4) >> sub) - 1;
  a.maxY = ((fp->mi_rows * 4) >> sub) - 1;
  a.edgeFilter = 0;
  a.filterType = 0;
  predict_intra(plane, stride, x, y, a, pred, 64);
  return block_satd(plane + y * stride + x, stride, pred, 1 << log2W, 1 << log2H);
}

INLINE int mv_bits8(int mvr, int mvc) {
  return 8 * (2 + 2 * FloorLog2(k_abs(mvr) + 1) + 2 * FloorLog2(k_abs(mvc) + 1));
}

/* Evaluates the block (miRow, miCol, bsize); the inter candidate uses the
 * motion vector of square node `mvNode` of the superblock. Returns the cost
 * and fills the decision in *out. */
INLINE long eval_block(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                       DEV const int32_t *mvField, int mvNode, int miRow, int miCol, int bsize,
                       DEV int32_t *scratch, THR BlockInfo *out) {
  DEV uint8_t *pred = (DEV uint8_t *)(scratch + A_PRED);
  int lw = Mi_Width_Log2[bsize] + 2, lh = Mi_Height_Log2[bsize] + 2;
  int w = 1 << lw, h = 1 << lh;
  int x = miCol * 4, y = miRow * 4;
  int lam = fp->ac_q[0];

  /* luma intra */
  long best = -1;
  int bestMode = DC_PRED, bestDelta = 0;
  for (int mode = DC_PRED; mode <= PAETH_PRED; mode++) {
    long c = intra_satd(fp, src + fp->offset[0], fp->stride[0], 0, x, y, lw, lh, mode, 0, pred) +
             (long)lam * mode_bits8(mode) / 8;
    if (best < 0 || c < best) { best = c; bestMode = mode; }
  }
  if (intra_is_directional(bestMode)) {
    int base = bestMode;
    for (int d = -MAX_ANGLE_DELTA; d <= MAX_ANGLE_DELTA; d++) {
      if (d == 0) continue;
      long c = intra_satd(fp, src + fp->offset[0], fp->stride[0], 0, x, y, lw, lh, base, d, pred) +
               (long)lam * (mode_bits8(base) + 16) / 8;
      if (c < best) { best = c; bestDelta = d; }
    }
  }
  /* chroma intra */
  int cx = x >> 1, cy = y >> 1, clw = lw - 1, clh = lh - 1;
  long bestUv = -1;
  int bestUvMode = DC_PRED;
  for (int m = DC_PRED; m <= PAETH_PRED; m++) {
    if (intra_is_directional(m) && m != V_PRED && m != H_PRED && m != bestMode) continue;
    int delta = (m == bestMode) ? bestDelta : 0;
    long c = intra_satd(fp, src + fp->offset[1], fp->stride[1], 1, cx, cy, clw, clh, m, delta, pred) +
             intra_satd(fp, src + fp->offset[2], fp->stride[2], 2, cx, cy, clw, clh, m, delta, pred) +
             (long)lam * (m == DC_PRED ? 8 : 24) / 8;
    if (bestUv < 0 || c < bestUv) { bestUv = c; bestUvMode = m; }
  }
  if (k_max(lw, lh) <= 5) {
    /* CfL estimated from source luma (recon picks the final alphas). */
    DEV int32_t *dc = scratch + A_TMP;
    int cw = 1 << clw, ch = 1 << clh;
    long c = (long)lam * 40 / 8;
    for (int p = 1; p < 3; p++) {
      intra_satd(fp, src + fp->offset[p], fp->stride[p], p, cx, cy, clw, clh, DC_PRED, 0, pred);
      for (int i = 0; i < ch; i++)
        for (int j = 0; j < cw; j++) dc[i * cw + j] = pred[i * 64 + j];
      DEV const uint8_t *sp = src + fp->offset[p] + cy * fp->stride[p] + cx;
      int al = cfl_best_alpha(src + fp->offset[0], fp->stride[0], cx, cy, x + w, y + h, clw, clh, dc, sp,
                              fp->stride[p], pred, 64, 0);
      cfl_predict(src + fp->offset[0], fp->stride[0], cx, cy, x + w, y + h, clw, clh, al, dc, pred, 64);
      c += block_satd(sp, fp->stride[p], pred, cw, ch);
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
    DEV const int32_t *mvp = mvField + (sbIdx * NUM_NODES + mvNode) * 2;
    int mvr = mvp[0], mvc = mvp[1];
    DEV int32_t *tmp = scratch + A_TMP;
    predict_inter_block(ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, pred, 64, x, y, w, h,
                        mvr, mvc, 0, 0, EIGHTTAP, tmp);
    long interCost = block_satd(src + fp->offset[0] + y * fp->stride[0] + x, fp->stride[0], pred, w, h);
    for (int p = 1; p < 3; p++) {
      predict_inter_block(ref + fp->offset[p], fp->stride[p], fp->frame_w, fp->frame_h, pred, 64, cx, cy,
                          w >> 1, h >> 1, mvr, mvc, 1, 1, EIGHTTAP, tmp);
      interCost += block_satd(src + fp->offset[p] + cy * fp->stride[p] + cx, fp->stride[p], pred, w >> 1, h >> 1);
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

/* Number of blocks of partition p and the k-th block's offset / size
 * (decode order, spec decode_partition). half/quarter in mi units. */
INLINE int part_block(int p, int bSize, int k, int half, THR int *dr, THR int *dc) {
  int sub = Partition_Subsize[p][bSize], split = Partition_Subsize[PARTITION_SPLIT][bSize];
  int q = half >> 1;
  *dr = 0;
  *dc = 0;
  switch (p) {
    case PARTITION_HORZ: *dr = k * half; return sub;
    case PARTITION_VERT: *dc = k * half; return sub;
    case PARTITION_HORZ_A:
      if (k < 2) { *dc = k * half; return split; }
      *dr = half; return sub;
    case PARTITION_HORZ_B:
      if (k == 0) return sub;
      *dr = half; *dc = (k - 1) * half; return split;
    case PARTITION_VERT_A:
      if (k < 2) { *dr = k * half; return split; }
      *dc = half; return sub;
    case PARTITION_VERT_B:
      if (k == 0) return sub;
      *dc = half; *dr = (k - 1) * half; return split;
    case PARTITION_HORZ_4: *dr = k * q; return sub;
    case PARTITION_VERT_4: *dc = k * q; return sub;
    default: return sub; /* NONE */
  }
}

/* Blocks actually decoded for partition p at (r, c) (handles frame edges). */
INLINE int part_count(DEV const FrameParams *fp, int p, int r, int c, int half) {
  int hasRows = (r + half) < fp->mi_rows, hasCols = (c + half) < fp->mi_cols;
  if (p == PARTITION_NONE) return 1;
  if (p == PARTITION_HORZ) return hasRows ? 2 : 1;
  if (p == PARTITION_VERT) return hasCols ? 2 : 1;
  if (p == PARTITION_HORZ_4) return (r + (half >> 1) * 3 < fp->mi_rows) ? 4 : 3;
  if (p == PARTITION_VERT_4) return (c + (half >> 1) * 3 < fp->mi_cols) ? 4 : 3;
  return 3;
}

/* Cost of a rectangular block of the partition, memoised per call site. */
INLINE long part_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                      DEV const int32_t *mvField, int node, int p, int r, int c, int bSize, int half,
                      THR const long *noneCost, int childBase, int cn, int ci, DEV int32_t *scratch) {
  long total = 0;
  int n = part_count(fp, p, r, c, half);
  for (int k = 0; k < n; k++) {
    int dr, dc;
    int bs = part_block(p, bSize, k, half, &dr, &dc);
    if (bs == Partition_Subsize[PARTITION_SPLIT][bSize] && p != PARTITION_HORZ_4 && p != PARTITION_VERT_4 &&
        p != PARTITION_HORZ && p != PARTITION_VERT) {
      /* square quarter: reuse the child's NONE cost */
      long cc = noneCost[childBase + ci + (dr ? cn : 0) + (dc ? 1 : 0)];
      if (cc < 0) return -1;
      total += cc;
    } else {
      BlockInfo b;
      total += eval_block(fp, src, ref, mvField, node, r + dr, c + dc, bs, scratch, &b);
    }
  }
  return total;
}

/*
 * Analyses superblock `sbIdx` and writes its blocks (decode order) to
 * blocks[sbIdx * MAX_BLOCKS_PER_SB ...] and the partition of every square
 * node to sbs[sbIdx].part.
 */
INLINE void analyze_sb(DEV const FrameParams *fp, int sbIdx, DEV const uint8_t *src, DEV const uint8_t *ref,
                       DEV const int32_t *mvField, DEV BlockInfo *blocks, DEV SbInfo *sbs,
                       DEV int32_t *scratch) {
  int sbRow = sbIdx / fp->sb_cols, sbCol = sbIdx % fp->sb_cols;
  long cost[NUM_NODES];
  long noneCost[NUM_NODES];
  int lam = fp->ac_q[0];
  DEV SbInfo *sb = &sbs[sbIdx];

  for (int level = 3; level >= 0; level--) {
    int n = 1 << level;
    int size4 = 16 >> level;
    int bSize = level_bsize(level);
    for (int i = 0; i < n * n; i++) {
      int node = NODE_BASE(level) + i;
      int r = sbRow * 16 + (i / n) * size4, c = sbCol * 16 + (i % n) * size4;
      noneCost[node] = -1;
      if (r >= fp->mi_rows || c >= fp->mi_cols) {
        cost[node] = 0;
        sb->part[node] = -1;
        continue;
      }
      int halfSz = size4 >> 1;
      int hasRows = (r + halfSz) < fp->mi_rows, hasCols = (c + halfSz) < fp->mi_cols;
      int cn = n * 2, ci = (i / n) * 2 * cn + (i % n) * 2, cb = NODE_BASE(level + 1);
      long bestC = -1;
      int bestP = PARTITION_SPLIT;
      if (level < 3) {
        bestC = cost[cb + ci] + cost[cb + ci + 1] + cost[cb + ci + cn] + cost[cb + ci + cn + 1] +
                (long)lam * 16 / 8;
      }
      for (int p = PARTITION_NONE; p <= PARTITION_VERT_4; p++) {
        if (p == PARTITION_SPLIT) continue;
        if (level == 3 && p != PARTITION_NONE) continue;          /* blocks stay >= 8x8 */
        if (level == 0 && (p == PARTITION_HORZ_4 || p == PARTITION_VERT_4)) {
          /* 64x16 / 16x64 are allowed */
        }
        if (level == 2 && (p == PARTITION_HORZ_4 || p == PARTITION_VERT_4)) continue; /* 16x4 */
        if (!(hasRows && hasCols)) {
          if (!((p == PARTITION_HORZ && hasCols && !hasRows) || (p == PARTITION_VERT && hasRows && !hasCols)))
            continue;
        }
        long pc = part_cost(fp, src, ref, mvField, node, p, r, c, bSize, halfSz, noneCost, cb, cn, ci, scratch);
        if (pc < 0) continue;
        pc += (long)lam * (p == PARTITION_NONE ? 8 : 24) / 8;
        if (p == PARTITION_NONE) noneCost[node] = pc;
        if (bestC < 0 || pc <= bestC) {
          bestC = pc;
          bestP = p;
        }
      }
      cost[node] = bestC;
      sb->part[node] = bestP;
    }
  }

  /* Emit blocks in decode order: walk 8x8 positions in Z order; the first
   * non-SPLIT node on the path owns the position; its blocks are emitted
   * when the walk reaches its top-left corner. */
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
      int p = sb->part[node];
      if (p < 0) break;
      if (p == PARTITION_SPLIT) continue;
      if ((zx & ((1 << shift) - 1)) == 0 && (zy & ((1 << shift) - 1)) == 0) {
        int size4 = 16 >> level, half = size4 >> 1;
        int r = sbRow * 16 + (zy >> shift) * size4, c = sbCol * 16 + (zx >> shift) * size4;
        int nb = part_count(fp, p, r, c, half);
        for (int k = 0; k < nb; k++) {
          int dr, dc;
          int bs = part_block(p, level_bsize(level), k, half, &dr, &dc);
          BlockInfo b;
          eval_block(fp, src, ref, mvField, node, r + dr, c + dc, bs, scratch, &b);
          int w = Block_Width(bs), h = Block_Height(bs);
          b.coef_off[0] = coefOff;
          coefOff += k_min(32, w) * k_min(32, h);
          b.coef_off[1] = coefOff;
          coefOff += (w >> 1) * (h >> 1);
          b.coef_off[2] = coefOff;
          coefOff += (w >> 1) * (h >> 1);
          b.eob[0] = b.eob[1] = b.eob[2] = 0;
          blocks[sbIdx * MAX_BLOCKS_PER_SB + count] = b;
          count++;
        }
      }
      break;
    }
  }
  sb->first_block = sbIdx * MAX_BLOCKS_PER_SB;
  sb->num_blocks = count;
  sb->sb_row = sbRow;
  sb->sb_col = sbCol;
}

#endif
