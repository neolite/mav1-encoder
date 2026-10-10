/* Motion search: for every partition node of a superblock, a full-pel SAD
 * search followed by half- and quarter-pel SATD refinement through the
 * normative inter predictor. Results are even 1/8-pel MVs (quarter-pel),
 * as allow_high_precision_mv is 0. Written to mvField[(sb * 85 + node) * 2]. */
#ifndef K_ME_H
#define K_ME_H

#include "k_analysis.h"

/* SAD of a w x w block against the reference at full-pel offset (dy, dx),
 * with reference coordinates clamped to the frame like the predictor. */
INLINE long me_sad(DEV const uint8_t *src, DEV const uint8_t *ref, int stride, int frameW, int frameH, int x, int y,
                   int w, int dy, int dx, long limit) {
  long sad = 0;
  for (int i = 0; i < w; i++) {
    int ry = Clip3(0, frameH - 1, y + i + dy);
    DEV const uint8_t *s = src + (y + i) * stride + x;
    DEV const uint8_t *r = ref + ry * stride;
    for (int j = 0; j < w; j++) {
      int rx = Clip3(0, frameW - 1, x + j + dx);
      sad += k_abs((int)s[j] - (int)r[rx]);
    }
    if (sad >= limit) return sad;
  }
  return sad;
}

INLINE long me_subpel_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, int x, int y,
                           int w, int mvr, int mvc, DEV int32_t *scratch) {
  DEV uint8_t *pred = (DEV uint8_t *)(scratch + A_PRED);
  predict_inter_block(ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, pred, 64, x, y, w, w, mvr, mvc, 0,
                      0, EIGHTTAP, scratch + A_TMP);
  return block_satd(src + fp->offset[0] + y * fp->stride[0] + x, fp->stride[0], pred, w, w);
}

INLINE void me_sb(DEV const FrameParams *fp, int sbIdx, DEV const uint8_t *src, DEV const uint8_t *ref,
                  DEV int32_t *mvField, DEV int32_t *scratch) {
  int sbRow = sbIdx / fp->sb_cols, sbCol = sbIdx % fp->sb_cols;
  int stride = fp->stride[0];
  DEV const uint8_t *s0 = src + fp->offset[0];
  DEV const uint8_t *r0 = ref + fp->offset[0];
  int lam = fp->ac_q[0];
  for (int level = 0; level < 4; level++) {
    int n = 1 << level, w = 64 >> level;
    for (int i = 0; i < n * n; i++) {
      int node = NODE_BASE(level) + i;
      DEV int32_t *out = mvField + (sbIdx * NUM_NODES + node) * 2;
      int x = sbCol * 64 + (i % n) * w, y = sbRow * 64 + (i / n) * w;
      if (x >= fp->frame_w || y >= fp->frame_h) {
        out[0] = out[1] = 0;
        continue;
      }
      /* Search centre: the parent's result (full-pel part), range shrinks
       * with depth. */
      int cy = 0, cx = 0, range = fp->me_range;
      if (level > 0) {
        int pi = ((i / n) >> 1) * (n >> 1) + ((i % n) >> 1);
        DEV const int32_t *par = mvField + (sbIdx * NUM_NODES + NODE_BASE(level - 1) + pi) * 2;
        cy = par[0] >> 3;
        cx = par[1] >> 3;
        range = k_min(range, 4);
      }
      long best = me_sad(s0, r0, stride, fp->frame_w, fp->frame_h, x, y, w, 0, 0, 1L << 40);
      int by = 0, bx = 0;
      for (int pass = 0; pass < 2; pass++) {
        int oy = pass ? cy : 0, ox = pass ? cx : 0, rg = pass ? range : (level == 0 ? range : 0);
        for (int dy = oy - rg; dy <= oy + rg; dy++)
          for (int dx = ox - rg; dx <= ox + rg; dx++) {
            long bits = 2 + 2 * FloorLog2(k_abs(dy) * 8 + 1) + 2 * FloorLog2(k_abs(dx) * 8 + 1);
            long c = me_sad(s0, r0, stride, fp->frame_w, fp->frame_h, x, y, w, dy, dx, 1L << 40) * 1 +
                     (long)lam * bits / 64;
            long bc = best + (long)lam * (2 + 2 * FloorLog2(k_abs(by) * 8 + 1) + 2 * FloorLog2(k_abs(bx) * 8 + 1)) / 64;
            if (c < bc) {
              best = c - (long)lam * bits / 64;
              by = dy;
              bx = dx;
            }
          }
      }
      /* sub-pel refinement: steps of 4 (half) then 2 (quarter) in 1/8 units */
      int mvr = by * 8, mvc = bx * 8;
      long bestS = me_subpel_cost(fp, src, ref, x, y, w, mvr, mvc, scratch);
      for (int step = 4; step >= 2; step >>= 1) {
        int cr = mvr, cc = mvc;
        for (int k = 0; k < 8; k++) {
          int dr = (k < 3) ? -step : (k < 5 ? 0 : step);
          int dc = (k == 0 || k == 3 || k == 5) ? -step : ((k == 1 || k == 6) ? 0 : step);
          long c = me_subpel_cost(fp, src, ref, x, y, w, cr + dr, cc + dc, scratch);
          if (c < bestS) {
            bestS = c;
            mvr = cr + dr;
            mvc = cc + dc;
          }
        }
      }
      out[0] = mvr;
      out[1] = mvc;
    }
  }
}

#endif
