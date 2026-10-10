/* Motion search: for every partition node of a superblock, a full-pel SAD
 * search followed by half- and quarter-pel SATD refinement through the
 * normative inter predictor. Results are even 1/8-pel MVs (quarter-pel),
 * as allow_high_precision_mv is 0. Written to mvField[(sb * 85 + node) * 2].
 *
 * The full-pel search is the first minimum of cost(k) = SAD + lambda * bits
 * over an ordered candidate list k = 0 .. count-1 (level 0: a coarse grid of
 * spacing 4, then a second list refining +-3 around its winner). The CPU walks the list
 * serially; the GPU gives each lane of a SIMD group a stride of candidates
 * and reduces (cost, k) lexicographically, which yields the same candidate.
 * The levels (64, 32, 16, 8) depend on each other (children search around
 * the parent's vector) and run as separate dispatches. */
#ifndef K_ME_H
#define K_ME_H

#include "k_analysis.h"

/* SAD of a w x w block against the reference at full-pel offset (dy, dx),
 * with reference coordinates clamped to the frame like the predictor. */
INLINE int me_sad(DEV const uint8_t *src, DEV const uint8_t *ref, int stride, int frameW, int frameH, int x, int y,
                  int w, int dy, int dx) {
  int sad = 0;
  for (int i = 0; i < w; i++) {
    int ry = Clip3(0, frameH - 1, y + i + dy);
    DEV const uint8_t *s = src + (y + i) * stride + x;
    DEV const uint8_t *r = ref + ry * stride;
    for (int j = 0; j < w; j++) {
      int rx = Clip3(0, frameW - 1, x + j + dx);
      sad += k_abs((int)s[j] - (int)r[rx]);
    }
  }
  return sad;
}

/* One search node: block position and the candidate grid. */
typedef struct {
  int valid;
  int x, y, w;
  int cy, cx, range; /* grid centre and radius */
  int step;          /* grid spacing */
  int stages;        /* level 0: coarse grid, then a +-3 refinement */
  int count;         /* candidates of the current stage */
} MeNode;

#define ME_COARSE_STEP 4
#define ME_REFINE 3

INLINE MeNode me_node(DEV const FrameParams *fp, int sbIdx, int level, int i, DEV const int32_t *mvField) {
  MeNode m;
  int sbRow = sbIdx / fp->sb_cols, sbCol = sbIdx % fp->sb_cols;
  int n = 1 << level;
  m.w = 64 >> level;
  m.x = sbCol * 64 + (i % n) * m.w;
  m.y = sbRow * 64 + (i / n) * m.w;
  m.valid = m.x < fp->frame_w && m.y < fp->frame_h;
  m.cy = 0;
  m.cx = 0;
  m.range = fp->me_range;
  if (level > 0) {
    int pi = ((i / n) >> 1) * (n >> 1) + ((i % n) >> 1);
    DEV const int32_t *par = mvField + (sbIdx * NUM_NODES + NODE_BASE(level - 1) + pi) * 2;
    m.cy = par[0] >> 3;
    m.cx = par[1] >> 3;
    m.range = k_min(m.range, 4);
  }
  m.step = 1;
  m.stages = 1;
  if (level == 0) {
    /* stage 0: grid of spacing 4 over +-range around (0, 0) */
    m.step = ME_COARSE_STEP;
    m.range = (m.range / ME_COARSE_STEP) * ME_COARSE_STEP;
    m.stages = 2;
  }
  int side = 2 * (m.range / m.step) + 1;
  m.count = 1 + side * side;
  return m;
}

/* Second stage of level 0: +-3 full-pel around the coarse winner. */
INLINE void me_refine_stage(THR MeNode *m, int wy, int wx) {
  m->cy = wy;
  m->cx = wx;
  m->range = ME_REFINE;
  m->step = 1;
  int side = 2 * ME_REFINE + 1;
  m->count = 1 + side * side;
}

/* Candidate k of the current stage: 0 is the grid centre ((0, 0) at level 0,
 * the parent's vector below, the coarse winner when refining), then the grid
 * around the centre in raster order. */
INLINE void me_candidate(THR const MeNode *m, int k, THR int *dy, THR int *dx) {
  if (k == 0) {
    *dy = m->cy;
    *dx = m->cx;
    return;
  }
  k -= 1;
  int side = 2 * (m->range / m->step) + 1;
  *dy = m->cy - m->range + (k / side) * m->step;
  *dx = m->cx - m->range + (k % side) * m->step;
}

INLINE int me_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, THR const MeNode *m,
                   int dy, int dx) {
  int bits = 2 + 2 * FloorLog2(k_abs(dy) * 8 + 1) + 2 * FloorLog2(k_abs(dx) * 8 + 1);
  return me_sad(src + fp->offset[0], ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, m->x, m->y, m->w,
                dy, dx) +
         fp->ac_q[0] * bits / 64;
}

/* Scratch of one sub-pel evaluation: a w x w byte prediction and the
 * (w + 7) x w filter intermediate. Nodes of a level run concurrently, and on
 * the GPU 9 lanes per node evaluate sub-pel candidates concurrently. */
INLINE int me_slot(int level) {
  int w = 64 >> level;
  return (w * w + 3) / 4 + (w + 7) * w;
}
INLINE int me_node_scratch(int level, int i) { return i * me_slot(level); }
#define ME_LANES 9
#define ME_SB_INTS (64 * ME_LANES * (16 + 15 * 8)) /* level 3 is the largest */

/* Sub-pel neighbour k (0..7) at distance step, in 1/8 units. */
INLINE void me_neighbour(int k, int step, THR int *dr, THR int *dc) {
  *dr = (k < 3) ? -step : (k < 5 ? 0 : step);
  *dc = (k == 0 || k == 3 || k == 5) ? -step : ((k == 1 || k == 6) ? 0 : step);
}

INLINE int me_subpel_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                          THR const MeNode *m, int mvr, int mvc, DEV int32_t *scratch) {
  DEV uint8_t *pred = (DEV uint8_t *)scratch;
  predict_inter_block(ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, pred, m->w, m->x, m->y, m->w,
                      m->w, mvr, mvc, 0, 0, EIGHTTAP, scratch + (m->w * m->w + 3) / 4);
  return block_satd_s(src + fp->offset[0] + m->y * fp->stride[0] + m->x, fp->stride[0], pred, m->w, m->w, m->w);
}

/* Sub-pel refinement around the full-pel winner: steps of 4 (half) then 2
 * (quarter) in 1/8 units, first minimum of the 8 neighbours per step. */
INLINE void me_finish(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, THR const MeNode *m,
                      int by, int bx, DEV int32_t *out, DEV int32_t *scratch) {
  int mvr = by * 8, mvc = bx * 8;
  int bestS = me_subpel_cost(fp, src, ref, m, mvr, mvc, scratch);
  for (int step = 4; step >= 2; step >>= 1) {
    int cr = mvr, cc = mvc;
    for (int k = 0; k < 8; k++) {
      int dr, dc;
      me_neighbour(k, step, &dr, &dc);
      int c = me_subpel_cost(fp, src, ref, m, cr + dr, cc + dc, scratch);
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

/* Serial search of one node (CPU backend). */
INLINE void me_node_serial(DEV const FrameParams *fp, int sbIdx, int level, int i, DEV const uint8_t *src,
                           DEV const uint8_t *ref, DEV int32_t *mvField, DEV int32_t *scratch) {
  MeNode m = me_node(fp, sbIdx, level, i, mvField);
  DEV int32_t *out = mvField + (sbIdx * NUM_NODES + NODE_BASE(level) + i) * 2;
  if (!m.valid) {
    out[0] = out[1] = 0;
    return;
  }
  int by = 0, bx = 0;
  for (int stage = 0; stage < m.stages; stage++) {
    if (stage > 0) me_refine_stage(&m, by, bx);
    int best = -1;
    for (int k = 0; k < m.count; k++) {
      int dy, dx;
      me_candidate(&m, k, &dy, &dx);
      int c = me_cost(fp, src, ref, &m, dy, dx);
      if (best < 0 || c < best) {
        best = c;
        by = dy;
        bx = dx;
      }
    }
  }
  me_finish(fp, src, ref, &m, by, bx, out, scratch + me_node_scratch(level, i));
}

#endif
