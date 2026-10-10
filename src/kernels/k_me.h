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
#ifdef __METAL_VERSION__
/* 4x4 SATD of the rows of differences r0..r3 (vertical butterflies on the
 * row vectors, then horizontal ones): equals satd4x4. */
INLINE int satd4x4_v(int4 r0, int4 r1, int4 r2, int4 r3) {
  int4 a0 = r0 + r1, a1 = r0 - r1, a2 = r2 + r3, a3 = r2 - r3;
  int4 t[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
  int s = 0;
  for (int i = 0; i < 4; i++) {
    int4 u = t[i].xxzz + t[i].yyww * int4(1, -1, 1, -1);
    int4 v = u.xyxy + u.zwzw * int4(1, 1, -1, -1);
    uint4 a = uint4(abs(v));
    s += (int)(a.x + a.y + a.z + a.w);
  }
  return s;
}
#endif

INLINE int me_sad(DEV const uint8_t *src, DEV const uint8_t *ref, int stride, int frameW, int frameH, int x, int y,
                  int w, int dy, int dx) {
#ifdef __METAL_VERSION__
  if (x + dx >= 0 && x + dx + w <= frameW && y + dy >= 0 && y + dy + w <= frameH) {
    /* no clamping: 4 samples per load */
    uint4 acc = 0;
    for (int i = 0; i < w; i++) {
      DEV const packed_uchar4 *s = (DEV const packed_uchar4 *)(src + (y + i) * stride + x);
      DEV const packed_uchar4 *r = (DEV const packed_uchar4 *)(ref + (y + i + dy) * stride + x + dx);
      for (int j = 0; j < (w >> 2); j++) acc += uint4(absdiff(uchar4(s[j]), uchar4(r[j])));
    }
    return (int)(acc.x + acc.y + acc.z + acc.w);
  }
#endif
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
  int pr, pc;        /* parent's vector (1/8 pel; levels > 0) */
  int copy;          /* turbo: take the parent's vector without searching */
} MeNode;

/* turbo bits (FrameParams.turbo) of the motion search */
#define T_ME_CR1 2     /* children search +-1 full-pel around the parent */
#define T_ME_L3COPY 4  /* 8x8 nodes take the 16x16 parent's vector */
#define T_ME_KEEP 8    /* children keep the parent's sub-pel vector when the full-pel winner is the parent's */
#define T_ME_HPEL 16   /* sub-pel refinement on precomputed quarter-pel planes */
#define T_ME_CR2 64    /* children search +-2 full-pel around the parent */
#define T_ME_L3SUB 4096 /* 8x8 nodes: no full-pel search, sub-pel steps around the parent's vector */
#define T_ME_REF2 512  /* 64x64 full-pel refinement +-2 around the coarse winner (instead of +-3) */
#define T_ME_DEC 32    /* 64x64 coarse grid (step 4) on 4x4-decimated planes */
#define T_ME_CROSS 8192 /* sub-pel steps try the 4 axial neighbours only */
#define T_ME_G3COPY 16384 /* leaf frames: 8x8 nodes of the second reference take the 16x16 parent's vector */
INLINE int me_gold_copy(DEV const FrameParams *fp, int level) {
  return level == 3 && (fp->turbo & T_ME_G3COPY) && fp->leaf && !fp->frame_is_intra;
}

/* Plane buffer of the turbo motion search: ME_QP_ENTRIES entries, each the
 * 16 quarter-pel planes of one reference followed by its 4x4-decimated
 * plane (T_ME_DEC), then the decimated source. Entries are reused across
 * frames for the same reference picture. */
#define ME_QP_ENTRIES 3
INLINE int dec_w(DEV const FrameParams *fp) { return (fp->frame_w + 3) >> 2; }
INLINE int dec_h(DEV const FrameParams *fp) { return (fp->frame_h + 3) >> 2; }
INLINE long qp_entry_bytes(DEV const FrameParams *fp) {
  return 16L * fp->frame_w * fp->frame_h + (long)dec_w(fp) * dec_h(fp);
}
/* decimated plane of an entry */
INLINE DEV const uint8_t *qp_dec(DEV const FrameParams *fp, DEV const uint8_t *entry) {
  return entry + 16L * fp->frame_w * fp->frame_h;
}
INLINE void dec_build(DEV const FrameParams *fp, DEV const uint8_t *pl, DEV uint8_t *out, int y, int x) {
  int s = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      s += pl[fp->offset[0] + Clip3(0, fp->frame_h - 1, 4 * y + i) * fp->stride[0] + Clip3(0, fp->frame_w - 1, 4 * x + j)];
  out[(size_t)y * dec_w(fp) + x] = (uint8_t)((s + 8) >> 4);
}
/* SAD x16 of the w x w block at (x, y) on the decimated planes, offset
 * (dy, dx) (multiples of 4). */
INLINE int dec_sad(DEV const FrameParams *fp, DEV const uint8_t *ds, DEV const uint8_t *dr, int x, int y, int w, int dy,
                   int dx) {
  int W = dec_w(fp), H = dec_h(fp), n = w >> 2, x0 = x >> 2, y0 = y >> 2, oy = dy >> 2, ox = dx >> 2, sad = 0;
  for (int i = 0; i < n; i++) {
    int sy = k_min(y0 + i, H - 1), ry = Clip3(0, H - 1, y0 + i + oy);
    for (int j = 0; j < n; j++)
      sad += k_abs((int)ds[sy * W + k_min(x0 + j, W - 1)] - (int)dr[ry * W + Clip3(0, W - 1, x0 + j + ox)]);
  }
  return sad * 16;
}

/* Turbo: the node's search ends with the parent's vector (copy node, or the
 * full-pel winner is the parent's own position). */
INLINE int me_take_parent(DEV const FrameParams *fp, int level, THR const MeNode *m, int by, int bx) {
  return level > 0 && (m->copy || ((fp->turbo & T_ME_KEEP) && by == m->cy && bx == m->cx));
}

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
  m.pr = m.pc = 0;
  m.copy = 0;
  if (level > 0) {
    int pi = ((i / n) >> 1) * (n >> 1) + ((i % n) >> 1);
    DEV const int32_t *par = mvField + (sbIdx * NUM_NODES + NODE_BASE(level - 1) + pi) * 2;
    m.pr = par[0];
    m.pc = par[1];
    m.cy = par[0] >> 3;
    m.cx = par[1] >> 3;
    m.range = k_min(m.range, (fp->turbo & T_ME_CR1) ? 1 : (fp->turbo & T_ME_CR2) ? 2 : 4);
    m.copy = ((fp->turbo & T_ME_L3COPY) || leaf_no8(fp)) && level == 3;
  }
  m.step = 1;
  m.stages = 1;
  if (level == 0) {
    /* stage 0: grid of spacing 4 over +-range around (0, 0) */
    m.step = ME_COARSE_STEP;
    m.range = (m.range / ME_COARSE_STEP) * ME_COARSE_STEP;
    m.stages = 2;
  }
  /* turbo T_ME_L3SUB: 8x8 nodes refine the parent's vector at sub-pel only */
  if (level == 3 && (fp->turbo & T_ME_L3SUB)) m.stages = 0;
  int side = 2 * (m.range / m.step) + 1;
  m.count = 1 + side * side;
  return m;
}

/* Second stage of level 0: +-3 full-pel around the coarse winner. */
INLINE void me_refine_stage(DEV const FrameParams *fp, THR MeNode *m, int wy, int wx) {
  m->cy = wy;
  m->cx = wx;
  m->range = (fp->turbo & T_ME_REF2) ? 2 : ME_REFINE;
  m->step = 1;
  int side = 2 * m->range + 1;
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

INLINE int me_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, DEV const uint8_t *ds,
                   DEV const uint8_t *dr, THR const MeNode *m, int dy, int dx) {
  int bits = 2 + 2 * FloorLog2(k_abs(dy) * 8 + 1) + 2 * FloorLog2(k_abs(dx) * 8 + 1);
  if ((fp->turbo & T_ME_DEC) && m->step == ME_COARSE_STEP)
    return dec_sad(fp, ds, dr, m->x, m->y, m->w, dy, dx) + fp->ac_q[0] * bits / 64;
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

/* Sub-pel neighbour j (0 .. me_nb_count - 1) of the turbo set. */
INLINE int me_nb_count(DEV const FrameParams *fp) { return (fp->turbo & T_ME_CROSS) ? 4 : 8; }
INLINE void me_nb(DEV const FrameParams *fp, int j, int step, THR int *dr, THR int *dc) {
  me_neighbour((fp->turbo & T_ME_CROSS) ? (j == 0 ? 1 : j == 1 ? 3 : j == 2 ? 4 : 6) : j, step, dr, dc);
}

INLINE int me_unit_satd_hp(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                           DEV const uint8_t *hp, THR const MeNode *m, int mvr, int mvc, int u);
INLINE int me_subpel_cost(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                          DEV const uint8_t *hp, THR const MeNode *m, int mvr, int mvc, DEV int32_t *scratch) {
  if (fp->turbo & T_ME_HPEL) {
    int s = 0, nu = (m->w >> 2) * (m->w >> 2);
    for (int u = 0; u < nu; u++) s += me_unit_satd_hp(fp, src, ref, hp, m, mvr, mvc, u);
    return s;
  }
  DEV uint8_t *pred = (DEV uint8_t *)scratch;
  predict_inter_block(ref + fp->offset[0], fp->stride[0], fp->frame_w, fp->frame_h, pred, m->w, m->x, m->y, m->w,
                      m->w, mvr, mvc, 0, 0, EIGHTTAP, scratch + (m->w * m->w + 3) / 4);
  return block_satd_s(src + fp->offset[0] + m->y * fp->stride[0] + m->x, fp->stride[0], pred, m->w, m->w, m->w);
}

/* Sub-pel refinement around the full-pel winner: steps of 4 (half) then 2
 * (quarter) in 1/8 units, first minimum of the 8 neighbours per step. */
INLINE void me_finish(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref, DEV const uint8_t *hp,
                      THR const MeNode *m, int by, int bx, DEV int32_t *out, DEV int32_t *scratch) {
  int mvr = m->stages ? by * 8 : m->pr, mvc = m->stages ? bx * 8 : m->pc;
  int bestS = me_subpel_cost(fp, src, ref, hp, m, mvr, mvc, scratch);
  for (int step = 4; step >= 2; step >>= 1) {
    int cr = mvr, cc = mvc;
    for (int k = 0; k < me_nb_count(fp); k++) {
      int dr, dc;
      me_nb(fp, k, step, &dr, &dc);
      int c = me_subpel_cost(fp, src, ref, hp, m, cr + dr, cc + dc, scratch);
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

/* Quarter-pel planes (turbo T_ME_HPEL): for every luma sample (y, x) of a
 * reference, plane fy * 4 + fx holds the EIGHTTAP prediction at
 * (y + fy / 4, x + fx / 4), computed per sample like predict_inter_block
 * (plane 0 is the reference itself). Layout: qp[((r * 16) + k) * W * H].
 * The motion search reads its sub-pel candidates from them; the coded
 * prediction stays normative. */
INLINE void qpel_build(DEV const FrameParams *fp, DEV const uint8_t *ref, DEV uint8_t *qp, int y, int x) {
  DEV const uint8_t *rp = ref + fp->offset[0];
  int stride = fp->stride[0], W = fp->frame_w, H = fp->frame_h;
  int hrow[4][8];
  for (int t = 0; t < 8; t++) {
    DEV const uint8_t *row = rp + Clip3(0, H - 1, y + t - 3) * stride;
    int px[8];
    for (int u = 0; u < 8; u++) px[u] = row[Clip3(0, W - 1, x + u - 3)];
    for (int fx = 0; fx < 4; fx++) {
      KCONST int32_t *f = Subpel_Filters[EIGHTTAP][4 * fx];
      int s = 0;
      for (int u = 0; u < 8; u++) s += f[u] * px[u];
      hrow[fx][t] = Round2(s, 3);
    }
  }
  size_t o = (size_t)y * W + x, n = (size_t)W * H;
  for (int fy = 0; fy < 4; fy++) {
    KCONST int32_t *f = Subpel_Filters[EIGHTTAP][4 * fy];
    for (int fx = 0; fx < 4; fx++) {
      int s = 0;
      for (int t = 0; t < 8; t++) s += f[t] * hrow[fx][t];
      qp[(size_t)(fy * 4 + fx) * n + o] = (uint8_t)Clip1(Round2(s, 11));
    }
  }
}

/* SATD of 4x4 unit u of the node against the quarter-pel-plane prediction
 * at (mvr, mvc) (even, 1/8 units). */
INLINE int me_unit_satd_hp(DEV const FrameParams *fp, DEV const uint8_t *src, DEV const uint8_t *ref,
                           DEV const uint8_t *qp, THR const MeNode *m, int mvr, int mvc, int u) {
  int W = fp->frame_w, H = fp->frame_h;
  int nu = m->w >> 2;
  int ux = m->x + (u % nu) * 4, uy = m->y + (u / nu) * 4;
  DEV const uint8_t *pl = qp + (size_t)(((mvr >> 1) & 3) * 4 + ((mvc >> 1) & 3)) * W * H;
  int iy = uy + (mvr >> 3), ix = ux + (mvc >> 3);
  DEV const uint8_t *sp = src + fp->offset[0] + uy * fp->stride[0] + ux;
#ifdef __METAL_VERSION__
  if (iy >= 0 && iy + 4 <= H && ix >= 0 && ix + 4 <= W) {
    int4 r[4];
    for (int k = 0; k < 4; k++)
      r[k] = int4(uchar4(*(DEV const packed_uchar4 *)(sp + k * fp->stride[0]))) -
             int4(uchar4(*(DEV const packed_uchar4 *)(pl + (size_t)(iy + k) * W + ix)));
    return satd4x4_v(r[0], r[1], r[2], r[3]);
  }
#endif
  int32_t d[16];
  for (int r = 0; r < 4; r++) {
    DEV const uint8_t *row = pl + (size_t)Clip3(0, H - 1, iy + r) * W;
    for (int c = 0; c < 4; c++) d[r * 4 + c] = (int)sp[r * fp->stride[0] + c] - (int)row[Clip3(0, W - 1, ix + c)];
  }
  return satd4x4(d);
}

/* Serial search of one node (CPU backend). */
INLINE void me_node_serial(DEV const FrameParams *fp, int sbIdx, int level, int i, DEV const uint8_t *src,
                           DEV const uint8_t *ref, DEV const uint8_t *hp, DEV const uint8_t *ds, DEV const uint8_t *dr,
                           DEV int32_t *mvField, DEV int32_t *scratch, int isGold) {
  MeNode m = me_node(fp, sbIdx, level, i, mvField);
  if (isGold && me_gold_copy(fp, level)) m.copy = 1;
  DEV int32_t *out = mvField + (sbIdx * NUM_NODES + NODE_BASE(level) + i) * 2;
  if (!m.valid) {
    out[0] = out[1] = 0;
    return;
  }
  int by = 0, bx = 0;
  if (m.copy) {
    out[0] = m.pr;
    out[1] = m.pc;
    return;
  }
  for (int stage = 0; stage < m.stages; stage++) {
    if (stage > 0) me_refine_stage(fp, &m, by, bx);
    int best = -1;
    for (int k = 0; k < m.count; k++) {
      int dy, dx;
      me_candidate(&m, k, &dy, &dx);
      int c = me_cost(fp, src, ref, ds, dr, &m, dy, dx);
      if (best < 0 || c < best) {
        best = c;
        by = dy;
        bx = dx;
      }
    }
  }
  if (me_take_parent(fp, level, &m, by, bx)) {
    out[0] = m.pr;
    out[1] = m.pc;
    return;
  }
  me_finish(fp, src, ref, hp, &m, by, bx, out, scratch + me_node_scratch(level, i));
}

#endif
