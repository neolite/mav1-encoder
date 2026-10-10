/* Motion vector prediction (spec 7.10.2, single reference, no temporal
 * MVs, identity global motion) and inter_block_mode_info( ) writing. */
#include "mvpred.h"

#include <stdlib.h>
#include <string.h>

#include "enc_internal.h"

typedef struct {
  struct Mav1Encoder *e;
  int miRow, miCol, miSize, bw4, bh4;
  int numMvFound, newMvCount, foundMatch;
  int stack[MAX_REF_MV_STACK_SIZE + 2][2];
  int weight[MAX_REF_MV_STACK_SIZE + 2];
  int drlCtx[MAX_REF_MV_STACK_SIZE];
  int newMvContext, refMvContext, zeroMvContext;
} MvStack;

static const MiInfo *mi_at(const struct Mav1Encoder *e, int r, int c) { return &e->mi[r * e->mi_cols + c]; }

void mvpred_frame_start(struct Mav1Encoder *e) {
  memset(e->mi, 0, sizeof(MiInfo) * (size_t)e->mi_rows * e->mi_cols);
}

static int is_inside(const MvStack *s, int r, int c) {
  return c >= 0 && c < s->e->mi_cols && r >= 0 && r < s->e->mi_rows;
}

static void lower_precision(int *mv) {
  /* allow_high_precision_mv = 0, force_integer_mv = 0 */
  for (int i = 0; i < 2; i++)
    if (mv[i] & 1) mv[i] += mv[i] > 0 ? -1 : 1;
}

static int has_newmv(int mode) {
  return mode == NEWMV || mode == NEW_NEWMV || mode == NEAR_NEWMV || mode == NEW_NEARMV || mode == NEAREST_NEWMV ||
         mode == NEW_NEARESTMV;
}

/* 7.10.2.7 / 7.10.2.8 (RefFrames[][1] is always NONE here). */
static void add_ref_mv_candidate(MvStack *s, int r, int c, int weight) {
  const MiInfo *m = mi_at(s->e, r, c);
  if (!m->is_inter || m->ref0 != LAST_FRAME) return;
  int cand[2] = {m->mv_row, m->mv_col}; /* GLOBALMV with identity motion is a plain MV */
  lower_precision(cand);
  if (has_newmv(m->y_mode)) s->newMvCount++;
  s->foundMatch = 1;
  for (int i = 0; i < s->numMvFound; i++)
    if (s->stack[i][0] == cand[0] && s->stack[i][1] == cand[1]) {
      s->weight[i] += weight;
      return;
    }
  if (s->numMvFound < MAX_REF_MV_STACK_SIZE) {
    s->stack[s->numMvFound][0] = cand[0];
    s->stack[s->numMvFound][1] = cand[1];
    s->weight[s->numMvFound] = weight;
    s->numMvFound++;
  }
}

static void scan_row(MvStack *s, int deltaRow) {
  int bw4 = s->bw4;
  int end4 = k_min(k_min(bw4, s->e->mi_cols - s->miCol), 16);
  int deltaCol = 0, useStep16 = bw4 >= 16;
  if (k_abs(deltaRow) > 1) {
    deltaRow += s->miRow & 1;
    deltaCol = 1 - (s->miCol & 1);
  }
  for (int i = 0; i < end4;) {
    int r = s->miRow + deltaRow, c = s->miCol + deltaCol + i;
    if (!is_inside(s, r, c)) break;
    int len = k_min(bw4, Num_4x4_Blocks_Wide[mi_at(s->e, r, c)->bsize]);
    if (k_abs(deltaRow) > 1) len = k_max(2, len);
    if (useStep16) len = k_max(4, len);
    add_ref_mv_candidate(s, r, c, len * 2);
    i += len;
  }
}

static void scan_col(MvStack *s, int deltaCol) {
  int bh4 = s->bh4;
  int end4 = k_min(k_min(bh4, s->e->mi_rows - s->miRow), 16);
  int deltaRow = 0, useStep16 = bh4 >= 16;
  if (k_abs(deltaCol) > 1) {
    deltaRow = 1 - (s->miRow & 1);
    deltaCol += s->miCol & 1;
  }
  for (int i = 0; i < end4;) {
    int r = s->miRow + deltaRow + i, c = s->miCol + deltaCol;
    if (!is_inside(s, r, c)) break;
    int len = k_min(bh4, Num_4x4_Blocks_High[mi_at(s->e, r, c)->bsize]);
    if (k_abs(deltaCol) > 1) len = k_max(2, len);
    if (useStep16) len = k_max(4, len);
    add_ref_mv_candidate(s, r, c, len * 2);
    i += len;
  }
}

static void scan_point(MvStack *s, int deltaRow, int deltaCol) {
  int r = s->miRow + deltaRow, c = s->miCol + deltaCol;
  if (is_inside(s, r, c) && mi_at(s->e, r, c)->written) add_ref_mv_candidate(s, r, c, 4);
}

static void sort_stack(MvStack *s, int start, int end) {
  while (end > start) {
    int newEnd = start;
    for (int i = start + 1; i < end; i++)
      if (s->weight[i - 1] < s->weight[i]) {
        int t = s->weight[i - 1]; s->weight[i - 1] = s->weight[i]; s->weight[i] = t;
        for (int k = 0; k < 2; k++) { t = s->stack[i - 1][k]; s->stack[i - 1][k] = s->stack[i][k]; s->stack[i][k] = t; }
        newEnd = i;
      }
    end = newEnd;
  }
}

/* 7.10.2.12 / 7.10.2.13 for single prediction. */
static void extra_search(MvStack *s) {
  int w4 = k_min(k_min(16, s->bw4), s->e->mi_cols - s->miCol);
  int h4 = k_min(k_min(16, s->bh4), s->e->mi_rows - s->miRow);
  int num4x4 = k_min(w4, h4);
  for (int pass = 0; pass < 2; pass++) {
    int idx = 0;
    while (idx < num4x4 && s->numMvFound < 2) {
      int r = pass == 0 ? s->miRow - 1 : s->miRow + idx;
      int c = pass == 0 ? s->miCol + idx : s->miCol - 1;
      if (!is_inside(s, r, c)) break;
      const MiInfo *m = mi_at(s->e, r, c);
      if (m->is_inter && m->ref0 > INTRA_FRAME) { /* candList 1 is NONE */
        int cand[2] = {m->mv_row, m->mv_col}; /* equal sign bias: single LAST ref */
        int i;
        for (i = 0; i < s->numMvFound; i++)
          if (cand[0] == s->stack[i][0] && cand[1] == s->stack[i][1]) break;
        if (i == s->numMvFound) {
          s->stack[i][0] = cand[0];
          s->stack[i][1] = cand[1];
          s->weight[i] = 2;
          s->numMvFound++;
        }
      }
      idx += pass == 0 ? Num_4x4_Blocks_Wide[m->bsize] : Num_4x4_Blocks_High[m->bsize];
    }
  }
  for (int idx = s->numMvFound; idx < 2; idx++) s->stack[idx][0] = s->stack[idx][1] = 0; /* GlobalMvs[0] */
}

static void find_mv_stack(MvStack *s) {
  s->numMvFound = 0;
  s->newMvCount = 0;
  s->foundMatch = 0;
  scan_row(s, -1);
  int foundAboveMatch = s->foundMatch;
  s->foundMatch = 0;
  scan_col(s, -1);
  int foundLeftMatch = s->foundMatch;
  s->foundMatch = 0;
  if (k_max(s->bw4, s->bh4) <= 16) scan_point(s, -1, s->bw4);
  if (s->foundMatch) foundAboveMatch = 1;
  int closeMatches = foundAboveMatch + foundLeftMatch;
  int numNearest = s->numMvFound, numNew = s->newMvCount;
  for (int i = 0; i < numNearest; i++) s->weight[i] += REF_CAT_LEVEL;
  s->zeroMvContext = 0; /* use_ref_frame_mvs = 0 */
  s->foundMatch = 0;
  scan_point(s, -1, -1);
  if (s->foundMatch) foundAboveMatch = 1;
  s->foundMatch = 0;
  scan_row(s, -3);
  if (s->foundMatch) foundAboveMatch = 1;
  s->foundMatch = 0;
  scan_col(s, -3);
  if (s->foundMatch) foundLeftMatch = 1;
  s->foundMatch = 0;
  if (s->bh4 > 1) scan_row(s, -5);
  if (s->foundMatch) foundAboveMatch = 1;
  s->foundMatch = 0;
  if (s->bw4 > 1) scan_col(s, -5);
  if (s->foundMatch) foundLeftMatch = 1;
  int totalMatches = foundAboveMatch + foundLeftMatch;
  sort_stack(s, 0, numNearest);
  sort_stack(s, numNearest, s->numMvFound);
  if (s->numMvFound < 2) extra_search(s);

  /* 7.10.2.14 context and clamping */
  for (int i = 0; i < s->numMvFound; i++) {
    int z = 0;
    if (i + 1 < s->numMvFound) {
      int w0 = s->weight[i], w1 = s->weight[i + 1];
      if (w0 >= REF_CAT_LEVEL) {
        if (w1 < REF_CAT_LEVEL) z = 1;
      } else {
        z = 2;
      }
    }
    s->drlCtx[i] = z;
  }
  int bw = s->bw4 * 4, bh = s->bh4 * 4;
  int top = -(s->miRow * MI_SIZE * 8), bottom = (s->e->mi_rows - s->bh4 - s->miRow) * MI_SIZE * 8;
  int left = -(s->miCol * MI_SIZE * 8), right = (s->e->mi_cols - s->bw4 - s->miCol) * MI_SIZE * 8;
  for (int i = 0; i < s->numMvFound; i++) {
    s->stack[i][0] = Clip3(top - (MV_BORDER + bh * 8), bottom + MV_BORDER + bh * 8, s->stack[i][0]);
    s->stack[i][1] = Clip3(left - (MV_BORDER + bw * 8), right + MV_BORDER + bw * 8, s->stack[i][1]);
  }
  if (closeMatches == 0) {
    s->newMvContext = k_min(totalMatches, 1);
    s->refMvContext = totalMatches;
  } else if (closeMatches == 1) {
    s->newMvContext = 3 - k_min(numNew, 1);
    s->refMvContext = 2 + totalMatches;
  } else {
    s->newMvContext = 5 - k_min(numNew, 1);
    s->refMvContext = 5;
  }
}

/* Writes one MV component difference (allow_high_precision_mv = 0). */
static void write_mv_component(EntEnc *ec, CdfContext *cdf, int comp, int v) {
  int sign = v < 0, mag = sign ? -v : v;
  int off = mag - 1; /* hp bit is implicitly 1 */
  ec_symbol(ec, sign, cdf->mv_sign[comp], 2);
  if (off < CLASS0_SIZE << 3) {
    ec_symbol(ec, 0, cdf->mv_class[comp], MV_CLASSES);
    int bit = off >> 3, fr = (off >> 1) & 3;
    ec_symbol(ec, bit, cdf->mv_class0_bit[comp], 2);
    ec_symbol(ec, fr, cdf->mv_class0_fr[comp][bit], MV_JOINTS);
  } else {
    int cls = FloorLog2(off >> 3);
    int rem = off - (CLASS0_SIZE << (cls + 2));
    int d = rem >> 3, fr = (rem >> 1) & 3;
    ec_symbol(ec, cls, cdf->mv_class[comp], MV_CLASSES);
    for (int i = 0; i < cls; i++) ec_symbol(ec, (d >> i) & 1, cdf->mv_bit[comp][i], 2);
    ec_symbol(ec, fr, cdf->mv_fr[comp], MV_JOINTS);
  }
}

static void write_mv(EntEnc *ec, CdfContext *cdf, int dr, int dc) {
  int joint = (dr != 0) * 2 + (dc != 0); /* MV_JOINT_ZERO, HNZVZ=1, HZVNZ=2, HNZVNZ=3 */
  ec_symbol(ec, joint, cdf->mv_joint, MV_JOINTS);
  if (dr) write_mv_component(ec, cdf, 0, dr);
  if (dc) write_mv_component(ec, cdf, 1, dc);
}

int mvpred_valid_newmv(int mvRow, int mvCol) {
  return !(mvRow & 1) && !(mvCol & 1) && k_abs(mvRow) < (1 << 14) && k_abs(mvCol) < (1 << 14);
}

int write_inter_block(EntEnc *ec, CdfContext *cdf, struct Mav1Encoder *e, const BlockInfo *b) {
  MvStack s;
  memset(&s, 0, sizeof(s));
  s.e = e;
  s.miRow = b->mi_row;
  s.miCol = b->mi_col;
  s.miSize = b->bsize;
  s.bw4 = Num_4x4_Blocks_Wide[b->bsize];
  s.bh4 = Num_4x4_Blocks_High[b->bsize];
  int r = b->mi_row, c = b->mi_col;

  /* read_ref_frames: single_ref_p1 = 0, p3 = 0, p4 = 0 -> LAST_FRAME.
   * All three contexts reduce to ref_count_ctx(lastCount, 0). */
  int lastCount = 0;
  if (r > 0) lastCount += mi_at(e, r - 1, c)->ref0 == LAST_FRAME;
  if (c > 0) lastCount += mi_at(e, r, c - 1)->ref0 == LAST_FRAME;
  int rctx = lastCount == 0 ? 1 : 2;
  ec_symbol(ec, 0, cdf->single_ref[rctx][0], 2);
  ec_symbol(ec, 0, cdf->single_ref[rctx][2], 2);
  ec_symbol(ec, 0, cdf->single_ref[rctx][3], 2);

  find_mv_stack(&s);

  int mr = b->mv_row, mc = b->mv_col;
  int mode = NEWMV, refMvIdx = 0;
  if (mr == s.stack[0][0] && mc == s.stack[0][1]) {
    mode = NEARESTMV;
  } else if (mr == 0 && mc == 0) {
    mode = GLOBALMV;
  } else {
    /* NEARMV with RefMvIdx = 1..3 if any candidate matches. */
    for (int idx = 1; idx < 4; idx++) {
      if (idx > 1 && s.numMvFound <= idx) break;
      if (idx == 1 || idx < s.numMvFound) {
        if (mr == s.stack[idx][0] && mc == s.stack[idx][1]) {
          mode = NEARMV;
          refMvIdx = idx;
          break;
        }
      }
    }
  }
  if (mode == NEWMV && !mvpred_valid_newmv(mr, mc)) abort(); /* analysis only emits even MVs */

  if (mode == NEWMV) {
    ec_symbol(ec, 0, cdf->new_mv[s.newMvContext], 2);
  } else {
    ec_symbol(ec, 1, cdf->new_mv[s.newMvContext], 2);
    if (mode == GLOBALMV) {
      ec_symbol(ec, 0, cdf->zero_mv[s.zeroMvContext], 2);
    } else {
      ec_symbol(ec, 1, cdf->zero_mv[s.zeroMvContext], 2);
      ec_symbol(ec, mode == NEARMV, cdf->ref_mv[s.refMvContext], 2);
    }
  }
  /* drl: NEWMV uses RefMvIdx 0 (first drl_mode = 0 when it is coded). */
  if (mode == NEWMV) {
    if (s.numMvFound > 1) ec_symbol(ec, 0, cdf->drl_mode[s.drlCtx[0]], 2);
  } else if (mode == NEARMV) {
    for (int idx = 1; idx < 3; idx++) {
      if (s.numMvFound > idx + 1) {
        int bit = refMvIdx != idx;
        ec_symbol(ec, bit, cdf->drl_mode[s.drlCtx[idx]], 2);
        if (!bit) break;
      }
    }
  }
  if (mode == NEWMV) write_mv(ec, cdf, mr - s.stack[0][0], mc - s.stack[0][1]);
  /* read_interintra_mode, read_motion_mode, read_compound_type and the
   * interpolation filter carry no syntax with the frame/sequence settings. */
  return mode;
}
