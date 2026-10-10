/* Motion vector prediction (spec 7.10.2, single reference, no temporal
 * MVs, identity global motion) and inter_block_mode_info( ) writing. */
#include "mvpred.h"

#include <stdlib.h>
#include <string.h>

#include "enc_internal.h"

typedef struct {
  struct Mav1Encoder *e;
  int miRow, miCol, miSize, bw4, bh4;
  int refFrame[2]; /* RefFrame[ 0 ], RefFrame[ 1 ] (0 for single prediction) */
  int isCompound;
  int numMvFound, newMvCount, foundMatch;
  int stack[MAX_REF_MV_STACK_SIZE + 2][2][2]; /* RefStackMv[ idx ][ list ][ comp ] */
  int weight[MAX_REF_MV_STACK_SIZE + 2];
  int drlCtx[MAX_REF_MV_STACK_SIZE];
  int newMvContext, refMvContext, zeroMvContext;
} MvStack;

static const MiInfo *mi_at(const struct Mav1Encoder *e, int r, int c) { return &e->mi[r * e->mi_cols + c]; }

void mvpred_frame_start(struct Mav1Encoder *e) {
  memset(e->mi, 0, sizeof(MiInfo) * (size_t)e->mi_rows * e->mi_cols);
}

static int is_inside(const MvStack *s, int r, int c) {
  const FrameParams *fp = ENC_PARAMS(s->e);
  int cs = tile_col_start(fp, s->miCol);
  return c >= cs && c < tile_col_end(fp, s->miCol) && r >= 0 && r < s->e->mi_rows;
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

static void mi_mv(const MiInfo *m, int list, int *mv) {
  mv[0] = list ? m->mv1_row : m->mv_row;
  mv[1] = list ? m->mv1_col : m->mv_col;
}

static int mi_ref(const MiInfo *m, int list) { return list ? m->ref1 : m->ref0; }

/* 7.10.2.7 - 7.10.2.9 (GLOBALMV with identity motion is a plain MV). */
static void add_ref_mv_candidate(MvStack *s, int r, int c, int weight) {
  const MiInfo *m = mi_at(s->e, r, c);
  if (!m->is_inter) return;
  if (!s->isCompound) {
    for (int candList = 0; candList < 2; candList++) {
      if (mi_ref(m, candList) != s->refFrame[0]) continue;
      int cand[2];
      mi_mv(m, candList, cand);
      lower_precision(cand);
      if (has_newmv(m->y_mode)) s->newMvCount++;
      s->foundMatch = 1;
      int i;
      for (i = 0; i < s->numMvFound; i++)
        if (s->stack[i][0][0] == cand[0] && s->stack[i][0][1] == cand[1]) break;
      if (i < s->numMvFound) {
        s->weight[i] += weight;
      } else if (s->numMvFound < MAX_REF_MV_STACK_SIZE) {
        s->stack[i][0][0] = cand[0];
        s->stack[i][0][1] = cand[1];
        s->weight[i] = weight;
        s->numMvFound++;
      }
    }
    return;
  }
  if (m->ref0 != s->refFrame[0] || m->ref1 != s->refFrame[1]) return;
  int cand[2][2];
  mi_mv(m, 0, cand[0]);
  mi_mv(m, 1, cand[1]);
  lower_precision(cand[0]);
  lower_precision(cand[1]);
  s->foundMatch = 1;
  int i;
  for (i = 0; i < s->numMvFound; i++)
    if (s->stack[i][0][0] == cand[0][0] && s->stack[i][0][1] == cand[0][1] && s->stack[i][1][0] == cand[1][0] &&
        s->stack[i][1][1] == cand[1][1])
      break;
  if (i < s->numMvFound) {
    s->weight[i] += weight;
  } else if (s->numMvFound < MAX_REF_MV_STACK_SIZE) {
    memcpy(s->stack[i], cand, sizeof cand);
    s->weight[i] = weight;
    s->numMvFound++;
  }
  if (has_newmv(m->y_mode)) s->newMvCount++;
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
        int tmp[2][2];
        memcpy(tmp, s->stack[i - 1], sizeof tmp);
        memcpy(s->stack[i - 1], s->stack[i], sizeof tmp);
        memcpy(s->stack[i], tmp, sizeof tmp);
        newEnd = i;
      }
    end = newEnd;
  }
}

/* 7.10.2.12 / 7.10.2.13 (GlobalMvs are zero). */
static void extra_search(MvStack *s) {
  struct Mav1Encoder *e = s->e;
  int idMvs[2][2][2], diffMvs[2][2][2], idCount[2] = {0, 0}, diffCount[2] = {0, 0};
  int w4 = k_min(k_min(16, s->bw4), e->mi_cols - s->miCol);
  int h4 = k_min(k_min(16, s->bh4), e->mi_rows - s->miRow);
  int num4x4 = k_min(w4, h4);
  for (int pass = 0; pass < 2; pass++) {
    int idx = 0;
    while (idx < num4x4 && s->numMvFound < 2) {
      int r = pass == 0 ? s->miRow - 1 : s->miRow + idx;
      int c = pass == 0 ? s->miCol + idx : s->miCol - 1;
      if (!is_inside(s, r, c)) break;
      const MiInfo *m = mi_at(e, r, c);
      for (int candList = 0; candList < 2 && m->is_inter; candList++) {
        int candRef = mi_ref(m, candList);
        if (candRef <= INTRA_FRAME) continue;
        if (s->isCompound) {
          for (int list = 0; list < 2; list++) {
            int cand[2];
            mi_mv(m, candList, cand);
            if (candRef == s->refFrame[list] && idCount[list] < 2) {
              idMvs[list][idCount[list]][0] = cand[0];
              idMvs[list][idCount[list]][1] = cand[1];
              idCount[list]++;
            } else if (diffCount[list] < 2) {
              if (e->sign_bias[candRef] != e->sign_bias[s->refFrame[list]]) { cand[0] *= -1; cand[1] *= -1; }
              diffMvs[list][diffCount[list]][0] = cand[0];
              diffMvs[list][diffCount[list]][1] = cand[1];
              diffCount[list]++;
            }
          }
        } else {
          int cand[2];
          mi_mv(m, candList, cand);
          if (e->sign_bias[candRef] != e->sign_bias[s->refFrame[0]]) { cand[0] *= -1; cand[1] *= -1; }
          int i;
          for (i = 0; i < s->numMvFound; i++)
            if (cand[0] == s->stack[i][0][0] && cand[1] == s->stack[i][0][1]) break;
          if (i == s->numMvFound) {
            s->stack[i][0][0] = cand[0];
            s->stack[i][0][1] = cand[1];
            s->weight[i] = 2;
            s->numMvFound++;
          }
        }
      }
      idx += pass == 0 ? Num_4x4_Blocks_Wide[m->bsize] : Num_4x4_Blocks_High[m->bsize];
    }
  }
  if (s->isCompound) {
    int comb[2][2][2]; /* [idx][list][comp] */
    for (int list = 0; list < 2; list++) {
      int n = 0;
      for (int i = 0; i < idCount[list]; i++, n++) { comb[n][list][0] = idMvs[list][i][0]; comb[n][list][1] = idMvs[list][i][1]; }
      for (int i = 0; i < diffCount[list] && n < 2; i++, n++) {
        comb[n][list][0] = diffMvs[list][i][0];
        comb[n][list][1] = diffMvs[list][i][1];
      }
      for (; n < 2; n++) comb[n][list][0] = comb[n][list][1] = 0;
    }
    if (s->numMvFound == 1) {
      int k = (comb[0][0][0] == s->stack[0][0][0] && comb[0][0][1] == s->stack[0][0][1] &&
               comb[0][1][0] == s->stack[0][1][0] && comb[0][1][1] == s->stack[0][1][1]) ? 1 : 0;
      memcpy(s->stack[1], comb[k], sizeof comb[k]);
      s->weight[1] = 2;
      s->numMvFound++;
    } else {
      for (int k = 0; k < 2; k++) {
        memcpy(s->stack[s->numMvFound], comb[k], sizeof comb[k]);
        s->weight[s->numMvFound] = 2;
        s->numMvFound++;
      }
    }
  } else {
    for (int idx = s->numMvFound; idx < 2; idx++) s->stack[idx][0][0] = s->stack[idx][0][1] = 0; /* GlobalMvs[0] */
  }
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
  for (int list = 0; list < 1 + s->isCompound; list++)
    for (int i = 0; i < s->numMvFound; i++) {
      s->stack[i][list][0] = Clip3(top - (MV_BORDER + bh * 8), bottom + MV_BORDER + bh * 8, s->stack[i][list][0]);
      s->stack[i][list][1] = Clip3(left - (MV_BORDER + bw * 8), right + MV_BORDER + bw * 8, s->stack[i][list][1]);
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

static int ref_count_ctx(int c0, int c1) { return c0 < c1 ? 0 : c0 == c1 ? 1 : 2; }

/* count_refs( ) of the CDF selection process (9.3) */
static int count_refs(struct Mav1Encoder *e, int r, int c, int ft) {
  int n = 0;
  if (r > 0) n += (mi_at(e, r - 1, c)->ref0 == ft) + (mi_at(e, r - 1, c)->ref1 == ft);
  if (c > tile_col_start(ENC_PARAMS(e), c)) n += (mi_at(e, r, c - 1)->ref0 == ft) + (mi_at(e, r, c - 1)->ref1 == ft);
  return n;
}

/* single_ref_p1..p5 for LAST_FRAME, GOLDEN_FRAME or ALTREF_FRAME. */
static void write_single_ref(EntEnc *ec, CdfContext *cdf, struct Mav1Encoder *e, int r, int c, int ref) {
  int cnt[ALTREF_FRAME + 1];
  for (int f = LAST_FRAME; f <= ALTREF_FRAME; f++) cnt[f] = count_refs(e, r, c, f);
  int fwd = cnt[LAST_FRAME] + cnt[LAST2_FRAME] + cnt[LAST3_FRAME] + cnt[GOLDEN_FRAME];
  int bwd = cnt[BWDREF_FRAME] + cnt[ALTREF2_FRAME] + cnt[ALTREF_FRAME];
  int p1 = ref >= BWDREF_FRAME;
  ec_symbol(ec, p1, cdf->single_ref[ref_count_ctx(fwd, bwd)][0], 2);
  if (p1) {
    if (ref != ALTREF_FRAME) abort();
    ec_symbol(ec, 1, cdf->single_ref[ref_count_ctx(cnt[BWDREF_FRAME] + cnt[ALTREF2_FRAME], cnt[ALTREF_FRAME])][1], 2);
    return;
  }
  if (ref != LAST_FRAME && ref != GOLDEN_FRAME) abort();
  int p3 = ref == GOLDEN_FRAME;
  ec_symbol(ec, p3, cdf->single_ref[ref_count_ctx(cnt[LAST_FRAME] + cnt[LAST2_FRAME], cnt[LAST3_FRAME] + cnt[GOLDEN_FRAME])][2], 2);
  if (p3) ec_symbol(ec, 1, cdf->single_ref[ref_count_ctx(cnt[LAST3_FRAME], cnt[GOLDEN_FRAME])][4], 2);
  else ec_symbol(ec, 0, cdf->single_ref[ref_count_ctx(cnt[LAST_FRAME], cnt[LAST2_FRAME])][3], 2);
}

/* AboveRefFrame / LeftRefFrame helpers for the compound contexts. */
typedef struct { int avail, ref0, ref1; } Nb;
static int nb_intra(Nb n) { return n.ref0 <= INTRA_FRAME; }
static int nb_single(Nb n) { return n.ref1 <= INTRA_FRAME; }
static int check_backward(int r) { return r >= BWDREF_FRAME && r <= ALTREF_FRAME; }
static int samedir(int a, int b) { return (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME); }

static int comp_mode_ctx(Nb a, Nb l) {
  if (a.avail && l.avail) {
    if (nb_single(a) && nb_single(l)) return check_backward(a.ref0) ^ check_backward(l.ref0);
    if (nb_single(a)) return 2 + (check_backward(a.ref0) || nb_intra(a));
    if (nb_single(l)) return 2 + (check_backward(l.ref0) || nb_intra(l));
    return 4;
  }
  if (a.avail) return nb_single(a) ? check_backward(a.ref0) : 3;
  if (l.avail) return nb_single(l) ? check_backward(l.ref0) : 3;
  return 1;
}

static int comp_ref_type_ctx(Nb a, Nb l) {
  int aboveComp = a.avail && !nb_intra(a) && !nb_single(a), leftComp = l.avail && !nb_intra(l) && !nb_single(l);
  int aboveUni = aboveComp && samedir(a.ref0, a.ref1), leftUni = leftComp && samedir(l.ref0, l.ref1);
  if (a.avail && !nb_intra(a) && l.avail && !nb_intra(l)) {
    int sd = samedir(a.ref0, l.ref0);
    if (!aboveComp && !leftComp) return 1 + 2 * sd;
    if (!aboveComp) return !leftUni ? 1 : 3 + sd;
    if (!leftComp) return !aboveUni ? 1 : 3 + sd;
    if (!aboveUni && !leftUni) return 0;
    if (!aboveUni || !leftUni) return 2;
    return 3 + ((a.ref0 == BWDREF_FRAME) == (l.ref0 == BWDREF_FRAME));
  }
  if (a.avail && l.avail) {
    if (aboveComp) return 1 + 2 * aboveUni;
    if (leftComp) return 1 + 2 * leftUni;
    return 2;
  }
  if (aboveComp) return 4 * aboveUni;
  if (leftComp) return 4 * leftUni;
  return 2;
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
  s.refFrame[0] = b->ref_frame;
  s.refFrame[1] = b->ref_frame1;
  s.isCompound = b->ref_frame1 > INTRA_FRAME;
  int r = b->mi_row, c = b->mi_col;

  /* read_ref_frames( ) */
  if (e->hdr.reference_select) {
    Nb a = {r > 0, 0, 0}, l = {c > tile_col_start(ENC_PARAMS(e), c), 0, 0};
    if (a.avail) { a.ref0 = mi_at(e, r - 1, c)->ref0; a.ref1 = mi_at(e, r - 1, c)->ref1; }
    if (l.avail) { l.ref0 = mi_at(e, r, c - 1)->ref0; l.ref1 = mi_at(e, r, c - 1)->ref1; }
    ec_symbol(ec, s.isCompound, cdf->comp_mode[comp_mode_ctx(a, l)], 2);
    if (s.isCompound) {
      /* BIDIR_COMP_REFERENCE, LAST_FRAME + ALTREF_FRAME */
      if (b->ref_frame != LAST_FRAME || b->ref_frame1 != ALTREF_FRAME) abort();
      ec_symbol(ec, 1, cdf->comp_ref_type[comp_ref_type_ctx(a, l)], 2);
      int cnt[ALTREF_FRAME + 1];
      for (int f = LAST_FRAME; f <= ALTREF_FRAME; f++) cnt[f] = count_refs(e, r, c, f);
      ec_symbol(ec, 0, cdf->comp_ref[ref_count_ctx(cnt[LAST_FRAME] + cnt[LAST2_FRAME], cnt[LAST3_FRAME] + cnt[GOLDEN_FRAME])][0], 2);
      ec_symbol(ec, 0, cdf->comp_ref[ref_count_ctx(cnt[LAST_FRAME], cnt[LAST2_FRAME])][1], 2);
      ec_symbol(ec, 1, cdf->comp_bwd_ref[ref_count_ctx(cnt[BWDREF_FRAME] + cnt[ALTREF2_FRAME], cnt[ALTREF_FRAME])][0], 2);
    }
  }
  if (!s.isCompound) write_single_ref(ec, cdf, e, r, c, b->ref_frame);

  find_mv_stack(&s);

  int mr = b->mv_row, mc = b->mv_col;
  if (s.isCompound) {
    int mr1 = b->mv1_row, mc1 = b->mv1_col;
    int mode = NEW_NEWMV;
    if (mr == s.stack[0][0][0] && mc == s.stack[0][0][1] && mr1 == s.stack[0][1][0] && mc1 == s.stack[0][1][1])
      mode = NEAREST_NEARESTMV;
    else if (mr == 0 && mc == 0 && mr1 == 0 && mc1 == 0)
      mode = GLOBAL_GLOBALMV;
    if (mode == NEW_NEWMV && (!mvpred_valid_newmv(mr, mc) || !mvpred_valid_newmv(mr1, mc1))) abort();
    int ctx = Compound_Mode_Ctx_Map[s.refMvContext >> 1][k_min(s.newMvContext, COMP_NEWMV_CTXS - 1)];
    ec_symbol(ec, mode - NEAREST_NEARESTMV, cdf->compound_mode[ctx], COMPOUND_MODES);
    if (mode == NEW_NEWMV) {
      /* RefMvIdx 0: the first drl_mode (NumMvFound >= 2 for compound) is 0 */
      if (s.numMvFound > 1) ec_symbol(ec, 0, cdf->drl_mode[s.drlCtx[0]], 2);
      write_mv(ec, cdf, mr - s.stack[0][0][0], mc - s.stack[0][0][1]);
      write_mv(ec, cdf, mr1 - s.stack[0][1][0], mc1 - s.stack[0][1][1]);
    }
    return mode;
  }

  int mode = NEWMV, refMvIdx = 0;
  if (mr == s.stack[0][0][0] && mc == s.stack[0][0][1]) {
    mode = NEARESTMV;
  } else if (mr == 0 && mc == 0) {
    mode = GLOBALMV;
  } else {
    /* NEARMV with RefMvIdx = 1..3 if any candidate matches. */
    for (int idx = 1; idx < 4; idx++) {
      if (idx > 1 && s.numMvFound <= idx) break;
      if (idx == 1 || idx < s.numMvFound) {
        if (mr == s.stack[idx][0][0] && mc == s.stack[idx][0][1]) {
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
  if (mode == NEWMV) write_mv(ec, cdf, mr - s.stack[0][0][0], mc - s.stack[0][0][1]);
  /* read_interintra_mode, read_motion_mode, read_compound_type and the
   * interpolation filter carry no syntax with the frame/sequence settings. */
  return mode;
}
