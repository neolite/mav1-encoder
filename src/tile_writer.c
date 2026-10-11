/*
 * Entropy coding of one tile (the whole frame). This mirrors the decoding
 * process of spec sections 5.11 (syntax) and 8.3 (CDF selection), walking the
 * blocks chosen by the analysis/reconstruction kernels in decode order.
 */
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

#include "enc_internal.h"
#include "kernels/k_intra.h"
#include "kernels/k_scan.h"
#include "mvpred.h"
#include "kernels/k_analysis.h"
#include "kernels/k_recon.h"
#include "kernels/k_lr.h"

typedef struct {
  struct Mav1Encoder *e;
  EntEnc ec;
  CdfContext cdf;
  uint8_t *aboveLevel[3], *aboveDc[3], *leftLevel[3], *leftDc[3];
  int qctx;
  int refLrWiener[3][2][3], refSgrXqd[3][2];
} Tile;

static MiInfo *mi_at(const struct Mav1Encoder *e, int r, int c) { return &e->mi[r * e->mi_cols + c]; }

/* ------------------------------------------------------------------------ */
/* Coefficients (5.11.39)                                                    */

static int tx_class(int t) {
  if (t == V_DCT || t == V_ADST || t == V_FLIPADST) return TX_CLASS_VERT;
  if (t == H_DCT || t == H_ADST || t == H_FLIPADST) return TX_CLASS_HORIZ;
  return TX_CLASS_2D;
}

static int coeff_base_eob_ctx(int txSz, int c) {
  int adj = Adjusted_Tx_Size[txSz];
  int area = Tx_Height[adj] << Tx_Width_Log2[adj];
  if (c == 0) return 0;
  if (c <= area / 8) return 1;
  if (c <= area / 4) return 2;
  return 3;
}

static int inv_index(KCONST int32_t *tab, int n, int v) {
  for (int i = 0; i < n; i++)
    if (tab[i] == v) return i;
  abort();
}

static void write_tx_type(Tile *t, int txSz, int isInter, int yMode, int txType) {
  int set = get_tx_set(txSz, isInter, 0);
  if (set <= 0 || t->e->hdr.base_q_idx <= 0) return;
  int sqr = Tx_Size_Sqr[txSz];
  if (isInter) {
    if (set == TX_SET_INTER_1)
      ec_symbol(&t->ec, inv_index(Tx_Type_Inter_Inv_Set1, 16, txType), t->cdf.inter_tx_set1[sqr], 16);
    else if (set == TX_SET_INTER_2)
      ec_symbol(&t->ec, inv_index(Tx_Type_Inter_Inv_Set2, 12, txType), t->cdf.inter_tx_set2, 12);
    else
      ec_symbol(&t->ec, inv_index(Tx_Type_Inter_Inv_Set3, 2, txType), t->cdf.inter_tx_set3[sqr], 2);
  } else {
    if (set == TX_SET_INTRA_1)
      ec_symbol(&t->ec, inv_index(Tx_Type_Intra_Inv_Set1, 7, txType), t->cdf.intra_tx_set1[sqr][yMode], 7);
    else
      ec_symbol(&t->ec, inv_index(Tx_Type_Intra_Inv_Set2, 5, txType), t->cdf.intra_tx_set2[sqr][yMode], 5);
  }
}

static void write_golomb(EntEnc *ec, int x) {
  /* x >= 1: (length-1) zero bits, a one bit, then the low length-1 bits. */
  int length = 0;
  while ((x >> length) > 1) length++;
  length++;
  for (int i = 0; i < length - 1; i++) ec_literal(ec, 1, 0);
  ec_literal(ec, 1, 1);
  for (int i = length - 2; i >= 0; i--) ec_literal(ec, 1, (x >> i) & 1);
}

/*
 * coeffs( plane, startX, startY, txSz ) with the levels in `q` (layout
 * i * tw + j) and the plane transform type.
 */
static void write_coeffs(Tile *t, int plane, int startX, int startY, int txSz, int bsize, const int32_t *q,
                         int eob, int planeTxType, int isInter, int yMode) {
  struct Mav1Encoder *e = t->e;
  int x4 = startX >> 2, y4 = startY >> 2;
  int w4 = Tx_Width[txSz] >> 2, h4 = Tx_Height[txSz] >> 2;
  int txSzCtx = (Tx_Size_Sqr[txSz] + Tx_Size_Sqr_Up[txSz] + 1) >> 1;
  int ptype = plane > 0;
  int maxX4 = e->mi_cols, maxY4 = e->mi_rows;
  if (plane > 0) { maxX4 >>= 1; maxY4 >>= 1; }

  /* all_zero context */
  int w = Tx_Width[txSz], h = Tx_Height[txSz];
  int psz = Subsampled_Size[bsize][plane > 0][plane > 0];
  int bw = Block_Width(psz), bh = Block_Height(psz);
  int ctx;
  if (plane == 0) {
    int top = 0, left = 0;
    for (int k = 0; k < w4; k++)
      if (x4 + k < maxX4 && t->aboveLevel[plane][x4 + k] > top) top = t->aboveLevel[plane][x4 + k];
    for (int k = 0; k < h4; k++)
      if (y4 + k < maxY4 && t->leftLevel[plane][y4 + k] > left) left = t->leftLevel[plane][y4 + k];
    int mx = top > left ? top : left, mn = top < left ? top : left;
    if (bw == w && bh == h) ctx = 0;
    else if (top == 0 && left == 0) ctx = 1;
    else if (top == 0 || left == 0) ctx = 2 + (mx > 3);
    else if (mx <= 3) ctx = 4;
    else if (mn <= 3) ctx = 5;
    else ctx = 6;
  } else {
    int above = 0, left = 0;
    for (int i = 0; i < w4; i++)
      if (x4 + i < maxX4) above |= t->aboveLevel[plane][x4 + i] | t->aboveDc[plane][x4 + i];
    for (int i = 0; i < h4; i++)
      if (y4 + i < maxY4) left |= t->leftLevel[plane][y4 + i] | t->leftDc[plane][y4 + i];
    ctx = 7 + (above != 0) + (left != 0);
    if (bw * bh > w * h) ctx += 3;
  }
  ec_symbol(&t->ec, eob == 0, t->cdf.txb_skip[txSzCtx][ctx], 2);

  int culLevel = 0, dcCategory = 0;
  if (eob > 0) {
    if (plane == 0) write_tx_type(t, txSz, isInter, yMode, planeTxType);
    KCONST int32_t *scan = get_scan(txSz, planeTxType);
    int txClass = tx_class(planeTxType);
    int eobMultisize = (Tx_Width_Log2[txSz] < 5 ? Tx_Width_Log2[txSz] : 5) +
                       (Tx_Height_Log2[txSz] < 5 ? Tx_Height_Log2[txSz] : 5) - 4;
    int eobPt = eob <= 2 ? eob : FloorLog2(eob - 1) + 2;
    int ectx = txClass == TX_CLASS_2D ? 0 : 1;
    int sym = eobPt - 1;
    switch (eobMultisize) {
      case 0: ec_symbol(&t->ec, sym, t->cdf.eob_pt_16[ptype][ectx], 5); break;
      case 1: ec_symbol(&t->ec, sym, t->cdf.eob_pt_32[ptype][ectx], 6); break;
      case 2: ec_symbol(&t->ec, sym, t->cdf.eob_pt_64[ptype][ectx], 7); break;
      case 3: ec_symbol(&t->ec, sym, t->cdf.eob_pt_128[ptype][ectx], 8); break;
      case 4: ec_symbol(&t->ec, sym, t->cdf.eob_pt_256[ptype][ectx], 9); break;
      case 5: ec_symbol(&t->ec, sym, t->cdf.eob_pt_512[ptype], 10); break;
      default: ec_symbol(&t->ec, sym, t->cdf.eob_pt_1024[ptype], 11); break;
    }
    int eobShift = eobPt - 3;
    if (eobShift >= 0) {
      int rem = eob - ((1 << (eobPt - 2)) + 1);
      ec_symbol(&t->ec, (rem >> eobShift) & 1, t->cdf.eob_extra[txSzCtx][ptype][eobPt - 3], 2);
      for (int i = 1; i < eobPt - 2; i++) {
        int sh = eobPt - 2 - 1 - i;
        ec_literal(&t->ec, 1, (rem >> sh) & 1);
      }
    }
    /* Levels, from the last coefficient backwards. `lv` mirrors the
     * decoder's Quant[] (levels of already decoded positions) inside the
     * (at most 32 x 32) adjusted transform, with 4 zero columns and rows of
     * padding so that the context neighbours need no bounds checks. */
    int adj = Adjusted_Tx_Size[txSz];
    int bwl = Tx_Width_Log2[adj], aw = 1 << bwl, ah = Tx_Height[adj], S = aw + 4;
    int32_t lv[36 * 36];
    memset(lv, 0, sizeof(int32_t) * (size_t)S * (ah + 4));
    int sig[SIG_REF_DIFF_OFFSET_NUM], mag3[3];
    for (int i = 0; i < SIG_REF_DIFF_OFFSET_NUM; i++)
      sig[i] = Sig_Ref_Diff_Offset[txClass][i][0] * S + Sig_Ref_Diff_Offset[txClass][i][1];
    for (int i = 0; i < 3; i++)
      mag3[i] = Mag_Ref_Offset_With_Tx_Class[txClass][i][0] * S + Mag_Ref_Offset_With_Tx_Class[txClass][i][1];
    KCONST int32_t(*baseOff)[5] = Coeff_Base_Ctx_Offset[txSz];
    uint16_t(*cdfBase)[4 + 1] = t->cdf.coeff_base[txSzCtx][ptype];
    uint16_t(*cdfBr)[BR_CDF_SIZE + 1] = t->cdf.coeff_br[txSzCtx < TX_32X32 ? txSzCtx : TX_32X32][ptype];
    for (int c = eob - 1; c >= 0; c--) {
      int pos = scan[c];
      int level = abs(q[pos]);
      int row = pos >> bwl, col = pos & (aw - 1);
      int32_t *l = lv + row * S + col;
      int coded = level < 3 ? level : 3;
      if (c == eob - 1) {
        ec_symbol(&t->ec, coded - 1, t->cdf.coeff_base_eob[txSzCtx][ptype][coeff_base_eob_ctx(txSz, c)], 3);
      } else {
        /* get_coeff_base_ctx( ) */
        int mag = 0;
        for (int i = 0; i < SIG_REF_DIFF_OFFSET_NUM; i++) {
          int v = l[sig[i]];
          mag += v < 3 ? v : 3;
        }
        int ctx = (mag + 1) >> 1;
        if (ctx > 4) ctx = 4;
        if (txClass == TX_CLASS_2D) ctx = pos == 0 ? 0 : ctx + baseOff[row < 4 ? row : 4][col < 4 ? col : 4];
        else {
          int idx = txClass == TX_CLASS_VERT ? row : col;
          ctx += Coeff_Base_Pos_Ctx_Offset[idx < 2 ? idx : 2];
        }
        ec_symbol(&t->ec, coded, cdfBase[ctx], 4);
      }
      if (coded > NUM_BASE_LEVELS) {
        /* coeff_br context */
        int mag = 0;
        for (int i = 0; i < 3; i++) {
          int v = l[mag3[i]];
          mag += v < COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1 ? v : COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;
        }
        mag = (mag + 1) >> 1;
        if (mag > 6) mag = 6;
        int bctx = pos == 0 ? mag
                   : txClass == TX_CLASS_2D ? ((row < 2 && col < 2) ? mag + 7 : mag + 14)
                   : txClass == TX_CLASS_HORIZ ? (col == 0 ? mag + 7 : mag + 14)
                                               : (row == 0 ? mag + 7 : mag + 14);
        for (int idx = 0; idx < COEFF_BASE_RANGE / (BR_CDF_SIZE - 1); idx++) {
          int rem = level - coded;
          int br = rem < BR_CDF_SIZE - 1 ? rem : BR_CDF_SIZE - 1;
          ec_symbol(&t->ec, br, cdfBr[bctx], BR_CDF_SIZE);
          coded += br;
          if (br < BR_CDF_SIZE - 1) break;
        }
      }
      *l = coded;
    }
    /* Signs and Golomb remainders, in forward scan order. */
    int dctx = 0;
    {
      int dcSign = 0;
      for (int k = 0; k < w4; k++)
        if (x4 + k < maxX4) {
          int s = t->aboveDc[plane][x4 + k];
          dcSign += (s == 2) - (s == 1);
        }
      for (int k = 0; k < h4; k++)
        if (y4 + k < maxY4) {
          int s = t->leftDc[plane][y4 + k];
          dcSign += (s == 2) - (s == 1);
        }
      dctx = dcSign < 0 ? 1 : dcSign > 0 ? 2 : 0;
    }
    for (int c = 0; c < eob; c++) {
      int pos = scan[c];
      int v = q[pos];
      int level = abs(v);
      if (level != 0) {
        int sign = v < 0;
        if (c == 0) ec_symbol(&t->ec, sign, t->cdf.dc_sign[ptype][dctx], 2);
        else ec_literal(&t->ec, 1, sign);
        if (level > NUM_BASE_LEVELS + COEFF_BASE_RANGE)
          write_golomb(&t->ec, level - (NUM_BASE_LEVELS + COEFF_BASE_RANGE));
        if (pos == 0) dcCategory = sign ? 1 : 2;
        culLevel += level & 0xFFFFF;
      }
    }
    if (culLevel > 63) culLevel = 63;
  }
  for (int i = 0; i < w4; i++) {
    t->aboveLevel[plane][x4 + i] = (uint8_t)culLevel;
    t->aboveDc[plane][x4 + i] = (uint8_t)dcCategory;
  }
  for (int i = 0; i < h4; i++) {
    t->leftLevel[plane][y4 + i] = (uint8_t)culLevel;
    t->leftDc[plane][y4 + i] = (uint8_t)dcCategory;
  }
}

/* ------------------------------------------------------------------------ */
/* Block syntax                                                              */

static void write_intra_modes(Tile *t, const BlockInfo *b, int availU, int availL, int isKey) {
  struct Mav1Encoder *e = t->e;
  int bs = b->bsize;
  if (isKey) {
    int above = Intra_Mode_Context[availU ? mi_at(e, b->mi_row - 1, b->mi_col)->y_mode : DC_PRED];
    int left = Intra_Mode_Context[availL ? mi_at(e, b->mi_row, b->mi_col - 1)->y_mode : DC_PRED];
    ec_symbol(&t->ec, b->y_mode, t->cdf.intra_frame_y_mode[above][left], INTRA_MODES);
  } else {
    ec_symbol(&t->ec, b->y_mode, t->cdf.y_mode[Size_Group[bs]], INTRA_MODES);
  }
  if (intra_is_directional(b->y_mode))
    ec_symbol(&t->ec, b->angle_y + MAX_ANGLE_DELTA, t->cdf.angle_delta[b->y_mode - V_PRED], 2 * MAX_ANGLE_DELTA + 1);
  /* HasChroma: always for blocks of at least 8x8. */
  int bw = Block_Width(bs), bh = Block_Height(bs);
  (void)bw; (void)bh;
  if (cfl_allowed(bs, e->hdr.base_q_idx == 0))
    ec_symbol(&t->ec, b->uv_mode, t->cdf.uv_mode_cfl_allowed[b->y_mode], UV_INTRA_MODES_CFL_ALLOWED);
  else
    ec_symbol(&t->ec, b->uv_mode, t->cdf.uv_mode_cfl_not_allowed[b->y_mode], UV_INTRA_MODES_CFL_NOT_ALLOWED);
  if (b->uv_mode == UV_CFL_PRED) {
    /* read_cfl_alphas: signs 0 = zero, 1 = neg, 2 = pos; joint = signU*3+signV-1 */
    int au = b->cfl_alpha_u, av = b->cfl_alpha_v;
    int signU = au == 0 ? 0 : (au < 0 ? 1 : 2), signV = av == 0 ? 0 : (av < 0 ? 1 : 2);
    ec_symbol(&t->ec, signU * 3 + signV - 1, t->cdf.cfl_sign, CFL_JOINT_SIGNS);
    if (signU) ec_symbol(&t->ec, k_abs(au) - 1, t->cdf.cfl_alpha[(signU - 1) * 3 + signV], CFL_ALPHABET_SIZE);
    if (signV) ec_symbol(&t->ec, k_abs(av) - 1, t->cdf.cfl_alpha[(signV - 1) * 3 + signU], CFL_ALPHABET_SIZE);
  }
  if (intra_is_directional(b->uv_mode))
    ec_symbol(&t->ec, b->angle_uv + MAX_ANGLE_DELTA, t->cdf.angle_delta[b->uv_mode - V_PRED], 2 * MAX_ANGLE_DELTA + 1);
}

static void write_block(Tile *t, const BlockInfo *b) {
  struct Mav1Encoder *e = t->e;
  int r = b->mi_row, c = b->mi_col, bs = b->bsize;
  int bw4 = Num_4x4_Blocks_Wide[bs], bh4 = Num_4x4_Blocks_High[bs];
  int availU = r > 0, availL = c > tile_col_start(ENC_PARAMS(e), c);
  int isKey = e->hdr.frame_type == KEY_FRAME;
  int skip = b->eob[0] == 0 && b->eob[1] == 0 && b->eob[2] == 0;

  if (!isKey) {
    /* inter_frame_mode_info: skip_mode is not present (reference_select = 0). */
  }
  int sctx = (availU ? mi_at(e, r - 1, c)->skip : 0) + (availL ? mi_at(e, r, c - 1)->skip : 0);
  ec_symbol(&t->ec, skip, t->cdf.skip[sctx], 2);
  if (e->cfg.cdef && !skip && e->hdr.base_q_idx > 0) {
    int sbi = (r >> 4) * e->sb_cols + (c >> 4);
    if (e->cdef_idx[sbi] == -1) {
      e->cdef_idx[sbi] = e->cdef_choice[sbi];
      ec_literal(&t->ec, e->hdr.cdef_bits, e->cdef_choice[sbi]);
    }
  }
  int interMode = NEWMV;
  if (isKey) {
    write_intra_modes(t, b, availU, availL, 1);
  } else {
    int ctx;
    int aboveIntra = availU ? !mi_at(e, r - 1, c)->is_inter : 0;
    int leftIntra = availL ? !mi_at(e, r, c - 1)->is_inter : 0;
    if (availU && availL) ctx = (leftIntra && aboveIntra) ? 3 : (leftIntra || aboveIntra);
    else if (availU || availL) ctx = 2 * (availU ? aboveIntra : leftIntra);
    else ctx = 0;
    ec_symbol(&t->ec, b->is_inter, t->cdf.is_inter[ctx], 2);
    if (b->is_inter) {
      interMode = write_inter_block(&t->ec, &t->cdf, e, b);
      if (e->cfg.switchable_interp) {
        /* enable_dual_filter = 1; single reference, so RefFrame[1] = NONE.
         * needs_interp_filter( ): no skip mode, SIMPLE motion; GLOBALMV with
         * GmType IDENTITY (not TRANSLATION) on blocks >= 8x8 is not coded and
         * uses EIGHTTAP (its zero MV makes the filter irrelevant). */
        int glob = interMode == GLOBALMV || interMode == GLOBAL_GLOBALMV;
        if (glob) ((BlockInfo *)b)->interp_filter &= ~15;
        for (int dir = 0; dir < 2 && !glob; dir++) {
          int ctx = ((dir & 1) * 2 + (b->ref_frame1 > INTRA_FRAME)) * 4;
          int leftType = 3, aboveType = 3;
          const MiInfo *lm = availL ? mi_at(e, r, c - 1) : NULL, *am = availU ? mi_at(e, r - 1, c) : NULL;
          if (lm && (lm->ref0 == b->ref_frame || lm->ref1 == b->ref_frame)) leftType = lm->interp[dir];
          if (am && (am->ref0 == b->ref_frame || am->ref1 == b->ref_frame)) aboveType = am->interp[dir];
          if (leftType == aboveType) ctx += leftType;
          else if (leftType == 3) ctx += aboveType;
          else if (aboveType == 3) ctx += leftType;
          else ctx += 3;
          int f = dir == 0 ? (b->interp_filter & 3) : ((b->interp_filter >> 2) & 3);
          ec_symbol(&t->ec, f, t->cdf.interp_filter[ctx], INTERP_FILTERS);
        }
      }
    }
    else write_intra_modes(t, b, availU, availL, 0);
  }

  /* read_block_tx_size( ) */
  int lossless = e->hdr.base_q_idx == 0;
  int txSize = lossless ? TX_4X4 : b->is_inter ? Max_Tx_Size_Rect[bs] : depth_tx_size(bs, b->tx_depth);
  if (e->cfg.tx_depth && !lossless) {
    if (b->is_inter && !skip) {
      /* read_var_tx_size( MiRow, MiCol, maxTxSz, 0 ): one unit (blocks <= 64),
       * never split */
      int txSz = Max_Tx_Size_Rect[bs];
      if (txSz != TX_4X4) {
        int aboveW = !availU ? 64
                     : (mi_at(e, r - 1, c)->skip && mi_at(e, r - 1, c)->is_inter)
                         ? Block_Width(mi_at(e, r - 1, c)->bsize)
                         : Tx_Width[mi_at(e, r - 1, c)->tx_size];
        int leftH = !availL ? 64
                    : (mi_at(e, r, c - 1)->skip && mi_at(e, r, c - 1)->is_inter)
                        ? Block_Height(mi_at(e, r, c - 1)->bsize)
                        : Tx_Height[mi_at(e, r, c - 1)->tx_size];
        int size = k_min(64, k_max(Block_Width(bs), Block_Height(bs)));
        int maxTxSz = FloorLog2(size) - 2; /* find_tx_size( size, size ) */
        int ctx = (Tx_Size_Sqr_Up[txSz] != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
                  (aboveW < Tx_Width[txSz]) + (leftH < Tx_Height[txSz]);
        ec_symbol(&t->ec, 0, t->cdf.txfm_split[ctx], 2);
      }
    } else if (!b->is_inter) {
      /* read_tx_size( 1 ) */
      int maxRect = Max_Tx_Size_Rect[bs];
      int maxTxDepth = Max_Tx_Depth[bs];
      int aboveW = 0, leftH = 0;
      if (availU) {
        const MiInfo *a = mi_at(e, r - 1, c);
        aboveW = a->is_inter ? Block_Width(a->bsize) : Tx_Width[a->tx_size];
      }
      if (availL) {
        const MiInfo *l = mi_at(e, r, c - 1);
        leftH = l->is_inter ? Block_Height(l->bsize) : Tx_Height[l->tx_size];
      }
      int ctx = (aboveW >= Tx_Width[maxRect]) + (leftH >= Tx_Height[maxRect]);
      if (maxTxDepth == 4) ec_symbol(&t->ec, b->tx_depth, t->cdf.tx_64x64[ctx], MAX_TX_DEPTH + 1);
      else if (maxTxDepth == 3) ec_symbol(&t->ec, b->tx_depth, t->cdf.tx_32x32[ctx], MAX_TX_DEPTH + 1);
      else if (maxTxDepth == 2) ec_symbol(&t->ec, b->tx_depth, t->cdf.tx_16x16[ctx], MAX_TX_DEPTH + 1);
      else ec_symbol(&t->ec, b->tx_depth, t->cdf.tx_8x8[ctx], MAX_TX_DEPTH);
    }
  }

  /* Update the mode-info grid (used by later context derivations). */
  for (int y = 0; y < bh4 && r + y < e->mi_rows; y++)
    for (int x = 0; x < bw4 && c + x < e->mi_cols; x++) {
      MiInfo *m = mi_at(e, r + y, c + x);
      m->bsize = (int8_t)bs;
      m->is_inter = (int8_t)b->is_inter;
      m->skip = (int8_t)skip;
      m->y_mode = (int8_t)(b->is_inter ? interMode : b->y_mode);
      m->written = 1;
      m->interp[0] = (int8_t)(b->interp_filter & 3);
      m->interp[1] = (int8_t)((b->interp_filter >> 2) & 3);
      m->ref0 = (int8_t)(b->is_inter ? b->ref_frame : INTRA_FRAME);
      m->tx_size = (int8_t)txSize; /* InterTxSizes */
      m->mv_row = (int16_t)(b->is_inter ? b->mv_row : 0);
      m->mv_col = (int16_t)(b->is_inter ? b->mv_col : 0);
      m->ref1 = (int8_t)(b->is_inter ? b->ref_frame1 : 0);
      m->mv1_row = (int16_t)(b->is_inter ? b->mv1_row : 0);
      m->mv1_col = (int16_t)(b->is_inter ? b->mv1_col : 0);
    }

  if (skip) {
    for (int plane = 0; plane < 3; plane++) {
      int sub = plane > 0;
      for (int i = c >> sub; i < ((c + bw4) >> sub); i++) t->aboveLevel[plane][i] = t->aboveDc[plane][i] = 0;
      for (int i = r >> sub; i < ((r + bh4) >> sub); i++) t->leftLevel[plane][i] = t->leftDc[plane][i] = 0;
    }
    return;
  }
  const int32_t *coefs = ENC_COEFS(e);
  for (int plane = 0; plane < 3; plane++) {
    int sub = plane > 0;
    int txSz = lossless ? TX_4X4 : (plane == 0 ? txSize : uv_tx_size(bs, 1, 1));
    int maxX = (e->mi_cols * 4) >> sub, maxY = (e->mi_rows * 4) >> sub;
    int planeSz = Subsampled_Size[bs][sub][sub];
    int tw = Tx_Width[txSz], th = Tx_Height[txSz];
    int txIdx = 0;
    for (int y = 0; y < Block_Height(planeSz); y += th)
      for (int x = 0; x < Block_Width(planeSz); x += tw) {
    int startX = (c >> sub) * 4 + x, startY = (r >> sub) * 4 + y;
    if (startX >= maxX || startY >= maxY) continue;
    const int32_t *q = coefs + b->coef_off[plane] + txIdx * tw * th;
    int planeTxType = lossless ? DCT_DCT
                      : plane == 0 ? tx_type_get(b, txIdx)
                                   : chroma_tx_type(txSz, b->is_inter, b->uv_mode, b->tx_type, 0);
    txIdx++;
    int eob = find_eob(q, get_scan(txSz, planeTxType), seg_eob(txSz));
    write_coeffs(t, plane, startX, startY, txSz, bs, q, eob, planeTxType,
                 b->is_inter, b->y_mode);
      }
  }
}

/* ------------------------------------------------------------------------ */
/* Loop restoration units (5.11.57 read_lr)                                  */

static void write_lr(Tile *t, int r, int c) {
  struct Mav1Encoder *e = t->e;
  const FrameParams *fp = ENC_PARAMS(e);
  const LrUnit *units = (const LrUnit *)e->lrunits.base;
  int w = 16, h = 16; /* Num_4x4_Blocks_Wide/High[BLOCK_64X64] */
  for (int plane = 0; plane < 3; plane++) {
    if (e->hdr.lr_type[plane] == RESTORE_NONE) continue;
    int sub = plane > 0;
    int unitSize = fp->lr_unit_size[plane];
    int unitRows = fp->lr_unit_rows[plane], unitCols = fp->lr_unit_cols[plane];
    int unitRowStart = (r * (MI_SIZE >> sub) + unitSize - 1) / unitSize;
    int unitRowEnd = k_min(unitRows, ((r + h) * (MI_SIZE >> sub) + unitSize - 1) / unitSize);
    int unitColStart = (c * (MI_SIZE >> sub) + unitSize - 1) / unitSize;
    int unitColEnd = k_min(unitCols, ((c + w) * (MI_SIZE >> sub) + unitSize - 1) / unitSize);
    for (int ur = unitRowStart; ur < unitRowEnd; ur++)
      for (int uc = unitColStart; uc < unitColEnd; uc++) {
        const LrUnit *u = &units[fp->lr_unit_off[plane] + ur * unitCols + uc];
        /* FrameRestorationType is RESTORE_SWITCHABLE: restoration_type */
        ec_symbol(&t->ec, u->type, t->cdf.restoration_type, RESTORE_SWITCHABLE);
        lr_unit_params(&t->ec, plane, u, t->refLrWiener[plane], t->refSgrXqd[plane]);
      }
  }
}

/* ------------------------------------------------------------------------ */
/* Partition (5.11.4)                                                        */

static uint16_t *partition_cdf(Tile *t, int bsl, int ctx) {
  if (bsl == 1) return t->cdf.partition_w8[ctx];
  if (bsl == 2) return t->cdf.partition_w16[ctx];
  if (bsl == 3) return t->cdf.partition_w32[ctx];
  return t->cdf.partition_w64[ctx];
}

static int pprob(const uint16_t *cdf, int p) { return cdf[p] - (p > 0 ? cdf[p - 1] : 0); }

static void write_partition(Tile *t, int r, int c, int bSize, const BlockInfo *blocks, int *k, int n) {
  struct Mav1Encoder *e = t->e;
  if (r >= e->mi_rows || c >= e->mi_cols) return;
  int num4x4 = Num_4x4_Blocks_Wide[bSize];
  int half = num4x4 >> 1;
  int hasRows = (r + half) < e->mi_rows, hasCols = (c + half) < e->mi_cols;
  int level = 4 - Mi_Width_Log2[bSize]; /* 64x64 -> 0 ... 8x8 -> 3 */
  int nl = 1 << level;
  int node = NODE_BASE(level) + ((r & 15) / num4x4) * nl + (c & 15) / num4x4;
  int p = ENC_SBS(e)[(r >> 4) * e->sb_cols + (c >> 4)].part[node];
  int bsl = Mi_Width_Log2[bSize];
  int availU = r > 0, availL = c > tile_col_start(ENC_PARAMS(e), c);
  int above = availU && Mi_Width_Log2[mi_at(e, r - 1, c)->bsize] < bsl;
  int left = availL && Mi_Height_Log2[mi_at(e, r, c - 1)->bsize] < bsl;
  int ctx = left * 2 + above;
  if (hasRows && hasCols) {
    int nsym = bsl == 1 ? 4 : 10;
    ec_symbol(&t->ec, p, partition_cdf(t, bsl, ctx), nsym);
  } else if (hasCols) {
    const uint16_t *pc = partition_cdf(t, bsl, ctx);
    int psum = pprob(pc, PARTITION_VERT) + pprob(pc, PARTITION_SPLIT) + pprob(pc, PARTITION_HORZ_A) +
               pprob(pc, PARTITION_VERT_A) + pprob(pc, PARTITION_VERT_B) + pprob(pc, PARTITION_VERT_4);
    uint16_t tmp[3] = {(uint16_t)((1 << 15) - psum), 1 << 15, 0};
    if (p != PARTITION_SPLIT && p != PARTITION_HORZ) abort();
    ec_symbol(&t->ec, p == PARTITION_SPLIT, tmp, 2);
  } else if (hasRows) {
    const uint16_t *pc = partition_cdf(t, bsl, ctx);
    int psum = pprob(pc, PARTITION_HORZ) + pprob(pc, PARTITION_SPLIT) + pprob(pc, PARTITION_HORZ_A) +
               pprob(pc, PARTITION_HORZ_B) + pprob(pc, PARTITION_VERT_A) + pprob(pc, PARTITION_HORZ_4);
    uint16_t tmp[3] = {(uint16_t)((1 << 15) - psum), 1 << 15, 0};
    if (p != PARTITION_SPLIT && p != PARTITION_VERT) abort();
    ec_symbol(&t->ec, p == PARTITION_SPLIT, tmp, 2);
  } else if (p != PARTITION_SPLIT) {
    abort();
  }
  if (p == PARTITION_SPLIT) {
    int sub = Partition_Subsize[PARTITION_SPLIT][bSize];
    write_partition(t, r, c, sub, blocks, k, n);
    write_partition(t, r, c + half, sub, blocks, k, n);
    write_partition(t, r + half, c, sub, blocks, k, n);
    write_partition(t, r + half, c + half, sub, blocks, k, n);
    return;
  }
  int nb = part_count(ENC_PARAMS(e), p, r, c, half);
  for (int i = 0; i < nb; i++) {
    int dr, dc;
    int bs = part_block(p, bSize, i, half, &dr, &dc);
    if (*k >= n || blocks[*k].mi_row != r + dr || blocks[*k].mi_col != c + dc || blocks[*k].bsize != bs) abort();
    write_block(t, &blocks[*k]);
    (*k)++;
  }
}

/* Codes tile column tc (decode_tile( ) for MiColStart..MiColEnd, all rows). */
static size_t write_one_tile(struct Mav1Encoder *e, int tc, uint8_t **out) {
  Tile t;
  memset(&t, 0, sizeof(t));
  t.e = e;
  ec_init(&t.ec);
  cdf_init(&t.cdf, e->hdr.base_q_idx);
  int aw = e->mi_cols + 32, lh = e->mi_rows + 32;
  for (int p = 0; p < 3; p++) {
    t.aboveLevel[p] = calloc(aw, 1);
    t.aboveDc[p] = calloc(aw, 1);
    t.leftLevel[p] = calloc(lh, 1);
    t.leftDc[p] = calloc(lh, 1);
  }
  for (int p = 0; p < 3; p++)
    for (int pass = 0; pass < 2; pass++) {
      t.refSgrXqd[p][pass] = Sgrproj_Xqd_Mid[pass];
      for (int i = 0; i < WIENER_COEFFS; i++) t.refLrWiener[p][pass][i] = Wiener_Taps_Mid[i];
    }

  const BlockInfo *blocks = ENC_BLOCKS(e);
  const SbInfo *sbs = ENC_SBS(e);
  int sc0 = tc * e->tile_w_sb, sc1 = sc0 + e->tile_w_sb < e->sb_cols ? sc0 + e->tile_w_sb : e->sb_cols;
  for (int sr = 0; sr < e->sb_rows; sr++) {
    for (int p = 0; p < 3; p++) {
      memset(t.leftLevel[p], 0, lh);
      memset(t.leftDc[p], 0, lh);
    }
    for (int sc = sc0; sc < sc1; sc++) {
      const SbInfo *sb = &sbs[sr * e->sb_cols + sc];
      int k = 0;
      write_lr(&t, sr * 16, sc * 16);
      write_partition(&t, sr * 16, sc * 16, BLOCK_64X64, blocks + sb->first_block, &k, sb->num_blocks);
      if (k != sb->num_blocks) abort();
    }
  }
  size_t n = ec_finish(&t.ec, out);
  ec_free(&t.ec);
  for (int p = 0; p < 3; p++) {
    free(t.aboveLevel[p]);
    free(t.aboveDc[p]);
    free(t.leftLevel[p]);
    free(t.leftDc[p]);
  }
  return n;
}

/* All tiles of the frame; every tile but the last is preceded by
 * tile_size_minus_1 (le(4)). Tiles touch disjoint columns of e->mi and are
 * coded concurrently. */
size_t write_tile(struct Mav1Encoder *e, uint8_t **out) {
  memset(e->mi, 0, sizeof(MiInfo) * e->mi_rows * e->mi_cols);
  for (int i = 0; i < e->num_sbs; i++) e->cdef_idx[i] = -1;
  mvpred_frame_start(e);
  int n = e->tile_cols;
  uint8_t **bufs = calloc((size_t)n, sizeof *bufs);
  size_t *sizes = calloc((size_t)n, sizeof *sizes);
  dispatch_apply((size_t)n, DISPATCH_APPLY_AUTO, ^(size_t tc) { sizes[tc] = write_one_tile(e, (int)tc, &bufs[tc]); });
  size_t total = 0;
  for (int i = 0; i < n; i++) total += sizes[i] + (i < n - 1 ? 4 : 0);
  uint8_t *o = malloc(total), *q = o;
  for (int i = 0; i < n; i++) {
    if (i < n - 1) {
      uint32_t m1 = (uint32_t)sizes[i] - 1;
      for (int b = 0; b < 4; b++) *q++ = (uint8_t)(m1 >> (8 * b));
    }
    memcpy(q, bufs[i], sizes[i]);
    q += sizes[i];
    free(bufs[i]);
  }
  free(bufs);
  free(sizes);
  *out = o;
  return total;
}
