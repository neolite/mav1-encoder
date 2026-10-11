/*
 * Loop restoration parameter search. Runs on the host after CDEF (both
 * backends share it, so their bitstreams stay identical); candidates are
 * evaluated with the exact kernel code of k_lr.h.
 *
 * Per restoration unit: RESTORE_NONE, a Wiener filter fitted by separable
 * least squares (alternating vertical / horizontal solves, then a +-1 tap
 * refinement) and every self-guided parameter set with least-squares
 * projection weights. The choice minimises 16 * SSE + lambda * bits.
 *
 * The kernels are instantiated per bitdepth (MAV1_BD 10 then 8) and this
 * search dispatches on fp->bitdepth; its own pixel reads go through
 * px_get( ), which handles both sample sizes.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "enc_internal.h"

#undef MAV1_BD
#define MAV1_BD 10
#include "kernels/kernel_set.h"
#undef MAV1_BD
#define MAV1_BD 8
#include "kernels/kernel_set.h"

/* reads the sample at element offset o of a frame buffer of either depth */
static int px_get(const Mav1Encoder *e, const uint8_t *base, size_t o) {
  return e->pxsize == 2 ? *(const uint16_t *)(base + o * 2) : base[o];
}

/* ---- subexp coding (decode_signed_subexp_with_ref_bool inverse) -------- */

static int lit(EntEnc *ec, int n, int v) {
  if (ec && n > 0) ec_literal(ec, n, (uint32_t)v);
  return n;
}

static int recenter_nonneg(int r, int v) {
  if (v > (r << 1)) return v;
  if (v >= r) return (v - r) << 1;
  return ((r - v) << 1) - 1;
}

/* Writes (when ec != NULL) and returns the number of bits of v, coded as
 * decode_signed_subexp_with_ref_bool( low, high, k, r ). */
int lr_subexp(EntEnc *ec, int low, int high, int k, int r, int v) {
  int mx = high - low;
  r -= low;
  v -= low;
  int x = (r << 1) <= mx ? recenter_nonneg(r, v) : recenter_nonneg(mx - 1 - r, mx - 1 - v);
  int bits = 0, i = 0, mk = 0;
  for (;;) {
    int b2 = i ? k + i - 1 : k;
    int a = 1 << b2;
    if (mx <= mk + 3 * a) {
      /* NS( mx - mk ) */
      int n = mx - mk, val = x - mk;
      int w = FloorLog2(n) + 1, m = (1 << w) - n;
      if (val < m) return bits + lit(ec, w - 1, val);
      int tt = val + m;
      bits += lit(ec, w - 1, tt >> 1);
      return bits + lit(ec, 1, tt & 1);
    }
    int more = x >= mk + a;
    bits += lit(ec, 1, more);
    if (!more) return bits + lit(ec, b2, x - mk);
    i++;
    mk += a;
  }
}

/* Bits of the parameters of one unit given the running references; updates
 * the references (as read_lr_unit does). */
int lr_unit_params(EntEnc *ec, int plane, const void *unit, int refW[2][3], int refX[2]) {
  const LrUnit *u = (const LrUnit *)unit;
  int bits = 0;
  if (u->type == RESTORE_WIENER) {
    for (int pass = 0; pass < 2; pass++)
      for (int j = plane ? 1 : 0; j < 3; j++) {
        bits += lr_subexp(ec, Wiener_Taps_Min[j], Wiener_Taps_Max[j] + 1, Wiener_Taps_K[j], refW[pass][j],
                          u->wiener[pass][j]);
        refW[pass][j] = u->wiener[pass][j];
      }
  } else if (u->type == RESTORE_SGRPROJ) {
    bits += lit(ec, SGRPROJ_PARAMS_BITS, u->set);
    for (int i = 0; i < 2; i++) {
      if (Sgr_Params[u->set][i * 2])
        bits += lr_subexp(ec, Sgrproj_Xqd_Min[i], Sgrproj_Xqd_Max[i] + 1, SGRPROJ_PRJ_SUBEXP_K, refX[i], u->xqd[i]);
      refX[i] = u->xqd[i];
    }
  }
  return bits;
}

/* ---- search -------------------------------------------------------------- */

typedef struct {
  Mav1Encoder *e;
  const FrameParams *fp;
  int plane;
  const int *blk; /* (row, col) pairs of the 4x4 blocks of the unit */
  int nblk;
} UnitCtx;

static long unit_sse(UnitCtx *c, const LrUnit *u) {
  Mav1Encoder *e = c->e;
  int bd = c->fp->bitdepth;
  long sse = 0;
  int stride = e->stride[c->plane];
  for (int b = 0; b < c->nblk; b++) {
    LrBlock lb;
    KNAME(lr_block_setup)(c->fp, c->plane, c->blk[2 * b], c->blk[2 * b + 1], &lb);
    if (bd > 8)
      lr_apply_10(c->fp, c->plane, &lb, (const uint16_t *)e->rec.base, (const uint16_t *)e->dbk.base,
                  (uint16_t *)e->lrout.base, u);
    else
      lr_apply_8(c->fp, c->plane, &lb, (const uint8_t *)e->rec.base, (const uint8_t *)e->dbk.base,
                 (uint8_t *)e->lrout.base, u);
    for (int i = 0; i < lb.h; i++)
      for (int j = 0; j < lb.w; j++) {
        size_t o = e->offset[c->plane] + (size_t)(lb.y + i) * stride + lb.x + j;
        int d = px_get(e, e->lrout.base, o) - px_get(e, e->src.base, o);
        sse += d * d;
      }
  }
  return sse;
}

/* Solves the n x n system M x = v (Gaussian elimination, partial pivoting). */
static int solve(int n, double M[3][3], double v[3], double *x) {
  for (int i = 0; i < n; i++) {
    int p = i;
    for (int r = i + 1; r < n; r++)
      if (fabs(M[r][i]) > fabs(M[p][i])) p = r;
    if (fabs(M[p][i]) < 1e-9) return 0;
    if (p != i) {
      for (int k = 0; k < n; k++) {
        double t = M[i][k];
        M[i][k] = M[p][k];
        M[p][k] = t;
      }
      double t = v[i];
      v[i] = v[p];
      v[p] = t;
    }
    for (int r = i + 1; r < n; r++) {
      double f = M[r][i] / M[i][i];
      for (int k = i; k < n; k++) M[r][k] -= f * M[i][k];
      v[r] -= v[i] * f;
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    double s = v[i];
    for (int k = i + 1; k < n; k++) s -= M[i][k] * x[k];
    x[i] = s / M[i][i];
  }
  return 1;
}

/* Separable least-squares Wiener fit; out[pass][j] in coded units. */
static void wiener_fit(UnitCtx *c, int out[2][3]) {
  Mav1Encoder *e = c->e;
  int bd = c->fp->bitdepth;
  int first = c->plane ? 1 : 0, nc = 3 - first;
  double f[7] = {0, 0, 0, 1, 0, 0, 0}, g[7] = {0, 0, 0, 1, 0, 0, 0}; /* vertical, horizontal */
  int stride = e->stride[c->plane];
  for (int iter = 0; iter < 4; iter++) {
    int vertical = !(iter & 1);
    double M[3][3] = {{0}}, V[3] = {0};
    for (int b = 0; b < c->nblk; b++) {
      LrBlock lb;
      KNAME(lr_block_setup)(c->fp, c->plane, c->blk[2 * b], c->blk[2 * b + 1], &lb);
      for (int i = 0; i < lb.h; i++)
        for (int j = 0; j < lb.w; j++) {
          double S[7][7];
          for (int v = 0; v < 7; v++)
            for (int h = 0; h < 7; h++)
              S[v][h] = bd > 8 ? lr_src_10(c->fp, c->plane, &lb, (const uint16_t *)e->rec.base,
                                           (const uint16_t *)e->dbk.base, lb.x + j + h - 3, lb.y + i + v - 3)
                               : lr_src_8(c->fp, c->plane, &lb, (const uint8_t *)e->rec.base,
                                          (const uint8_t *)e->dbk.base, lb.x + j + h - 3, lb.y + i + v - 3);
          double Z[7];
          for (int k = 0; k < 7; k++) {
            double s = 0;
            for (int t = 0; t < 7; t++) s += vertical ? g[t] * S[k][t] : f[t] * S[t][k];
            Z[k] = s;
          }
          double phi[3];
          for (int k = first; k < 3; k++) phi[k - first] = Z[k] + Z[6 - k] - 2 * Z[3];
          double tgt = px_get(e, e->src.base, e->offset[c->plane] + (size_t)(lb.y + i) * stride + lb.x + j) - Z[3];
          for (int a = 0; a < nc; a++) {
            V[a] += phi[a] * tgt;
            for (int bb = 0; bb < nc; bb++) M[a][bb] += phi[a] * phi[bb];
          }
        }
    }
    double x[3] = {0, 0, 0};
    for (int a = 0; a < nc; a++) M[a][a] += 1e-3; /* regularisation */
    if (!solve(nc, M, V, x)) break;
    double *dst = vertical ? f : g;
    double sum = 0;
    for (int k = 0; k < 3; k++) {
      double v = k < first ? 0 : x[k - first];
      dst[k] = dst[6 - k] = v;
      sum += v;
    }
    dst[3] = 1 - 2 * sum;
  }
  for (int pass = 0; pass < 2; pass++) {
    double *src = pass == 0 ? f : g;
    for (int j = 0; j < 3; j++) {
      int v = j < first ? 0 : (int)lround(src[j] * 128);
      out[pass][j] = j < first ? 0 : k_max(Wiener_Taps_Min[j], k_min(Wiener_Taps_Max[j], v));
    }
  }
}

/* Least-squares projection weights for self-guided set `set`. */
static void sgr_fit(UnitCtx *c, int set, int xqd[2]) {
  Mav1Encoder *e = c->e;
  int bd = c->fp->bitdepth;
  int r0 = Sgr_Params[set][0], r1 = Sgr_Params[set][2];
  int stride = e->stride[c->plane];
  double H00 = 0, H01 = 0, H11 = 0, C0 = 0, C1 = 0;
  for (int b = 0; b < c->nblk; b++) {
    LrBlock lb;
    KNAME(lr_block_setup)(c->fp, c->plane, c->blk[2 * b], c->blk[2 * b + 1], &lb);
    int flt0[16], flt1[16];
    if (bd > 8) {
      lr_box_10(c->fp, c->plane, &lb, (const uint16_t *)e->rec.base, (const uint16_t *)e->dbk.base, set, 0, flt0);
      lr_box_10(c->fp, c->plane, &lb, (const uint16_t *)e->rec.base, (const uint16_t *)e->dbk.base, set, 1, flt1);
    } else {
      lr_box_8(c->fp, c->plane, &lb, (const uint8_t *)e->rec.base, (const uint8_t *)e->dbk.base, set, 0, flt0);
      lr_box_8(c->fp, c->plane, &lb, (const uint8_t *)e->rec.base, (const uint8_t *)e->dbk.base, set, 1, flt1);
    }
    for (int i = 0; i < lb.h; i++)
      for (int j = 0; j < lb.w; j++) {
        size_t o = e->offset[c->plane] + (size_t)(lb.y + i) * stride + lb.x + j;
        double u = px_get(e, e->rec.base, o) << SGRPROJ_RST_BITS;
        double s = (px_get(e, e->src.base, o) << SGRPROJ_RST_BITS) - u;
        double f0 = r0 ? flt0[i * 4 + j] - u : 0, f1 = r1 ? flt1[i * 4 + j] - u : 0;
        H00 += f0 * f0;
        H01 += f0 * f1;
        H11 += f1 * f1;
        C0 += f0 * s;
        C1 += f1 * s;
      }
  }
  double a = 0, bb = 0;
  if (r0 && r1) {
    double det = H00 * H11 - H01 * H01;
    if (fabs(det) > 1e-9) {
      a = (C0 * H11 - C1 * H01) / det;
      bb = (C1 * H00 - C0 * H01) / det;
    }
  } else if (r0) {
    if (H00 > 0) a = C0 / H00;
  } else if (r1) {
    if (H11 > 0) bb = C1 / H11;
  }
  int w0 = (int)lround(a * (1 << SGRPROJ_PRJ_BITS)), w2 = (int)lround(bb * (1 << SGRPROJ_PRJ_BITS));
  if (!r0) w0 = 0;
  xqd[0] = k_max(Sgrproj_Xqd_Min[0], k_min(Sgrproj_Xqd_Max[0], w0));
  if (!r1) {
    xqd[1] = k_max(Sgrproj_Xqd_Min[1], k_min(Sgrproj_Xqd_Max[1], (1 << SGRPROJ_PRJ_BITS) - xqd[0]));
  } else {
    xqd[1] = k_max(Sgrproj_Xqd_Min[1], k_min(Sgrproj_Xqd_Max[1], (1 << SGRPROJ_PRJ_BITS) - xqd[0] - w2));
  }
}

static long rd(const FrameParams *fp, long sse, int bits) { return sse * 16 + (long)fp->lambda * bits; }

void lr_select(Mav1Encoder *e) {
  FrameParams *fp = ENC_PARAMS(e);
  LrUnit *units = (LrUnit *)e->lrunits.base;
  int rows4 = (e->h + 3) >> 2, cols4 = (e->w + 3) >> 2;
  int *blk = malloc(sizeof(int) * 2 * rows4 * cols4);
  int *cnt = malloc(sizeof(int) * (fp->lr_unit_rows[0] * fp->lr_unit_cols[0] + 1) * 2);
  for (int plane = 0; plane < 3; plane++) {
    int nUnits = fp->lr_unit_rows[plane] * fp->lr_unit_cols[plane];
    int *start = cnt;
    int *fill = cnt + nUnits + 1;
    /* group the 4x4 blocks by restoration unit (counting sort) */
    memset(start, 0, sizeof(int) * (nUnits + 1));
    for (int r = 0; r < rows4; r++)
      for (int c = 0; c < cols4; c++) {
        LrBlock lb;
        if (!KNAME(lr_block_setup)(fp, plane, r, c, &lb)) continue;
        start[lb.unitRow * fp->lr_unit_cols[plane] + lb.unitCol + 1]++;
      }
    for (int u = 0; u < nUnits; u++) start[u + 1] += start[u];
    memset(fill, 0, sizeof(int) * nUnits);
    for (int r = 0; r < rows4; r++)
      for (int c = 0; c < cols4; c++) {
        LrBlock lb;
        if (!KNAME(lr_block_setup)(fp, plane, r, c, &lb)) continue;
        int u = lb.unitRow * fp->lr_unit_cols[plane] + lb.unitCol;
        int k = start[u] + fill[u]++;
        blk[2 * k] = r;
        blk[2 * k + 1] = c;
      }
    int refW[2][3], refX[2];
    for (int pass = 0; pass < 2; pass++) {
      refX[pass] = Sgrproj_Xqd_Mid[pass];
      for (int j = 0; j < 3; j++) refW[pass][j] = Wiener_Taps_Mid[j];
    }
    int used = 0;
    for (int u = 0; u < nUnits; u++) {
      UnitCtx c = {e, fp, plane, blk + 2 * start[u], start[u + 1] - start[u]};
      LrUnit best;
      memset(&best, 0, sizeof(best));
      best.type = RESTORE_NONE;
      /* restoration_type symbol: roughly 1 bit for NONE, 2 otherwise */
      long bestCost = rd(fp, unit_sse(&c, &best), 1);
      int tw[2][3], tx[2];
      /* Wiener */
      LrUnit w;
      memset(&w, 0, sizeof(w));
      w.type = RESTORE_WIENER;
      wiener_fit(&c, w.wiener);
      memcpy(tw, refW, sizeof(tw));
      memcpy(tx, refX, sizeof(tx));
      long wc = rd(fp, unit_sse(&c, &w), 2 + lr_unit_params(NULL, plane, &w, tw, tx));
      for (int pass = 0; pass < 2; pass++)
        for (int j = plane ? 1 : 0; j < 3; j++)
          for (int d = -1; d <= 1; d += 2) {
            LrUnit t = w;
            t.wiener[pass][j] += d;
            if (t.wiener[pass][j] < Wiener_Taps_Min[j] || t.wiener[pass][j] > Wiener_Taps_Max[j]) continue;
            memcpy(tw, refW, sizeof(tw));
            memcpy(tx, refX, sizeof(tx));
            long tc = rd(fp, unit_sse(&c, &t), 2 + lr_unit_params(NULL, plane, &t, tw, tx));
            if (tc < wc) {
              wc = tc;
              w = t;
            }
          }
      if (wc < bestCost) {
        bestCost = wc;
        best = w;
      }
      /* self-guided */
      for (int set = 0; set < (1 << SGRPROJ_PARAMS_BITS); set++) {
        LrUnit s;
        memset(&s, 0, sizeof(s));
        s.type = RESTORE_SGRPROJ;
        s.set = set;
        sgr_fit(&c, set, s.xqd);
        memcpy(tw, refW, sizeof(tw));
        memcpy(tx, refX, sizeof(tx));
        long sc = rd(fp, unit_sse(&c, &s), 2 + lr_unit_params(NULL, plane, &s, tw, tx));
        if (sc < bestCost) {
          bestCost = sc;
          best = s;
        }
      }
      if (e->cfg.lr_test) {
        /* exercise both filters: Wiener on even units, the best self-guided set on odd ones */
        if (!(u & 1)) {
          best = w;
        } else {
          long bs = -1;
          for (int set = 0; set < (1 << SGRPROJ_PARAMS_BITS); set++) {
            LrUnit s;
            memset(&s, 0, sizeof(s));
            s.type = RESTORE_SGRPROJ;
            s.set = set;
            sgr_fit(&c, set, s.xqd);
            long v = unit_sse(&c, &s);
            if (bs < 0 || v < bs) {
              bs = v;
              best = s;
            }
          }
        }
      }
      lr_unit_params(NULL, plane, &best, refW, refX);
      e->last.tool_use[9] += best.type == RESTORE_WIENER;
      e->last.tool_use[10] += best.type == RESTORE_SGRPROJ;
      units[fp->lr_unit_off[plane] + u] = best;
      used |= best.type != RESTORE_NONE;
    }
    fp->lr_type[plane] = used ? RESTORE_SWITCHABLE : RESTORE_NONE;
    e->hdr.lr_type[plane] = fp->lr_type[plane];
  }
  free(blk);
  free(cnt);
}
