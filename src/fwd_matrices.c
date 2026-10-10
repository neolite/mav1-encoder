/*
 * Derivation of the forward 1D transform matrices.
 *
 * For each kernel (DCT, ADST, identity) and length n, the normative integer
 * inverse kernel is applied to scaled unit impulses to obtain its linear
 * matrix A (n outputs x Min(32,n) inputs). The forward matrix is the
 * least-squares inverse F = (A^T A)^-1 A^T, stored in Q(FWD_BITS).
 */
#include "fwd_matrices.h"

#include <math.h>
#include <string.h>

#include "kernels/k_itx.h"
#include "kernels/k_fwd.h"

static void build_one(int kind, int log2n, int32_t *out) {
  int n = 1 << log2n, m = n > 32 ? 32 : n;
  static double A[64][32], N[32][64];
  const double S = 65536.0;
  for (int j = 0; j < m; j++) {
    int32_t T[64] = {0};
    T[j] = (int32_t)S;
    itx_1d(T, kind, log2n, 31);
    for (int i = 0; i < n; i++) A[i][j] = T[i] / S;
  }
  /* N = [A^T A | A^T], then Gauss-Jordan elimination. */
  static double M[32][32 + 64];
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < m; c++) {
      double s = 0;
      for (int i = 0; i < n; i++) s += A[i][r] * A[i][c];
      M[r][c] = s;
    }
    for (int i = 0; i < n; i++) M[r][m + i] = A[i][r];
  }
  for (int p = 0; p < m; p++) {
    int best = p;
    for (int r = p + 1; r < m; r++)
      if (fabs(M[r][p]) > fabs(M[best][p])) best = r;
    if (best != p)
      for (int c = 0; c < m + n; c++) { double t = M[p][c]; M[p][c] = M[best][c]; M[best][c] = t; }
    double inv = 1.0 / M[p][p];
    for (int c = 0; c < m + n; c++) M[p][c] *= inv;
    for (int r = 0; r < m; r++) {
      if (r == p || M[r][p] == 0) continue;
      double f = M[r][p];
      for (int c = 0; c < m + n; c++) M[r][c] -= f * M[p][c];
    }
  }
  for (int k = 0; k < m; k++)
    for (int i = 0; i < n; i++) N[k][i] = M[k][m + i];
  for (int k = 0; k < m; k++)
    for (int i = 0; i < n; i++) out[k * n + i] = (int32_t)lrint(N[k][i] * (double)(1 << FWD_BITS));
}

void fwd_build_matrices(int32_t *mats) {
  memset(mats, 0, sizeof(int32_t) * FWD_SLOT_SIZE * FWD_NUM_SLOTS);
  for (int kind = 0; kind < 3; kind++)
    for (int log2n = 2; log2n <= 6; log2n++) {
      if (kind == ITX_ADST && log2n > 4) continue;
      if (kind == ITX_IDTX && log2n > 5) continue;
      build_one(kind, log2n, mats + FWD_SLOT(kind, log2n));
    }
}
