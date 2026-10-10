/* Forward/inverse transform consistency over every tx size and type. */
#include <stdio.h>
#include <stdlib.h>
#include "../src/kernels/k_fwd.h"
#include "../src/kernels/k_scan.h"
#include "../src/fwd_matrices.h"

int main(void) {
  static int32_t mats[FWD_SLOT_SIZE * FWD_NUM_SLOTS];
  static int32_t resid[4096], coef[4096], dq[4096], out[4096], tmp[8192];
  fwd_build_matrices(mats);
  unsigned seed = 7;
  int bad = 0;
  for (int txSz = 0; txSz < TX_SIZES_ALL; txSz++) {
    int w = Tx_Width[txSz], h = Tx_Height[txSz];
    for (int t = 0; t < 16; t++) {
      if (Tx_Size_Sqr_Up[txSz] == TX_64X64 && t != DCT_DCT) continue;
      if (Tx_Size_Sqr_Up[txSz] == TX_32X32 && t != DCT_DCT && t != IDTX) continue;
      double maxerr = 0, sumsq = 0;
      for (int trial = 0; trial < 20; trial++) {
        int smooth = trial & 1;
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++) {
            seed = seed * 1103515245 + 12345;
            int v = (int)((seed >> 16) % 511) - 255;
            if (smooth || w == 64 || h == 64) v = (int)(120 * __builtin_sin(0.07 * (i + 1) * (trial % 5 + 1)) + 60 * __builtin_cos(0.05 * j * (trial % 5 + 2)));
            resid[i * 64 + j] = v;
          }
        int fud = itx_flip_ud(t), flr = itx_flip_lr(t);
        static int32_t fl[4096];
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++) fl[i * 64 + j] = resid[(fud ? h - 1 - i : i) * 64 + (flr ? w - 1 - j : j)];
        fwd_2d(fl, coef, txSz, t, mats, tmp);
        for (int k = 0; k < 4096; k++) dq[k] = 0;
        for (int i = 0; i < 32 && i < h; i++)
          for (int j = 0; j < 32 && j < w; j++) {
            int v = coef[i * 64 + j];
            dq[i * 64 + j] = (v + (v >= 0 ? 8 : -8)) / 16;
          }
        itx_2d(dq, out, txSz, t, 0);
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++) {
            double e = out[i * 64 + j] - fl[i * 64 + j];
            if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
            sumsq += e * e;
          }
      }
      double rms = __builtin_sqrt(sumsq / (20.0 * w * h));
      int ok = (w == 64 || h == 64) ? rms < 3.0 : rms < 2.0;
      if (!ok) { bad++; printf("txSz %2d (%dx%d) type %2d: maxerr %.0f rms %.2f\n", txSz, w, h, t, maxerr, rms); }
    }
  }
  printf(bad ? "FAIL (%d)\n" : "transform consistency OK\n", bad);
  return bad != 0;
}
