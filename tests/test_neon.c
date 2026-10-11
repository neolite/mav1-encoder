/* Differential test: the NEON kernels must be bit-identical to the scalar
 * reference implementations (itx_2d_ref, fwd_2d_ref) and to the reference
 * formulas of quantize/dequant. aarch64 only. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/kernels/k_itx.h"
#include "../src/kernels/k_fwd.h"
#include "../src/fwd_matrices.h"

static unsigned seed = 12345;
static int rnd(void) { seed = seed * 1103515245 + 12345; return (int)(seed >> 16); }

int main(void) {
  static int32_t mats[FWD_SLOT_SIZE * FWD_NUM_SLOTS];
  static int32_t resid[4096], c1[4096], c2[4096], tmp[8192];
  static int32_t dq[4096], o1[4096], o2[4096];
  static int32_t q1[4096], q2[4096], d1[4096];
  fwd_build_matrices(mats);
  int bad = 0, cases = 0;

  for (int txSz = 0; txSz < TX_SIZES_ALL; txSz++) {
    int w = Tx_Width[txSz], h = Tx_Height[txSz];
    int tw = w > 32 ? 32 : w, th = h > 32 ? 32 : h;
    for (int t = 0; t < 16; t++) {
      for (int trial = 0; trial < 40; trial++) {
        /* fwd: random residual */
        int sparse = trial & 1;
        memset(resid, 0, sizeof resid);
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++)
            resid[i * 64 + j] = sparse ? ((rnd() & 7) == 0 ? (rnd() % 1021) - 510 : 0) : (rnd() % 511) - 255;
        fwd_2d_ref(resid, c1, txSz, t, mats, tmp);
        fwd_2d_neon(resid, c2, txSz, t, mats, tmp);
        if (memcmp(c1, c2, tw * th * sizeof(int32_t))) { bad++; printf("fwd %dx%d t%d tr%d\n", w, h, t, trial); }
        cases++;

        /* itx: random dequant in range */
        memset(dq, 0, sizeof dq);
        for (int i = 0; i < th; i++)
          for (int j = 0; j < tw; j++)
            dq[i * 64 + j] = sparse ? ((rnd() & 3) == 0 ? (rnd() % 65535) - 32767 : 0) : (rnd() % 65535) - 32767;
        memset(o1, -7, sizeof o1); memset(o2, -3, sizeof o2);
        itx_2d_ref(dq, o1, txSz, t, 0);
        itx_2d_neon(dq, o2, txSz, t, 0);
        int diff = 0;
        for (int i = 0; i < h && !diff; i++)
          for (int j = 0; j < w; j++)
            if (o1[i * 64 + j] != o2[i * 64 + j]) diff = 1;
        if (diff) { bad++; printf("itx %dx%d t%d tr%d\n", w, h, t, trial); }
        cases++;
      }
      /* lossless wht */
      memset(dq, 0, sizeof dq);
      for (int i = 0; i < 16; i++) dq[i / 4 * 64 + i % 4] = (rnd() % 2000000) - 1000000;
      itx_2d_ref(dq, o1, TX_4X4, 0, 1);
      itx_2d_neon(dq, o2, TX_4X4, 0, 1);
      int bd=0;
      for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(o1[i*64+j]!=o2[i*64+j]){bd=1;printf("  [%d,%d] ref %d neon %d\n",i,j,o1[i*64+j],o2[i*64+j]);}
      if(bd){bad++;printf("wht txSz%d t%d\n",txSz,t);}
    }
  }
  /* quantize/dequant differential vs reference formulas */
  for (int dc = 4; dc < 22000; dc += 997) {
    for (int ac = 4; ac < 30000; ac += 1993) {
      for (int txSz = 0; txSz < TX_SIZES_ALL; txSz++) {
        int w = Tx_Width[txSz], h = Tx_Height[txSz];
        int tw = w > 32 ? 32 : w, th = h > 32 ? 32 : h;
        memset(c1, 0, sizeof c1);
        for (int i = 0; i < th; i++)
          for (int j = 0; j < tw; j++) c1[i * 64 + j] = (rnd() % 4000000) - 2000000;
        memset(q1, 9, sizeof q1); memset(q2, 7, sizeof q2);
        int n1 = quantize_block(c1, q1, txSz, dc, ac, 26, 22);
        /* scalar reference implementation of quantize */
        int denom = dq_denom(txSz), nz2 = 0;
        for (int i = 0; i < th; i++)
          for (int j = 0; j < tw; j++) {
            int q = (i == 0 && j == 0) ? dc : ac, rnd2 = (i == 0 && j == 0) ? 26 : 22;
            int v = c1[i * 64 + j];
            long a = (long)(v < 0 ? -v : v) * denom;
            long step = (long)q * 16;
            int level = (int)((a + (step * rnd2 >> 6)) / step);
            if (level > 0xFFFF) level = 0xFFFF;
            q2[i * tw + j] = v < 0 ? -level : level;
            nz2 += level != 0;
          }
        if (n1 != nz2 || memcmp(q1, q2, tw * th * sizeof(int32_t))) { bad++; printf("quant dc%d ac%d sz%d\n", dc, ac, txSz); }
        dequant_block(q1, d1, txSz, dc, ac);
        for (int i = 0; i < th; i++)
          for (int j = 0; j < tw; j++) {
            int q = (i == 0 && j == 0) ? dc : ac;
            long d = (long)q1[i * tw + j] * q, ad = d < 0 ? -d : d;
            long dv = (ad & 0xFFFFFF) / denom;
            if (d < 0) dv = -dv;
            int cl = dv < -(1 << 15) ? -(1 << 15) : (dv > (1 << 15) - 1 ? (1 << 15) - 1 : (int)dv);
            if (d1[i * 64 + j] != cl) { bad++; printf("dequant dc%d ac%d sz%d\n", dc, ac, txSz); i = th = 0; }
          }
        cases++;
      }
    }
  }
  printf("%s (%d bad / %d cases)\n", bad ? "FAIL" : "all identical", bad, cases);
  return bad != 0;
}
