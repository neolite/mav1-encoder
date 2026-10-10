/* Round-trip test: entenc output decoded by a literal transcription of the
 * spec symbol decoder (8.2.2, 8.2.6, exit process 8.2.4). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/entenc.h"

typedef struct { const uint8_t *d; size_t pos; int64_t val, rng, maxbits; } Dec;
static int fbits(Dec *s, int n) { int x = 0; for (int i = 0; i < n; i++) { int b = (s->d[s->pos >> 3] >> (7 - (s->pos & 7))) & 1; s->pos++; x = 2 * x + b; } return x; }
static int floorlog2(int64_t x) { int s = 0; while (x > 1) { x >>= 1; s++; } return s; }
static void dinit(Dec *s, const uint8_t *d, size_t sz) {
  s->d = d; s->pos = 0; int nb = sz * 8 < 15 ? sz * 8 : 15; int buf = fbits(s, nb);
  s->val = ((1 << 15) - 1) ^ (buf << (15 - nb)); s->rng = 1 << 15; s->maxbits = 8 * (int64_t)sz - 15;
}
static int dsym(Dec *s, uint16_t *cdf, int N, int adapt) {
  int64_t cur = s->rng, prev; int sym = -1;
  do { sym++; prev = cur; int64_t f = (1 << 15) - cdf[sym];
       cur = ((s->rng >> 8) * (f >> 6)) >> 1; cur += 4 * (N - sym - 1); } while (s->val < cur);
  s->rng = prev - cur; s->val -= cur;
  int bits = 15 - floorlog2(s->rng); s->rng <<= bits;
  int numBits = bits < (s->maxbits > 0 ? s->maxbits : 0) ? bits : (int)(s->maxbits > 0 ? s->maxbits : 0);
  int64_t nd = fbits(s, numBits); int64_t pd = nd << (bits - numBits);
  s->val = pd ^ (((s->val + 1) << bits) - 1); s->maxbits -= bits;
  if (adapt) {
    int rate = 3 + (cdf[N] > 15) + (cdf[N] > 31) + (floorlog2(N) < 2 ? floorlog2(N) : 2); int tmp = 0;
    for (int i = 0; i < N - 1; i++) { tmp = (i == sym) ? (1 << 15) : tmp;
      if (tmp < cdf[i]) cdf[i] -= ((cdf[i] - tmp) >> rate); else cdf[i] += ((tmp - cdf[i]) >> rate); }
    cdf[N] += (cdf[N] < 32);
  }
  return sym;
}
static int dexit_ok(Dec *s, size_t sz) {
  int64_t trailing = (int64_t)s->pos - (15 < s->maxbits + 15 ? 15 : s->maxbits + 15);
  if (s->maxbits < -14) return 0;
  int64_t end = s->pos + (s->maxbits > 0 ? s->maxbits : 0);
  if (end != (int64_t)sz * 8) return 0;
  for (int64_t x = trailing; x < end; x++) { int b = (s->d[x >> 3] >> (7 - (x & 7))) & 1; if (b != (x == trailing)) return 0; }
  return 1;
}
static void mkcdf(uint16_t *cdf, int n, unsigned *seed) {
  int acc = 0; for (int i = 0; i < n - 1; i++) { *seed = *seed * 1103515245 + 12345; acc += 1 + (*seed >> 16) % ((32768 - acc) / (n - i)); cdf[i] = acc; }
  cdf[n - 1] = 32768; cdf[n] = 0;
}
int main(void) {
  unsigned seed = 1; int fails = 0;
  for (int trial = 0; trial < 2000; trial++) {
    int nsym = trial % 3 == 0 ? 0 : 1 + (trial * 7919) % 3000;
    int n = 2 + trial % 15; uint16_t ce[17], cd[17]; mkcdf(ce, n, &seed); memcpy(cd, ce, sizeof ce);
    int *syms = malloc(sizeof(int) * (nsym + 1)); int *kinds = malloc(sizeof(int) * (nsym + 1));
    EntEnc e; ec_init(&e);
    for (int i = 0; i < nsym; i++) { seed = seed * 1103515245 + 12345; kinds[i] = (seed >> 20) & 3;
      seed = seed * 1103515245 + 12345;
      if (kinds[i] == 0) { syms[i] = (seed >> 16) & 1; ec_bool(&e, syms[i]); }
      else { int skew = (seed >> 16) % 4 == 0; seed = seed * 1103515245 + 12345; syms[i] = skew ? 0 : (int)((seed >> 16) % n); ec_symbol(&e, syms[i], ce, n); } }
    uint8_t *buf; size_t sz = ec_finish(&e, &buf);
    Dec d; dinit(&d, buf, sz);
    for (int i = 0; i < nsym; i++) {
      int s;
      if (kinds[i] == 0) { uint16_t b[3] = {16384, 32768, 0}; s = dsym(&d, b, 2, 0); }
      else s = dsym(&d, cd, n, 1);
      if (s != syms[i]) { printf("trial %d sym %d: got %d want %d\n", trial, i, s, syms[i]); fails++; break; }
    }
    if (!dexit_ok(&d, sz)) { printf("trial %d: bad trailing bits (sz=%zu)\n", trial, sz); fails++; }
    free(buf); free(syms); free(kinds); ec_free(&e);
    if (fails > 5) break;
  }
  printf(fails ? "FAIL\n" : "entropy round-trip OK\n");
  return fails != 0;
}
