/* Multi-symbol arithmetic encoder matching the AV1 symbol decoder (spec 8.2),
 * with the spec's CDF adaptation (8.2.6 / 8.3.2). CDFs use the spec layout:
 * N cumulative values ending in 32768, followed by the adaptation counter. */
#ifndef ENTENC_H
#define ENTENC_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
  uint16_t *precarry;
  size_t cap, offs;
  uint32_t low;
  uint16_t rng;
  int16_t cnt;
  int disable_update;   /* disable_cdf_update */
} EntEnc;

void ec_init(EntEnc *e);
void ec_free(EntEnc *e);
void ec_grow(EntEnc *e);

#ifndef EC_PROB_SHIFT
#define EC_PROB_SHIFT (6)
#define EC_MIN_PROB (4)
#endif
static inline int ilog_nz(uint32_t v) { return 32 - __builtin_clz(v); }

/* ec_symbol( ): encodes symbol `s` with an N-symbol cdf and adapts it unless
 * disabled; ec_bool( ): read_bool( ), an equiprobable bit; ec_literal( ):
 * read_literal(n), n equiprobable bits, MSB first (inline: hot paths). */
static inline void ec_push(EntEnc *e, uint16_t v) {
  if (e->offs >= e->cap) ec_grow(e);
  e->precarry[e->offs++] = v;
}

static inline void ec_normalize(EntEnc *e, uint32_t low, unsigned rng) {
  int d = 16 - ilog_nz(rng);
  int c = e->cnt;
  int s = c + d;
  if (s >= 0) {
    c += 16;
    uint32_t m = (1u << c) - 1;
    if (s >= 8) {
      ec_push(e, (uint16_t)(low >> c));
      low &= m;
      c -= 8;
      m >>= 8;
    }
    ec_push(e, (uint16_t)(low >> c));
    s = c + d - 24;
    low &= m;
  }
  e->low = low << d;
  e->rng = (uint16_t)(rng << d);
  e->cnt = (int16_t)s;
}

/* fl/fh are inverse cdf values (32768 - cdf); s is the symbol, n symbols. */
static inline void ec_encode_q15(EntEnc *e, unsigned fl, unsigned fh, int s, int n) {
  uint32_t l = e->low;
  unsigned r = e->rng;
  const int N = n - 1;
  if (fl < 32768u) {
    unsigned u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (N - (s - 1));
    unsigned v = (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (N - s);
    l += r - u;
    r = u - v;
  } else {
    r -= (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (N - s);
  }
  ec_normalize(e, l, r);
}

static inline void cdf_adapt(uint16_t *cdf, int s, int N) {
  int rate = 3 + (cdf[N] > 15) + (cdf[N] > 31) + (N >= 4 ? 2 : (N >= 2 ? 1 : 0));
  int tmp = 0;
  for (int i = 0; i < N - 1; i++) {
    tmp = (i == s) ? (1 << 15) : tmp;
    if (tmp < cdf[i]) cdf[i] -= (uint16_t)((cdf[i] - tmp) >> rate);
    else cdf[i] += (uint16_t)((tmp - cdf[i]) >> rate);
  }
  cdf[N] += (cdf[N] < 32);
}

static inline void ec_symbol(EntEnc *e, int s, uint16_t *cdf, int n) {
  unsigned fl = s > 0 ? 32768u - cdf[s - 1] : 32768u;
  unsigned fh = 32768u - cdf[s];
  ec_encode_q15(e, fl, fh, s, n);
  if (!e->disable_update) cdf_adapt(cdf, s, n);
}

static inline void ec_bool(EntEnc *e, int bit) {
  uint16_t cdf[3] = {1 << 14, 1 << 15, 0};
  unsigned fl = bit ? 32768u - cdf[0] : 32768u;
  unsigned fh = 32768u - cdf[bit];
  ec_encode_q15(e, fl, fh, bit, 2);
}

static inline void ec_literal(EntEnc *e, int n, uint32_t v) {
  for (int i = n - 1; i >= 0; i--) ec_bool(e, (v >> i) & 1);
}

/* Finishes the tile (exit_symbol + trailing bits); returns the size and
 * stores a malloc'ed buffer in *out. */
size_t ec_finish(EntEnc *e, uint8_t **out);
/* Approximate cost in 1/8 bits of coding `s` with `cdf` (no state change). */
int ec_cost(const uint16_t *cdf, int n, int s);

#endif
