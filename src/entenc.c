/*
 * Range encoder producing the bitstream consumed by the AV1 symbol decoder.
 * This is the classic Daala/libaom "precarry" formulation: output bytes are
 * buffered as 16-bit values so that carries can be propagated backwards
 * once the tile is finished.
 */
#include "entenc.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EC_PROB_SHIFT 6
#define EC_MIN_PROB 4

static int ilog_nz(uint32_t v) { return 32 - __builtin_clz(v); }

void ec_init(EntEnc *e) {
  memset(e, 0, sizeof(*e));
  e->cap = 1024;
  e->precarry = malloc(e->cap * sizeof(uint16_t));
  e->low = 0;
  e->rng = 0x8000;
  e->cnt = -9;
}

void ec_free(EntEnc *e) {
  free(e->precarry);
  e->precarry = NULL;
}

static void ec_push(EntEnc *e, uint16_t v) {
  if (e->offs >= e->cap) {
    e->cap *= 2;
    e->precarry = realloc(e->precarry, e->cap * sizeof(uint16_t));
  }
  e->precarry[e->offs++] = v;
}

static void ec_normalize(EntEnc *e, uint32_t low, unsigned rng) {
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
static void ec_encode_q15(EntEnc *e, unsigned fl, unsigned fh, int s, int n) {
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

static void cdf_adapt(uint16_t *cdf, int s, int N) {
  int rate = 3 + (cdf[N] > 15) + (cdf[N] > 31) + (N >= 4 ? 2 : (N >= 2 ? 1 : 0));
  int tmp = 0;
  for (int i = 0; i < N - 1; i++) {
    tmp = (i == s) ? (1 << 15) : tmp;
    if (tmp < cdf[i]) cdf[i] -= (uint16_t)((cdf[i] - tmp) >> rate);
    else cdf[i] += (uint16_t)((tmp - cdf[i]) >> rate);
  }
  cdf[N] += (cdf[N] < 32);
}

void ec_symbol(EntEnc *e, int s, uint16_t *cdf, int n) {
  assert(s >= 0 && s < n);
  assert(cdf[n - 1] == 32768);
  unsigned fl = s > 0 ? 32768u - cdf[s - 1] : 32768u;
  unsigned fh = 32768u - cdf[s];
  ec_encode_q15(e, fl, fh, s, n);
  if (!e->disable_update) cdf_adapt(cdf, s, n);
}

void ec_bool(EntEnc *e, int bit) {
  uint16_t cdf[3] = {1 << 14, 1 << 15, 0};
  unsigned fl = bit ? 32768u - cdf[0] : 32768u;
  unsigned fh = 32768u - cdf[bit];
  ec_encode_q15(e, fl, fh, bit, 2);
}

void ec_literal(EntEnc *e, int n, uint32_t v) {
  for (int i = n - 1; i >= 0; i--) ec_bool(e, (v >> i) & 1);
}

size_t ec_finish(EntEnc *e, uint8_t **out) {
  uint32_t l = e->low;
  int c = e->cnt;
  int s = 10;
  uint32_t m = 0x3FFF;
  uint32_t en = ((l + m) & ~m) | (m + 1);
  s += c;
  if (s > 0) {
    uint32_t n = (1u << (c + 16)) - 1;
    do {
      ec_push(e, (uint16_t)(en >> (c + 16)));
      en &= n;
      s -= 8;
      c -= 8;
      n >>= 8;
    } while (s > 0);
  }
  size_t nb = e->offs;
  uint8_t *buf = malloc(nb ? nb : 1);
  unsigned carry = 0;
  for (size_t i = nb; i-- > 0;) {
    carry = e->precarry[i] + carry;
    buf[i] = (uint8_t)carry;
    carry >>= 8;
  }
  *out = buf;
  return nb;
}

int ec_cost(const uint16_t *cdf, int n, int s) {
  int lo = s > 0 ? cdf[s - 1] : 0;
  int p = cdf[s] - lo;
  if (p < 1) p = 1;
  return (int)lrint(-8.0 * log2((double)p / 32768.0));
}
