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

void ec_grow(EntEnc *e) {
  e->cap *= 2;
  e->precarry = realloc(e->precarry, e->cap * sizeof(uint16_t));
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
