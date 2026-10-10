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
/* Encodes symbol `s` with an N-symbol cdf and adapts it unless disabled. */
void ec_symbol(EntEnc *e, int s, uint16_t *cdf, int n);
/* read_bool(): equiprobable bit, no adaptation. */
void ec_bool(EntEnc *e, int bit);
/* read_literal(n): n equiprobable bits, MSB first. */
void ec_literal(EntEnc *e, int n, uint32_t v);
/* Finishes the tile (exit_symbol + trailing bits); returns the size and
 * stores a malloc'ed buffer in *out. */
size_t ec_finish(EntEnc *e, uint8_t **out);
/* Approximate cost in 1/8 bits of coding `s` with `cdf` (no state change). */
int ec_cost(const uint16_t *cdf, int n, int s);

#endif
