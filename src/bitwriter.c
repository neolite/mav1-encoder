#include "bitwriter.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void bw_reserve(BitWriter *w, size_t bytes) {
  if (bytes <= w->cap) return;
  size_t cap = w->cap ? w->cap : 256;
  while (cap < bytes) cap *= 2;
  w->buf = realloc(w->buf, cap);
  memset(w->buf + w->cap, 0, cap - w->cap);
  w->cap = cap;
}

void bw_init(BitWriter *w) {
  memset(w, 0, sizeof(*w));
  bw_reserve(w, 256);
}

void bw_free(BitWriter *w) {
  free(w->buf);
  memset(w, 0, sizeof(*w));
}

static void bw_bit(BitWriter *w, int b) {
  bw_reserve(w, w->bitpos / 8 + 1);
  if (b) w->buf[w->bitpos >> 3] |= (uint8_t)(0x80 >> (w->bitpos & 7));
  else w->buf[w->bitpos >> 3] &= (uint8_t)~(0x80 >> (w->bitpos & 7));
  w->bitpos++;
}

void bw_f(BitWriter *w, int n, uint32_t v) {
  for (int i = n - 1; i >= 0; i--) bw_bit(w, (v >> i) & 1);
}

void bw_su(BitWriter *w, int n, int32_t v) {
  /* su(n): n-bit two's complement, value range [-(1<<(n-1)), (1<<(n-1))-1]. */
  bw_f(w, n, (uint32_t)v & ((1u << n) - 1));
}

void bw_ns(BitWriter *w, uint32_t n, uint32_t v) {
  /* ns(n) from spec 4.10.7. */
  int bits = 0;
  while ((1u << bits) <= n) bits++;   /* w = FloorLog2(n) + 1 */
  uint32_t m = (1u << bits) - n;
  if (v < m) {
    bw_f(w, bits - 1, v);
  } else {
    uint32_t extra = v - m;
    bw_f(w, bits - 1, m + (extra >> 1));
    bw_f(w, 1, extra & 1);
  }
}

void bw_byte_align(BitWriter *w) {
  while (w->bitpos & 7) bw_bit(w, 0);
}

void bw_trailing_bits(BitWriter *w) {
  bw_bit(w, 1);
  while (w->bitpos & 7) bw_bit(w, 0);
}

size_t bw_bytes(const BitWriter *w) { return (w->bitpos + 7) >> 3; }

void bw_bytes_append(BitWriter *w, const uint8_t *p, size_t n) {
  assert((w->bitpos & 7) == 0);
  bw_reserve(w, w->bitpos / 8 + n);
  memcpy(w->buf + w->bitpos / 8, p, n);
  w->bitpos += n * 8;
}

int leb128_write(uint8_t *out, uint64_t v) {
  int n = 0;
  do {
    uint8_t b = v & 0x7f;
    v >>= 7;
    if (v) b |= 0x80;
    out[n++] = b;
  } while (v);
  return n;
}
