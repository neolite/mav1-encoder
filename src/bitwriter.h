/* MSB-first bit writer for OBU headers (spec 4.10 descriptors f(n), su(n),
 * ns(n), leb128, uvlc). */
#ifndef BITWRITER_H
#define BITWRITER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
  uint8_t *buf;
  size_t cap;
  size_t bitpos;
} BitWriter;

void bw_init(BitWriter *w);
void bw_free(BitWriter *w);
void bw_f(BitWriter *w, int n, uint32_t v);         /* f(n) */
void bw_su(BitWriter *w, int n, int32_t v);         /* su(n) */
void bw_ns(BitWriter *w, uint32_t n, uint32_t v);   /* ns(n) */
void bw_byte_align(BitWriter *w);                   /* zero bits to byte boundary */
void bw_trailing_bits(BitWriter *w);                /* trailing_bits() */
size_t bw_bytes(const BitWriter *w);
/* Appends raw bytes; the writer must be byte aligned. */
void bw_bytes_append(BitWriter *w, const uint8_t *p, size_t n);

/* leb128 encoding of `v` into `out`; returns the number of bytes written. */
int leb128_write(uint8_t *out, uint64_t v);

#endif
