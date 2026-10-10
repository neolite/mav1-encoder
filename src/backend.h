/* Compute backends: the same kernels run either on CPU threads or on the GPU
 * through Metal. Every stage reads and writes the encoder's shared buffers. */
#ifndef BACKEND_H
#define BACKEND_H

#include <stddef.h>

struct Mav1Encoder;

typedef struct Backend {
  const char *name;
  void (*motion_search)(struct Backend *, struct Mav1Encoder *);
  void (*analyze)(struct Backend *, struct Mav1Encoder *);
  void (*recon)(struct Backend *, struct Mav1Encoder *);
  /* Deblocks `rec` in place, then writes CDEF output for every parameter
   * set k into cdefbuf + k * frame_bytes. */
  void (*filter)(struct Backend *, struct Mav1Encoder *);
  /* Loop restoration of `rec` (CDEF output) using `dbk` into `lrout`. */
  void (*restore)(struct Backend *, struct Mav1Encoder *);
  void (*destroy)(struct Backend *);
} Backend;

Backend *backend_cpu_create(struct Mav1Encoder *e);
/* The serial-dependency stage (wavefront reconstruction) runs best on the
 * CPU cores; the Metal backend uses it too (unified memory, no copies). */
void cpu_backend_recon(Backend *b, struct Mav1Encoder *e);
void cpu_backend_analyze(Backend *b, struct Mav1Encoder *e);
Backend *backend_metal_create(struct Mav1Encoder *e, char *err, size_t errlen);

#endif
