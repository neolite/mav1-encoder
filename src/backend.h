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
  void (*filter)(struct Backend *, struct Mav1Encoder *);
  void (*destroy)(struct Backend *);
} Backend;

Backend *backend_cpu_create(struct Mav1Encoder *e);
Backend *backend_metal_create(struct Mav1Encoder *e, char *err, size_t errlen);

#endif
