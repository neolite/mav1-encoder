/* Used for builds without Metal (e.g. sanitizer runs of the CPU path). */
#include <stdio.h>

#include "backend.h"

Backend *backend_metal_create(struct Mav1Encoder *e, char *err, size_t errlen) {
  (void)e;
  snprintf(err, errlen, "built without Metal support");
  return NULL;
}
