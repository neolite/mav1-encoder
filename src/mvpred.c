#include "mvpred.h"

#include <stdlib.h>

#include "enc_internal.h"

void mvpred_frame_start(struct Mav1Encoder *e) { (void)e; }

void write_inter_block(EntEnc *ec, CdfContext *cdf, struct Mav1Encoder *e, const BlockInfo *b) {
  (void)ec; (void)cdf; (void)e; (void)b;
  abort(); /* implemented in the inter milestone */
}
