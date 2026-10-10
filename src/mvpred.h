/* Motion vector prediction (spec 7.10.2 find_mv_stack) and inter block
 * syntax writing. */
#ifndef MVPRED_H
#define MVPRED_H

#include "cdfs.h"
#include "entenc.h"
#include "kernels/k_common.h"

struct Mav1Encoder;

void mvpred_frame_start(struct Mav1Encoder *e);
/* Writes inter_block_mode_info( ) for a single-reference LAST_FRAME block. */
/* Returns the chosen YMode. */
int write_inter_block(EntEnc *ec, CdfContext *cdf, struct Mav1Encoder *e, const BlockInfo *b);

#endif
