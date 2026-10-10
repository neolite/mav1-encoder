#ifndef FWD_MATRICES_H
#define FWD_MATRICES_H
#include <stdint.h>
/* Fills `mats` (FWD_SLOT_SIZE * FWD_NUM_SLOTS int32 values). */
void fwd_build_matrices(int32_t *mats);
#endif
