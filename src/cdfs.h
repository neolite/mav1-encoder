/* Adaptive CDF state for one tile (spec 7.20 init_non_coeff_cdfs /
 * init_coeff_cdfs), restricted to the syntax elements this encoder emits. */
#ifndef CDFS_H
#define CDFS_H

#include "spec_tables.h"

typedef struct {
  /* intra syntax */
  uint16_t intra_frame_y_mode[INTRA_MODE_CONTEXTS][INTRA_MODE_CONTEXTS][INTRA_MODES + 1];
  uint16_t y_mode[BLOCK_SIZE_GROUPS][INTRA_MODES + 1];
  uint16_t uv_mode_cfl_not_allowed[INTRA_MODES][UV_INTRA_MODES_CFL_NOT_ALLOWED + 1];
  uint16_t uv_mode_cfl_allowed[INTRA_MODES][UV_INTRA_MODES_CFL_ALLOWED + 1];
  uint16_t cfl_sign[CFL_JOINT_SIGNS + 1];
  uint16_t cfl_alpha[CFL_ALPHA_CONTEXTS][CFL_ALPHABET_SIZE + 1];
  uint16_t angle_delta[DIRECTIONAL_MODES][(2 * MAX_ANGLE_DELTA + 1) + 1];
  uint16_t partition_w8[PARTITION_CONTEXTS][5];
  uint16_t partition_w16[PARTITION_CONTEXTS][11];
  uint16_t partition_w32[PARTITION_CONTEXTS][11];
  uint16_t partition_w64[PARTITION_CONTEXTS][11];
  uint16_t skip[SKIP_CONTEXTS][3];
  uint16_t intra_tx_set1[2][INTRA_MODES][8];
  uint16_t intra_tx_set2[3][INTRA_MODES][6];
  uint16_t inter_tx_set1[2][17];
  uint16_t inter_tx_set2[13];
  uint16_t inter_tx_set3[4][3];
  /* inter syntax */
  uint16_t is_inter[IS_INTER_CONTEXTS][3];
  uint16_t new_mv[NEW_MV_CONTEXTS][3];
  uint16_t zero_mv[ZERO_MV_CONTEXTS][3];
  uint16_t ref_mv[REF_MV_CONTEXTS][3];
  uint16_t drl_mode[DRL_MODE_CONTEXTS][3];
  uint16_t single_ref[REF_CONTEXTS][SINGLE_REFS - 1][3];
  uint16_t mv_joint[MV_JOINTS + 1];
  uint16_t mv_class[2][MV_CLASSES + 1];
  uint16_t mv_class0_bit[2][3];
  uint16_t mv_class0_fr[2][CLASS0_SIZE][MV_JOINTS + 1];
  uint16_t mv_class0_hp[2][3];
  uint16_t mv_sign[2][3];
  uint16_t mv_bit[2][MV_OFFSET_BITS][3];
  uint16_t mv_fr[2][MV_JOINTS + 1];
  uint16_t mv_hp[2][3];
  uint16_t tx_8x8[TX_SIZE_CONTEXTS][MAX_TX_DEPTH + 1];
  uint16_t tx_16x16[TX_SIZE_CONTEXTS][MAX_TX_DEPTH + 2];
  uint16_t tx_32x32[TX_SIZE_CONTEXTS][MAX_TX_DEPTH + 2];
  uint16_t tx_64x64[TX_SIZE_CONTEXTS][MAX_TX_DEPTH + 2];
  uint16_t txfm_split[TXFM_PARTITION_CONTEXTS][3];
  uint16_t interp_filter[INTERP_FILTER_CONTEXTS][INTERP_FILTERS + 1];
  uint16_t use_wiener[3], use_sgrproj[3], restoration_type[RESTORE_SWITCHABLE + 1];
  /* coefficients (already indexed by the quantizer context) */
  uint16_t txb_skip[TX_SIZES][TXB_SKIP_CONTEXTS][3];
  uint16_t eob_pt_16[PLANE_TYPES][2][6];
  uint16_t eob_pt_32[PLANE_TYPES][2][7];
  uint16_t eob_pt_64[PLANE_TYPES][2][8];
  uint16_t eob_pt_128[PLANE_TYPES][2][9];
  uint16_t eob_pt_256[PLANE_TYPES][2][10];
  uint16_t eob_pt_512[PLANE_TYPES][11];
  uint16_t eob_pt_1024[PLANE_TYPES][12];
  uint16_t eob_extra[TX_SIZES][PLANE_TYPES][EOB_COEF_CONTEXTS][3];
  uint16_t dc_sign[PLANE_TYPES][DC_SIGN_CONTEXTS][3];
  uint16_t coeff_base_eob[TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS_EOB][4];
  uint16_t coeff_base[TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS][5];
  uint16_t coeff_br[TX_SIZES][PLANE_TYPES][LEVEL_CONTEXTS][BR_CDF_SIZE + 1];
} CdfContext;

void cdf_init(CdfContext *c, int base_q_idx);

#endif
