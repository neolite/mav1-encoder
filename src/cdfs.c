#include "cdfs.h"

#include <string.h>

#define COPY(dst, src)                                         \
  do {                                                         \
    _Static_assert(sizeof(dst) == sizeof(src), #dst " size");  \
    memcpy(dst, src, sizeof(dst));                             \
  } while (0)

void cdf_init(CdfContext *c, int base_q_idx) {
  COPY(c->intra_frame_y_mode, Default_Intra_Frame_Y_Mode_Cdf);
  COPY(c->y_mode, Default_Y_Mode_Cdf);
  COPY(c->uv_mode_cfl_not_allowed, Default_Uv_Mode_Cfl_Not_Allowed_Cdf);
  COPY(c->uv_mode_cfl_allowed, Default_Uv_Mode_Cfl_Allowed_Cdf);
  COPY(c->cfl_sign, Default_Cfl_Sign_Cdf);
  COPY(c->cfl_alpha, Default_Cfl_Alpha_Cdf);
  COPY(c->angle_delta, Default_Angle_Delta_Cdf);
  COPY(c->partition_w8, Default_Partition_W8_Cdf);
  COPY(c->partition_w16, Default_Partition_W16_Cdf);
  COPY(c->partition_w32, Default_Partition_W32_Cdf);
  COPY(c->partition_w64, Default_Partition_W64_Cdf);
  COPY(c->skip, Default_Skip_Cdf);
  COPY(c->intra_tx_set1, Default_Intra_Tx_Type_Set1_Cdf);
  COPY(c->intra_tx_set2, Default_Intra_Tx_Type_Set2_Cdf);
  COPY(c->inter_tx_set1, Default_Inter_Tx_Type_Set1_Cdf);
  COPY(c->inter_tx_set2, Default_Inter_Tx_Type_Set2_Cdf);
  COPY(c->inter_tx_set3, Default_Inter_Tx_Type_Set3_Cdf);
  COPY(c->is_inter, Default_Is_Inter_Cdf);
  COPY(c->new_mv, Default_New_Mv_Cdf);
  COPY(c->zero_mv, Default_Zero_Mv_Cdf);
  COPY(c->ref_mv, Default_Ref_Mv_Cdf);
  COPY(c->drl_mode, Default_Drl_Mode_Cdf);
  COPY(c->single_ref, Default_Single_Ref_Cdf);
  COPY(c->comp_mode, Default_Comp_Mode_Cdf);
  COPY(c->comp_ref_type, Default_Comp_Ref_Type_Cdf);
  COPY(c->comp_ref, Default_Comp_Ref_Cdf);
  COPY(c->comp_bwd_ref, Default_Comp_Bwd_Ref_Cdf);
  COPY(c->compound_mode, Default_Compound_Mode_Cdf);
  /* MvCtx 0 only (intrabc is not used). */
  COPY(c->mv_joint, Default_Mv_Joint_Cdf);
  for (int comp = 0; comp < 2; comp++) {
    COPY(c->mv_class[comp], Default_Mv_Class_Cdf[comp]);
    COPY(c->mv_class0_bit[comp], Default_Mv_Class0_Bit_Cdf);
    COPY(c->mv_class0_fr[comp], Default_Mv_Class0_Fr_Cdf[comp]);
    COPY(c->mv_class0_hp[comp], Default_Mv_Class0_Hp_Cdf);
    COPY(c->mv_sign[comp], Default_Mv_Sign_Cdf);
    COPY(c->mv_bit[comp], Default_Mv_Bit_Cdf);
    COPY(c->mv_fr[comp], Default_Mv_Fr_Cdf[comp]);
    COPY(c->mv_hp[comp], Default_Mv_Hp_Cdf);
  }
  COPY(c->tx_8x8, Default_Tx_8x8_Cdf);
  COPY(c->tx_16x16, Default_Tx_16x16_Cdf);
  COPY(c->tx_32x32, Default_Tx_32x32_Cdf);
  COPY(c->tx_64x64, Default_Tx_64x64_Cdf);
  COPY(c->txfm_split, Default_Txfm_Split_Cdf);
  COPY(c->interp_filter, Default_Interp_Filter_Cdf);
  COPY(c->use_wiener, Default_Use_Wiener_Cdf);
  COPY(c->use_sgrproj, Default_Use_Sgrproj_Cdf);
  COPY(c->restoration_type, Default_Restoration_Type_Cdf);
  int idx = base_q_idx <= 20 ? 0 : base_q_idx <= 60 ? 1 : base_q_idx <= 120 ? 2 : 3;
  COPY(c->txb_skip, Default_Txb_Skip_Cdf[idx]);
  COPY(c->eob_pt_16, Default_Eob_Pt_16_Cdf[idx]);
  COPY(c->eob_pt_32, Default_Eob_Pt_32_Cdf[idx]);
  COPY(c->eob_pt_64, Default_Eob_Pt_64_Cdf[idx]);
  COPY(c->eob_pt_128, Default_Eob_Pt_128_Cdf[idx]);
  COPY(c->eob_pt_256, Default_Eob_Pt_256_Cdf[idx]);
  COPY(c->eob_pt_512, Default_Eob_Pt_512_Cdf[idx]);
  COPY(c->eob_pt_1024, Default_Eob_Pt_1024_Cdf[idx]);
  COPY(c->eob_extra, Default_Eob_Extra_Cdf[idx]);
  COPY(c->dc_sign, Default_Dc_Sign_Cdf[idx]);
  COPY(c->coeff_base_eob, Default_Coeff_Base_Eob_Cdf[idx]);
  COPY(c->coeff_base, Default_Coeff_Base_Cdf[idx]);
  COPY(c->coeff_br, Default_Coeff_Br_Cdf[idx]);
}
