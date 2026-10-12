/*
 * Shared definitions for the pixel kernels.
 *
 * Every file under src/kernels/ is compiled twice: once as C for the CPU
 * backend and once as Metal Shading Language for the GPU backend (the Metal
 * backend concatenates the sources and compiles them at runtime). The code
 * therefore sticks to the common subset of C11 and MSL: fixed-width integer
 * types, no recursion, no dynamic allocation, and explicit address-space
 * qualifiers on every pointer.
 */
#ifndef K_COMMON_H
#define K_COMMON_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
/* turbo mask: compile-time constant inside Metal kernels (the library is
 * built with -DMAV1_TURBO_MASK=...), fp->turbo on the CPU side. */
#ifndef MAV1_TURBO_MASK
#define MAV1_TURBO_MASK 0
#endif
#define FPT(fp) (MAV1_TURBO_MASK)
using namespace metal;
#define DEV device            /* read/write buffer memory */
#define KCONST constant       /* constant tables */
#define THR thread            /* function-local memory */
#define INLINE inline
#else
#include <stdint.h>
#include <string.h>
#define DEV
#define KCONST const
#define THR
#define INLINE static inline
#define FPT(fp) ((fp)->turbo)
#endif

#include "../spec_tables.h"

INLINE int k_min(int a, int b) { return a < b ? a : b; }
INLINE int k_max(int a, int b) { return a > b ? a : b; }
INLINE int k_abs(int a) { return a < 0 ? -a : a; }
INLINE int Clip3(int lo, int hi, int x) { return x < lo ? lo : (x > hi ? hi : x); }
INLINE int Clip1(int x) { return Clip3(0, 255, x); }
INLINE int Round2(int x, int n) { return n == 0 ? x : ((x + (1 << (n - 1))) >> n); }
INLINE long Round2L(long x, int n) { return n == 0 ? x : ((x + ((long)1 << (n - 1))) >> n); }
INLINE int Round2Signed(int x, int n) { return x >= 0 ? Round2(x, n) : -Round2(-x, n); }
INLINE int FloorLog2(int x) { int s = 0; while (x > 1) { x >>= 1; s++; } return s; }
INLINE int CeilLog2(int x) { if (x < 2) return 0; int i = 1; int p = 2; while (p < x) { i++; p <<= 1; } return i; }

/* Frame-level parameters shared by every kernel invocation. All fields are
 * 32-bit so that the C and MSL struct layouts are identical. */
typedef struct {
  int32_t mi_cols, mi_rows;
  int32_t sb_cols, sb_rows;
  int32_t plane_w[3], plane_h[3];     /* allocated size (MiCols*4 >> ss) */
  int32_t stride[3], offset[3];       /* layout of a frame buffer */
  int32_t base_q_idx;
  int32_t dc_q[3], ac_q[3];
  int32_t satd_lam;                   /* SATD/SAD-domain rate multiplier (ac_q scaled by bitdepth) */
  int32_t frame_is_intra;
  int32_t reduced_tx_set;
  int32_t enable_intra_edge_filter;
  int32_t tx_search;                  /* 1: closed-loop luma tx type search */
  int32_t lambda;                     /* rate-distortion multiplier (Q8) */
  int32_t wave;                       /* current wavefront index */
  int32_t num_blocks;
  int32_t frame_w, frame_h;           /* real (cropped) luma frame size */
  int32_t me_range;                   /* full-pel motion search range */
  int32_t lossless;                   /* CodedLossless (base_q_idx == 0) */
  int32_t tx_mode_select;             /* TxMode == TX_MODE_SELECT */
  int32_t tile_w_sb;                  /* uniform tile column width in SBs */
  int32_t tile_h_sb;                  /* uniform tile row height in SBs */
  int32_t fast;                       /* reduced analysis search */
  int32_t leaf;                       /* non-reference frame: inter only */
  int32_t depth_thr;                  /* depth early exit: SATD per sample x16 */
  int32_t switchable_interp;          /* interpolation_filter == SWITCHABLE (dual filter) */
  int32_t golden;                     /* inter blocks may also predict from a second reference */
  int32_t ref2_frame;                 /* that reference: GOLDEN_FRAME or ALTREF_FRAME */
  int32_t compound;                   /* LAST + ref2 compound (COMPOUND_AVERAGE) blocks allowed */
  int32_t lf_level[4];                /* loop_filter_level[] */
  int32_t lf_sharpness;
  int32_t cdef_damping, cdef_bits;
  int32_t cdef_y_pri[8], cdef_y_sec[8], cdef_uv_pri[8], cdef_uv_sec[8];
  /* loop restoration: FrameRestorationType and the unit grid of each plane */
  int32_t lr_type[3], lr_unit_size[3], lr_unit_rows[3], lr_unit_cols[3], lr_unit_off[3];
  int32_t turbo;                      /* speed mode (cfg.turbo): reduced ME and analysis */
  int32_t bitdepth;                   /* 8 or 10 */
  int32_t px_max;                     /* (1 << bitdepth) - 1 */
  int32_t bd_mid;                     /* 0x80 << (bitdepth - 8): filter midpoint */
  int32_t bd_half;                    /* 1 << (bitdepth - 1): filter clamp range */
  int32_t coeff_shift;                /* bitdepth - 8 (CDEF strengths, SGR sums) */
} FrameParams;

typedef struct {
  int32_t type;         /* LrType: RESTORE_NONE / RESTORE_WIENER / RESTORE_SGRPROJ */
  int32_t set;          /* LrSgrSet */
  int32_t xqd[2];       /* LrSgrXqd */
  int32_t wiener[2][3]; /* LrWiener[pass][j]: pass 0 vertical, pass 1 horizontal */
  int32_t pad[2];
} LrUnit;

typedef struct {
  int x, y, w, h;          /* block in plane samples */
  int stripeStart, stripeEnd;
  int planeEndX, planeEndY;
  int unitRow, unitCol;
} LrBlock;

/* Tile columns (uniform spacing, one tile row): MiColStart / MiColEnd of
 * the tile containing miCol. */
INLINE int tile_col_start(DEV const FrameParams *fp, int miCol) {
  return ((miCol >> 4) / fp->tile_w_sb) * fp->tile_w_sb * 16;
}
INLINE int tile_col_end(DEV const FrameParams *fp, int miCol) {
  return k_min(fp->mi_cols, tile_col_start(fp, miCol) + fp->tile_w_sb * 16);
}
INLINE int tile_row_start(DEV const FrameParams *fp, int miRow) {
  return fp->tile_h_sb > 0 ? ((miRow >> 4) / fp->tile_h_sb) * fp->tile_h_sb * 16 : 0;
}
INLINE int tile_row_end(DEV const FrameParams *fp, int miRow) {
  return fp->tile_h_sb > 0 ? k_min(fp->mi_rows, tile_row_start(fp, miRow) + fp->tile_h_sb * 16) : fp->mi_rows;
}

/* One coded block, in decode order. Filled by the CPU mode decision, then
 * completed by the reconstruction kernel. */
typedef struct {
  int32_t mi_row, mi_col, bsize;
  int32_t is_inter;
  int32_t y_mode, uv_mode, angle_y, angle_uv;
  int32_t mv_row, mv_col;               /* single reference (ref_frame) */
  /* kernel outputs */
  int32_t tx_type;                      /* luma tx type (DCT_DCT if eob==0) */
  int32_t eob[3];
  int32_t coef_off[3];                  /* offsets into the coefficient buffer */
  int32_t cfl_alpha_u, cfl_alpha_v;     /* CflAlphaU/V when uv_mode == UV_CFL_PRED */
  int32_t tx_depth;                     /* intra tx_depth (TX_MODE_SELECT) */
  int32_t tx_types_lo, tx_types_hi;     /* luma TxType of each transform, 4 bits each (raster) */
  int32_t interp_filter;                /* interp_filter[0] | interp_filter[1] << 2;
                                         * bit 4/5: get_filter_type( ) for luma/chroma */
  int32_t ref_frame;                    /* RefFrame[ 0 ] of inter blocks (LAST_FRAME or fp->ref2_frame) */
  int32_t ref_frame1;                   /* RefFrame[ 1 ] of compound blocks (fp->ref2_frame), else 0 */
  int32_t mv1_row, mv1_col;             /* Mv[ 1 ] of compound blocks */
} BlockInfo;

/* Per-superblock index range into the block list. */
typedef struct {
  int32_t first_block, num_blocks;
  int32_t sb_row, sb_col;
  int32_t part[85];                     /* partition per square node (-1 outside) */
} SbInfo;

/* Luma transform size of an intra block for the given tx_depth. */
INLINE int depth_tx_size(int bs, int depth) {
  int t = Max_Tx_Size_Rect[bs];
  for (int i = 0; i < depth; i++) t = Split_Tx_Size[t];
  return t;
}

#endif /* K_COMMON_H */

/*
 * Bitdepth instantiation, outside the include guard so that every inclusion
 * of this header re-derives it from MAV1_BD (both kernel instantiations can
 * live in one translation unit). Every kernel that touches frame pixels is
 * compiled per bitdepth (8: uint8_t pixels, 10: uint16_t pixels); set MAV1_BD
 * before including a kernel header. The pixel functions are named KNAME(fn)
 * (fn_8 / fn_10). The integer arithmetic is identical for both depths; only
 * the pixel element size and the pixel-range clamps differ (all of them
 * derived from FrameParams.bitdepth).
 */
#undef KSB
#undef KPIX
#undef KNAME
#if !defined(MAV1_BD)
#define MAV1_BD 8
#endif
#if MAV1_BD == 10
#define KPIX uint16_t
#define KSB _10
#else
#define KPIX uint8_t
#define KSB _8
#endif
#ifndef K_MCAT_
#define K_MCAT_(a, b) a##b
#define K_MCAT(a, b) K_MCAT_(a, b)
#endif
#define KNAME(n) K_MCAT(n, KSB)
/* selects one of two instantiations at run time: KSEL(bd, fn)(args) */
#define KSEL(bd, n) ((bd) > 8 ? K_MCAT(n, _10) : K_MCAT(n, _8))
