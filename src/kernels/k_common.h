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
  int32_t frame_is_intra;
  int32_t reduced_tx_set;
  int32_t enable_intra_edge_filter;
  int32_t tx_search;                  /* 1: closed-loop luma tx type search */
  int32_t lambda;                     /* rate-distortion multiplier (Q8) */
  int32_t wave;                       /* current wavefront index */
  int32_t num_blocks;
  int32_t frame_w, frame_h;           /* real (cropped) luma frame size */
  int32_t me_range;                   /* full-pel motion search range */
} FrameParams;

/* One coded block, in decode order. Filled by the CPU mode decision, then
 * completed by the reconstruction kernel. */
typedef struct {
  int32_t mi_row, mi_col, bsize;
  int32_t is_inter;
  int32_t y_mode, uv_mode, angle_y, angle_uv;
  int32_t mv_row, mv_col;               /* single reference (LAST_FRAME) */
  /* kernel outputs */
  int32_t tx_type;                      /* luma tx type (DCT_DCT if eob==0) */
  int32_t eob[3];
  int32_t coef_off[3];                  /* offsets into the coefficient buffer */
  int32_t pad0;
} BlockInfo;

/* Per-superblock index range into the block list. */
typedef struct {
  int32_t first_block, num_blocks;
  int32_t sb_row, sb_col;
} SbInfo;

#endif
