/* Internal encoder state shared by the frame driver, the tile writer and
 * the backends. */
#ifndef ENC_INTERNAL_H
#define ENC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "kernels/k_common.h"
#include "cdfs.h"
#include "entenc.h"
#include "mav1.h"

typedef struct Backend Backend;

/* Per 4x4 (mi) information kept for context derivation. */
typedef struct {
  int8_t bsize;     /* MiSizes */
  int8_t is_inter;  /* IsInters */
  int8_t skip;      /* Skips */
  int8_t y_mode;    /* YModes */
  int8_t ref0;      /* RefFrames[0] (INTRA_FRAME = 0) */
  int8_t tx_size;   /* TxSizes (luma) */
  int16_t mv_row, mv_col; /* Mvs[0] */
  int8_t written;   /* decoded in the current frame */
  int8_t interp[2]; /* InterpFilters */
} MiInfo;

/* Values signalled in the frame header. */
typedef struct {
  int frame_type;          /* KEY_FRAME or INTER_FRAME */
  int base_q_idx;
  int order_hint;
  int refresh_frame_flags;
  int lf_level[4];
  int lf_sharpness;
  int cdef_damping, cdef_bits;
  int cdef_y_pri[8], cdef_y_sec[8], cdef_uv_pri[8], cdef_uv_sec[8];
  int lr_type[3];           /* FrameRestorationType (RESTORE_NONE or RESTORE_SWITCHABLE) */
} FrameHdr;

typedef struct {
  uint8_t *base;    /* page aligned allocation */
  size_t size;
} Buf;

struct Mav1Encoder {
  Mav1Config cfg;
  int w, h;
  int mi_cols, mi_rows, sb_cols, sb_rows, num_sbs;
  int tile_cols_log2, tile_cols, tile_w_sb, min_log2_tile_cols, max_log2_tile_cols;
  int stride[3], offset[3], plane_h[3];
  size_t frame_bytes;

  /* Shared (CPU/GPU) buffers. */
  Buf src, rec, ref, params, blocks, sbs, coefs, mats, mvfield, scratch, lfmask, cdefbuf, mescratch, acosts, ainfos, ascratch;
  Buf dbk, lrout, lrunits;  /* deblocked copy, LR output, LrUnit per unit */

  MiInfo *mi;               /* mi_rows * mi_cols */
  int8_t *cdef_idx;         /* per 64x64 superblock, -1 when not coded */
  int8_t *cdef_choice;      /* per superblock: chosen CDEF parameter set */
  int num_cdef_presets;     /* 1 << cdef_bits */
  FrameHdr hdr;
  int frame_num;
  int nonref; /* current frame is not kept as a reference */
  int order_hint;
  int have_ref;
  Backend *be;

  /* statistics of the last frame */
  Mav1FrameStats last;
};

#define ENC_PARAMS(e) ((FrameParams *)(e)->params.base)
#define ENC_BLOCKS(e) ((BlockInfo *)(e)->blocks.base)
#define ENC_SBS(e) ((SbInfo *)(e)->sbs.base)
#define ENC_COEFS(e) ((int32_t *)(e)->coefs.base)

/* tile_writer.c: entropy codes the whole frame as one tile. */
size_t write_tile(struct Mav1Encoder *e, uint8_t **out);

/* lr_search.c */
void lr_select(struct Mav1Encoder *e);
int lr_unit_params(EntEnc *ec, int plane, const void *unit, int refW[2][3], int refX[2]);

/* obu.c */
size_t write_sequence_header(const struct Mav1Encoder *e, uint8_t *out);
size_t write_frame_obu(const struct Mav1Encoder *e, const uint8_t *tile, size_t tileSize, uint8_t *out);
size_t write_temporal_delimiter(uint8_t *out);

#endif
