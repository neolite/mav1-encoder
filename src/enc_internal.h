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
struct Mav1Encoder;

/* Per 4x4 (mi) information kept for context derivation. */
typedef struct {
  int8_t bsize;     /* MiSizes */
  int8_t is_inter;  /* IsInters */
  int8_t skip;      /* Skips */
  int8_t y_mode;    /* YModes */
  int8_t ref0;      /* RefFrames[0] (INTRA_FRAME = 0) */
  int8_t ref1;      /* RefFrames[1] of compound blocks (else 0) */
  int8_t tx_size;   /* TxSizes (luma) */
  int16_t mv_row, mv_col; /* Mvs[0] */
  int16_t mv1_row, mv1_col; /* Mvs[1] */
  int8_t written;   /* decoded in the current frame */
  int8_t interp[2]; /* InterpFilters */
} MiInfo;

/* Values signalled in the frame header. */
typedef struct {
  int frame_type;          /* KEY_FRAME or INTER_FRAME */
  int base_q_idx;
  int order_hint;
  int refresh_frame_flags;
  int show_frame, showable_frame;
  int ref_idx[7];          /* ref_frame_idx[ LAST_FRAME - 1 .. ALTREF_FRAME - 1 ] */
  int reference_select;
  int skip_mode_allowed;   /* skip_mode_present is coded (always 0) */
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

/* Writer context (turbo): the per-frame buffers the tile writer reads,
 * handed over from the frame core so that the entropy coding of a frame runs
 * on the writer queue while the next frame's core runs. */
typedef struct WCtx {
  Buf blocks, sbs, coefs, params, lrunits;
  MiInfo *mi;
  int8_t *cdef_idx, *cdef_choice;
  void *free;              /* dispatch semaphore: the context is not in use */
  struct Mav1Encoder *snap; /* encoder state the writer job sees */
} WCtx;
#define NUM_WCTX 3

struct Mav1Encoder {
  Mav1Config cfg;
  int w, h;
  int mi_cols, mi_rows, sb_cols, sb_rows, num_sbs;
  int tile_cols_log2, tile_cols, tile_w_sb, min_log2_tile_cols, max_log2_tile_cols;
  int stride[3], offset[3], plane_h[3];
  size_t frame_bytes;

  /* Shared (CPU/GPU) buffers. */
  Buf src, rec, ref, gold, mvgold, params, blocks, sbs, coefs, mats, mvfield, scratch, lfmask, cdefbuf, mescratch, acosts, ainfos, ascratch;
  Buf dbk, lrout, lrunits;
  unsigned ref_gen, gold_gen; /* generation of the pictures in ref / gold (0: unknown) */
  unsigned slot_gen[8], gen;  /* generation of each slot's picture (bumped per refresh) */
  Buf hpel;                 /* turbo: quarter-pel planes of the motion search references */  /* deblocked copy, LR output, LrUnit per unit */

  MiInfo *mi;               /* mi_rows * mi_cols */
  int8_t *cdef_idx;         /* per 64x64 superblock, -1 when not coded */
  int8_t *cdef_choice;      /* per superblock: chosen CDEF parameter set */
  int num_cdef_presets;     /* 1 << cdef_bits */
  FrameHdr hdr;
  int frame_num;
  int nonref; /* current frame is not kept as a reference */
  int leafopt, leafopt_cfg; /* leaf-frame savings of the current frame / of leaf frames (cfg.leaf) */
  int order_hint;
  int have_ref;
  int last_id, gold_id; /* frame_num of the frames held in the LAST / GOLDEN slots */
  int golden_refresh;   /* current frame refreshes the GOLDEN slot */
  Backend *be;
  int sign_bias[8];     /* RefFrameSignBias[ ] of the current frame */

  /* ALTREF mini-GOP driver (cfg.altref > 0) */
  int frame_kind;       /* 0 regular, 1 hidden ALTREF, 2 frame inside the mini-GOP */
  Buf slot[8];          /* reference slots (0, 2 and 3 are used) */
  int slot_hint[8];     /* OrderHint of each slot */
  int anchor_slot;      /* slot of the last shown reference frame */
  uint8_t **inq;        /* queued input frames (packed I420), display order */
  int inq_n, in_disp, key_disp;
  struct AltPacket { uint8_t *data; size_t size; uint8_t *recon; Mav1FrameStats st; } *outq;
  int out_n, out_pos, out_cap;
  uint8_t *shown_recon; /* recon of the frame shown by the last received packet */
  int pending;          /* non-altref send/receive: a packet waits */
  uint8_t *pend_data; size_t pend_size;

  /* turbo: entropy coding and bitstream assembly on a serial queue */
  void *wq;             /* dispatch queue (NULL: everything inline) */
  WCtx wctx[NUM_WCTX];
  int wnext, flushing;
  void *outmu;          /* pthread mutex of outq */
  /* turbo: frame engines (complete encoders without driver) coding frames
   * concurrently; frames wait for the slots they reference */
#define NUM_ENG 3 /* engine 0 codes the reference frames, 1 and 2 the others */
  struct Mav1Encoder *eng[NUM_ENG];
  int eng_next, is_engine;
  void *eng_q[NUM_ENG], *eng_busy[NUM_ENG]; /* serial dispatch queue, group of its jobs */
  /* slot versions: slot_ver[k] (dispatch group, NULL: written) completes
   * when the slot's current picture is written; slot_rd[k] holds the
   * frames reading that picture (a later write waits for them) */
  void *slot_ver[8], *slot_rd[8];
  void *inflight;               /* dispatch semaphore: frames in flight */
  /* GPU priority of reference frames: a non-reference frame's core starts
   * once the next reference frame's GPU stages are done (gate_pending: the
   * gate of the last such frame, handed to the next reference frame) */
  void *gate_pending;
  void *gate_out;               /* engine: semaphore to signal when the GPU core is done */

  /* statistics of the last frame */
  Mav1FrameStats last;
};

#define ENC_PARAMS(e) ((FrameParams *)(e)->params.base)
#define ENC_BLOCKS(e) ((BlockInfo *)(e)->blocks.base)
#define ENC_SBS(e) ((SbInfo *)(e)->sbs.base)
#define ENC_COEFS(e) ((int32_t *)(e)->coefs.base)

/* tile_writer.c: entropy codes the whole frame as one tile. */
size_t write_tile(struct Mav1Encoder *e, uint8_t **out);

/* encoder.c: edge filter types of the blocks of SB rows [row0, row1) */
void edge_types_rows(struct Mav1Encoder *e, uint8_t *ys, uint8_t *uvs, int row0, int row1);

/* lr_search.c */
void lr_select(struct Mav1Encoder *e);
int lr_unit_params(EntEnc *ec, int plane, const void *unit, int refW[2][3], int refX[2]);

/* obu.c */
size_t write_sequence_header(const struct Mav1Encoder *e, uint8_t *out);
size_t write_frame_obu(const struct Mav1Encoder *e, const uint8_t *tile, size_t tileSize, uint8_t *out);
size_t write_temporal_delimiter(uint8_t *out);
size_t write_show_existing(int slot, uint8_t *out);

#endif
