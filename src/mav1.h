/*
 * mav1 - an AV1 encoder whose pixel processing runs on Metal (or on the CPU
 * through the very same kernel sources).
 *
 * Output: 4:2:0, profile 0 (8 or 10 bit, Mav1Config.bitdepth), low-delay (key
 * frame followed by inter frames predicting from the previous frame). Every
 * temporal unit returned by mav1_encode() starts with a temporal delimiter;
 * key frames also carry a sequence header (and, with cfg.hdr, the HDR10
 * metadata OBUs).
 *
 * Planes are described as byte pointers at both depths: at 10 bits each
 * sample is a native-endian uint16_t (strides in bytes), the I420/P010-style
 * layout FFmpeg hands to an encoder.
 */
#ifndef MAV1_H
#define MAV1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* rate control modes */
#define MAV1_RC_CQP 0 /* constant qindex (cfg.qindex) */
#define MAV1_RC_VBR 1 /* single-pass average bitrate target */
#define MAV1_RC_CBR 2 /* VBR plus a buffer constraint (capped peak) */

typedef struct {
  int width, height;
  int bitdepth;          /* 8 (default) or 10 */
  int qindex;          /* base_q_idx, 1..255 */
  int use_metal;       /* 1: Metal backend, 0: CPU backend */
  int keyint;          /* key frame interval (0: only the first frame) */
  int tx_search;       /* closed-loop luma transform type search */
  int tx_depth;        /* TX_MODE_SELECT: intra transform split (tx_depth) RD search */
  int loop_filter;     /* deblocking filter */
  int cdef;            /* constrained directional enhancement filter */
  int restoration;     /* loop restoration (Wiener / self-guided) */
  int tile_cols_log2;    /* requested log2 of tile columns */
  int intra_edge_filter; /* enable_intra_edge_filter */
  int nonref_odd;       /* odd frames are non-reference leaf frames */
  int fast;              /* reduced mode/partition search */
  int multi_ref;         /* GOLDEN reference (slot 1, refreshed every 4th frame) besides LAST */
  int altref;            /* ALTREF mini-GOP length (0: off): hidden future frame, shown by show_existing_frame */
  int leaf;              /* with altref: pyramid mini-GOPs, non-reference leaf frames coded cheaply */
  int switchable_interp; /* per-block dual interpolation filter choice */
  int lr_test;         /* testing: force Wiener / self-guided on alternate units, ignoring rate */
  int me_range;        /* full-pel motion search range */
  int threads;         /* CPU backend worker threads (0: auto) */
  int turbo;           /* speed mode: implies fast, altref 2, leaf; reduced ME and analysis */
  /* color description (CICP; written to the sequence header when != -1) */
  int color_primaries, transfer_characteristics, matrix_coefficients; /* -1: unspecified */
  int color_range;     /* 0 limited, 1 full */
  /* HDR10 metadata (cfg.hdr): mastering display + content light level OBUs
   * after every sequence header. Chromaticities in 1/65536 units (0.16
   * fixed); luminances in the signal's fixed point (max 1/256 cd/m^2,
   * min 1/16384 cd/m^2). */
  int hdr;                 /* emit HDR10 metadata OBUs (implies a color description) */
  int mastering_display[8]; /* r, g, b, white: x, y in 1/65536 units (0: none) */
  int max_luma, min_luma;  /* mastering luminance, 0: defaults 1000 nits / 1 centinit */
  int max_cll, max_fall;   /* content light level in cd/m^2 (-1: none) */
  /* rate control */
  int rc;              /* MAV1_RC_CQP (default), MAV1_RC_VBR, MAV1_RC_CBR */
  int bitrate;         /* target bitrate in kbit/s (rc != CQP) */
  int vbv_size;        /* rate buffer in kbits (0: 1000 ms worth) */
  int fps_num, fps_den; /* frame rate for the rate model (0: 30/1) */
} Mav1Config;

typedef struct {
  int frame_type;      /* 0 key, 1 inter */
  size_t bytes;
  int num_blocks, num_inter_blocks;
  int tool_use[16]; /* per-frame block counts per coding tool, see mav1_tool_names */
  double t_analysis, t_recon, t_filter, t_entropy; /* seconds */
} Mav1FrameStats;

typedef struct Mav1Encoder Mav1Encoder;

void mav1_default_config(Mav1Config *cfg, int width, int height);
Mav1Encoder *mav1_create(const Mav1Config *cfg, char *err, size_t errlen);
/* Encodes one frame. `planes`/`strides` describe the I420 input. *out is a
 * malloc'ed buffer the caller frees. Returns 0 on success. */
int mav1_encode(Mav1Encoder *e, const uint8_t *const planes[3], const int strides[3],
                uint8_t **out, size_t *outSize);
/* Queue interface (needed with altref, which reorders frames): send input
 * frames (planes == NULL at the end of the stream), then receive temporal
 * units until it returns 1. Every temporal unit shows exactly one frame, in
 * display order; mav1_get_recon / mav1_last_stats then describe it. */
int mav1_send_frame(Mav1Encoder *e, const uint8_t *const planes[3], const int strides[3]);
int mav1_receive_packet(Mav1Encoder *e, uint8_t **out, size_t *outSize);
/* like mav1_receive_packet but waits up to timeout_ms for a packet */
int mav1_receive_packet_wait(Mav1Encoder *e, uint8_t **out, size_t *outSize, int timeout_ms);
/* number of finished packets queued for receive (peek, non-destructive) */
int mav1_pending_packets(Mav1Encoder *e);
/* nonzero while input frames or engine/writer work remain in flight */
int mav1_busy(Mav1Encoder *e);
/* Writes the sequence header OBU (with OBU header) and returns its size;
 * `out` must hold at least 64 bytes. */
size_t mav1_sequence_header(const Mav1Encoder *e, uint8_t *out);
/* Copies the reconstruction of the last encoded frame (what a decoder
 * outputs) into I420 planes. */
void mav1_get_recon(const Mav1Encoder *e, uint8_t *const planes[3], const int strides[3]);
const Mav1FrameStats *mav1_last_stats(const Mav1Encoder *e);
const char *mav1_backend_name(const Mav1Encoder *e);
extern const char *const mav1_tool_names[16];
void mav1_destroy(Mav1Encoder *e);

#ifdef __cplusplus
}
#endif

#endif
