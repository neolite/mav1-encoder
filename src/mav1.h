/*
 * mav1 - an AV1 encoder whose pixel processing runs on Metal (or on the CPU
 * through the very same kernel sources).
 *
 * Output: 8-bit 4:2:0, profile 0, low-delay (key frame followed by inter
 * frames predicting from the previous frame). Every temporal unit returned
 * by mav1_encode() starts with a temporal delimiter; key frames also carry
 * a sequence header.
 */
#ifndef MAV1_H
#define MAV1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  int qindex;          /* base_q_idx, 1..255 */
  int use_metal;       /* 1: Metal backend, 0: CPU backend */
  int keyint;          /* key frame interval (0: only the first frame) */
  int tx_search;       /* closed-loop luma transform type search */
  int tx_depth;        /* TX_MODE_SELECT: intra transform split (tx_depth) RD search */
  int loop_filter;     /* deblocking filter */
  int cdef;            /* constrained directional enhancement filter */
  int restoration;     /* loop restoration (Wiener / self-guided) */
  int switchable_interp; /* per-block dual interpolation filter choice */
  int lr_test;         /* testing: force Wiener / self-guided on alternate units, ignoring rate */
  int me_range;        /* full-pel motion search range */
  int threads;         /* CPU backend worker threads (0: auto) */
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
