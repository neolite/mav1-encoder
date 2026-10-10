/* OBU writing: sequence header (5.5), frame header (5.9) and frame OBU. */
#include <string.h>

#include "bitwriter.h"
#include "enc_internal.h"

#define ORDER_HINT_BITS 7

static int seq_level_idx(int w, int h) {
  long s = (long)w * h;
  if (s <= 147456) return 0;      /* 2.0 */
  if (s <= 278784) return 1;      /* 2.1 */
  if (s <= 665856) return 4;      /* 3.0 */
  if (s <= 1065024) return 5;     /* 3.1 */
  if (s <= 2359296) return 8;     /* 4.0 */
  if (s <= 8912896) return 12;    /* 5.0 */
  if (s <= 35651584) return 16;   /* 6.0 */
  return 31;
}

/* Wraps a payload into an OBU with obu_has_size_field = 1. */
static size_t wrap_obu(int type, const uint8_t *payload, size_t n, uint8_t *out) {
  size_t pos = 0;
  out[pos++] = (uint8_t)((type << 3) | 2);
  pos += leb128_write(out + pos, n);
  memcpy(out + pos, payload, n);
  return pos + n;
}

static void seq_header_payload(const struct Mav1Encoder *e, BitWriter *w) {
  bw_f(w, 3, 0);   /* seq_profile */
  bw_f(w, 1, 0);   /* still_picture */
  bw_f(w, 1, 0);   /* reduced_still_picture_header */
  bw_f(w, 1, 0);   /* timing_info_present_flag */
  bw_f(w, 1, 0);   /* initial_display_delay_present_flag */
  bw_f(w, 5, 0);   /* operating_points_cnt_minus_1 */
  bw_f(w, 12, 0);  /* operating_point_idc[0] */
  int lvl = seq_level_idx(e->w, e->h);
  bw_f(w, 5, lvl);
  if (lvl > 7) bw_f(w, 1, 0); /* seq_tier */
  bw_f(w, 4, 15);  /* frame_width_bits_minus_1 */
  bw_f(w, 4, 15);  /* frame_height_bits_minus_1 */
  bw_f(w, 16, e->w - 1);
  bw_f(w, 16, e->h - 1);
  bw_f(w, 1, 0);   /* frame_id_numbers_present_flag */
  bw_f(w, 1, 0);   /* use_128x128_superblock */
  bw_f(w, 1, 0);   /* enable_filter_intra */
  bw_f(w, 1, 0);   /* enable_intra_edge_filter */
  bw_f(w, 1, 0);   /* enable_interintra_compound */
  bw_f(w, 1, 0);   /* enable_masked_compound */
  bw_f(w, 1, 0);   /* enable_warped_motion */
  bw_f(w, 1, 0);   /* enable_dual_filter */
  bw_f(w, 1, 1);   /* enable_order_hint */
  bw_f(w, 1, 0);   /* enable_jnt_comp */
  bw_f(w, 1, 0);   /* enable_ref_frame_mvs */
  bw_f(w, 1, 0);   /* seq_choose_screen_content_tools */
  bw_f(w, 1, 0);   /* seq_force_screen_content_tools */
  bw_f(w, 3, ORDER_HINT_BITS - 1);
  bw_f(w, 1, 0);   /* enable_superres */
  bw_f(w, 1, e->cfg.cdef ? 1 : 0);
  bw_f(w, 1, 0);   /* enable_restoration */
  /* color_config */
  bw_f(w, 1, 0);   /* high_bitdepth */
  bw_f(w, 1, 0);   /* mono_chrome */
  bw_f(w, 1, 0);   /* color_description_present_flag */
  bw_f(w, 1, 0);   /* color_range */
  bw_f(w, 2, 0);   /* chroma_sample_position = CSP_UNKNOWN */
  bw_f(w, 1, 0);   /* separate_uv_delta_q */
  bw_f(w, 1, 0);   /* film_grain_params_present */
  bw_trailing_bits(w);
}

size_t write_sequence_header(const struct Mav1Encoder *e, uint8_t *out) {
  BitWriter w;
  bw_init(&w);
  seq_header_payload(e, &w);
  size_t n = wrap_obu(1 /* OBU_SEQUENCE_HEADER */, w.buf, bw_bytes(&w), out);
  bw_free(&w);
  return n;
}

size_t write_temporal_delimiter(uint8_t *out) {
  out[0] = (2 << 3) | 2;
  out[1] = 0;
  return 2;
}

static int tile_log2(int blk, int target) {
  int k = 0;
  while ((blk << k) < target) k++;
  return k;
}

static void frame_header(const struct Mav1Encoder *e, BitWriter *w) {
  const FrameHdr *h = &e->hdr;
  int intra = h->frame_type == KEY_FRAME;
  bw_f(w, 1, 0);                 /* show_existing_frame */
  bw_f(w, 2, h->frame_type);
  bw_f(w, 1, 1);                 /* show_frame */
  if (!intra) bw_f(w, 1, 0);     /* error_resilient_mode */
  bw_f(w, 1, 0);                 /* disable_cdf_update */
  bw_f(w, 1, 0);                 /* frame_size_override_flag */
  bw_f(w, ORDER_HINT_BITS, h->order_hint & ((1 << ORDER_HINT_BITS) - 1));
  if (!intra) bw_f(w, 3, PRIMARY_REF_NONE);
  if (!intra) bw_f(w, 8, h->refresh_frame_flags);
  if (intra) {
    bw_f(w, 1, 0);               /* render_and_frame_size_different */
  } else {
    bw_f(w, 1, 0);               /* frame_refs_short_signaling */
    for (int i = 0; i < REFS_PER_FRAME; i++) bw_f(w, 3, 0); /* ref_frame_idx */
    bw_f(w, 1, 0);               /* render_and_frame_size_different */
    bw_f(w, 1, 0);               /* allow_high_precision_mv */
    bw_f(w, 1, 0);               /* is_filter_switchable */
    bw_f(w, 2, EIGHTTAP);        /* interpolation_filter */
    bw_f(w, 1, 0);               /* is_motion_mode_switchable */
  }
  bw_f(w, 1, 1);                 /* disable_frame_end_update_cdf */

  /* tile_info: a single tile. */
  int sbCols = e->sb_cols, sbRows = e->sb_rows;
  int maxTileWidthSb = MAX_TILE_WIDTH >> 6;
  int maxTileAreaSb = MAX_TILE_AREA >> 12;
  int minLog2TileCols = tile_log2(maxTileWidthSb, sbCols);
  int maxLog2TileCols = tile_log2(1, sbCols < MAX_TILE_COLS ? sbCols : MAX_TILE_COLS);
  int maxLog2TileRows = tile_log2(1, sbRows < MAX_TILE_ROWS ? sbRows : MAX_TILE_ROWS);
  int minLog2Tiles = tile_log2(maxTileAreaSb, sbRows * sbCols);
  if (minLog2TileCols > minLog2Tiles) minLog2Tiles = minLog2TileCols;
  bw_f(w, 1, 1);                 /* uniform_tile_spacing_flag */
  int tileColsLog2 = minLog2TileCols;
  if (tileColsLog2 < maxLog2TileCols) bw_f(w, 1, 0);
  int minLog2TileRows = minLog2Tiles - tileColsLog2 > 0 ? minLog2Tiles - tileColsLog2 : 0;
  if (minLog2TileRows < maxLog2TileRows) bw_f(w, 1, 0);

  /* quantization_params */
  bw_f(w, 8, h->base_q_idx);
  bw_f(w, 1, 0);                 /* DeltaQYDc */
  bw_f(w, 1, 0);                 /* DeltaQUDc */
  bw_f(w, 1, 0);                 /* DeltaQUAc */
  bw_f(w, 1, 0);                 /* using_qmatrix */
  bw_f(w, 1, 0);                 /* segmentation_enabled */
  bw_f(w, 1, 0);                 /* delta_q_present (base_q_idx > 0) */

  /* loop_filter_params */
  bw_f(w, 6, h->lf_level[0]);
  bw_f(w, 6, h->lf_level[1]);
  if (h->lf_level[0] || h->lf_level[1]) {
    bw_f(w, 6, h->lf_level[2]);
    bw_f(w, 6, h->lf_level[3]);
  }
  bw_f(w, 3, h->lf_sharpness);
  bw_f(w, 1, 0);                 /* loop_filter_delta_enabled */

  /* cdef_params */
  if (e->cfg.cdef) {
    bw_f(w, 2, h->cdef_damping - 3);
    bw_f(w, 2, h->cdef_bits);
    for (int i = 0; i < (1 << h->cdef_bits); i++) {
      bw_f(w, 4, h->cdef_y_pri[i]);
      bw_f(w, 2, h->cdef_y_sec[i] == 4 ? 3 : h->cdef_y_sec[i]);
      bw_f(w, 4, h->cdef_uv_pri[i]);
      bw_f(w, 2, h->cdef_uv_sec[i] == 4 ? 3 : h->cdef_uv_sec[i]);
    }
  }
  bw_f(w, 1, 0);                 /* tx_mode_select -> TX_MODE_LARGEST */
  if (!intra) bw_f(w, 1, 0);     /* reference_select */
  bw_f(w, 1, 0);                 /* reduced_tx_set */
  if (!intra)
    for (int ref = LAST_FRAME; ref <= ALTREF_FRAME; ref++) bw_f(w, 1, 0); /* is_global */
}

size_t write_frame_obu(const struct Mav1Encoder *e, const uint8_t *tile, size_t tileSize, uint8_t *out) {
  BitWriter w;
  bw_init(&w);
  frame_header(e, &w);
  bw_byte_align(&w);
  bw_bytes_append(&w, tile, tileSize);
  size_t n = wrap_obu(6 /* OBU_FRAME */, w.buf, bw_bytes(&w), out);
  bw_free(&w);
  return n;
}
