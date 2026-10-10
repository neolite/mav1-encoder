/* Metal entry points. Each wraps a kernel shared with the CPU backend. */
#include "k_analysis.h"
#include "k_recon.h"
#include "k_me.h"
#include "k_lf.h"
#include "k_cdef.h"

kernel void mav1_lf(device const FrameParams *fp [[buffer(0)]],
                    device uint8_t *rec [[buffer(1)]],
                    device const uint8_t *info [[buffer(2)]],
                    constant int &pass [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  int lines = pass == 0 ? fp->mi_rows : fp->mi_cols;
  if ((int)gid >= 3 * lines) return;
  int plane = (int)gid / lines, line = (int)gid % lines;
  if (plane > 0 && (line & 1)) return;
  lf_line(fp, rec, info, plane, pass, line);
}

kernel void mav1_cdef(device const FrameParams *fp [[buffer(0)]],
                      device const uint8_t *cur [[buffer(1)]],
                      device uint8_t *out [[buffer(2)]],
                      device const uint8_t *info [[buffer(3)]],
                      constant int &idx [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  int bc = (fp->mi_cols + 1) >> 1, br = (fp->mi_rows + 1) >> 1;
  if ((int)gid >= bc * br) return;
  cdef_block(fp, cur, out, info, ((int)gid / bc) * 2, ((int)gid % bc) * 2, idx);
}

kernel void mav1_me(device const FrameParams *fp [[buffer(0)]],
                    device const uint8_t *src [[buffer(1)]],
                    device const uint8_t *ref [[buffer(2)]],
                    device int32_t *mvField [[buffer(3)]],
                    device int32_t *scratch [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= fp->sb_cols * fp->sb_rows) return;
  me_sb(fp, (int)gid, src, ref, mvField, scratch + (long)gid * SCRATCH_INTS);
}

kernel void mav1_analyze(device const FrameParams *fp [[buffer(0)]],
                         device const uint8_t *src [[buffer(1)]],
                         device const uint8_t *ref [[buffer(2)]],
                         device const int32_t *mvField [[buffer(3)]],
                         device BlockInfo *blocks [[buffer(4)]],
                         device SbInfo *sbs [[buffer(5)]],
                         device int32_t *scratch [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
  int n = fp->sb_cols * fp->sb_rows;
  if ((int)gid >= n) return;
  analyze_sb(fp, (int)gid, src, ref, mvField, blocks, sbs, scratch + (long)gid * SCRATCH_INTS);
}

kernel void mav1_recon(device const FrameParams *fp [[buffer(0)]],
                       device const uint8_t *src [[buffer(1)]],
                       device uint8_t *rec [[buffer(2)]],
                       device const uint8_t *ref [[buffer(3)]],
                       device BlockInfo *blocks [[buffer(4)]],
                       device const SbInfo *sbs [[buffer(5)]],
                       device int32_t *coefs [[buffer(6)]],
                       device const int32_t *mats [[buffer(7)]],
                       device int32_t *scratch [[buffer(8)]],
                       constant int2 &wave [[buffer(9)]],   /* x: wave index t, y: first row */
                       uint gid [[thread_position_in_grid]]) {
  int r = wave.y + (int)gid, c = wave.x - 2 * r;
  if (r >= fp->sb_rows || c < 0 || c >= fp->sb_cols) return;
  int idx = r * fp->sb_cols + c;
  recon_sb(fp, sbs[idx], blocks, src, rec, ref, coefs, mats, scratch + (long)idx * SCRATCH_INTS);
}
