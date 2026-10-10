/* CPU backend: runs the kernels as plain C on Grand Central Dispatch
 * worker threads, following the same dispatch structure as the Metal
 * backend (one work item per superblock, wavefront order for the
 * reconstruction). */
#include <dispatch/dispatch.h>
#include <stdlib.h>

#include "backend.h"
#include "enc_internal.h"
#include "kernels/k_analysis.h"
#include "kernels/k_recon.h"
#include "kernels/k_me.h"
#include "kernels/k_lf.h"
#include "kernels/k_cdef.h"
#include "kernels/k_lr.h"

typedef struct {
  Backend base;
} CpuBackend;

static void cpu_motion_search(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  int nref = fp->golden ? 2 : 1;
  if (fp->turbo & T_ME_HPEL) {
    int W = fp->frame_w, H = fp->frame_h;
    dispatch_apply((size_t)H * nref, DISPATCH_APPLY_AUTO, ^(size_t g) {
      int r = (int)g / H, y = (int)g % H;
      for (int x = 0; x < W; x++) qpel_build(fp, r ? e->gold.base : e->ref.base, e->hpel.base + (size_t)r * 16 * W * H, y, x);
    });
  }
  for (int level = 0; level < 4; level++) {
    int nn = 1 << (2 * level), nodes = e->num_sbs * nn;
    /* LAST then GOLDEN nodes; each reference uses its half of the SB scratch */
    dispatch_apply((size_t)nodes * nref, DISPATCH_APPLY_AUTO, ^(size_t g) {
      int r = (int)g / nodes, sb = ((int)g % nodes) / nn, i = (int)g % nn;
      me_node_serial(fp, sb, level, i, e->src.base, r ? e->gold.base : e->ref.base,
                     e->hpel.base + (size_t)r * 16 * fp->frame_w * fp->frame_h,
                     (int32_t *)(r ? e->mvgold.base : e->mvfield.base),
                     (int32_t *)e->scratch.base + (size_t)sb * SCRATCH_INTS + r * (SCRATCH_INTS / 2));
    });
  }
}

void cpu_backend_analyze(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  long *costs = (long *)e->acosts.base;
  BlockInfo *infos = (BlockInfo *)e->ainfos.base;
  dispatch_apply((size_t)e->num_sbs, DISPATCH_APPLY_AUTO, ^(size_t i) {
    for (int level = 0; level < 4; level++) {
      int ns = level_slots(fp, level);
      for (int node = 0; node < (1 << (2 * level)); node++)
        for (int slot = 0; slot < ns; slot++)
          analyze_item(fp, (int)i, level, node, slot, e->src.base, e->ref.base, e->gold.base,
                       (const int32_t *)e->mvfield.base, (const int32_t *)e->mvgold.base,
                       costs, infos, (int32_t *)e->scratch.base + i * SCRATCH_INTS);
    }
    analyze_sb_table(fp, (int)i, costs, infos, ENC_BLOCKS(e), ENC_SBS(e));
  });
}

void cpu_backend_recon(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  int waves = e->sb_cols + 2 * (e->sb_rows - 1);
  for (int t = 0; t < waves; t++) {
    /* SBs with col + 2 * row == t */
    int rmin = t - (e->sb_cols - 1);
    rmin = rmin > 0 ? (rmin + 1) / 2 : 0;
    int rmax = t / 2 < e->sb_rows - 1 ? t / 2 : e->sb_rows - 1;
    if (rmax < rmin) continue;
    dispatch_apply((size_t)(rmax - rmin + 1), DISPATCH_APPLY_AUTO, ^(size_t k) {
      int r = rmin + (int)k, c = t - 2 * r;
      int idx = r * e->sb_cols + c;
      recon_sb(fp, ENC_SBS(e)[idx], ENC_BLOCKS(e), e->src.base, e->rec.base, e->ref.base, e->gold.base, ENC_COEFS(e),
               (const int32_t *)e->mats.base, (int32_t *)e->scratch.base + (size_t)idx * SCRATCH_INTS);
    });
  }
}

static void cpu_filter(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  const uint8_t *info = e->lfmask.base;
  for (int pass = 0; pass < 2; pass++) {
    int lines = pass == 0 ? e->mi_rows : e->mi_cols;
    /* work item = (plane, line); chroma lines step by 2 mi */
    dispatch_apply((size_t)(3 * lines), DISPATCH_APPLY_AUTO, ^(size_t i) {
      int plane = (int)i / lines, line = (int)i % lines;
      if (plane > 0 && (line & 1)) return;
      lf_line(fp, e->rec.base, info, plane, pass, line);
    });
  }
  int bc = (e->mi_cols + 1) >> 1, br = (e->mi_rows + 1) >> 1;
  for (int k = 0; k < e->num_cdef_presets; k++) {
    uint8_t *out = e->cdefbuf.base + (size_t)k * e->frame_bytes;
    dispatch_apply((size_t)(bc * br), DISPATCH_APPLY_AUTO, ^(size_t i) {
      cdef_block(fp, e->rec.base, out, info, (int)(i / bc) * 2, (int)(i % bc) * 2, k);
    });
  }
}

static void cpu_restore(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  int rows4 = (e->h + 3) >> 2, cols4 = (e->w + 3) >> 2, n = rows4 * cols4;
  dispatch_apply((size_t)(3 * n), DISPATCH_APPLY_AUTO, ^(size_t g) {
    int plane = (int)g / n, i = (int)g % n;
    lr_block(fp, plane, i / cols4, i % cols4, e->rec.base, e->dbk.base, e->lrout.base,
             (const LrUnit *)e->lrunits.base);
  });
}

static void cpu_destroy(Backend *b) { free(b); }

Backend *backend_cpu_create(struct Mav1Encoder *e) {
  (void)e;
  CpuBackend *c = calloc(1, sizeof(*c));
  c->base.name = "cpu";
  c->base.motion_search = cpu_motion_search;
  c->base.analyze = cpu_backend_analyze;
  c->base.recon = cpu_backend_recon;
  c->base.filter = cpu_filter;
  c->base.restore = cpu_restore;
  c->base.destroy = cpu_destroy;
  return &c->base;
}
