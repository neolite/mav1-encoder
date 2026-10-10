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

typedef struct {
  Backend base;
} CpuBackend;

static void cpu_motion_search(Backend *b, struct Mav1Encoder *e) { (void)b; (void)e; }

static void cpu_analyze(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  const FrameParams *fp = ENC_PARAMS(e);
  dispatch_apply((size_t)e->num_sbs, DISPATCH_APPLY_AUTO, ^(size_t i) {
    analyze_sb(fp, (int)i, e->src.base, e->ref.base, (const int32_t *)e->mvfield.base, ENC_BLOCKS(e), ENC_SBS(e),
               (int32_t *)e->scratch.base + i * SCRATCH_INTS);
  });
}

static void cpu_recon(Backend *b, struct Mav1Encoder *e) {
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
      recon_sb(fp, ENC_SBS(e)[idx], ENC_BLOCKS(e), e->src.base, e->rec.base, e->ref.base, ENC_COEFS(e),
               (const int32_t *)e->mats.base, (int32_t *)e->scratch.base + (size_t)idx * SCRATCH_INTS);
    });
  }
}

static void cpu_filter(Backend *b, struct Mav1Encoder *e) { (void)b; (void)e; }

static void cpu_destroy(Backend *b) { free(b); }

Backend *backend_cpu_create(struct Mav1Encoder *e) {
  (void)e;
  CpuBackend *c = calloc(1, sizeof(*c));
  c->base.name = "cpu";
  c->base.motion_search = cpu_motion_search;
  c->base.analyze = cpu_analyze;
  c->base.recon = cpu_recon;
  c->base.filter = cpu_filter;
  c->base.destroy = cpu_destroy;
  return &c->base;
}
