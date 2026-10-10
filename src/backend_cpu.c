/* CPU backend: runs the kernels as plain C on Grand Central Dispatch
 * worker threads, following the same dispatch structure as the Metal
 * backend (one work item per superblock, wavefront order for the
 * reconstruction). */
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <Block.h>

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
  /* plane buffer entries 0 (LAST) and 1 (second reference), rebuilt every frame */
  uint8_t *qp = e->hpel.base, *srcDec = qp + ME_QP_ENTRIES * qp_entry_bytes(fp);
  if (fp->turbo & T_ME_HPEL) {
    int W = fp->frame_w, H = fp->frame_h;
    dispatch_apply((size_t)H * (nref + 1), DISPATCH_APPLY_AUTO, ^(size_t g) {
      int r = (int)g / H, y = (int)g % H;
      for (int x = 0; x < W; x++) {
        int dec = (fp->turbo & T_ME_DEC) && y < dec_h(fp) && x < dec_w(fp);
        if (r == nref) {
          if (dec) dec_build(fp, e->src.base, srcDec, y, x);
          continue;
        }
        uint8_t *ent = qp + r * qp_entry_bytes(fp);
        qpel_build(fp, r ? e->gold.base : e->ref.base, ent, y, x);
        if (dec) dec_build(fp, r ? e->gold.base : e->ref.base, ent + 16L * W * H, y, x);
      }
    });
  }
  for (int level = 0; level < 4; level++) {
    int nn = 1 << (2 * level), nodes = e->num_sbs * nn;
    /* LAST then GOLDEN nodes; each reference uses its half of the SB scratch */
    dispatch_apply((size_t)nodes * nref, DISPATCH_APPLY_AUTO, ^(size_t g) {
      int r = (int)g / nodes, sb = ((int)g % nodes) / nn, i = (int)g % nn;
      me_node_serial(fp, sb, level, i, e->src.base, r ? e->gold.base : e->ref.base,
                     qp + r * qp_entry_bytes(fp), srcDec, qp_dec(fp, qp + r * qp_entry_bytes(fp)),
                     (int32_t *)(r ? e->mvgold.base : e->mvfield.base),
                     (int32_t *)e->scratch.base + (size_t)sb * SCRATCH_INTS + r * (SCRATCH_INTS / 2), r);
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

/* Superblock (r, c) needs its left and above-right neighbours inside its
 * tile column (above when it is the tile's last column): intra edges never
 * cross tile columns. Each finished SB releases the SBs waiting on it, so
 * no global wave barrier idles the cores and the tile columns proceed in
 * parallel. */
static int recon_prereq(const struct Mav1Encoder *e, int r, int c, int *pre) {
  int ts = (c / e->tile_w_sb) * e->tile_w_sb, te = k_min(e->sb_cols, ts + e->tile_w_sb), n = 0;
  if (c > ts) pre[n++] = r * e->sb_cols + c - 1;
  if (r > 0) pre[n++] = (r - 1) * e->sb_cols + (c + 1 < te ? c + 1 : c);
  return n;
}

typedef struct {
  struct Mav1Encoder *e;
  _Atomic int *deps;
  int (*next)[3]; /* dependents: count, ids */
  dispatch_queue_t q;
  dispatch_group_t g;
  void (^run)(int);
} ReconSched;

void *recon_sched_begin(struct Mav1Encoder *e, int extra) {
  ReconSched *rs = calloc(1, sizeof *rs);
  int n = e->num_sbs;
  rs->e = e;
  rs->deps = calloc((size_t)n, sizeof *rs->deps);
  rs->next = calloc((size_t)n, sizeof *rs->next);
  for (int i = 0; i < n; i++) {
    int pre[2], k = recon_prereq(e, i / e->sb_cols, i % e->sb_cols, pre);
    rs->deps[i] = k + extra;
    for (int j = 0; j < k; j++) rs->next[pre[j]][1 + rs->next[pre[j]][0]++] = i;
  }
  rs->q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  rs->g = dispatch_group_create();
  const FrameParams *fp = ENC_PARAMS(e);
  _Atomic int *deps = rs->deps;
  int (*next)[3] = rs->next;
  dispatch_queue_t q = rs->q;
  dispatch_group_t g = rs->g;
  __block void (^run)(int);
  run = Block_copy(^(int idx) {
    recon_sb(fp, ENC_SBS(e)[idx], ENC_BLOCKS(e), e->src.base, e->rec.base, e->ref.base, e->gold.base, ENC_COEFS(e),
             (const int32_t *)e->mats.base, (int32_t *)e->scratch.base + (size_t)idx * SCRATCH_INTS);
    for (int j = 0; j < next[idx][0]; j++) {
      int d = next[idx][1 + j];
      if (atomic_fetch_sub(&deps[d], 1) == 1) dispatch_group_async(g, q, ^{ run(d); });
    }
  });
  rs->run = run;
  /* collect the ready set first: launched tasks decrement deps meanwhile */
  int *ready = malloc(sizeof(int) * (size_t)n), nr = 0;
  for (int i = 0; i < n; i++)
    if (rs->deps[i] == 0) ready[nr++] = i;
  for (int k = 0; k < nr; k++) {
    int i = ready[k];
    dispatch_group_async(g, q, ^{ run(i); });
  }
  free(ready);
  return rs;
}

void recon_sched_release(void *p, int sb0, int nsb) {
  ReconSched *rs = p;
  void (^run)(int) = rs->run;
  for (int i = sb0; i < sb0 + nsb; i++)
    if (atomic_fetch_sub(&rs->deps[i], 1) == 1) dispatch_group_async(rs->g, rs->q, ^{ run(i); });
}

void recon_sched_end(void *p) {
  ReconSched *rs = p;
  dispatch_group_wait(rs->g, DISPATCH_TIME_FOREVER);
  Block_release(rs->run);
  free((void *)rs->deps);
  free(rs->next);
  free(rs);
}

void cpu_backend_recon(Backend *b, struct Mav1Encoder *e) {
  (void)b;
  recon_sched_end(recon_sched_begin(e, 0));
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
  if (e->num_cdef_presets)
    dispatch_apply((size_t)(bc * br), DISPATCH_APPLY_AUTO, ^(size_t i) {
      cdef_block_all(fp, e->rec.base, e->cdefbuf.base, (long)e->frame_bytes, info, (int)(i / bc) * 2, (int)(i % bc) * 2,
                     e->num_cdef_presets);
    });
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
