/* Metal backend: the kernel sources shared with the CPU backend are compiled
 * at runtime as MSL; encoder buffers are wrapped without copies (they are
 * page aligned, unified memory). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "backend.h"
#include "enc_internal.h"
#include "kernels_src.h"
#include "kernels/k_analysis.h"
#include "kernels/k_me.h"
#include <dispatch/dispatch.h>

@interface Mav1MetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> analyze, analyzeCoop, analyzeL3, recon, me, meTg, hpel, lf, cdef, lr;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLBuffer>> *buffers;
@end
@implementation Mav1MetalState
@end

typedef struct {
  Backend base;
  void *state; /* retained Mav1MetalState */
  /* plane buffer entries: reference generation held (0: none), last use */
  unsigned qpGen[ME_QP_ENTRIES], qpUse[ME_QP_ENTRIES], qpClock;
  int qpLast, qpGold; /* entries of the current frame's references */
  int qpcHeld[2];     /* shared cache entries held by the frame (-1: none) */
} MetalBackend;

/* the plane buffer holding the current references' entries */
static const Buf *qp_buf(struct Mav1Encoder *e) { return e->qpc ? &e->qpc->buf : &e->hpel; }

static void qpc_drop(MetalBackend *m, struct Mav1Encoder *e) {
  for (int i = 0; i < 2; i++)
    if (m->qpcHeld[i] >= 0) {
      qpc_release(e->qpc, m->qpcHeld[i]);
      m->qpcHeld[i] = -1;
    }
}

static Mav1MetalState *ST(Backend *b) { return (__bridge Mav1MetalState *)((MetalBackend *)b)->state; }

/* Returns (and caches) a no-copy MTLBuffer for a shared encoder buffer. */
static id<MTLBuffer> wrap(Mav1MetalState *s, const Buf *buf) {
  NSNumber *key = @((uintptr_t)buf->base);
  id<MTLBuffer> m = s.buffers[key];
  if (!m || m.length != buf->size) {
    m = [s.device newBufferWithBytesNoCopy:buf->base
                                    length:buf->size
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
    s.buffers[key] = m;
  }
  return m;
}

/* The GOLDEN buffers exist only with multi_ref; otherwise bind LAST's. */
static const Buf *gold_buf(struct Mav1Encoder *e) { return e->gold.base ? &e->gold : &e->ref; }

static void run(Mav1MetalState *s, void (^encode)(id<MTLComputeCommandEncoder>)) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (getenv("MAV1_GPUT")) fprintf(stderr, "gpu %.2fms\n", (cb.GPUEndTime - cb.GPUStartTime) * 1000);
    if (getenv("MAV1_GPUTL")) fprintf(stderr, "gpu %p run %.2f %.2f\n", (__bridge void *)s, fmod(cb.GPUStartTime, 1000) * 1e3, fmod(cb.GPUEndTime, 1000) * 1e3);
    if (cb.error) fprintf(stderr, "mav1 metal: command buffer error: %s\n", cb.error.localizedDescription.UTF8String);
  }
}

/* Encoders of the motion search / analysis stages for superblocks
 * [sb0, sb0 + nsb) (bands of SB rows; the whole frame otherwise). */
/* Entry holding the planes of reference generation `gen` (0: never
 * cached), never `avoid`; *build when it must be (re)computed. */
static int qp_pick(MetalBackend *m, unsigned gen, int avoid, int *build) {
  m->qpClock++;
  for (int k = 0; k < ME_QP_ENTRIES; k++)
    if (gen && m->qpGen[k] == gen && k != avoid) {
      m->qpUse[k] = m->qpClock;
      *build = 0;
      return k;
    }
  int best = -1;
  for (int k = 0; k < ME_QP_ENTRIES; k++)
    if (k != avoid && (best < 0 || m->qpUse[k] < m->qpUse[best])) best = k;
  m->qpGen[best] = gen;
  m->qpUse[best] = m->qpClock;
  *build = 1;
  return best;
}

static void enc_qpel(Backend *b, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e) {
  MetalBackend *m = (MetalBackend *)b;
  Mav1MetalState *s = ST(b);
  const FrameParams *fp = ENC_PARAMS(e);
  long eb = qp_entry_bytes(fp);
  int buildLast, buildGold = 0;
  NSUInteger n = (NSUInteger)fp->frame_w * fp->frame_h;
  if (e->qpc) {
    /* shared entries: missing ones built by this frame (its own command
     * buffer, done before any other reader proceeds); then the decimated
     * source of this frame */
    m->qpcHeld[0] = m->qpLast = qpc_acquire(e->qpc, e->ref_gen, -1, &buildLast);
    m->qpcHeld[1] = -1;
    if (fp->golden) m->qpcHeld[1] = m->qpGold = qpc_acquire(e->qpc, e->gold_gen, m->qpLast, &buildGold);
    else m->qpGold = m->qpLast;
    if (buildLast || buildGold) {
      id<MTLCommandBuffer> cb = [s.queue commandBuffer];
      id<MTLComputeCommandEncoder> be = [cb computeCommandEncoder];
      [be setComputePipelineState:s.hpel];
      [be setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
      [be setBuffer:wrap(s, &e->src) offset:0 atIndex:4];
      [be setBuffer:wrap(s, &e->hpel) offset:(NSUInteger)(ME_QP_ENTRIES * eb) atIndex:5];
      for (int r = 0; r < 2; r++) {
        if (!(r ? buildGold : buildLast)) continue;
        int mode[2] = {1, 0};
        [be setBuffer:wrap(s, r ? gold_buf(e) : &e->ref) offset:0 atIndex:1];
        [be setBuffer:wrap(s, &e->qpc->buf) offset:(NSUInteger)((r ? m->qpGold : m->qpLast) * eb) atIndex:3];
        [be setBytes:mode length:sizeof mode atIndex:6];
        [be dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      [be endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (buildLast) qpc_ready(e->qpc, m->qpLast);
      if (buildGold) qpc_ready(e->qpc, m->qpGold);
    }
    int mode[2] = {0, 1};
    [enc setComputePipelineState:s.hpel];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:4];
    [enc setBuffer:wrap(s, &e->hpel) offset:(NSUInteger)(ME_QP_ENTRIES * eb) atIndex:5];
    [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->hpel) offset:0 atIndex:3];
    [enc setBytes:mode length:sizeof mode atIndex:6];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return;
  }
  m->qpLast = qp_pick(m, e->ref_gen, -1, &buildLast);
  m->qpGold = fp->golden ? qp_pick(m, e->gold_gen, m->qpLast, &buildGold) : m->qpLast;
  [enc setComputePipelineState:s.hpel];
  [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
  [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:4];
  [enc setBuffer:wrap(s, &e->hpel) offset:(NSUInteger)(ME_QP_ENTRIES * eb) atIndex:5];
  for (int r = 0; r < 2; r++) {
    int bld = r ? buildGold : buildLast;
    if (r && !bld) continue;
    int mode[2] = {bld, r == 0};
    [enc setBuffer:wrap(s, r ? gold_buf(e) : &e->ref) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->hpel) offset:(NSUInteger)((r ? m->qpGold : m->qpLast) * eb) atIndex:3];
    [enc setBytes:mode length:sizeof mode atIndex:6];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* prune: the analysis of the levels above is done; depth-pruned nodes are
 * skipped (frame core) */
static void enc_me(Backend *b, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e, int level, int sb0,
                   int nsb, int prune) {
  MetalBackend *mb = (MetalBackend *)b;
  Mav1MetalState *s = ST(b);
  long eb = qp_entry_bytes(ENC_PARAMS(e));
  [enc setComputePipelineState:s.me];
  [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
  [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
  [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:2];
  [enc setBuffer:wrap(s, &e->mvfield) offset:0 atIndex:3];
  [enc setBuffer:wrap(s, &e->mescratch) offset:0 atIndex:4];
  [enc setBuffer:wrap(s, gold_buf(e)) offset:0 atIndex:6];
  [enc setBuffer:wrap(s, &e->mvgold) offset:0 atIndex:7];
  [enc setBuffer:wrap(s, qp_buf(e)) offset:(NSUInteger)(mb->qpLast * eb) atIndex:8];
  [enc setBuffer:wrap(s, qp_buf(e)) offset:(NSUInteger)(mb->qpGold * eb) atIndex:10];
  [enc setBuffer:wrap(s, &e->hpel) offset:(NSUInteger)(ME_QP_ENTRIES * eb) atIndex:11];
  int band[2] = {sb0, nsb};
  [enc setBytes:band length:sizeof band atIndex:9];
  [enc setBuffer:wrap(s, &e->acosts) offset:0 atIndex:12];
  [enc setBytes:&prune length:sizeof prune atIndex:13];
  int nref = ENC_PARAMS(e)->golden ? 2 : 1;
  /* one SIMD group per node and reference */
  [enc setBytes:&level length:sizeof level atIndex:5];
  NSUInteger groups = ((NSUInteger)nsb << (2 * level)) * nref;
  static int tgw[2] = {-1, -1};
  if (tgw[0] < 0) {
    tgw[0] = getenv("MAV1_METG0") ? atoi(getenv("MAV1_METG0")) : 128;
    tgw[1] = getenv("MAV1_METG1") ? atoi(getenv("MAV1_METG1")) : 0;
  }
  if (level < 2 && tgw[level] >= 32) {
    /* large nodes: a threadgroup per node */
    [enc setComputePipelineState:s.meTg];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)tgw[level], 1, 1)];
  } else
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void enc_ana(Backend *b, Mav1MetalState *s, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e, int level, int sb0,
                    int nsb) {
  [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
  [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
  [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:2];
  [enc setBuffer:wrap(s, &e->mvfield) offset:0 atIndex:3];
  [enc setBuffer:wrap(s, &e->acosts) offset:0 atIndex:4];
  [enc setBuffer:wrap(s, &e->ainfos) offset:0 atIndex:5];
  [enc setBuffer:wrap(s, &e->ascratch) offset:0 atIndex:6];
  [enc setBuffer:wrap(s, gold_buf(e)) offset:0 atIndex:8];
  [enc setBuffer:wrap(s, &e->mvgold) offset:0 atIndex:9];
  int band[2] = {sb0, nsb};
  [enc setBytes:band length:sizeof band atIndex:10];
  [enc setBytes:&level length:sizeof level atIndex:7];
  const FrameParams *fp = ENC_PARAMS(e);
  /* inter luma predictions from the motion search's quarter-pel planes */
  {
    MetalBackend *mb = (MetalBackend *)b;
    int useHp = !fp->frame_is_intra && (fp->turbo & T_ME_HPEL) && !getenv("MAV1_NOAHP");
    long eb = qp_entry_bytes(fp);
    const Buf *qb = e->qpc ? &e->qpc->buf : &e->hpel;
    [enc setBuffer:wrap(s, qb) offset:(NSUInteger)(useHp ? mb->qpLast * eb : 0) atIndex:11];
    [enc setBuffer:wrap(s, qb) offset:(NSUInteger)(useHp ? mb->qpGold * eb : 0) atIndex:12];
    [enc setBytes:&useHp length:sizeof useHp atIndex:13];
  }
  NSUInteger n = (NSUInteger)nsb * (1u << (2 * level)) * level_slots(fp, level);
  if (level == 3 && (fp->turbo & T_INTRA) && fp->fast && !fp->frame_is_intra && !getenv("MAV1_NOL3O")) {
    /* 8x8: one SIMD group per 16x16 node, 8 lanes per block */
    [enc setComputePipelineState:s.analyzeL3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nsb * 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  } else if (level <= 2) {
    /* blocks >= 16x16: one SIMD group per candidate block */
    [enc setComputePipelineState:s.analyzeCoop];
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  } else {
    [enc setComputePipelineState:s.analyze];
    NSUInteger tg = MIN((NSUInteger)64, s.analyze.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void mtl_motion_search(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  int one = getenv("MAV1_MELEVEL") ? 1 : 0; /* profiling: one command buffer per level (and the planes) */
  if (one && (ENC_PARAMS(e)->turbo & T_ME_HPEL)) run(s, ^(id<MTLComputeCommandEncoder> enc) { enc_qpel(b, enc, e); });
  for (int L = 0; L <= (one ? 3 : 0); L++)
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    if ((ENC_PARAMS(e)->turbo & T_ME_HPEL) && L == 0 && !one) enc_qpel(b, enc, e);
    /* levels depend on their parents: one dispatch each */
    for (int level = one ? L : 0; level < (one ? L + 1 : 4); level++) enc_me(b, enc, e, level, 0, e->num_sbs, 0);
  });
}

static void mtl_analyze(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  int one = getenv("MAV1_ALEVEL") ? 1 : 0; /* profiling: one command buffer per level */
  for (int L = 0; L <= (one ? 3 : 0); L++) {
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    /* levels are independent; they share the scratch buffer, so they run as
     * consecutive (serial) dispatches */
    for (int level = one ? L : 0; level <= (one ? L : 3); level++) enc_ana(b, s, enc, e, level, 0, e->num_sbs);
  });
  }
  if (e->qpc) qpc_drop((MetalBackend *)b, e); /* the planes are read by the analysis too */
  /* pass 2 (partition decision on the table) on the CPU cores */
  const FrameParams *fp = ENC_PARAMS(e);
  if (getenv("MAV1_DBG")) {
    long *gc = malloc(sizeof(long) * A_ITEMS * e->num_sbs);
    memcpy(gc, e->acosts.base, sizeof(long) * A_ITEMS * e->num_sbs);
    cpu_backend_analyze(b, e);
    long *cc = (long *)e->acosts.base;
    int bad = 0;
    for (long i = 0; i < (long)A_ITEMS * e->num_sbs && bad < 5; i++)
      if (gc[i] != cc[i]) { fprintf(stderr, "item %ld (sb %ld node %ld slot %ld): gpu %ld cpu %ld\n", i, i / A_ITEMS, (i % A_ITEMS) / A_SLOTS, i % A_SLOTS, gc[i], cc[i]); bad++; }
    free(gc);
  }
  dispatch_apply((size_t)e->num_sbs, DISPATCH_APPLY_AUTO, ^(size_t i) {
    analyze_sb_table(fp, (int)i, (const long *)e->acosts.base, (const BlockInfo *)e->ainfos.base, ENC_BLOCKS(e),
                     ENC_SBS(e));
  });
}

/* Turbo frame core: motion search, analysis and reconstruction overlap.
 * Levels 0-1 of both stages run on the whole frame; levels 2-3 run in bands
 * of SB rows, one command buffer each. As each band completes, the CPU
 * makes its partition decisions and edge types and releases its
 * superblocks to the dependency-driven reconstruction, while the GPU works
 * on the next band. The results are those of the sequential stages. */
/* profiling (MAV1_KPROF): GPU ms per stage, reference / leaf frames */
static double kprof[2][12];
static int kprof_n[2];
static void kprof_dump(void) {
  static const char *nm[12] = {"qpel", "me0", "ana0", "me1", "ana1", "me2", "ana2", "me3", "ana3", "filter", "", ""};
  for (int r = 0; r < 2; r++) {
    fprintf(stderr, "%s frames %d:", r ? "leaf" : "ref", kprof_n[r]);
    for (int i = 0; i < 10; i++) fprintf(stderr, " %s %.2f", nm[i], kprof[r][i] / (kprof_n[r] ? kprof_n[r] : 1));
    fprintf(stderr, "\n");
  }
}
static double prof_run(Mav1MetalState *s, void (^encode)(id<MTLComputeCommandEncoder>)) {
  id<MTLCommandBuffer> cb = [s.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encode(enc);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}
static int kprof_on(void) {
  static int on = -1;
  if (on < 0) {
    on = getenv("MAV1_KPROF") != NULL;
    if (on) atexit(kprof_dump);
  }
  return on;
}

static void mtl_frame_core(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  const FrameParams *fp = ENC_PARAMS(e);
  int inter = !fp->frame_is_intra;
  if (kprof_on() && inter) {
    int pr0 = fp->fast && fp->depth_thr > 0, r = e->nonref != 0;
    @autoreleasepool {
      if (fp->turbo & T_ME_HPEL) { double t = prof_run(s, ^(id<MTLComputeCommandEncoder> enc) { enc_qpel(b, enc, e); }); kprof[r][0] += t; }
      for (int l = 0; l < 4; l++) {
        kprof[r][1 + 2 * l] += prof_run(s, ^(id<MTLComputeCommandEncoder> enc) { enc_me(b, enc, e, l, 0, e->num_sbs, pr0); });
        kprof[r][2 + 2 * l] += prof_run(s, ^(id<MTLComputeCommandEncoder> enc) { enc_ana(b, s, enc, e, l, 0, e->num_sbs); });
      }
    }
    kprof_n[r]++;
    if (e->qpc) qpc_drop((MetalBackend *)b, e);
    dispatch_apply((size_t)e->num_sbs, DISPATCH_APPLY_AUTO, ^(size_t i) {
      analyze_sb_table(fp, (int)i, (const long *)e->acosts.base, (const BlockInfo *)e->ainfos.base, ENC_BLOCKS(e), ENC_SBS(e));
    });
    if (e->cfg.intra_edge_filter) {
      uint8_t *ys = calloc((size_t)e->mi_rows * e->mi_cols, 1), *uvs = calloc((size_t)e->mi_rows * e->mi_cols, 1);
      edge_types_rows(e, ys, uvs, 0, e->sb_rows);
      free(ys);
      free(uvs);
    }
    e->be->recon(e->be, e);
    if (e->gate_out) { dispatch_semaphore_signal((__bridge dispatch_semaphore_t)e->gate_out); e->gate_out = NULL; }
    return;
  }
  int rowsPer = getenv("MAV1_BAND") ? atoi(getenv("MAV1_BAND")) : e->sb_rows >= 16 ? 4 : 2; /* bigger bands from 1080p on */
  if (rowsPer < 1) rowsPer = 1;
  int nb = (e->sb_rows + rowsPer - 1) / rowsPer;
  int pr = fp->fast && fp->depth_thr > 0;
  /* levels below lw run on the whole frame before the bands */
  int lw = getenv("MAV1_LW") ? atoi(getenv("MAV1_LW")) : 2;
  NSMutableArray<dispatch_semaphore_t> *done = [NSMutableArray arrayWithCapacity:(NSUInteger)nb + 1];
  for (int k = 0; k <= nb; k++) [done addObject:dispatch_semaphore_create(0)];
  @autoreleasepool {
    for (int k = -1; k < nb; k++) {
      id<MTLCommandBuffer> cb = [s.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      if (k < 0) {
        if (inter && (fp->turbo & T_ME_HPEL)) enc_qpel(b, enc, e);
        /* each level's analysis before the next level's search (depth pruning) */
        for (int l = 0; l < lw; l++) {
          if (inter) enc_me(b, enc, e, l, 0, e->num_sbs, pr);
          enc_ana(b, s, enc, e, l, 0, e->num_sbs);
        }
      } else {
        int sb0 = k * rowsPer * e->sb_cols, nsb = k_min(e->num_sbs - sb0, rowsPer * e->sb_cols);
        for (int l = lw; l < 4; l++) {
          if (inter) enc_me(b, enc, e, l, sb0, nsb, pr);
          enc_ana(b, s, enc, e, l, sb0, nsb);
        }
      }
      [enc endEncoding];
      dispatch_semaphore_t d = done[k + 1];
      int kk = k;
      [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if (getenv("MAV1_GPUTL")) fprintf(stderr, "gpu %p cb%d %.2f %.2f\n", (void *)b, kk, fmod(c.GPUStartTime, 1000) * 1e3, fmod(c.GPUEndTime, 1000) * 1e3);
        if (c.error) fprintf(stderr, "mav1 metal: command buffer error: %s\n", c.error.localizedDescription.UTF8String);
        dispatch_semaphore_signal(d);
      }];
      [cb commit];
    }
  }
  int cols = e->mi_cols, rows = e->mi_rows;
  uint8_t *ys = NULL, *uvs = NULL;
  if (e->cfg.intra_edge_filter) {
    ys = calloc((size_t)rows * cols, 1);
    uvs = calloc((size_t)rows * cols, 1);
  }
  int ct = getenv("MAV1_CT") != NULL;
  double c0 = CFAbsoluteTimeGetCurrent();
  void *rs = recon_sched_begin(e, 1);
  dispatch_semaphore_wait(done[0], DISPATCH_TIME_FOREVER);
  if (ct) fprintf(stderr, "core: cb0 %.1f", (CFAbsoluteTimeGetCurrent() - c0) * 1e3);
  for (int k = 0; k < nb; k++) {
    dispatch_semaphore_wait(done[k + 1], DISPATCH_TIME_FOREVER);
    if (ct) fprintf(stderr, " b%d %.1f", k, (CFAbsoluteTimeGetCurrent() - c0) * 1e3);
    int sb0 = k * rowsPer * e->sb_cols, nsb = k_min(e->num_sbs - sb0, rowsPer * e->sb_cols);
    dispatch_apply((size_t)nsb, DISPATCH_APPLY_AUTO, ^(size_t i) {
      analyze_sb_table(fp, sb0 + (int)i, (const long *)e->acosts.base, (const BlockInfo *)e->ainfos.base,
                       ENC_BLOCKS(e), ENC_SBS(e));
    });
    if (ys) edge_types_rows(e, ys, uvs, k * rowsPer, k_min(e->sb_rows, (k + 1) * rowsPer));
    recon_sched_release(rs, sb0, nsb);
  }
  if (e->qpc) qpc_drop((MetalBackend *)b, e); /* the motion search is done */
  if (e->gate_out) { /* GPU stages done: a waiting non-reference frame may start */
    dispatch_semaphore_signal((__bridge dispatch_semaphore_t)e->gate_out);
    e->gate_out = NULL;
  }
  if (ct) fprintf(stderr, " rel %.1f", (CFAbsoluteTimeGetCurrent() - c0) * 1e3);
  recon_sched_end(rs);
  if (ct) fprintf(stderr, " end %.1f\n", (CFAbsoluteTimeGetCurrent() - c0) * 1e3);
  free(ys);
  free(uvs);
}

static void mtl_recon(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:s.recon];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->rec) offset:0 atIndex:2];
    [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:3];
    [enc setBuffer:wrap(s, &e->blocks) offset:0 atIndex:4];
    [enc setBuffer:wrap(s, &e->sbs) offset:0 atIndex:5];
    [enc setBuffer:wrap(s, &e->coefs) offset:0 atIndex:6];
    [enc setBuffer:wrap(s, &e->mats) offset:0 atIndex:7];
    [enc setBuffer:wrap(s, &e->scratch) offset:0 atIndex:8];
    [enc setBuffer:wrap(s, gold_buf(e)) offset:0 atIndex:10];
    NSUInteger tg = MIN((NSUInteger)32, s.recon.maxTotalThreadsPerThreadgroup);
    int waves = e->sb_cols + 2 * (e->sb_rows - 1);
    /* Serial dispatch type: each wave sees the previous waves' writes. */
    for (int t = 0; t < waves; t++) {
      int rmin = t - (e->sb_cols - 1);
      rmin = rmin > 0 ? (rmin + 1) / 2 : 0;
      int rmax = t / 2 < e->sb_rows - 1 ? t / 2 : e->sb_rows - 1;
      if (rmax < rmin) continue;
      int wave[2] = {t, rmin};
      [enc setBytes:wave length:sizeof wave atIndex:9];
      [enc dispatchThreads:MTLSizeMake(rmax - rmin + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  });
}

static void mtl_filter(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:s.lf];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->rec) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->lfmask) offset:0 atIndex:2];
    NSUInteger tg = MIN((NSUInteger)64, s.lf.maxTotalThreadsPerThreadgroup);
    for (int pass = 0; pass < 2; pass++) {
      int lines = pass == 0 ? e->mi_rows : e->mi_cols, segs = ((pass == 0 ? e->mi_cols : e->mi_rows) + 15) >> 4;
      [enc setBytes:&pass length:sizeof pass atIndex:3];
      [enc dispatchThreads:MTLSizeMake(3 * lines * segs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    if (e->num_cdef_presets) {
      int bc = (e->mi_cols + 1) >> 1, br = (e->mi_rows + 1) >> 1;
      [enc setComputePipelineState:s.cdef];
      [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
      [enc setBuffer:wrap(s, &e->rec) offset:0 atIndex:1];
      [enc setBuffer:wrap(s, &e->lfmask) offset:0 atIndex:3];
      NSUInteger tc = MIN((NSUInteger)64, s.cdef.maxTotalThreadsPerThreadgroup);
      int np = e->num_cdef_presets;
      long fb = (long)e->frame_bytes;
      [enc setBuffer:wrap(s, &e->cdefbuf) offset:0 atIndex:2];
      [enc setBytes:&np length:sizeof np atIndex:4];
      [enc setBytes:&fb length:sizeof fb atIndex:5];
      [enc dispatchThreads:MTLSizeMake(bc * br, 1, 1) threadsPerThreadgroup:MTLSizeMake(tc, 1, 1)];
    }
  });
}

static void mtl_restore(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:s.lr];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->rec) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->dbk) offset:0 atIndex:2];
    [enc setBuffer:wrap(s, &e->lrout) offset:0 atIndex:3];
    [enc setBuffer:wrap(s, &e->lrunits) offset:0 atIndex:4];
    NSUInteger n = (NSUInteger)3 * ((e->h + 3) >> 2) * ((e->w + 3) >> 2);
    NSUInteger tg = MIN((NSUInteger)64, s.lr.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  });
}

static void mtl_destroy(Backend *b) {
  MetalBackend *m = (MetalBackend *)b;
  if (m->state) CFRelease(m->state);
  free(m);
}

Backend *backend_metal_create(struct Mav1Encoder *e, char *err, size_t errlen) {
  (void)e;
  @autoreleasepool {
    /* the library and pipelines are built once per process; every backend
     * (frame engine) has its own command queue and buffer wrappers */
    static Mav1MetalState *proto;
    static dispatch_once_t once;
    static char perr[256];
    dispatch_once(&once, ^{
      Mav1MetalState *s = [Mav1MetalState new];
      perr[0] = 0;
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev) {
        snprintf(perr, sizeof perr, "no Metal device");
        return;
      }
      double t0i = CFAbsoluteTimeGetCurrent();
      MTLCompileOptions *opt = [MTLCompileOptions new];
      opt.mathMode = MTLMathModeSafe;
      NSError *nerr = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:@(kMav1KernelSource) options:opt error:&nerr];
      if (!lib) {
        snprintf(perr, sizeof perr, "Metal kernel compilation failed: %s", nerr.localizedDescription.UTF8String);
        return;
      }
      if (getenv("MAV1_TL")) fprintf(stderr, "init: device+compile %.1f ms\n", (CFAbsoluteTimeGetCurrent() - t0i) * 1e3);
      s.device = dev;
      s.analyze = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_analyze"] error:&nerr];
      s.analyzeCoop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_analyze_coop"] error:&nerr];
      s.recon = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_recon"] error:&nerr];
      s.analyzeL3 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_analyze_l3o"] error:&nerr];
      s.me = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_me"] error:&nerr];
      s.meTg = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_me_tg"] error:&nerr];
      s.hpel = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_hpel"] error:&nerr];
      s.lf = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lf"] error:&nerr];
      s.cdef = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_cdef"] error:&nerr];
      s.lr = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lr"] error:&nerr];
      if (!s.analyze || !s.analyzeCoop || !s.analyzeL3 || !s.meTg || !s.recon || !s.me || !s.hpel || !s.lf || !s.cdef || !s.lr) {
        snprintf(perr, sizeof perr, "pipeline creation failed: %s", nerr.localizedDescription.UTF8String);
        return;
      }
      proto = s;
      if (getenv("MAV1_TL")) fprintf(stderr, "init: pipelines %.1f ms\n", (CFAbsoluteTimeGetCurrent() - t0i) * 1e3);
      if (getenv("MAV1_PSO"))
        fprintf(stderr, "maxThreads: analyze %lu coop %lu recon %lu me %lu hpel %lu lf %lu cdef %lu\n",
                (unsigned long)s.analyze.maxTotalThreadsPerThreadgroup, (unsigned long)s.analyzeCoop.maxTotalThreadsPerThreadgroup,
                (unsigned long)s.recon.maxTotalThreadsPerThreadgroup, (unsigned long)s.me.maxTotalThreadsPerThreadgroup,
                (unsigned long)s.hpel.maxTotalThreadsPerThreadgroup, (unsigned long)s.lf.maxTotalThreadsPerThreadgroup,
                (unsigned long)s.cdef.maxTotalThreadsPerThreadgroup);
    });
    if (!proto) {
      snprintf(err, errlen, "%s", perr);
      return NULL;
    }
    Mav1MetalState *s = [Mav1MetalState new];
    s.device = proto.device;
    s.analyze = proto.analyze;
    s.analyzeCoop = proto.analyzeCoop;
    s.analyzeL3 = proto.analyzeL3;
    s.recon = proto.recon;
    s.me = proto.me;
    s.meTg = proto.meTg;
    s.hpel = proto.hpel;
    s.lf = proto.lf;
    s.cdef = proto.cdef;
    s.lr = proto.lr;
    s.queue = [s.device newCommandQueue];
    s.buffers = [NSMutableDictionary new];
    MetalBackend *m = calloc(1, sizeof(*m));
    m->base.name = "metal";
    m->qpcHeld[0] = m->qpcHeld[1] = -1;
    m->base.motion_search = mtl_motion_search;
    m->base.analyze = getenv("MAV1_CPU_ANALYZE") ? cpu_backend_analyze : mtl_analyze;
    /* Hybrid: GPU for the massively parallel stages (motion search,
   * analysis, filters), CPU for the wavefront reconstruction. MAV1_GPU_RECON=1
   * keeps the GPU wavefront (for the backend comparison tests). */
  m->base.recon = getenv("MAV1_GPU_RECON") ? mtl_recon : cpu_backend_recon;
    m->base.filter = mtl_filter;
  m->base.restore = mtl_restore;
    m->base.destroy = mtl_destroy;
    m->base.frame_core = getenv("MAV1_GPU_RECON") || getenv("MAV1_CPU_ANALYZE") || getenv("MAV1_NOCORE") ? NULL : mtl_frame_core;
    m->state = (void *)CFBridgingRetain(s);
    return &m->base;
  }
}
