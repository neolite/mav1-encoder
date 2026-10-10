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
@property(nonatomic, strong) id<MTLComputePipelineState> analyze, analyzeCoop, recon, me, hpel, lf, cdef, lr;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLBuffer>> *buffers;
@end
@implementation Mav1MetalState
@end

typedef struct {
  Backend base;
  void *state; /* retained Mav1MetalState */
} MetalBackend;

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
    if (cb.error) fprintf(stderr, "mav1 metal: command buffer error: %s\n", cb.error.localizedDescription.UTF8String);
  }
}

/* Encoders of the motion search / analysis stages for superblocks
 * [sb0, sb0 + nsb) (bands of SB rows; the whole frame otherwise). */
static void enc_qpel(Mav1MetalState *s, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e) {
  const FrameParams *fp = ENC_PARAMS(e);
  [enc setComputePipelineState:s.hpel];
  [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
  [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:1];
  [enc setBuffer:wrap(s, gold_buf(e)) offset:0 atIndex:2];
  [enc setBuffer:wrap(s, &e->hpel) offset:0 atIndex:3];
  [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:4];
  NSUInteger n = (NSUInteger)2 * fp->frame_w * fp->frame_h;
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void enc_me(Mav1MetalState *s, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e, int level, int sb0,
                   int nsb) {
  [enc setComputePipelineState:s.me];
  [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
  [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
  [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:2];
  [enc setBuffer:wrap(s, &e->mvfield) offset:0 atIndex:3];
  [enc setBuffer:wrap(s, &e->mescratch) offset:0 atIndex:4];
  [enc setBuffer:wrap(s, gold_buf(e)) offset:0 atIndex:6];
  [enc setBuffer:wrap(s, &e->mvgold) offset:0 atIndex:7];
  [enc setBuffer:wrap(s, &e->hpel) offset:0 atIndex:8];
  int band[2] = {sb0, nsb};
  [enc setBytes:band length:sizeof band atIndex:9];
  int nref = ENC_PARAMS(e)->golden ? 2 : 1;
  /* one SIMD group per node and reference */
  [enc setBytes:&level length:sizeof level atIndex:5];
  NSUInteger groups = ((NSUInteger)nsb << (2 * level)) * nref;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void enc_ana(Mav1MetalState *s, id<MTLComputeCommandEncoder> enc, struct Mav1Encoder *e, int level, int sb0,
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
  NSUInteger n = (NSUInteger)nsb * (1u << (2 * level)) * level_slots(fp, level);
  if (level <= 2) {
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
  if (one && (ENC_PARAMS(e)->turbo & T_ME_HPEL)) run(s, ^(id<MTLComputeCommandEncoder> enc) { enc_qpel(s, enc, e); });
  for (int L = 0; L <= (one ? 3 : 0); L++)
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    if ((ENC_PARAMS(e)->turbo & T_ME_HPEL) && L == 0 && !one) enc_qpel(s, enc, e);
    /* levels depend on their parents: one dispatch each */
    for (int level = one ? L : 0; level < (one ? L + 1 : 4); level++) enc_me(s, enc, e, level, 0, e->num_sbs);
  });
}

static void mtl_analyze(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  int one = getenv("MAV1_ALEVEL") ? 1 : 0; /* profiling: one command buffer per level */
  for (int L = 0; L <= (one ? 3 : 0); L++) {
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    /* levels are independent; they share the scratch buffer, so they run as
     * consecutive (serial) dispatches */
    for (int level = one ? L : 0; level <= (one ? L : 3); level++) enc_ana(s, enc, e, level, 0, e->num_sbs);
  });
  }
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
static void mtl_frame_core(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  const FrameParams *fp = ENC_PARAMS(e);
  int inter = !fp->frame_is_intra;
  int rowsPer = getenv("MAV1_BAND") ? atoi(getenv("MAV1_BAND")) : 2;
  if (rowsPer < 1) rowsPer = 1;
  int nb = (e->sb_rows + rowsPer - 1) / rowsPer;
  NSMutableArray<dispatch_semaphore_t> *done = [NSMutableArray arrayWithCapacity:(NSUInteger)nb + 1];
  for (int k = 0; k <= nb; k++) [done addObject:dispatch_semaphore_create(0)];
  @autoreleasepool {
    for (int k = -1; k < nb; k++) {
      id<MTLCommandBuffer> cb = [s.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      if (k < 0) {
        if (inter && (fp->turbo & T_ME_HPEL)) enc_qpel(s, enc, e);
        for (int l = 0; l < 2; l++) if (inter) enc_me(s, enc, e, l, 0, e->num_sbs);
        for (int l = 0; l < 2; l++) enc_ana(s, enc, e, l, 0, e->num_sbs);
      } else {
        int sb0 = k * rowsPer * e->sb_cols, nsb = k_min(e->num_sbs - sb0, rowsPer * e->sb_cols);
        for (int l = 2; l < 4; l++) if (inter) enc_me(s, enc, e, l, sb0, nsb);
        for (int l = 2; l < 4; l++) enc_ana(s, enc, e, l, sb0, nsb);
      }
      [enc endEncoding];
      dispatch_semaphore_t d = done[k + 1];
      [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
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
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      snprintf(err, errlen, "no Metal device");
      return NULL;
    }
    MTLCompileOptions *opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeSafe;
    NSError *nerr = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kMav1KernelSource) options:opt error:&nerr];
    if (!lib) {
      snprintf(err, errlen, "Metal kernel compilation failed: %s", nerr.localizedDescription.UTF8String);
      return NULL;
    }
    Mav1MetalState *s = [Mav1MetalState new];
    s.device = dev;
    s.queue = [dev newCommandQueue];
    s.buffers = [NSMutableDictionary new];
    s.analyze = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_analyze"] error:&nerr];
    s.analyzeCoop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_analyze_coop"] error:&nerr];
    s.recon = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_recon"] error:&nerr];
    s.me = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_me"] error:&nerr];
    s.hpel = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_hpel"] error:&nerr];
    s.lf = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lf"] error:&nerr];
    s.cdef = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_cdef"] error:&nerr];
    s.lr = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lr"] error:&nerr];
    if (!s.analyze || !s.analyzeCoop || !s.recon || !s.me || !s.hpel || !s.lf || !s.cdef || !s.lr) {
      snprintf(err, errlen, "pipeline creation failed: %s", nerr.localizedDescription.UTF8String);
      return NULL;
    }
    MetalBackend *m = calloc(1, sizeof(*m));
    m->base.name = "metal";
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
