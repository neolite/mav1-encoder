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
#include <dispatch/dispatch.h>

@interface Mav1MetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> analyze, analyzeCoop, recon, me, lf, cdef, lr;
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

static void mtl_motion_search(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:s.me];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:2];
    [enc setBuffer:wrap(s, &e->mvfield) offset:0 atIndex:3];
    [enc setBuffer:wrap(s, &e->mescratch) offset:0 atIndex:4];
    /* levels depend on their parents: one dispatch each, one SIMD group per node */
    for (int level = 0; level < 4; level++) {
      [enc setBytes:&level length:sizeof level atIndex:5];
      NSUInteger groups = (NSUInteger)e->num_sbs << (2 * level);
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  });
}

static void mtl_analyze(Backend *b, struct Mav1Encoder *e) {
  Mav1MetalState *s = ST(b);
  int one = getenv("MAV1_ALEVEL") ? 1 : 0; /* profiling: one command buffer per level */
  for (int L = 3; L >= (one ? 0 : 3); L--) {
  run(s, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:s.analyze];
    [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
    [enc setBuffer:wrap(s, &e->src) offset:0 atIndex:1];
    [enc setBuffer:wrap(s, &e->ref) offset:0 atIndex:2];
    [enc setBuffer:wrap(s, &e->mvfield) offset:0 atIndex:3];
    [enc setBuffer:wrap(s, &e->acosts) offset:0 atIndex:4];
    [enc setBuffer:wrap(s, &e->ainfos) offset:0 atIndex:5];
    [enc setBuffer:wrap(s, &e->ascratch) offset:0 atIndex:6];
    NSUInteger tg = MIN((NSUInteger)64, s.analyze.maxTotalThreadsPerThreadgroup);
    const FrameParams *fp = ENC_PARAMS(e);
    /* levels are independent; they share the scratch buffer, so they run as
     * consecutive (serial) dispatches */
    for (int level = one ? L : 3; level >= (one ? L : 0); level--) {
      [enc setBytes:&level length:sizeof level atIndex:7];
      NSUInteger n = (NSUInteger)e->num_sbs * (1u << (2 * level)) * level_slots(fp, level);
      if (level <= 1) {
        /* big blocks: one SIMD group per candidate block */
        [enc setComputePipelineState:s.analyzeCoop];
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      } else {
        [enc setComputePipelineState:s.analyze];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
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
      int lines = pass == 0 ? e->mi_rows : e->mi_cols;
      [enc setBytes:&pass length:sizeof pass atIndex:3];
      [enc dispatchThreads:MTLSizeMake(3 * lines, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    if (e->num_cdef_presets) {
      int bc = (e->mi_cols + 1) >> 1, br = (e->mi_rows + 1) >> 1;
      [enc setComputePipelineState:s.cdef];
      [enc setBuffer:wrap(s, &e->params) offset:0 atIndex:0];
      [enc setBuffer:wrap(s, &e->rec) offset:0 atIndex:1];
      [enc setBuffer:wrap(s, &e->lfmask) offset:0 atIndex:3];
      NSUInteger tc = MIN((NSUInteger)64, s.cdef.maxTotalThreadsPerThreadgroup);
      for (int k = 0; k < e->num_cdef_presets; k++) {
        [enc setBuffer:wrap(s, &e->cdefbuf) offset:(NSUInteger)k * e->frame_bytes atIndex:2];
        [enc setBytes:&k length:sizeof k atIndex:4];
        [enc dispatchThreads:MTLSizeMake(bc * br, 1, 1) threadsPerThreadgroup:MTLSizeMake(tc, 1, 1)];
      }
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
    s.lf = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lf"] error:&nerr];
    s.cdef = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_cdef"] error:&nerr];
    s.lr = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mav1_lr"] error:&nerr];
    if (!s.analyze || !s.analyzeCoop || !s.recon || !s.me || !s.lf || !s.cdef || !s.lr) {
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
    m->state = (void *)CFBridgingRetain(s);
    return &m->base;
  }
}
