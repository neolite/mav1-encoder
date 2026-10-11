/*
 * The whole shared-kernel set for one bitdepth instantiation, in dependency
 * order. The pixel headers carry no include guards of their own (they are
 * compiled once per MAV1_BD; both instantiations coexist in one translation
 * unit via KNAME-suffixed functions), so every consumer includes the set
 * through this header and never the kernel headers directly. The Metal
 * source string is built from the same include list (tools/embed_kernels.py
 * inlines each file once).
 */
#include "k_common.h"
#include "k_itx.h"
#include "k_scan.h"
#include "k_fwd.h"
#include "k_intra.h"
#include "k_inter.h"
#include "k_analysis.h"
#include "k_me.h"
#include "k_recon.h"
#include "k_lf.h"
#include "k_cdef.h"
#include "k_lr.h"
