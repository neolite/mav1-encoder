# mav1 `--turbo` vs SVT-AV1 preset 8: profile, headroom, decisions

Date: 2026-10-11. Machine: Apple M1 Pro (8 cores). Clip: crowd_run, first 60 frames, 720p (bicubic downscale) and 1080p50.
Raw profiles and scripts: [`docs/svtprof/`](svtprof/) (file list at the end).

## 1. Current state (mav1 `--turbo`)

| | SVT-AV1 p8 (v3.1.2) | mav1 `--turbo` | mav1 BD-rate vs SVT |
|---|---|---|---|
| 720p, fps (mean of 4 q) | 84 | ≈92 | +7.7% |
| 1080p, fps (mean of 4 q) | 48 | ≈50 | +9.8% |

BD-rate is piecewise-linear log(bytes) against PSNR Y over the overlapping range, with SVT p8 as the anchor. +9.8% means mav1 needs 9.8% more bytes for the same PSNR.

These are full-process fps, including startup. Checked independently: on 60 frames, 720p takes 0.59–0.66 s and 1080p takes 1.04–1.41 s. At the heavy q70 point mav1 is slower than SVT at 1080p (≈42 vs 48 fps); the mean is above SVT because of the low-rate points.

All the speed changes keep the output exact:
- recon is bit-identical with dav1d and aomdec;
- the Metal bitstream equals the CPU one byte for byte;
- `tools/regress.sh` reports 12/12 for both default and `--turbo`.

### How we got here

| Stage | 720p fps | Quality |
|---|---|---|
| Start of this goal (`--fast`, 10 frames) | ~4.8 | 0.5–1 dB below SVT |
| Cooperative GPU analysis/ME, exact CfL in one pass | ~11.6 | |
| `--altref 4 --leaf` (pyramid, compound LAST+ALTREF) | 9.1 | BD +4.8% |
| `--turbo` v1 (square partitions, 1/4-res coarse ME, quarter-pel planes, fewer intra modes, banded GPU core) | 35–37 | BD +5.0% |
| Round 2 (writer queue, 3 frame engines, fused kernels) | ~60 | same |
| Round 3 (depth pruning by ancestor NONE cost, 2 tiles, half-pel at 64/32, recon skip of zero rows) | 80–82 | BD +7.7% / +9.8% |
| Round 4 (concurrent entropy, shared planes, QoS, inlined entropy coder, no 16 KB clear per tx block) | **≈92** | unchanged (bitstream identical) |

## 2. SVT-AV1 p8 profile

Method:
- **Rate matching:** mav1 `-q` is a qindex, not a crf, so bitrates are matched instead (mav1 q125 ≈ SVT crf32, q180 ≈ crf46).
- **Timing:** 3 runs each. "Steady fps" has startup (measured with `-n 1`) subtracted.
- **Hotspots:** `sample` at 1 ms with `--lp 1`. Per-stage ME numbers varied up to 2x between runs, probably from P-core vs E-core scheduling.

| clip | wall | user+sys | cores used | steady fps |
|---|---|---|---|---|
| 720p crf32 | 0.68 s | 3.1 s | 4.6 (57%) | ~105 |
| 720p crf46 | 0.62 s | 2.9 s | 4.7 | ~118 |
| 1080p crf32 | 1.21 s | 6.8 s | 5.6 (70%) | ~58 |
| 1080p crf46 | 1.08 s | 6.1 s | 5.6 | ~66 |

**Thread scaling** (1080p crf32): `--lp 1` 5.46 s, `--lp 2` 1.98 s, `--lp 4` 1.27 s, `--lp 8` 1.40 s, default 1.23 s. SVT stops scaling at about 4 threads. On a short clip it is limited by its pipeline and semaphore waits, not by compute.

**Hotspots** (single thread, 1080p crf32, 80 ms/frame total):
- **Mode decision:** 70%, 55.6 ms.
  - Regular PD1: 38.6 ms, of which `full_loop_core` is 25.4 ms and `quantize_inv_quantize` 8.3 ms.
  - Light PD1: 11.3 ms. Light PD0: 3.3 ms.
  - `encdec_update` (final encode + deblock): 1.5 ms.
- **ME/HME and temporal filter:** 8 ms (TF 1.1 ms).
- **Entropy:** 7.7 ms at crf32, 3.0 ms at crf46.
- **Smaller stages:** LR 3.3 ms, TPL 2.4 ms, CDEF 1.9 ms, picture analysis and RC under 1 ms.

720p has the same shape: 37 ms/frame in total, of which MD is 26 ms.

## 3. mav1 `--turbo` profile

| clip | wall | user+sys | steady fps | GPU busy | GPU ms/frame | CPU ms/frame |
|---|---|---|---|---|---|---|
| 720p q125 | 0.63 s | 2.5 s | ~115 | 87% | 7.75 | 37 |
| 720p q180 | 0.59 s | 1.9 s | | 86% | 7.6 | |
| 1080p q125 | 1.17 s | 5.4 s | ~59 | 83% | 15.0 | 83 |
| 1080p q180 | 1.06 s | 4.3 s | | 87% | 14.0 | |

At 1080p, per stage:
- **GPU** (`MAV1_KPROF`): ME 5.9 ms on reference frames and 10.5 ms on leaf frames; analysis (`ana0`–`ana3`) about 6 ms; filters about 2.9 ms.
- **CPU:** recon about 48 ms/frame (`code_plane`, itx, inter prediction); entropy about 40 ms/frame (mostly `write_block`); CDEF select 2.2 ms; `analyze_sb_table` 0.9 ms.
- **Pipeline:** about 30 ms per frame through the core across 3 engines, roughly 10 ms/frame of throughput.

## 4. Work per stage, 1080p (ms/frame)

| stage | SVT (one core) | mav1 |
|---|---|---|
| ME/HME + TF | 8.0 | GPU ~8 |
| MD search (PD0+PD1) | 54 | GPU ~6 |
| final transform/quant/recon | 1.5 | **CPU 48** |
| entropy | 7.7 | **CPU 40** |
| deblock / CDEF / LR | 6.2 | GPU 2.9 + CPU 2.2 |
| TPL | 2.4 | none |
| total | 80 CPU-ms | GPU 15 + CPU 83 |

Findings:
1. **ME searches too much.** mav1's GPU ME costs about as much as all of SVT's ME, and the GPU is 6–10x one core. That makes mav1's ME roughly 50x more work.
2. **Recon is done twice.** The CPU redoes it in full, while SVT reuses the result from mode decision (48 vs 1.5 ms).
3. **Entropy is about 5x slower than SVT's** at the same bitrate.

## 5. Speedup estimate

The critical path is the largest of three things: GPU time, CPU time spread over about 7 effective cores, and per-frame pipeline throughput.

- **Today:** at 1080p the GPU takes 15 ms, the CPU about 12 and the pipeline about 10. That puts the ceiling at about 67 fps, against 59 achieved. At 720p the ceiling is about 129 fps, against 115.
- **After the changes below:**
  - 1080p: GPU about 8 ms, CPU about 6.5 ms. That gives about 125 fps, or 2.1x SVT; with 85% overlap it is about 105 fps, or **1.8x**.
  - 720p: best case about 210 fps (2.0x); realistically **1.7–1.9x**.
- **Bottom line:** about **1.8x faster than SVT p8** (range 1.6–2.1x). A 2.5x figure would also need SVT-level quality, so that the comparison is at equal BD rate rather than at mav1's current +8–10% BD.

## 6. Decisions and next steps (ranked by expected gain)

| # | Change | Expected gain | BD risk |
|---|---|---|---|
| 1 | SVT-style ME: 3-level HME on downscaled frames, small refine windows, reuse 8x8 SADs for larger blocks; leaf frames first (10.5 vs 5.9 ms) | −5 ms GPU/frame at 1080p, ≈+30% fps | +0.5–1% |
| 2 | Do not redo recon on the CPU: keep the winning mode's coefficients from GPU analysis, or run recon on the GPU, or NEON itx/quant | −20–25 CPU-ms/frame | 0 (exact) |
| 3 | NEON entropy: precomputed levels/contexts (`txb_init_levels`, `get_nz_map_contexts`), batched coefficient writes, cheaper CDF updates | −25–30 CPU-ms/frame | 0 |
| 4 | More frame overlap (more engines / deeper banding), GPU busy 85% → 95% | ≈+10% | 0 |
| 5 | Drop pruned analysis levels, merge small command buffers, fuse the filter pass | −2 ms GPU, ≈+10% | low to medium |

Items 1 and 4 remove the GPU limit; items 2 and 3 are needed so the CPU does not become the next limit. Proposed next round: **items 1 + 2**, under the same exactness checks.

To go beyond about 1.8x, BD-rate has to come down: better pruning criteria, altref temporal filtering, joint compound search. That quality can then be traded for speed.

### Tried and rejected (from all rounds)

- **Banded GPU/CPU pipeline at the band level:** it starves the GPU.
- **Open-loop ME on the previous source frame:** −0.2 to −0.27 dB.
- **Cooperative analysis at 16x16:** slower.
- **Two evaluations per SIMD group at 16x16:** no gain.
- **Holding leaf GPU work while a reference frame is on the GPU:** −2 to −5 fps.
- **More than 3 frame engines:** no gain.
- **Mini-GOP 3 or 4 under `--turbo`:** slower, or BD +13%.
- **Kept off by default** (knobs bits 1024 / 4096 / 8192 / 16384): no 8x8 in leaves, sub-pel-only 8x8, axial-only sub-pel, parent vectors for 8x8. Each costs about +2–5% BD for a few percent of speed.

## 7. Files in `docs/svtprof/`

| File | Content |
|---|---|
| `s_720.txt`, `s_720_46.txt`, `s_1080.txt`, `s_1080_46.txt` | SVT `sample` profiles (crf32 / crf46) |
| `t_*.txt` | SVT wall/CPU timing runs |
| `lp_scaling.txt` | SVT thread scaling (`--lp`, `--pin`) |
| `ms_720.txt`, `ms_1080.txt` | mav1 `sample` CPU profiles (720p caught only the end of the run) |
| `k_*_{125,180}.txt` | mav1 `MAV1_KPROF` GPU time per stage and level |
| `g_*_{125,180}.txt` | mav1 `MAV1_GPUTL` GPU timelines |
| `stages_ms_per_frame.tsv` | stage table (section 4) |
| `parse.py`, `tree.py`, `gtl.py` | profile and timeline parsers |

The encoded `.ivf` outputs (about 100 MB) are not kept in the repo.
