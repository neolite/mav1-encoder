# mav1 — AV1 encoder on Metal: specification, reference tests, coverage

## Sources
* **Specification:** AV1 Bitstream & Decoding Process Specification v1.0.0 with Errata 1 (`ref/av1-spec`, github.com/AOMediaCodec/av1-spec). The kernels transcribe its decoding-process sections (7.10 to 7.15). `src/spec_tables.h` is generated from the spec text by `tools/gen_spec_tables.py`: 190 constants and 214 tables.
* **Reference implementation:** libaom (`ref/aom`), used for the test suite and its test-vector list.

## What "correct" means for an encoder
AV1 normatively specifies only the decoder. An encoder is therefore correct when it meets three conditions:
1. Every bitstream it emits is conformant, so independent decoders accept it.
2. The encoder's own reconstruction is **bit-identical** to what conformant decoders output. Any mismatch would make predictions drift.
3. The GPU and CPU paths produce the same result. Both run the same kernel sources, so their bitstreams must be byte-identical.

## Reference test inventory

### A. Conformance streams (decoder side; they define the bit-exact behaviour mav1 must reproduce)
| Suite | Content | Use in this project |
|---|---|---|
| AOM CTC test vectors `av1-1-b8-00-quantizer-{00..63}` | Every base_q_idx class | Same space as the mav1 q sweep (1..255) |
| `av1-1-b8-01-size-{16,18,32,34,64,66,196..226}x{...}` | Odd and small frame sizes | Mirrored by the 203x131 clip (non-multiple-of-8) |
| `av1-1-b8-02-allintra`, `-03-sizeup/sizedown`, `-04-cdfupdate`, `-05-mv`, `-06-mfmv`, `-22-svc`, `-23-film_grain` | Tool-specific streams | mv and allintra are covered; the rest exercise tools mav1 does not emit |
| Argon conformance suite (AOM, about 2,500 streams) | Exhaustive coverage of the syntax and decoding process | Not run in this project (licensed suite). dav1d and libaom are themselves Argon-conformant, which is why they serve as the oracles here |
Full list: `ref/aom/test/test_vectors.cc` (176 entries).

### B. libaom encoder-side tests (`ref/aom/test/*.cc`) and the mav1 equivalent of each
| libaom test | Checks | mav1 equivalent |
|---|---|---|
| `boolcoder_test`, `ec_test` | Range-coder round trip | `tests/test_entropy.c`: random symbols and CDF adaptation, decoded by a literal transcription of the spec decoder, with a trailing-bit check |
| `av1_fwd_txfm1d/2d_test`, `av1_inv_txfm1d_test` | Transform accuracy | `tests/test_txfm.c`: forward then spec-inverse for all 19 tx sizes and 16 tx types |
| `quantize_func_test`, `av1_quantize_test` | Quantizer | Exercised end-to-end by `tools/regress.sh` q sweep |
| `end_to_end_psnr_test`, `allintra_end_to_end_test`, `rt_end_to_end_test` | Encode, decode, check PSNR | `tools/regress.sh`; `mav1enc` reports PSNR |
| `encode_small_width_height_test`, `frame_size_tests` | Odd and small sizes | 203x131 clip in the regression |
| `kf_test`, `force_key_frame_test` | Key-frame placement | `-g`/`--keyint` (checked through FFmpeg: K flags) |
| `motion_vector_test`, `mv_cost_test` | MV coding range | Inter regression; MVs are checked by is_mv_valid |
| `loopfilter_control_test`, `sharpness_test` | Deblocking | Filters-on regression (`regress.sh` default) |
| `coding_path_sync`, `av1_encoder_parms_get_to_decoder` | Encoder state equals decoder state | Recon equals dav1d output, frame by frame |
| `lossless_test`, `monochrome_test`, `screen_content_test`, `horz_superres_test`, `resize_test`, `svc_*`, `film_grain_table_test`, `tile_*`, `error_resilience_test`, `datarate_test`, `ratectrl_*`, `tpl_model_test`, `altref_test`, `gf_pyr_height_test` | Tools and features | Not applicable: these tools are not emitted (see the coverage table) |

### C. mav1's own tests
| Test | Command | Oracle |
|---|---|---|
| Entropy coder | `build/test_entropy` | Spec decoder transcription |
| Transforms | `build/test_txfm` | Spec inverse transform |
| Bit-exact regression (2 clips × 6 q incl. lossless × 5 frames, I and P frames; run in parallel) | `tools/regress.sh build/mav1enc [--lr-test] [--no-lf --no-cdef ...]` | dav1d, aomdec (libaom), Metal vs CPU byte compare |
| FFmpeg integration | `ffmpeg -c:v libmav1` into ivf, mkv, mp4 and webm; `-metal 0/1` | Packets identical to `mav1enc`; decoded output equals encoder recon |
| Speed and quality | `tools/bench.py` | Metal vs CPU, plus libaom, SVT-AV1 and rav1e |

## Coverage of the specification by mav1
| Spec area | Status |
|---|---|
| OBUs: temporal delimiter, sequence header, frame OBU | ✅ |
| Profile 0, 8-bit 4:2:0, any size from 16x16 up | ✅ |
| Key frames and inter frames (single reference, LAST) | ✅ |
| Partition: 64x64 superblocks, all 10 partition types (NONE, HORZ, VERT, SPLIT, HORZ_A/B, VERT_A/B, HORZ_4, VERT_4) | ✅ |
| Intra: all 13 modes, angle deltas, chroma-from-luma (CfL) | ✅ (filter-intra, intra edge filter and palette not emitted) |
| Inter: NEARESTMV, NEARMV (drl), GLOBALMV, NEWMV; find_mv_stack 7.10.2; MV coding | ✅ (compound, OBMC, warped and interintra not emitted) |
| Switchable dual interpolation filters (per block, 3x3 search) | ✅ |
| Motion search (full-pel plus 1/2 and 1/4 pel; GPU: SIMD-group parallel, bit-identical to CPU) | ✅ |
| Transforms: all sizes up to 64, all 16 types, tx-type RD search; TX_MODE_SELECT (intra tx_depth search, txfm_split for inter) | ✅ (inter var-tx split not emitted) |
| Quantization and dequantization, all q 0..255; lossless (q=0: WHT, 4x4, CodedLossless header) | ✅ (qmatrix and delta-q not emitted) |
| Coefficient coding (all contexts), CDF adaptation | ✅ |
| Deblocking loop filter 7.14 (including transform-split edges) | ✅ |
| CDEF 7.15 with per-SB selection over 4 strength sets | ✅ |
| Loop restoration 7.17: Wiener and self-guided, 64x64 units, least-squares parameter search, RD per unit | ✅ (`--lr-test` forces both filters for syntax coverage) |
| Superres, film grain, segmentation, tiles >1, 10/12-bit, 4:4:4, 128x128 SB, palette, intrabc, compound, multiple references | ❌ not emitted |

Every bitstream mav1 emits is verified bit-exact against two independent decoders.
