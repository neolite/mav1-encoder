# mav1

A fast AV1 encoder for Apple Silicon with a custom Metal GPU backend for the
analysis and motion-estimation hot paths, NEON kernels on the CPU side, and an
FFmpeg `libmav1` plugin.

On 1080p50 content the turbo preset runs ~1.4x faster than SVT-AV1 preset 8 at
a comparable bitrate, with a small BD-rate delta at high bitrates.

## Status

- AV1 output decodes in dav1d, libaom (`aomdec`) and FFmpeg's decoder.
- Bitdepth: 8-bit and 10-bit (profile 0, YUV 4:2:0) with native uint16 pixel paths
  in both CPU and Metal GPU kernels.
- HDR10 metadata: SMPTE 2084 (PQ), BT.2020 primaries, mastering display color
  volume (MDCV), and content light level (CLL) metadata OBUs.
- Rate control: CQP (default), VBR (single-pass target bitrate), and CBR (VBV buffer).
- The FFmpeg wrapper produces bit-exact output to the `mav1enc` CLI.
- Platform: macOS / Apple Silicon (Metal). A CPU-only path exists for
  correctness checking (`--cpu`, `MAV1_CPU=1`), the production path is Metal.

## Build

```sh
make            # build/libmav1.a + build/mav1enc + unit tests
make install    # stage include/mav1.h, lib/libmav1.a, libmav1.pc (PREFIX=...)
```

Requires a C11 toolchain, Xcode Metal framework, Python 3 (kernel embedding
script).

## CLI

```sh
# 8-bit SDR
build/mav1enc --metal -i in.y4m -o out.ivf -n 200 --turbo -q 125

# 10-bit HDR10 (BT.2020 + PQ, 1000 nits mastering display)
build/mav1enc --metal -i in_10bit.y4m -o hdr.ivf --bitdepth 10 --hdr 1000 --cll 1000,400 --turbo -q 110

# Single-pass VBR rate control
build/mav1enc --metal -i in.y4m -o vbr.ivf --rc vbr --b:v 2500 --turbo
```

Key options: `--turbo` (fast preset), `--q Q` (0-255 base_q_idx, CQP),
`--bitdepth 8|10`, `--hdr [NITS]`, `--cll MAX_CLL,MAX_FALL`,
`--cp N --tc N --mc N --range 0|1` (CICP color description),
`--rc cqp|vbr|cbr`, `--b:v KBPS`, `--vbv KBITS`,
`--keyint N`, `--tiles LOG2`, `--golden`, `--altref N`, `--no-lr`, `--no-cdef`,
`--no-txsearch`, `--no-txdepth`, `--no-switchable`, `--cpu`/`--metal`.

## FFmpeg integration

Apply `docs/ffmpeg-n8.1.3-libmav1.patch` to an FFmpeg tree and build with the
library installed:

```sh
make install PREFIX=/opt/mav1
cd ffmpeg && git apply /path/to/ffmpeg-n8.1.3-libmav1.patch
./configure --enable-libmav1 \
  --extra-cflags=-I/opt/mav1/include --extra-ldflags=-L/opt/mav1/lib
make -j8

# 10-bit HDR10 encode
./ffmpeg -thread_queue_size 512 -i in_10bit.y4m -c:v libmav1 -turbo 1 -hdr 1 -b:v 4M out.ivf
```

Codec options: `-metal 0|1`, `-turbo 0|1`, `-qindex N`, `-hdr 0|1`, `-lf`, `-cdef`,
`-txsearch`, `-txdepth`, `-lr`, `-dualfilter`, `-me_range N`, `-g` (GOP).
Pixel formats: `yuv420p`, `yuv420p10le`. Rate control via `-b:v` / `-maxrate` / `-bufsize`.

## API

`src/mav1.h` — `mav1_default_config`, `mav1_create`, `mav1_send_frame`,
`mav1_receive_packet` / `mav1_receive_packet_wait`, `mav1_get_recon`,
`mav1_destroy`. Altref mode reorders output (hidden ARF + show_existing_frame
temporal units), so consumers should use the queued send/receive API.

## HDR

10-bit 4:2:0 encoding (profile 0) with CICP colour description and static
HDR10 metadata (MDCV + CLL OBUs). See [docs/hdr.md](docs/hdr.md) for input
formats, CLI/FFmpeg flags, the iOS capture path, and the roadmap of
HDR-specific features compared to svt-av1-hdr.

## iOS

`ios/mav1cam` is a proof-of-concept camera app: `make ios` builds
`build/ios/libmav1.a` for `arm64-iphoneos`, `xcodegen generate` inside
`ios/mav1cam` regenerates the Xcode project. Captures 720p/1080p/4K
(optional HDR 10-bit), encodes AV1 on Metal, writes `.ivf` into the app
documents (visible in Files).

## Layout

- `src/encoder.c` — driver, engine scheduler, temporal-unit assembly
- `src/backend_metal.m`, `src/kernels/` — Metal backend and shared C/MSL kernels
- `src/backend_cpu.c` — CPU recon / filters / analysis
- `src/tile_writer.c`, `src/entenc.c`, `src/cdfs.c` — entropy coding
- `src/obu.c` — sequence/frame header OBUs
- `tools/mav1enc.c` — CLI
- `tests/` — entropy round-trip, transform and NEON-vs-scalar tests
- `docs/` — benchmark methodology, SVT profile, FFmpeg patch

## Debug knobs

`MAV1_TIMING` (stage timings), `MAV1_TL` (engine timeline),
`MAV1_DUMP_DEC=path` (per-8x8 decision dump), `MAV1_NOASYNC` (serialize
engines), `MAV1_CPU=1` (CPU backend), `MAV1_GPUTL` (command-buffer GPU times).

## License

MIT — see `LICENSE`. The FFmpeg wrapper file (`libavcodec/libmav1.c`, included
in the patch) is a FFmpeg derivative and stays under the LGPL.
