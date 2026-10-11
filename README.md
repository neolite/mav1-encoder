# mav1

A fast AV1 encoder for Apple Silicon with a custom Metal GPU backend for the
analysis and motion-estimation hot paths, NEON kernels on the CPU side, and an
FFmpeg `libmav1` plugin.

On 1080p50 content the turbo preset runs ~1.4x faster than SVT-AV1 preset 8 at
a comparable bitrate, with a small BD-rate delta at high bitrates.

## Status

- AV1 output decodes in dav1d, libaom (`aomdec`) and FFmpeg's decoder.
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
build/mav1enc --metal -i in.y4m -o out.ivf -n 200 --turbo -q 125
```

Key options: `--turbo` (fast preset), `--q Q` (0-255 base_q_idx, CQP),
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
./ffmpeg -thread_queue_size 512 -i in.y4m -c:v libmav1 -turbo 1 -qindex 125 out.ivf
```

> **Throughput note:** FFmpeg hands the encoder one input frame per
> `receive_packet` call and, with the default shallow thread queues, delivers
> input at consumption pace. That starves mav1's engine pipeline (~10 fps on
> 1080p). Passing `-thread_queue_size 512` (or larger) on the input lets the
> demuxer run ahead and restores near-CLI throughput (~55 fps on M1 Pro,
> ~1.4x libsvtav1). Without it the output is still correct, just slower.

Codec options: `-metal 0|1`, `-turbo 0|1`, `-qindex N`, `-lf`, `-cdef`,
`-txsearch`, `-txdepth`, `-lr`, `-dualfilter`, `-me_range N`, `-g` (GOP).

## API

`src/mav1.h` — `mav1_default_config`, `mav1_create`, `mav1_send_frame`,
`mav1_receive_packet` / `mav1_receive_packet_wait`, `mav1_get_recon`,
`mav1_destroy`. Altref mode reorders output (hidden ARF + show_existing_frame
temporal units), so consumers should use the queued send/receive API.

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
