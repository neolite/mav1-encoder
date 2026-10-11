# HDR encoding in mav1

mav1 encodes 10-bit 4:2:0 AV1 (profile 0, `high_bitdepth=1`) with full
HDR signalling: CICP colour description plus static HDR10 metadata
(SMPTE 2086 MDCV + Content Light Level OBUs) after every sequence
header.

## Input

| Source | Notes |
|--------|-------|
| y4m `C420p10` | `mav1enc` reads it directly; samples are LSB-justified `uint16_t` |
| P010 / `yuv420p10le` | via FFmpeg (`-pix_fmt yuv420p10le`) — swscale converts, or `--bitdepth 10` in the CLI |
| iOS camera | `ios/mav1cam` captures `420YpCbCr10BiPlanar` (P010, MSB-justified), shifts `>>6`, sends I420-10bit |

Careful: `p010le` is **MSB-justified** (10-bit in bits 6..15), while the
encoder API and `C420p10` y4m are **LSB-justified**. Shift `>>6` when
crossing (the iOS path does this in `CameraRecorder`).

## CLI

```sh
# HLG / HDR10 with defaults (BT.2020 + mastering display 1000 nits)
build/mav1enc -i in_10bit.y4m -o hdr.ivf --bitdepth 10 --hdr --turbo -q 110

# explicit CICP (BT.2020 primaries / PQ transfer / BT.2020 matrix)
build/mav1enc -i in.y4m -o out.ivf --bitdepth 10 --cp 9 --tc 16 --mc 9

# content light level
build/mav1enc -i in.y4m -o out.ivf --bitdepth 10 --hdr 1000 --cll 1000,400
```

Options: `--bitdepth 8|10`, `--hdr [NITS]` (mastering display + CLL OBUs),
`--cll MAX_CLL,MAX_FALL`, `--cp N --tc N --mc N --range 0|1`.

## FFmpeg

```sh
./ffmpeg -thread_queue_size 512 -i in_10bit.y4m \
  -c:v libmav1 -turbo 1 -hdr 1 -b:v 4M out.ivf
```

Pixel formats accepted: `yuv420p`, `yuv420p10le`.

## iOS (`ios/mav1cam`)

Toggle **HDR 10-bit** before recording. The app requests
`kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`, converts P010 to
LSB-justified I420-10bit, and signals **HLG** (`tc=18`, iPhone's native
HDR transfer) with BT.2020 primaries/matrix and CLL 1000/400. Falls
back to 8-bit if the format isn't delivered.

## What svt-av1-hdr does that we don't (yet)

Reviewed [SVT-AV1-HDR](https://github.com/juliobbv-p/SVT-AV1-HDR)
(svt-av1-psy fork focused on HDR). Relevant mechanisms, in rough
order of value/effort for mav1:

| Feature | What it is | Status for us |
|---------|-----------|---------------|
| **PQ-aware variance boost** | Per-SB delta-q driven by 8x8 luminance variance, curve fitted to the PQ transfer; auto-enabled when `tc=16`. Their main HDR win — better dark-area detail | Not implemented; we have per-SB analysis data on GPU already — the missing piece is a `delta_qindex` field in `BlockInfo`/`FrameHdr` + aq_coded syntax |
| **CDEF strength scaling** | Scalar control of CDEF filtering strength | Our CDEF is on/off; could scale `cdef_y_pri/sec` by a float — cheap |
| **Noise synthesis** | `--noise` injects grain on top of encoded content (their choice over AV1 film-grain synthesis) | We measured a denoise pre-filter; synthesis is the opposite direction — a post-analysis option |
| **Film grain tune** | Tune preset preserving grain temporally | Could map to our `MAV1_*` knobs (lower q on film-grain frames, weaker LR) |
| **QM-PSNR** | Quant matrices inside the RD distortion metric | We have no quant matrices; larger change |
| **HDR10+ / Dolby Vision RPU** | External dynamic metadata embedded via `--hdr10plus-json` / `--dolby-vision-rpu-path` | We emit static HDR10 only; dynamic metadata would need an RPU passthrough (AV1 can carry DV RPU in `metadata` OBUs) |
| **4:4:4 High Profile** | `seq_profile=1`, no chroma subsampling | All planes assume 420 (`offset/stride/plane_h` chroma shift); big refactor |

## Limitations

- YUV 4:2:0 only (profile 0). 4:4:4/4:2:2 need seq_profile changes.
- Static metadata only — no HDR10+/DV RPU passthrough.
- HLG vs PQ is signalling only: the encoder doesn't do transfer-aware
  quant decisions (that's the "PQ variance boost" item above).
- `p010le` input must be shifted `>>6` (done automatically in the iOS
  path and by FFmpeg's format negotiation, NOT in `mav1enc` — it wants
  `C420p10` y4m).
