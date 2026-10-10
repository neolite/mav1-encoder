#!/usr/bin/env python3
"""Compare the pixel data of two I420 y4m files frame by frame.
Exit status 0 when every frame is identical."""
import sys

def frames(path):
    with open(path, 'rb') as f:
        hdr = f.readline().split()
        w = int([t for t in hdr if t.startswith(b'W')][0][1:])
        h = int([t for t in hdr if t.startswith(b'H')][0][1:])
        size = w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)
        while True:
            line = f.readline()
            if not line:
                return
            data = f.read(size)
            if len(data) < size:
                return
            yield w, h, data

def main():
    a, b = sys.argv[1], sys.argv[2]
    fa, fb = list(frames(a)), list(frames(b))
    if len(fa) != len(fb):
        print(f"frame count differs: {len(fa)} vs {len(fb)}")
        return 1
    bad = 0
    for i, ((w, h, x), (_, _, y)) in enumerate(zip(fa, fb)):
        if x != y:
            diffs = [k for k in range(len(x)) if x[k] != y[k]]
            k = diffs[0]
            plane, off = (0, k) if k < w * h else ((1, k - w * h) if k < w * h + ((w + 1) // 2) * ((h + 1) // 2) else (2, k - w * h - ((w + 1) // 2) * ((h + 1) // 2)))
            pw = w if plane == 0 else (w + 1) // 2
            print(f"frame {i}: {len(diffs)} samples differ, first in plane {plane} at x={off % pw} y={off // pw}")
            bad += 1
    if not bad:
        print(f"identical ({len(fa)} frames)")
    return 1 if bad else 0

sys.exit(main())
