#!/usr/bin/env python3
"""Speed/quality comparison on this machine.

  * mav1 Metal vs mav1 CPU backend (same kernels; bitstreams must be identical)
  * mav1 vs the usual CPU AV1 encoders available in FFmpeg
    (libaom-av1, libsvtav1, librav1e)

Writes build/bench/results.md and results.csv."""
import csv, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, 'build')
OUT = os.path.join(B, 'bench')
OUR_FF = os.path.join(ROOT, 'ffmpeg-src', 'ffmpeg')
SYS_FF = 'ffmpeg'
os.makedirs(OUT, exist_ok=True)

CLIPS = [
    ('testsrc2 352x288', 'ts352.y4m', 'testsrc2=size=352x288:rate=30', 10),
    ('mandelbrot 640x360', 'mb640.y4m', 'mandelbrot=size=640x360:rate=30', 10),
    ('testsrc2 1280x720', 'ts720.y4m', 'testsrc2=size=1280x720:rate=30', 5),
]

ENCODERS = [
    # name, ffmpeg binary, codec args
    ('mav1 Metal (GPU)', OUR_FF, ['-c:v', 'libmav1', '-metal', '1', '-qindex', '100']),
    ('mav1 CPU (8 threads)', OUR_FF, ['-c:v', 'libmav1', '-metal', '0', '-qindex', '100']),
    ('libaom-av1 cpu-used=6', SYS_FF, ['-c:v', 'libaom-av1', '-cpu-used', '6', '-crf', '30', '-g', '9999', '-row-mt', '1']),
    ('libaom-av1 cpu-used=8 (rt)', SYS_FF, ['-c:v', 'libaom-av1', '-usage', 'realtime', '-cpu-used', '8', '-crf', '30', '-g', '9999', '-row-mt', '1']),
    ('libsvtav1 preset=8', SYS_FF, ['-c:v', 'libsvtav1', '-preset', '8', '-crf', '30', '-g', '9999']),
    ('librav1e speed=10', SYS_FF, ['-c:v', 'librav1e', '-speed', '10', '-qp', '100', '-g', '9999']),
]


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        raise RuntimeError(' '.join(cmd) + '\n' + r.stderr[-2000:])
    return r


def psnr(ref, enc):
    r = run([SYS_FF, '-v', 'info', '-i', enc, '-i', ref, '-lavfi', '[0:v][1:v]psnr', '-f', 'null', '-'])
    m = re.search(r'PSNR y:([\d.inf]+) u:([\d.inf]+) v:([\d.inf]+) average:([\d.inf]+)', r.stderr)
    return tuple(float(x) for x in m.groups())


def main():
    rows = []
    for title, name, src, n in CLIPS:
        clip = os.path.join(OUT, name)
        if not os.path.exists(clip):
            run([SYS_FF, '-v', 'error', '-y', '-f', 'lavfi', '-i', src, '-frames:v', str(n), '-pix_fmt', 'yuv420p', clip])
        streams = {}
        for ename, ff, args in ENCODERS:
            out = os.path.join(OUT, re.sub(r'\W+', '_', name + '_' + ename) + '.ivf')
            cmd = [ff, '-v', 'error', '-y', '-i', clip] + args + ['-threads', '8', out]
            try:
                t = []
                for _ in range(1):
                    t0 = time.time(); run(cmd); t.append(time.time() - t0)
            except RuntimeError as err:
                print('skip', ename, ':', str(err).splitlines()[-1], file=sys.stderr)
                continue
            sec = min(t)
            size = os.path.getsize(out)
            p = psnr(clip, out)
            streams[ename] = open(out, 'rb').read()
            row = dict(clip=title, encoder=ename, frames=n, seconds=round(sec, 3), fps=round(n / sec, 2),
                       kbps=round(size * 8 * 30 / n / 1000, 1), bytes=size, psnr_y=p[0], psnr_avg=p[3])
            rows.append(row)
            print(row, flush=True)
        if 'mav1 Metal (GPU)' in streams and 'mav1 CPU (8 threads)' in streams:
            same = streams['mav1 Metal (GPU)'] == streams['mav1 CPU (8 threads)']
            rows.append(dict(clip=title, encoder='mav1 Metal == CPU bitstream', frames=n, seconds='', fps='', kbps='',
                             bytes='', psnr_y='', psnr_avg='IDENTICAL' if same else 'DIFFERENT'))
    with open(os.path.join(OUT, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, 'results.md'), 'w') as f:
        f.write('| clip | encoder | fps | kbit/s | PSNR-Y | PSNR avg |\n|---|---|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['clip']} | {r['encoder']} | {r['fps']} | {r['kbps']} | {r['psnr_y']} | {r['psnr_avg']} |\n")
    print(open(os.path.join(OUT, 'results.md')).read())


main()
