#!/bin/zsh
# Bit-exactness regression: encoder recon vs dav1d and aomdec, Metal vs CPU bitstream.
# usage: regress.sh <encoder> [extra encoder flags...]
enc=$1; shift
fail=0
for clip in media/ts352.y4m media/mb203.y4m; do
  for q in 0 1 40 100 180 255; do
    $enc -i $clip -o r_cpu.ivf -r r_rec.y4m -n 5 --cpu -q $q "$@" >/dev/null 2>&1 || { echo "$clip q=$q encode FAILED"; fail=1; continue; }
    $enc -i $clip -o r_mtl.ivf -n 5 --metal -q $q "$@" >/dev/null 2>&1
    dav1d -q -i r_cpu.ivf -o r_d.y4m 2>/dev/null
    aomdec --rawvideo -o r_a.yuv r_cpu.ivf 2>/dev/null
    hl=$(head -1 r_rec.y4m | wc -c)
    python3 -c "
import sys
d=open('r_rec.y4m','rb').read(); a=open('r_a.yuv','rb').read()
fr=[x[x.index(b'\n')+1:] for x in d.split(b'FRAME')[1:]]
print('aom_ok' if b''.join(fr)==a else 'aom_DIFF')" > r_aom.txt
    d=$(python3 ../tools/y4mcmp.py r_rec.y4m r_d.y4m | head -1)
    cmp -s r_cpu.ivf r_mtl.ivf && m=metal==cpu || { m=METAL_DIFF; fail=1; }
    a=$(bat --plain r_aom.txt); [[ $a == aom_ok && $d == identical* ]] || fail=1
    echo "$(basename $clip) q=$q $(stat -f%z r_cpu.ivf)B dav1d:[$d] $a $m"
  done
done
exit $fail
