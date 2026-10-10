#!/bin/zsh
# Bit-exactness regression: encoder recon vs dav1d and aomdec, Metal vs CPU bitstream.
# Every (clip, q) case runs in its own directory, all cases in parallel.
# usage: regress.sh <encoder> [extra encoder flags...]   (JOBS=N to limit)
enc=${1:A}; shift
flags="$*"
tmp=$(mktemp -d)
sdir=${0:A:h}
export enc flags tmp sdir
one() {
  clip=$1 q=$2
  dir=$tmp/$(basename $clip .y4m)_$q; mkdir -p $dir
  ${=enc} -i $clip -o $dir/cpu.ivf -r $dir/rec.y4m -n 5 --cpu -q $q ${=flags} >/dev/null 2>&1 || { echo "FAIL $(basename $clip) q=$q encode"; return; }
  ${=enc} -i $clip -o $dir/mtl.ivf -n 5 --metal -q $q ${=flags} >/dev/null 2>&1
  dav1d -q -i $dir/cpu.ivf -o $dir/d.y4m 2>/dev/null
  aomdec --rawvideo -o $dir/a.yuv $dir/cpu.ivf 2>/dev/null
  a=$(python3 -c "
d=open('$dir/rec.y4m','rb').read(); a=open('$dir/a.yuv','rb').read()
fr=[x[x.index(b'\n')+1:] for x in d.split(b'FRAME')[1:]]
print('aom_ok' if b''.join(fr)==a else 'aom_DIFF')")
  d=$(python3 $sdir/y4mcmp.py $dir/rec.y4m $dir/d.y4m 2>&1 | head -1)
  cmp -s $dir/cpu.ivf $dir/mtl.ivf && m=metal==cpu || m=METAL_DIFF
  st=ok; [[ $a == aom_ok && $d == identical* && $m == metal==cpu ]] || st=FAIL
  echo "$st $(basename $clip) q=$q $(stat -f%z $dir/cpu.ivf)B dav1d:[$d] $a $m"
}
# test clips are generated on demand (make clean wipes build/)
mkdir -p media
[[ -f media/ts352.y4m ]] || ffmpeg -v error -f lavfi -i testsrc2=size=352x288:rate=30 -frames:v 10 -pix_fmt yuv420p media/ts352.y4m
[[ -f media/mb203.y4m ]] || ffmpeg -v error -f lavfi -i mandelbrot=size=203x131:rate=30 -frames:v 10 -pix_fmt yuv420p media/mb203.y4m
cases=()
for clip in media/ts352.y4m media/mb203.y4m; do
  for q in 0 1 40 100 180 255; do cases+=("$clip $q"); done
done
out=$(printf '%s\n' "${cases[@]}" | xargs -P ${JOBS:-$(sysctl -n hw.ncpu)} -L1 zsh -c "$(functions one); one \$1 \$2" _ | sort -k2,2 -k3,3V)
rm -rf $tmp
print -r -- "$out"
[[ $out != *FAIL* ]]
