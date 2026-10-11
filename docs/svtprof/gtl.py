import sys,re,collections
iv=[];lab=collections.Counter();cs={};core=0;wq=[]
for l in open(sys.argv[1]):
    m=re.match(r'gpu (\S+) (\S+) ([\d.]+) ([\d.]+)',l)
    if m: a,b=float(m[3]),float(m[4]); iv.append((a,b)); lab[m[2]]+=b-a; continue
    m=re.match(r'\s*([\d.]+) (e\d)\s+(d\d+) core (start|end)',l)
    if m:
        k=(m[2],m[3])
        if m[4]=='start': cs[k]=float(m[1])
        else: core+=float(m[1])-cs.pop(k)
    m=re.match(r'\s*([\d.]+) wq',l)
    if m: wq.append(float(m[1]))
iv.sort();u=0;s,e=iv[0]
for a,b in iv[1:]:
    if a>e: u+=e-s;s,e=a,b
    else: e=max(e,b)
u+=e-s;span=iv[-1][1]-iv[0][0]
print(f"span {span:.0f}ms gpu_busy {u:.0f}ms ({100*u/span:.0f}%) gpu/frame {u/60:.2f}ms core_cpu/frame {core/60:.1f}ms last_written {max(wq):.0f}")
for k,v in sorted(lab.items()): print(f"  {k} {v/60:.2f} ms/frame")
