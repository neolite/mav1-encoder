import re,sys,collections
pat=re.compile(r'^([\s+!:|]*)(\d+)\s+(\S+)')
lines=open(sys.argv[1]).read().split('Sort by top of stack')[0].splitlines()
K=sys.argv[2]; maxd=int(sys.argv[3])
stack=[];agg=collections.Counter()
for l in lines:
    m=pat.match(l)
    if not m: stack=[];continue
    d=len(m.group(1));c=int(m.group(2));n=m.group(3)
    while stack and stack[-1][0]>=d: stack.pop()
    stack.append((d,n,c))
    names=[s[1] for s in stack]
    if K in names[:-1]:
        i=names.index(K); rel=len(names)-1-i
        if rel<=maxd and 'wait' not in n and 'semaphore' not in n: agg[(rel,'/'.join(names[i+1:]))]+=c
for (r,p),v in sorted(agg.items(),key=lambda x:x[0][1]):
    if v>=20: print(f"{v:6d} {'  '*r}{p.split('/')[-1]}")
