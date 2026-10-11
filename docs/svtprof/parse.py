import re,sys,collections
WAIT=('semaphore_wait_trap','__psynch_cvwait','svt_block_on_semaphore','__psynch_mutexwait','swtch_pri','__semwait_signal')
pat=re.compile(r'^([\s+!:|]*)(\d+)\s+(\S+)')
lines=open(sys.argv[1]).read().split('Sort by top of stack')[0].splitlines()
stack=[]  # (depth,name,count)
kern=collections.Counter(); sub=collections.Counter()
SUBS=sys.argv[2].split(',') if len(sys.argv)>2 else []
for l in lines:
    m=pat.match(l)
    if not m or 'Thread_' in l: 
        if 'Thread_' in l: stack=[]
        continue
    d=len(m.group(1)); c=int(m.group(2)); n=m.group(3)
    while stack and stack[-1][0]>=d: stack.pop()
    stack.append((d,n,c))
    names=[s[1] for s in stack]
    k=next((x for x in names if x.endswith('_kernel')),None)
    if k is None: continue
    if n==k: kern[k]+=c
    if n in WAIT and WAIT[0] not in names[:-1] and not any(x in WAIT for x in names[:-1]):
        kern[k]-=c
        # subtract from sub too
        for s in SUBS:
            if s in names[:-1]: sub[(k,s)]-=c
    for s in SUBS:
        if n==s and names.count(s)==1: sub[(k,s)]+=c
tot=sum(kern.values())
for k,v in kern.most_common(): print(f"{v:6d} {100*v/tot:5.1f}% {k}")
print('total busy',tot)
for k,v in sorted(sub.items(),key=lambda x:-x[1]):
    if v>0: print(f"  {v:6d} {100*v/tot:5.1f}% {k[0]} > {k[1]}")
