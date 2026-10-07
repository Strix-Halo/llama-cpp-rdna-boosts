import subprocess, sys, os
files = subprocess.run(['git','diff','--name-only','--diff-filter=U'],capture_output=True,text=True).stdout.split()
total=0
for p in files:
    if not os.path.isfile(p): continue
    s=open(p, encoding='utf-8', errors='surrogateescape').read().split('\n')
    out=[]; i=0; n=0
    while i < len(s):
        if s[i].startswith('<<<<<<<'):
            n+=1; i+=1; ours=[]
            while not s[i].startswith('======='): ours.append(s[i]); i+=1
            i+=1; theirs=[]
            while not s[i].startswith('>>>>>>>'): theirs.append(s[i]); i+=1
            i+=1
            merged=list(ours)
            for l in theirs:
                if l not in merged: merged.append(l)
            out.extend(merged)
        else:
            out.append(s[i]); i+=1
    open(p,'w', encoding='utf-8', errors='surrogateescape').write('\n'.join(out))
    total+=n
    print(f"    resolved {n:2d} conflict(s) in {p}")
print(f"  total {total} conflict(s) across {len(files)} file(s)")
