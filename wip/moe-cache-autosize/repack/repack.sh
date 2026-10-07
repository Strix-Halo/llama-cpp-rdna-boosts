#!/bin/bash
set -u
cd /home/stew675/llama-repack
T=1a580f937447949e27f4f822b19714c1c8ebb826
git checkout -q -B repack06 ecc86b526
git cherry-pick cd98790ec >/dev/null 2>&1 || { echo "FATAL: block 06"; exit 1; }
python3 /tmp/extract_v4.py
python3 - <<'PY'
import subprocess, os, glob
for f in glob.glob('/tmp/subparts/*.diff'): os.remove(f)
os.makedirs('/tmp/subparts',exist_ok=True)
raw=open('/tmp/sub.diff').read(); ok=[];bad=[]
for p in ['diff --git '+x for x in raw.split('diff --git ') if x.strip()]:
    name=p.split(' b/')[-1].split('\n')[0].strip()
    fn='/tmp/subparts/'+name.replace('/','__')+'.diff'; open(fn,'w').write(p)
    r=subprocess.run(['git','apply','-3','--recount',fn],capture_output=True,text=True)
    (ok if r.returncode==0 else bad).append(name)
print(f"  applied {len(ok)} file(s); deferred: {bad}")
PY
git add -A && git commit -q --amend -C 848ac2802
echo "06' = $(git rev-parse --short HEAD) tree $(git rev-parse --short HEAD^{tree})  ($(git diff --stat HEAD~1 HEAD | tail -1 | sed 's/^ *//'))"
for c in 779ab4674 f43d9850b 2920d6630 277c07826 e6a87e590 63211896c c9d094908 0486b7cb2; do
  n=$(cd ~/llama.cpp && git log -1 --format="%h %s" $c | cut -c1-58)
  if git cherry-pick $c >/dev/null 2>&1; then echo "  OK        $n"
  else
    nf=$(git diff --name-only --diff-filter=U | wc -l)
    if [ "$nf" = 0 ]; then echo "  EMPTY     $n  <-- fully absorbed into an earlier block"; git cherry-pick --abort 2>/dev/null; git cherry-pick --skip 2>/dev/null || true
    else echo "  CONFLICT  $n  ($nf file(s))"; python3 /tmp/resolve_conflicts.py
      git add -A >/dev/null; git cherry-pick --continue --no-edit >/dev/null 2>&1 && echo "            -> $(git rev-parse --short HEAD)" || { echo "            FAILED"; git cherry-pick --abort; }
    fi
  fi
done
echo "=== 15' = the remainder to T_final ==="
git read-tree --reset $T && git checkout-index -a -f && git commit -q -C c3135532d && echo "15' = $(git rev-parse --short HEAD)"
echo "=== FINAL TREE ASSERTION ==="
echo "  got:      $(git rev-parse HEAD^{tree})"
echo "  expected: $T"
[ "$(git rev-parse HEAD^{tree})" = "$T" ] && echo "  ✅ TREE MATCHES T_final (zero code change)" || echo "  ❌ TREE MISMATCH"
echo "  blocks base..HEAD: $(git rev-list --count a55e952b8..HEAD)"
