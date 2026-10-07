import subprocess, re
T='1a580f937447949e27f4f822b19714c1c8ebb826'
SUB=re.compile(r'(moe_cache|moe-expert-cache|MOE_EXPERT_CACHE|moe_host_expert|host_expert|ggml_cuda_slab|slab_backed|slab_size|ggml_cuda_vmm|drop_compute_buffers|DROP_COMPUTE_BUFFERS|COMPUTE_BUFFER_MARGIN|alloc_buffer_usage|get_compute_margin_pct|slab_work_size|moe_cache_preflight|moe_host_expert_bytes)')
B09=re.compile(r'(Maximum number of views per statically allocated tensor|Views of the static tensors that are created|number of such views is proportional|much larger than 16 per static tensor|layer are created for the conv-state|Size the headroom accordingly|constexpr size_t compute_headroom)')
raw=subprocess.run(['git','diff','cd98790ec',T],capture_output=True,text=True).stdout
lines=raw.split('\n'); out=[]; files={}; hdr=[]; fname=None; i=0; dropped=0
while i < len(lines):
    l=lines[i]
    if l.startswith('diff --git'):
        fname=l.split(' b/')[-1]; hdr=[l]; j=i+1
        while j < len(lines) and not lines[j].startswith('@@') and not lines[j].startswith('diff --git'):
            hdr.append(lines[j]); j+=1
        i=j; continue
    if l.startswith('@@'):
        hunk=[l]; j=i+1
        while j < len(lines) and not lines[j].startswith('@@') and not lines[j].startswith('diff --git'):
            hunk.append(lines[j]); j+=1
        keep=[hunk[0]]
        for hl in hunk[1:]:
            if fname=='ggml/src/ggml-backend-meta.cpp' and hl[:1] in '+-' and B09.search(hl):
                dropped+=1; continue
            keep.append(hl)
        if fname!='ggml/src/ggml-cuda/mmvq.cu' and SUB.search('\n'.join(keep[1:])):
            out.extend(hdr); hdr=[]; out.extend(keep); files[fname]=files.get(fname,0)+1
        i=j; continue
    i+=1
open('/tmp/sub.diff','w').write('\n'.join(out)+'\n')
print(f"extraction v4: {sum(files.values())} hunks across {len(files)} files; {dropped} block-09 line(s) left for block 09")
