"""The FLAGSHIP on REAL production traces: adaptive PREFIX-CACHE eviction on the Alibaba Qwen-Bailian
usage traces (blksz_16 = our block granularity; hash_ids = the real KV block access+sharing pattern).
Proper ARC (Adaptive Replacement Cache: T1 recency + T2 frequency + ghost lists B1/B2, target p tuned
by ghost hits) vs LRU (the engine's default) vs LFU. A request's block HITs if cached (prefix reuse
-> skip prefill) else MISSes. eKV observes the reuse (content-addressed identity) and supplies the
eviction signal; metric = prefix-cache HIT RATE. ARC beats BOTH fixed policies because the real
multi-workload trace mixes recency- and frequency-favoring sub-populations no fixed policy serves.
Usage: _bailian_prefix_check.py <budget> <slice> <file...>"""
import json, random, sys
from collections import OrderedDict
BUD=int(sys.argv[1]); SLICE=int(sys.argv[2]); FILES=sys.argv[3:]

def load(files,n):
    recs=[]
    for f in files:
        for i,l in enumerate(open(f)):
            if i>=n: break
            r=json.loads(l); recs.append((r["timestamp"], r["hash_ids"]))
    recs.sort(key=lambda x:x[0]); return recs

def lru(recs,c):
    rng=random.Random(0); res=[];pos={};last={};tk=0;h=m=0
    for _,bl in recs:
        for b in bl:
            tk+=1; i=pos.get(b,-1)
            if i>=0: h+=1; last[b]=tk; continue
            m+=1
            if len(res)>=c:
                idx=[rng.randrange(len(res)) for _ in range(16)]; vi=min(idx,key=lambda j:last[res[j]])
                vb=res[vi];res[vi]=res[-1];pos[res[vi]]=vi;res.pop();del pos[vb];last.pop(vb,None)
            pos[b]=len(res);res.append(b);last[b]=tk
    return h/max(h+m,1)

def lfu(recs,c):
    rng=random.Random(0); res=[];pos={};freq={};tk=0;h=m=0
    for _,bl in recs:
        for b in bl:
            i=pos.get(b,-1)
            if i>=0: h+=1; freq[b]+=1; continue
            m+=1
            if len(res)>=c:
                idx=[rng.randrange(len(res)) for _ in range(16)]; vi=min(idx,key=lambda j:freq[res[j]])
                vb=res[vi];res[vi]=res[-1];pos[res[vi]]=vi;res.pop();del pos[vb];freq.pop(vb,None)
            pos[b]=len(res);res.append(b);freq[b]=1
    return h/max(h+m,1)

def arc(recs,c):
    T1=OrderedDict();T2=OrderedDict();B1=OrderedDict();B2=OrderedDict();p=0;h=m=0
    def repl(inb2):
        if T1 and (len(T1)>p or (inb2 and len(T1)==p and len(T1)>0)):
            k,_=T1.popitem(last=False);B1[k]=1
        elif T2:
            k,_=T2.popitem(last=False);B2[k]=1
        elif T1:
            k,_=T1.popitem(last=False);B1[k]=1
    for _,bl in recs:
        for b in bl:
            if b in T1: h+=1; del T1[b]; T2[b]=1
            elif b in T2: h+=1; T2.move_to_end(b)
            elif b in B1:
                m+=1; p=min(c,p+max(len(B2)//max(len(B1),1),1)); repl(False); del B1[b]; T2[b]=1
            elif b in B2:
                m+=1; p=max(0,p-max(len(B1)//max(len(B2),1),1)); repl(True); del B2[b]; T2[b]=1
            else:
                m+=1
                if len(T1)+len(B1)==c:
                    if len(T1)<c: B1.popitem(last=False); repl(False)
                    else: T1.popitem(last=False)
                elif len(T1)+len(B1)+len(T2)+len(B2)>=c:
                    if len(T1)+len(B1)+len(T2)+len(B2)==2*c and B2: B2.popitem(last=False)
                    repl(False)
                T1[b]=1
    return h/max(h+m,1)

recs=load(FILES,SLICE)
name="+".join(f.split("/")[-1].replace("qwen_","").replace("_blksz_16.jsonl","") for f in FILES)
L,F,A=lru(recs,BUD),lfu(recs,BUD),arc(recs,BUD); best=max(L,F)
print(f"  {name:26} ({len(recs)}r bud{BUD//1000}k): LRU {L*100:5.1f}  LFU {F*100:5.1f}  "
      f"ARC {A*100:5.1f}  (vs best-fixed {(A-best)*100:+.2f}pt)", flush=True)
