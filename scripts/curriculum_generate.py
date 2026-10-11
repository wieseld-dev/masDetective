from __future__ import annotations
import argparse, gc, random
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--round',type=int,required=True); ap.add_argument('--split',choices=['train','dev'],required=True)
    ap.add_argument('--root',required=True); ap.add_argument('--adapter',required=True); ap.add_argument('--v11-reference',default='artifacts/v11_policy')
    ap.add_argument('--group-size',type=int,default=6); ap.add_argument('--model-name',default='Qwen/Qwen2.5-1.5B-Instruct'); ap.add_argument('--seed',type=int,default=20260918)
    ap.add_argument('--temperature',type=float,default=.85); ap.add_argument('--top-p',type=float,default=.95)
    ap.add_argument('--max-new-tokens',type=int,default=192); ap.add_argument('--max-seq-length',type=int,default=4096); ap.add_argument('--local-retries',type=int,default=12)
    a=ap.parse_args(); root=Path(a.root)
    states_path=root/f"states_{a.split}_round_{a.round-1:02d}.jsonl"
    if not states_path.exists(): raise SystemExit(f"missing {states_path}")
    import torch
    current=CurriculumAdaptiveProbePolicyV11(a.model_name,adapter_path=a.adapter,trainable=False,use_4bit=False,max_seq_length=a.max_seq_length)
    reference=CurriculumAdaptiveProbePolicyV11(a.model_name,adapter_path=a.v11_reference,trainable=False,use_4bit=False,max_seq_length=a.max_seq_length)
    rows=[]
    states=read_jsonl(states_path)
    G=a.group_size if a.split=='train' else 1
    for si,raw in enumerate(states):
        assert_policy_safe(raw); st=PublicBeliefState.from_public_dict(raw); seen=[h.query for h in st.history]
        group=[]
        for j in range(G):
            last=None
            for retry in range(max(1,a.local_retries)):
                seed=stable_int(a.seed,a.split,a.round,st.session_id,j,retry)
                random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                obs={'belief_state':st.to_public_dict(),'candidate_index':j,'max_history_turns':5}
                sample=current.sample_probe(obs,temperature=a.temperature,top_p=a.top_p,max_new_tokens=a.max_new_tokens)
                q=str(sample.query or '').strip(); ok,why=surface_valid(q)
                valid=bool(sample.valid) and ok and q not in seen and all(q!=x.get('query') for x in group)
                last=(sample,q,valid,why,seed,retry)
                if valid: break
            sample,q,valid,why,seed,retry=last
            prompt=str(sample.prompt); completion=str(sample.completion)
            old_lp=getattr(sample,'old_mean_logprob',None)
            if old_lp is None and completion:
                old_lp=float(current.mean_logprob(prompt,completion).detach().cpu())
            if old_lp is None: old_lp=-20.0
            ref_lp=float(reference.mean_logprob(prompt,completion).detach().cpu()) if completion else -20.0
            rec={'session_id':st.session_id,'split':a.split,'round':a.round,'candidate_index':j,'group_id':f"{st.session_id}::r{a.round:02d}",
                 'query':q,'valid':bool(valid),'invalid_reason':'' if valid else (str(getattr(sample,'invalid_reason','')) or why or 'duplicate'),
                 'prompt':prompt,'completion':completion,'old_logp':float(old_lp),'ref_logp':float(ref_lp),'raw_completion':str(sample.raw_completion),
                 'seed':int(seed),'local_attempts':int(retry+1),'adapter':str(a.adapter)}
            group.append(rec)
            if valid: seen.append(q)
        valid_idx=[x['candidate_index'] for x in group if x['valid']]
        # Continuation is fixed before any MAS answers/rewards are known: no truth/reward leakage.
        if valid_idx:
            rr=random.Random(stable_int(a.seed,'continuation',a.split,a.round,st.session_id))
            chosen=rr.choice(valid_idx)
        else: chosen=-1
        for x in group:
            x['chosen_for_continuation']=bool(x['candidate_index']==chosen)
            rows.append(x)
        print(f"[gen] {si+1}/{len(states)} r={a.round} sid={st.session_id} valid={len(valid_idx)}/{G} continuation={chosen}",flush=True)
    out=root/f"candidates_{a.split}_round_{a.round:02d}.jsonl"; write_jsonl(out,rows)
    print(f"[done] {out} rows={len(rows)} valid={sum(int(x['valid']) for x in rows)}")
    del current,reference; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__': main()
