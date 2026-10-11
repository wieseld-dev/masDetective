from __future__ import annotations
import argparse, gc, json, math, os, random, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

from adaptive_v11_active.io import read_jsonl, write_json
from adaptive_v11_curriculum.hashguard import tree_sha256
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11


def load_rows(root: Path, horizon: int):
    rows=[]
    for r in range(1,horizon+1):
        p=root/f"train_rows_round_{r:02d}.jsonl"
        if not p.exists(): raise RuntimeError(f"missing {p}")
        rows.extend(read_jsonl(p))
    return rows


def add_mc_advantages(rows, gamma: float, mc_weight: float):
    chosen=defaultdict(list)
    for i,x in enumerate(rows):
        if x.get('chosen_for_continuation'):
            chosen[x['session_id']].append((int(x['round']),i,float(x['reward'])))
    rtg={}
    by_round=defaultdict(list)
    for sid,seq in chosen.items():
        seq=sorted(seq,reverse=True); g=0.0
        for rnd,i,rw in seq:
            g=rw+gamma*g; rtg[i]=g; by_round[rnd].append((i,g))
    mc={}
    for rnd,vals in by_round.items():
        xs=np.array([v for _,v in vals],dtype=float); m=float(xs.mean()); sd=float(xs.std())+1e-8
        for i,v in vals: mc[i]=(v-m)/sd if sd>1e-6 else 0.0
    out=[]
    for i,x in enumerate(rows):
        z=dict(x); z['mc_advantage']=float(mc.get(i,0.0)); z['combined_advantage']=float(np.clip(float(x.get('local_advantage',0.0))+mc_weight*z['mc_advantage'],-4.0,4.0)); out.append(z)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True); ap.add_argument('--horizon',type=int,required=True); ap.add_argument('--input-adapter',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--v11-reference',default='artifacts/v11_policy'); ap.add_argument('--v11-hash-file',required=True); ap.add_argument('--retention',required=True)
    ap.add_argument('--optimizer-in',default=''); ap.add_argument('--model-name',default='Qwen/Qwen2.5-1.5B-Instruct'); ap.add_argument('--max-seq-length',type=int,default=4096)
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--kl-beta',type=float,default=.03); ap.add_argument('--retention-weight',type=float,default=.10); ap.add_argument('--mc-weight',type=float,default=.35); ap.add_argument('--gamma',type=float,default=1.0)
    ap.add_argument('--clip-eps',type=float,default=.20); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--grad-accum',type=int,default=8); ap.add_argument('--seed',type=int,default=20260918)
    a=ap.parse_args(); root=Path(a.root); out=Path(a.output_dir); v11=Path(a.v11_reference)
    expected=Path(a.v11_hash_file).read_text().strip(); before=tree_sha256(v11)
    if before!=expected: raise RuntimeError(f"FROZEN V11 CHANGED before update: expected={expected} got={before}")
    if out.resolve()==v11.resolve(): raise RuntimeError('refusing to write over frozen V11')
    rows=add_mc_advantages(load_rows(root,a.horizon),a.gamma,a.mc_weight)
    rows=[x for x in rows if x.get('completion')]
    if len(rows)<20: raise RuntimeError(f"too few optimization rows: {len(rows)}")
    retention=read_jsonl(a.retention)
    if not retention: raise RuntimeError('empty retention buffer')
    import torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    pol=CurriculumAdaptiveProbePolicyV11(a.model_name,adapter_path=a.input_adapter,trainable=True,use_4bit=False,max_seq_length=a.max_seq_length)
    pol.model.train(); params=[p for p in pol.model.parameters() if p.requires_grad]
    if not params: raise RuntimeError('no trainable parameters')
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    if a.optimizer_in and Path(a.optimizer_in).is_file():
        state=torch.load(a.optimizer_in,map_location='cpu'); opt.load_state_dict(state)
        for g in opt.param_groups: g['lr']=a.lr
        print(f"[optimizer] resumed {a.optimizer_in}",flush=True)
    stats=[]; step_count=0; opt.zero_grad(set_to_none=True)
    for ep in range(a.epochs):
        order=list(range(len(rows))); random.Random(a.seed+ep).shuffle(order)
        for jj,idx in enumerate(order):
            x=rows[idx]; adv=float(x['combined_advantage']); old=float(x['old_logp']); ref=float(x['ref_logp'])
            new_lp=pol.mean_logprob(x['prompt'],x['completion'])
            log_ratio=torch.clamp(new_lp-new_lp.new_tensor(old),-8.0,8.0); ratio=torch.exp(log_ratio)
            adv_t=new_lp.new_tensor(adv); clipped=torch.clamp(ratio,1.0-a.clip_eps,1.0+a.clip_eps)
            pg=-torch.minimum(ratio*adv_t,clipped*adv_t)
            kl=0.5*(new_lp-new_lp.new_tensor(ref)).pow(2)
            loss=pg+a.kl_beta*kl
            ret_loss=new_lp.new_tensor(0.0)
            # Rehearse original V11 behavior throughout the continuation fine-tune.
            if jj % 4 == 0:
                rr=retention[(jj//4 + ep) % len(retention)]
                rlp=pol.mean_logprob(rr['prompt'],rr['completion'])
                ret_loss=-rlp
                loss=loss+a.retention_weight*ret_loss
            (loss/a.grad_accum).backward()
            if (jj+1)%a.grad_accum==0 or jj+1==len(order):
                torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True); step_count+=1
            stats.append((float(loss.detach().cpu()),float(pg.detach().cpu()),float(kl.detach().cpu()),float(ret_loss.detach().cpu()),adv))
    tmp=out.parent/(out.name+'.tmp')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True,exist_ok=True); pol.model.save_pretrained(tmp)
    if out.exists(): shutil.rmtree(out)
    tmp.rename(out)
    opt_path=out.parent/'optimizer.pt'; torch.save(opt.state_dict(),opt_path)
    after=tree_sha256(v11)
    if after!=expected: raise RuntimeError(f"FROZEN V11 CHANGED during update: expected={expected} got={after}")
    arr=np.asarray(stats,float)
    summary={'rows':len(rows),'epochs':a.epochs,'optimizer_steps':step_count,'lr':a.lr,'clip_eps':a.clip_eps,'kl_beta':a.kl_beta,'retention_weight':a.retention_weight,'mc_weight':a.mc_weight,'gamma':a.gamma,
             'mean_loss':float(arr[:,0].mean()),'mean_pg':float(arr[:,1].mean()),'mean_kl_surrogate_to_v11':float(arr[:,2].mean()),'mean_retention_nll_when_sampled':float(arr[:,3][arr[:,3]!=0].mean()) if np.any(arr[:,3]!=0) else 0.0,
             'mean_combined_advantage':float(arr[:,4].mean()),'input_adapter':a.input_adapter,'output_adapter':str(out),'optimizer':str(opt_path),'frozen_v11_sha256':after}
    write_json(out.parent/'update_summary.json',summary); print(json.dumps(summary,indent=2),flush=True)
    del pol; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__': main()
