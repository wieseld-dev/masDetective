from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import random
from dataclasses import dataclass
from pathlib import Path

from v10.checkpoint import commit_round_snapshot
from v11.role_universe import TRAIN_ROLES
from v11_2.hashcheck import assert_exact_v11_start
from v11_2.policy import HardConfuserProbePolicyV112

@dataclass
class TrainSample:
    role: str
    prompt: str
    completion: str
    old_logp: float
    ref_logp: float
    advantage: float
    reward: float
    elite: bool
    group_id: str

def _kl_k3(new_lp, ref_lp, torch):
    delta = ref_lp - new_lp
    return torch.exp(torch.clamp(delta, -6.0, 6.0)) - delta - 1.0

def _balanced_minibatches(by_role_samples, roles, *, batches: int, seed: int):
    n_batches=max(1,int(batches)); rng=random.Random(seed); shuffled={}
    for role in roles:
        xs=list(by_role_samples.get(role,[])); rng.shuffle(xs); shuffled[role]=xs
    out=[[] for _ in range(n_batches)]
    for role in roles:
        xs=shuffled[role]
        if not xs: continue
        for i,x in enumerate(xs): out[i % n_batches].append(x)
        for b in range(n_batches):
            if not any(t.role==role for t in out[b]): out[b].append(xs[b % len(xs)])
    for b in out: rng.shuffle(b)
    return [b for b in out if b]


def load_samples(path: str):
    by_role={r:[] for r in TRAIN_ROLES}
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rows.append(r)
            comp=str(r.get("completion","")).strip()
            role=str(r.get("target_role",""))
            if role in by_role and comp:
                by_role[role].append(TrainSample(
                    role=role,
                    prompt=str(r["prompt"]),
                    completion=comp,
                    old_logp=float(r["old_mean_logprob"]),
                    ref_logp=float(r.get("ref_mean_logprob", float("nan"))),
                    advantage=float(r["advantage"]),
                    reward=float(r["reward"]),
                    elite=bool(r.get("elite",False)),
                    group_id=str(r["group_id"]),
                ))
    return rows,by_role


def main():
    import torch

    ap=argparse.ArgumentParser(description="V11.2 split pipeline: short GPU optimization only")
    ap.add_argument("--round",type=int,required=True)
    ap.add_argument("--evaluated",required=True)
    ap.add_argument("--adapter",required=True)
    ap.add_argument("--optimizer-state",default="")
    ap.add_argument("--v11-frozen-summary",default="artifacts/v11_summary.json")
    ap.add_argument("--model-name",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--balanced-minibatches",type=int,default=4)
    ap.add_argument("--grpo-epochs",type=int,default=2)
    ap.add_argument("--elite-nll-weight",type=float,default=0.06)
    ap.add_argument("--lr",type=float,default=5e-6)
    ap.add_argument("--clip",type=float,default=0.20)
    ap.add_argument("--kl-beta",type=float,default=0.0)
    ap.add_argument("--seed",type=int,default=1010)
    ap.add_argument("--rounds-per-role",type=int,default=2)
    ap.add_argument("--checkpoint-root",default="artifacts/v11_2_split/checkpoints")
    ap.add_argument("--checkpoint-state",default="artifacts/v11_2_split/checkpoint.json")
    ap.add_argument("--output-dir",default="artifacts/v11_2_split/policy")
    ap.add_argument("--optimizer-mirror",default="artifacts/v11_2_split/optimizer.pt")
    ap.add_argument("--no-4bit",action="store_true")
    args=ap.parse_args()

    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    if args.round==1:
        fp=assert_exact_v11_start(args.adapter,args.v11_frozen_summary,strict=True)
        print("[start] exact trained V11 start match:",fp["policy_match"] and fp["summary_match"],flush=True)
    else:
        print("[start] continuation adapter:",args.adapter,flush=True)
    rows,by_role=load_samples(args.evaluated)
    print(f"[opt] round={args.round} evaluated_rows={len(rows)} samples={sum(len(v) for v in by_role.values())}",flush=True)

    policy=HardConfuserProbePolicyV112(args.model_name,adapter_path=args.adapter,trainable=True,use_4bit=not args.no_4bit,max_seq_length=1536)
    optimizer=torch.optim.AdamW(policy.trainable_parameters(),lr=args.lr,weight_decay=1e-4)
    if args.optimizer_state:
        optimizer.load_state_dict(torch.load(args.optimizer_state,map_location="cpu"))
        print("[opt] resumed optimizer:",args.optimizer_state,flush=True)
    else:
        print("[opt] fresh optimizer for V11.2 objective",flush=True)

    usable=[r for r in TRAIN_ROLES if by_role.get(r)]
    minibatches=_balanced_minibatches(by_role,usable,batches=args.balanced_minibatches,seed=args.seed+123457*args.round)
    update_idx=0
    for bi,buf in enumerate(minibatches,1):
        update_idx+=1; policy.model.train()
        print(f"[opt] minibatch={bi}/{len(minibatches)} n={len(buf)}",flush=True)
        for ep in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            n=max(len(buf),1); loss_sum=ratio_sum=kl_sum=0.0
            for j,tr in enumerate(buf,1):
                new_lp=policy.mean_logprob(tr.prompt,tr.completion)
                old_lp=torch.tensor(tr.old_logp,dtype=new_lp.dtype,device=new_lp.device)
                ratio=torch.exp(torch.clamp(new_lp-old_lp,-4.0,4.0))
                adv=torch.tensor(tr.advantage,dtype=new_lp.dtype,device=new_lp.device)
                unclipped=ratio*adv; clipped=torch.clamp(ratio,1.0-args.clip,1.0+args.clip)*adv
                pg=-torch.minimum(unclipped,clipped)
                if math.isfinite(tr.ref_logp) and args.kl_beta>0:
                    ref_lp=torch.tensor(tr.ref_logp,dtype=new_lp.dtype,device=new_lp.device); kl=_kl_k3(new_lp,ref_lp,torch)
                else: kl=new_lp*0.0
                elite_nll=(-new_lp) if tr.elite else new_lp*0.0
                loss=pg+args.kl_beta*kl+args.elite_nll_weight*elite_nll
                (loss/n).backward(); loss_sum+=float(loss.detach().cpu()); ratio_sum+=float(ratio.detach().cpu()); kl_sum+=float(kl.detach().cpu())
                if j%10==0 or j==len(buf):
                    print(f"[opt-step] mb={bi}/{len(minibatches)} epoch={ep+1}/{args.grpo_epochs} sample={j}/{len(buf)} running_loss={loss_sum/j:+.4f}",flush=True)
                del new_lp,old_lp,ratio,adv,unclipped,clipped,pg,kl,elite_nll,loss
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"[update] mb={bi}/{len(minibatches)} epoch={ep+1} loss={loss_sum/n:+.4f} ratio={ratio_sum/n:.3f} kl={kl_sum/n:.4f}",flush=True)
        policy.model.eval()

    state=commit_round_snapshot(
        policy=policy,optimizer=optimizer,round_i=args.round,total_rounds=args.rounds_per_role,
        checkpoint_root=args.checkpoint_root,checkpoint_state=args.checkpoint_state,
        public_output_dir=args.output_dir,log_path=args.evaluated,torch_module=torch,
    )
    mirror=Path(args.optimizer_mirror); mirror.parent.mkdir(parents=True,exist_ok=True)
    if Path(state["optimizer_state"]).resolve()!=mirror.resolve(): shutil.copy2(state["optimizer_state"],mirror)
    print(f"[opt-done] round={args.round} adapter={state['adapter_dir']} optimizer={state['optimizer_state']}",flush=True)


if __name__=="__main__":
    main()
