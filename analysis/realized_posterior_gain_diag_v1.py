#!/usr/bin/env python3
"""
realized_posterior_gain_diag_v1.py

Evaluate the EIG selector against the RIGHT downstream objective.

The previous replay compared MI_K to "number of new hidden roles exposed".
That is not the same objective as reducing uncertainty/error about K.

This diagnostic keeps the same held-out systems and trained observation model,
but for every candidate q:

  1) sample posterior particles Z_m ~ P_t(Z)
  2) predict P(Y | Z_m, q, b_t)
  3) reveal the recorded candidate outcome Y ONLY for retrospective evaluation
  4) Bayes-update particle weights:
       w'_m proportional to w_m P(Y | Z_m,q,b_t)
  5) measure realized improvement against ground truth:
       - K Brier gain
       - K log-probability gain
       - expected-|K-K*| reduction
       - role-map Brier gain

Then compare:
  max predicted MI_K / MI_ID / MI_Z
against
  random / existing selector / oracle best realized downstream gain.

Ground-truth Z and recorded Y are NEVER used to compute the candidate score;
they are used only after selection to evaluate what would have happened.

NOTE:
Y here is selected_roles_training_only, a latent training-time outcome.
So this remains a controlled diagnostic. A deployable black-box version should
replace Y with an observable fingerprint representation from the final answer.
"""

from __future__ import annotations
import argparse, json, random, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_obs():
    sys.path.insert(0, str(Path("analysis").resolve()))
    import observation_model_v1 as om
    return om


def suffix_esym(weights, kmax):
    n=len(weights)
    E=np.zeros((n+1,kmax+1),dtype=np.float64)
    E[n,0]=1.0
    for i in range(n-1,-1,-1):
        E[i,0]=1.0
        for k in range(1,kmax+1):
            E[i,k]=E[i+1,k]+weights[i]*E[i+1,k-1]
    return E


def sample_fixed_k(weights,k,rng,E):
    n=len(weights); z=np.zeros(n,dtype=np.float32); rem=int(k)
    for i in range(n):
        if rem==0: break
        if n-i==rem:
            z[i:]=1.0; break
        den=E[i,rem]
        if den<=0:
            ids=rng.choice(np.arange(i,n),size=rem,replace=False)
            z[ids]=1.0; break
        pin=float(np.clip(weights[i]*E[i+1,rem-1]/den,0,1))
        if rng.random()<pin:
            z[i]=1.0; rem-=1
    return z


def stratified_particles(k_probs,role_probs,roles,per_k,rng):
    p=np.array([float(role_probs[r]) for r in roles],dtype=np.float64)
    p=np.clip(p,1e-5,1-1e-5)
    odds=p/(1-p)
    kp=[(int(k),float(v)) for k,v in k_probs.items() if float(v)>0]
    zsum=sum(v for _,v in kp)
    kp=[(k,v/zsum) for k,v in kp]
    E=suffix_esym(odds,max(k for k,_ in kp))
    Z=[]; W=[]; K=[]
    for k,pk in kp:
        for _ in range(per_k):
            Z.append(sample_fixed_k(odds,k,rng,E))
            W.append(pk/per_k); K.append(k)
    return np.stack(Z).astype(np.float32),np.asarray(W,np.float64),np.asarray(K,np.int64),dict(kp)


def hbin(p):
    p=np.clip(np.asarray(p,np.float64),1e-8,1-1e-8)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return None
    if spearmanr is not None:
        v=spearmanr(a,b).correlation
        return None if np.isnan(v) else float(v)
    ra=np.empty(len(a)); rb=np.empty(len(b))
    ra[np.argsort(a)]=np.arange(len(a)); rb[np.argsort(b)]=np.arange(len(b))
    v=np.corrcoef(ra,rb)[0,1]
    return None if np.isnan(v) else float(v)


@torch.no_grad()
def particle_predictions(model,qemb_group,belief,kvec,round_norm,Z,device,batch_size):
    C,M=qemb_group.shape[0],Z.shape[0]; R=Z.shape[1]
    q=np.repeat(qemb_group[:,None,:],M,axis=1).reshape(C*M,-1)
    z=np.repeat(Z[None,:,:],C,axis=0).reshape(C*M,R)
    b=np.repeat(belief[None,:],C*M,axis=0)
    kv=np.repeat(kvec[None,:],C*M,axis=0)
    rr=np.full((C*M,1),round_norm,dtype=np.float32)
    outs=[]
    for s in range(0,C*M,batch_size):
        e=min(C*M,s+batch_size)
        logits=model(
            torch.from_numpy(q[s:e]).to(device),
            torch.from_numpy(z[s:e]).to(device),
            torch.from_numpy(b[s:e]).to(device),
            torch.from_numpy(kv[s:e]).to(device),
            torch.from_numpy(rr[s:e]).to(device),
        )
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs,axis=0).reshape(C,M,R).astype(np.float64)


def predicted_mi(P,W,Klabels,Kprior):
    C,M,R=P.shape
    w=W/W.sum()
    rbar=np.einsum("m,cmr->cr",w,P)
    Hm=hbin(rbar).sum(axis=1)
    Hz=np.einsum("m,cmr->c",w,hbin(P))
    Hk=np.zeros(C)
    for k,pk in Kprior.items():
        mask=(Klabels==int(k))
        wk=W[mask]; wk=wk/wk.sum()
        Pk=P[:,mask,:]
        rk=np.einsum("m,cmr->cr",wk,Pk)
        Hk += pk*hbin(rk).sum(axis=1)
    mi_k=np.maximum(Hm-Hk,0)
    mi_id=np.maximum(Hk-Hz,0)
    mi_z=np.maximum(Hm-Hz,0)
    return mi_k,mi_id,mi_z,rbar.sum(axis=1)


def posterior_update_for_observation(Pcand,Y,W):
    """
    Pcand [M,R], Y [R].
    Product Bernoulli likelihood in log-space.
    """
    eps=1e-6
    p=np.clip(Pcand,eps,1-eps)
    y=Y[None,:]
    loglik=(y*np.log(p)+(1-y)*np.log(1-p)).sum(axis=1)
    lw=np.log(np.clip(W,1e-300,None))+loglik
    lw -= np.max(lw)
    wp=np.exp(lw)
    s=wp.sum()
    if not np.isfinite(s) or s<=0:
        return W/W.sum()
    return wp/s


def metrics_from_weights(W,Z,Klabels,true_z,true_k,k_values):
    W=W/W.sum()
    pk=np.array([W[Klabels==k].sum() for k in k_values],dtype=np.float64)
    pk=pk/max(pk.sum(),1e-12)
    rm=np.einsum("m,mr->r",W,Z)

    target_k=np.array([1.0 if k==true_k else 0.0 for k in k_values])
    k_brier=float(np.square(pk-target_k).sum())
    ptrue=float(pk[k_values.index(true_k)]) if true_k in k_values else 1e-12
    k_log=float(np.log(max(ptrue,1e-12)))
    expected_abs=float(sum(pk[j]*abs(k-true_k) for j,k in enumerate(k_values)))
    role_brier=float(np.square(rm-true_z).sum())
    return {
        "k_probs":pk,
        "role_marginals":rm,
        "k_brier":k_brier,
        "k_logprob":k_log,
        "k_expected_abs_error":expected_abs,
        "role_brier":role_brier,
    }


def summarize(stats):
    def m(k):
        v=stats.get(k,[])
        return float(np.mean(v)) if v else None
    return {
        "groups":len(stats.get("oracle_gain",[])),
        "mean_oracle_gain":m("oracle_gain"),
        "mean_selected_gain":m("selected_gain"),
        "gain_optimal_hit_rate":m("optimal"),
        "mean_regret":m("regret"),
        "mean_within_group_spearman":m("spearman"),
        "spearman_groups":len(stats.get("spearman",[])),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--out",default="artifacts/realized_posterior_gain_v1")
    ap.add_argument("--particles-per-k",type=int,default=64)
    ap.add_argument("--batch-size",type=int,default=2048)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=23)
    args=ap.parse_args()
    seed_all(args.seed)
    om=load_obs()

    obs_dir=Path(args.obs_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(obs_dir/"model.pt",map_location="cpu")
    roles=ckpt["role_order"]; role_to_i={r:i for i,r in enumerate(roles)}
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device

    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
        hidden=ckpt["hidden"],dropout=ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    examples,roles2,_,_=om.collect_examples(Path(args.root))
    if roles2!=roles: raise RuntimeError("role order mismatch")
    split=json.loads((obs_dir/"split_systems.json").read_text())
    test_systems=set(split["test"])
    test_idx=[i for i,x in enumerate(examples) if x["system_id"] in test_systems]
    qemb=np.load(obs_dir/"query_embeddings.npy")
    max_round=max(x["round"] for x in examples)

    traj=defaultdict(lambda:defaultdict(list))
    for i in test_idx:
        x=examples[i]
        traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))

    methods=["mi_k","mi_identity","mi_z","exposure","existing_selector"]
    objectives=["k_brier_gain","k_log_gain","k_abs_gain","role_brier_gain"]
    stats={obj:{m:defaultdict(list) for m in methods} for obj in objectives}
    stats_r1={obj:{m:defaultdict(list) for m in methods} for obj in objectives}
    random_stats={obj:[] for obj in objectives}
    random_stats_r1={obj:[] for obj in objectives}
    per_group=[]
    total=sum(len(rm) for rm in traj.values()); done=0

    for (parent,sid),rm in traj.items():
        for rnd in sorted(rm):
            done+=1
            rows=sorted(rm[rnd],key=lambda x:x[1]["candidate_index"])
            if len(rows)<2: continue
            x0=rows[0][1]

            belief=np.array([x0["role_probs"][r] for r in roles],dtype=np.float32)
            kvec=np.array([x0["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
            rnorm=float(rnd/max(max_round,1))
            rng=np.random.default_rng(abs(hash((parent,sid,rnd,args.seed)))%(2**32))
            Z,W,Klabels,Kprior=stratified_particles(
                x0["k_probs"],x0["role_probs"],roles,args.particles_per_k,rng
            )
            k_values=sorted(Kprior.keys())

            qg=np.stack([qemb[i] for i,_ in rows]).astype(np.float32)
            P=particle_predictions(model,qg,belief,kvec,rnorm,Z,device,args.batch_size)
            mi_k,mi_id,mi_z,exp=predicted_mi(P,W,Klabels,Kprior)

            true_roles=set(x0["truth_roles"])
            true_k=len(true_roles)
            true_z=np.zeros(len(roles),dtype=np.float64)
            for r in true_roles: true_z[role_to_i[r]]=1.0

            prior=metrics_from_weights(W,Z,Klabels,true_z,true_k,k_values)

            cands=[]
            for j,(idx,x) in enumerate(rows):
                Y=np.zeros(len(roles),dtype=np.float64)
                for r in x["selected_roles"]:
                    Y[role_to_i[r]]=1.0

                Wp=posterior_update_for_observation(P[j],Y,W)
                post=metrics_from_weights(Wp,Z,Klabels,true_z,true_k,k_values)

                gains={
                    "k_brier_gain":prior["k_brier"]-post["k_brier"],
                    "k_log_gain":post["k_logprob"]-prior["k_logprob"],
                    "k_abs_gain":prior["k_expected_abs_error"]-post["k_expected_abs_error"],
                    "role_brier_gain":prior["role_brier"]-post["role_brier"],
                }
                scores={
                    "mi_k":float(mi_k[j]),
                    "mi_identity":float(mi_id[j]),
                    "mi_z":float(mi_z[j]),
                    "exposure":float(exp[j]),
                    "existing_selector":float(x["selector_predicted_utility"]) if x["selector_predicted_utility"] is not None else float("-inf"),
                }
                cands.append({
                    "candidate_index":x["candidate_index"],
                    "query":x["query"],
                    "scores":scores,
                    "realized_gains_eval_only":gains,
                    "selected_roles_eval_only":list(x["selected_roles"]),
                })

            grec={"parent":parent,"system_id":sid,"round":rnd,"prior_metrics":{
                "k_brier":prior["k_brier"],"k_logprob":prior["k_logprob"],
                "k_expected_abs_error":prior["k_expected_abs_error"],"role_brier":prior["role_brier"]
            },"candidates":cands}

            for obj in objectives:
                oracle=max(cands,key=lambda c:(c["realized_gains_eval_only"][obj],-c["candidate_index"]))
                og=oracle["realized_gains_eval_only"][obj]
                random_stats[obj].append(float(np.mean([c["realized_gains_eval_only"][obj] for c in cands])))
                if rnd==1:
                    random_stats_r1[obj].append(random_stats[obj][-1])

                for method in methods:
                    ch=max(cands,key=lambda c:(c["scores"][method],-c["candidate_index"]))
                    gain=ch["realized_gains_eval_only"][obj]
                    rho=safe_spearman([c["scores"][method] for c in cands],
                                      [c["realized_gains_eval_only"][obj] for c in cands])
                    for block in [stats[obj][method]]+([stats_r1[obj][method]] if rnd==1 else []):
                        block["oracle_gain"].append(og)
                        block["selected_gain"].append(gain)
                        block["optimal"].append(int(ch["candidate_index"]==oracle["candidate_index"]))
                        block["regret"].append(og-gain)
                        if rho is not None: block["spearman"].append(rho)

            per_group.append(grec)
            if done%100==0: print(f"processed {done}/{total} groups")

    summary={
        "setting":{
            "true_Z_used_for_candidate_scoring":False,
            "recorded_Y_used_only_for_realized_posterior_update_evaluation":True,
            "particles_per_K":args.particles_per_k,
            "test_systems":len(test_systems),
            "test_rows":len(test_idx),
        },
        "round_1_clean":{},
        "all_rounds_one_step_conditional_on_recorded_state":{},
        "important_note":"This evaluates MI against downstream posterior quality, not against new-role exposure count. Y is still a latent training-time outcome, so this is not yet the final deployable black-box evaluation."
    }

    for obj in objectives:
        summary["round_1_clean"][obj]={
            "methods":{m:summarize(stats_r1[obj][m]) for m in methods},
            "random_mean_gain":float(np.mean(random_stats_r1[obj])) if random_stats_r1[obj] else None,
        }
        summary["all_rounds_one_step_conditional_on_recorded_state"][obj]={
            "methods":{m:summarize(stats[obj][m]) for m in methods},
            "random_mean_gain":float(np.mean(random_stats[obj])) if random_stats[obj] else None,
        }

    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"per_group.jsonl").open("w",encoding="utf-8") as f:
        for g in per_group: f.write(json.dumps(g,ensure_ascii=False)+"\n")

    print("\n"+"="*100)
    print("REALIZED POSTERIOR GAIN DIAGNOSTIC")
    print("="*100)
    for scope in ["round_1_clean","all_rounds_one_step_conditional_on_recorded_state"]:
        print("\n"+scope)
        for obj in objectives:
            print("\nOBJECTIVE:",obj,"random=",summary[scope][obj]["random_mean_gain"])
            for m in methods:
                x=summary[scope][obj]["methods"][m]
                print(f"{m:17s} gain={x['mean_selected_gain']} regret={x['mean_regret']} rho={x['mean_within_group_spearman']}")
    print("\n[saved]",out/"summary.json")

if __name__=="__main__":
    main()
