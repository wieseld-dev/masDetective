#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_modules():
    sys.path.insert(0, str(Path("analysis").resolve()))
    import observation_model_v1 as om
    import matrix_selector_oracle_diag as ms
    return om, ms

def suffix_esym(weights, kmax):
    n=len(weights)
    E=np.zeros((n+1,kmax+1),dtype=np.float64)
    E[n,0]=1.0
    for i in range(n-1,-1,-1):
        E[i,0]=1.0
        for k in range(1,kmax+1):
            E[i,k]=E[i+1,k]+weights[i]*E[i+1,k-1]
    return E

def sample_fixed_k_subset(weights,k,rng,E):
    n=len(weights); z=np.zeros(n,dtype=np.float32); rem=int(k)
    for i in range(n):
        if rem==0: break
        left=n-i
        if left==rem:
            z[i:]=1.0; break
        denom=E[i,rem]
        if denom<=0:
            chosen=rng.choice(np.arange(i,n),size=rem,replace=False)
            z[chosen]=1.0; break
        pin=float(np.clip(weights[i]*E[i+1,rem-1]/denom,0,1))
        if rng.random()<pin:
            z[i]=1.0; rem-=1
    return z

def sample_particles(k_probs, role_probs, role_order, n_particles, rng):
    p=np.array([float(role_probs[r]) for r in role_order],dtype=np.float64)
    p=np.clip(p,1e-5,1-1e-5)
    odds=p/(1-p)
    ks=[]; ps=[]
    for k,v in k_probs.items():
        if float(v)>0:
            ks.append(int(k)); ps.append(float(v))
    ps=np.asarray(ps,dtype=np.float64); ps/=ps.sum()
    E=suffix_esym(odds,max(ks))
    out=[]
    for _ in range(n_particles):
        k=int(rng.choice(ks,p=ps))
        out.append(sample_fixed_k_subset(odds,k,rng,E))
    return np.stack(out).astype(np.float32)

def bern_entropy(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-8,1-1e-8)
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
def marginal_exposure(model,qemb_group,belief,kvec,round_norm,particles,device,batch_size):
    C,M=qemb_group.shape[0],particles.shape[0]
    N=particles.shape[1]
    q=np.repeat(qemb_group[:,None,:],M,axis=1).reshape(C*M,-1)
    z=np.repeat(particles[None,:,:],C,axis=0).reshape(C*M,N)
    b=np.repeat(belief[None,:],C*M,axis=0)
    kv=np.repeat(kvec[None,:],C*M,axis=0)
    rr=np.full((C*M,1),round_norm,dtype=np.float32)
    outs=[]
    for s in range(0,C*M,batch_size):
        e=min(C*M,s+batch_size)
        tq=torch.from_numpy(q[s:e]).to(device)
        tz=torch.from_numpy(z[s:e]).to(device)
        tb=torch.from_numpy(b[s:e]).to(device)
        tk=torch.from_numpy(kv[s:e]).to(device)
        tr=torch.from_numpy(rr[s:e]).to(device)
        outs.append(torch.sigmoid(model(tq,tz,tb,tk,tr)).cpu().numpy())
    return np.concatenate(outs,axis=0).reshape(C,M,N).mean(axis=1).astype(np.float64)

def covariance_scores(Sigma,rbar,noise):
    n=len(rbar); a=np.asarray(rbar,dtype=np.float64)
    if np.linalg.norm(a)<1e-12:
        return {"count_cov":0.0,"identity_cov":0.0,"eig1_cov":0.0}
    Sa=Sigma@a; denom=float(a@Sa+noise)
    one=np.ones(n)/math.sqrt(n)
    P=np.eye(n)-np.outer(one,one)
    count=float((one@Sa)**2/denom)
    identity=float(np.linalg.norm(P@Sa)**2/denom)
    Sid=P@Sigma@P
    vals,vecs=np.linalg.eigh(Sid)
    v1=vecs[:,int(np.argmax(vals))]
    eig1=float((v1@Sa)**2/denom)
    return {"count_cov":count,"identity_cov":identity,"eig1_cov":eig1}

def choose(cands,method):
    return max(cands,key=lambda c:(c["scores"][method],-c["candidate_index"]))

def summarize(stats):
    def m(k):
        v=stats.get(k,[])
        return float(np.mean(v)) if v else None
    return {
        "groups":len(stats.get("oracle_reward",[])),
        "mean_oracle_reward":m("oracle_reward"),
        "mean_selected_reward":m("selected_reward"),
        "reward_optimal_hit_rate":m("reward_optimal"),
        "exact_oracle_candidate_match_rate":m("exact_match"),
        "mean_regret":m("regret"),
        "mean_within_group_spearman":m("spearman"),
        "spearman_groups":len(stats.get("spearman",[])),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--out",default="artifacts/posterior_observation_selector_v1")
    ap.add_argument("--particles",type=int,default=64)
    ap.add_argument("--batch-size",type=int,default=2048)
    ap.add_argument("--noise",type=float,default=0.10)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=11)
    args=ap.parse_args()
    seed_all(args.seed)
    om,ms=load_modules()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    obs_dir=Path(args.obs_dir)
    ckpt=torch.load(obs_dir/"model.pt",map_location="cpu")
    role_order=ckpt["role_order"]
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    print("device:",device,"particles:",args.particles)
    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(role_order),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    examples,roles2,parents,skipped=om.collect_examples(Path(args.root))
    if roles2!=role_order: raise RuntimeError("role order mismatch")
    split_data=json.loads((obs_dir/"split_systems.json").read_text())
    test_systems=set(split_data["test"])
    test_idx=[i for i,x in enumerate(examples) if x["system_id"] in test_systems]
    qemb=np.load(obs_dir/"query_embeddings.npy")
    if len(qemb)!=len(examples): raise RuntimeError("embedding/example count mismatch")
    max_round=max(x["round"] for x in examples)

    traj=defaultdict(lambda:defaultdict(list))
    for i in test_idx:
        x=examples[i]
        traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))

    methods=["exposure","uncertainty","entropy","belief_novelty","count_cov","identity_cov","eig1_cov"]
    stats={m:defaultdict(list) for m in methods}
    stats_r1={m:defaultdict(list) for m in methods}
    baseline=defaultdict(list); baseline_r1=defaultdict(list)
    per_group=[]; done=0; total_groups=sum(len(v) for v in traj.values())

    for (parent,sid),rm in traj.items():
        seen_actual=set()
        for rnd in sorted(rm):
            done+=1
            rows=sorted(rm[rnd],key=lambda ix:ix[1]["candidate_index"])
            if len(rows)<2:
                actual=[x for _,x in rows if x["chosen"]]
                if len(actual)==1: seen_actual.update(actual[0]["selected_roles"])
                continue

            x0=rows[0][1]
            belief=np.array([x0["role_probs"][r] for r in role_order],dtype=np.float32)
            kvec=np.array([x0["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
            round_norm=float(rnd/max(max_round,1))
            rng=np.random.default_rng(abs(hash((parent,sid,rnd,args.seed)))%(2**32))
            particles=sample_particles(x0["k_probs"],x0["role_probs"],role_order,args.particles,rng)

            qg=np.stack([qemb[i] for i,_ in rows]).astype(np.float32)
            rbar=marginal_exposure(model,qg,belief,kvec,round_norm,particles,device,args.batch_size)
            _,Sigma,cov_info=ms.coherent_covariance(x0["k_probs"],x0["role_probs"],role_order)

            unc_w=4.0*belief*(1-belief)
            ent_w=bern_entropy(belief)
            nov_w=1.0-belief

            cands=[]
            for j,(i,x) in enumerate(rows):
                reward=len(set(x["selected_roles"])-seen_actual)
                scores={
                    "exposure":float(rbar[j].sum()),
                    "uncertainty":float(np.dot(unc_w,rbar[j])),
                    "entropy":float(np.dot(ent_w,rbar[j])),
                    "belief_novelty":float(np.dot(nov_w,rbar[j])),
                }
                scores.update(covariance_scores(Sigma,rbar[j],args.noise))
                cands.append({
                    "candidate_index":x["candidate_index"],
                    "query":x["query"],
                    "selected_roles_hidden_eval_only":list(x["selected_roles"]),
                    "oracle_new_hidden_eval_only":reward,
                    "scores":scores,
                    "existing_selector_score":x["selector_predicted_utility"],
                    "chosen_actual":x["chosen"],
                })

            oracle=max(cands,key=lambda c:(c["oracle_new_hidden_eval_only"],len(c["selected_roles_hidden_eval_only"]),-c["candidate_index"]))
            oracle_reward=oracle["oracle_new_hidden_eval_only"]
            group_rec={"parent":parent,"system_id":sid,"round":rnd,"pool_size":len(cands),
                       "seen_actual_hidden_eval_only":sorted(seen_actual),
                       "oracle_candidate_hidden_eval_only":oracle["candidate_index"],
                       "oracle_reward_hidden_eval_only":oracle_reward,"covariance_info":cov_info,"candidates":cands}

            for method in methods:
                ch=choose(cands,method)
                reward=ch["oracle_new_hidden_eval_only"]
                rho=safe_spearman([c["scores"][method] for c in cands],
                                  [c["oracle_new_hidden_eval_only"] for c in cands])
                for block in [stats[method]]+([stats_r1[method]] if rnd==1 else []):
                    block["oracle_reward"].append(oracle_reward)
                    block["selected_reward"].append(reward)
                    block["reward_optimal"].append(int(reward==oracle_reward))
                    block["exact_match"].append(int(ch["candidate_index"]==oracle["candidate_index"]))
                    block["regret"].append(oracle_reward-reward)
                    if rho is not None: block["spearman"].append(rho)
                group_rec[f"{method}_candidate"]=ch["candidate_index"]
                group_rec[f"{method}_reward"]=reward
                group_rec[f"{method}_regret"]=oracle_reward-reward
                group_rec[f"{method}_spearman"]=rho

            random_expected=float(np.mean([c["oracle_new_hidden_eval_only"] for c in cands]))
            baseline["random_reward"].append(random_expected)
            if rnd==1: baseline_r1["random_reward"].append(random_expected)

            with_old=[c for c in cands if c["existing_selector_score"] is not None]
            if len(with_old)==len(cands):
                old=max(with_old,key=lambda c:(float(c["existing_selector_score"]),-c["candidate_index"]))
                baseline["existing_reward"].append(old["oracle_new_hidden_eval_only"])
                if rnd==1: baseline_r1["existing_reward"].append(old["oracle_new_hidden_eval_only"])

            actual=[c for c in cands if c["chosen_actual"]]
            if len(actual)==1:
                baseline["actual_reward"].append(actual[0]["oracle_new_hidden_eval_only"])
                if rnd==1: baseline_r1["actual_reward"].append(actual[0]["oracle_new_hidden_eval_only"])
                seen_actual.update(actual[0]["selected_roles_hidden_eval_only"])

            per_group.append(group_rec)
            if done%100==0: print(f"processed {done}/{total_groups} groups")

    def base_summary(b):
        return {
            "random_mean_reward":float(np.mean(b["random_reward"])) if b["random_reward"] else None,
            "existing_selector_mean_reward":float(np.mean(b["existing_reward"])) if b["existing_reward"] else None,
            "actual_chosen_mean_reward":float(np.mean(b["actual_reward"])) if b["actual_reward"] else None,
        }

    summary={
        "setting":{"selection_uses_true_Z":False,"hidden_selected_roles_used_only_for_evaluation":True,
                   "posterior_particles":args.particles,"test_systems":len(test_systems),
                   "test_rows":len(test_idx),"teacher_forced_state_history":True},
        "round_1_clean":{"methods":{m:summarize(stats_r1[m]) for m in methods},"baselines":base_summary(baseline_r1)},
        "all_rounds_one_step_conditional_on_actual_history":{"methods":{m:summarize(stats[m]) for m in methods},"baselines":base_summary(baseline)},
        "notes":{
            "important":"Candidate scores marginalize the trained observation model over posterior-sampled hidden maps. Ground-truth Z is NOT an input to the selector at evaluation time.",
            "next_if_good":"Replace heuristic exposure scores with Monte-Carlo Expected Information Gain over K and/or Z."
        }
    }

    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out/"per_group.jsonl").open("w",encoding="utf-8") as f:
        for row in per_group: f.write(json.dumps(row,ensure_ascii=False)+"\n")

    print("\n"+"="*100)
    print("POSTERIOR-MARGINALIZED OBSERVATION SELECTOR V1")
    print("="*100)
    for block_name in ["round_1_clean","all_rounds_one_step_conditional_on_actual_history"]:
        print("\n"+block_name)
        block=summary[block_name]
        print("BASELINES:",block["baselines"])
        for m in methods:
            x=block["methods"][m]
            print(f"{m:15s} reward={x['mean_selected_reward']} opt={x['reward_optimal_hit_rate']} regret={x['mean_regret']} rho={x['mean_within_group_spearman']}")
    print("\n[saved]",out/"summary.json")

if __name__=="__main__":
    main()
