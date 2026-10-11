#!/usr/bin/env python3
"""
visible_adaptive_bridge_v1.py

First ALIGNED black-box adaptive-fingerprinting diagnostic.

Runtime-visible information:
    query q + final_output o

Training-only supervision:
    selected_roles_training_only, truth_roles_training_only

Pipeline tested:
    visible output -> fingerprint evidence E
    E + calibrated forward observation model -> Bayesian update over K
    current belief + candidate q -> expected VISIBLE information gain -> next-probe score

Important:
Later rounds are evaluated one-step conditional on the RECORDED state/history.
This is not yet a fully counterfactual multi-step replay.

No text embeddings are recomputed; reuses:
  artifacts/observation_model_v1/query_embeddings.npy
  artifacts/visible_output_fingerprint_v1/output_embeddings.npy
"""

from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, log_loss

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


def sigmoid(x):
    x=np.clip(x,-40,40)
    return 1/(1+np.exp(-x))


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def hbin(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    if spearmanr is not None:
        v=spearmanr(a,b).correlation
        return None if np.isnan(v) else float(v)
    return None


def suffix_esym(weights,kmax):
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
    s=sum(v for _,v in kp)
    kp=[(k,v/s) for k,v in kp]
    E=suffix_esym(odds,max(k for k,_ in kp))
    Z=[]; W=[]; K=[]
    for k,pk in kp:
        for _ in range(per_k):
            Z.append(sample_fixed_k(odds,k,rng,E))
            W.append(pk/per_k); K.append(k)
    return np.stack(Z).astype(np.float32),np.asarray(W,float),np.asarray(K,int),dict(kp)


def labels(examples,roles,field):
    ri={r:i for i,r in enumerate(roles)}
    Y=np.zeros((len(examples),len(roles)),dtype=np.int32)
    for i,x in enumerate(examples):
        for r in x[field]:
            if r in ri: Y[i,ri[r]]=1
    return Y


def fit_visible_detector(X,Y,train_idx,val_idx):
    """
    Balanced base classifier for discrimination, followed by per-role
    Platt calibration on held-out validation systems.
    """
    models=[]; calibrators=[]; thresholds=[]; tprs=[]; fprs=[]
    Pval=np.zeros((len(val_idx),Y.shape[1]),dtype=np.float64)

    for j in range(Y.shape[1]):
        yt=Y[train_idx,j]
        clf=LogisticRegression(
            max_iter=1500,class_weight="balanced",
            solver="liblinear",random_state=0
        )
        clf.fit(X[train_idx],yt)
        raw_val=clf.predict_proba(X[val_idx])[:,1]

        yv=Y[val_idx,j]
        cal=None
        if len(np.unique(yv))==2:
            cal=LogisticRegression(max_iter=500,solver="liblinear",random_state=0)
            cal.fit(logit(raw_val).reshape(-1,1),yv)
            pv=cal.predict_proba(logit(raw_val).reshape(-1,1))[:,1]
        else:
            pv=raw_val

        # threshold chosen ONLY on validation for F1
        best_t,best_f=0.5,-1
        for t in np.arange(0.05,0.951,0.05):
            f=f1_score(yv,(pv>=t).astype(int),zero_division=0)
            if f>best_f:
                best_f,best_t=f,float(t)

        ev=(pv>=best_t).astype(int)
        pos=max(int((yv==1).sum()),1)
        neg=max(int((yv==0).sum()),1)
        # smoothed detector channel
        tp=int(((ev==1)&(yv==1)).sum())
        fp=int(((ev==1)&(yv==0)).sum())
        tpr=(tp+1)/(pos+2)
        fpr=(fp+1)/(neg+2)

        models.append(clf); calibrators.append(cal); thresholds.append(best_t)
        tprs.append(tpr); fprs.append(fpr); Pval[:,j]=pv

    return models,calibrators,np.asarray(thresholds),np.asarray(tprs),np.asarray(fprs),Pval


def visible_probs(models,calibrators,X):
    P=np.zeros((len(X),len(models)),dtype=np.float64)
    for j,(clf,cal) in enumerate(zip(models,calibrators)):
        raw=clf.predict_proba(X)[:,1]
        if cal is not None:
            P[:,j]=cal.predict_proba(logit(raw).reshape(-1,1))[:,1]
        else:
            P[:,j]=raw
    return P


def detector_metrics(Y,P,T):
    pred=(P>=T[None,:]).astype(int)
    out={
        "micro_f1":float(f1_score(Y,pred,average="micro",zero_division=0)),
        "macro_f1":float(f1_score(Y,pred,average="macro",zero_division=0)),
        "brier":float(np.mean((P-Y)**2)),
        "true_cardinality":float(Y.sum(1).mean()),
        "pred_cardinality":float(pred.sum(1).mean()),
        "prob_cardinality":float(P.sum(1).mean()),
    }
    try:
        out["micro_auroc"]=float(roc_auc_score(Y,P,average="micro"))
    except Exception:
        out["micro_auroc"]=None
    return out


@torch.no_grad()
def particle_logits(model,qemb_group,belief,kvec,rnorm,Z,device,batch):
    C,M=qemb_group.shape[0],Z.shape[0]; R=Z.shape[1]
    q=np.repeat(qemb_group[:,None,:],M,axis=1).reshape(C*M,-1)
    z=np.repeat(Z[None,:,:],C,axis=0).reshape(C*M,R)
    b=np.repeat(belief[None,:],C*M,axis=0)
    kv=np.repeat(kvec[None,:],C*M,axis=0)
    rr=np.full((C*M,1),rnorm,dtype=np.float32)
    out=[]
    for s in range(0,C*M,batch):
        e=min(C*M,s+batch)
        out.append(model(
            torch.from_numpy(q[s:e]).to(device),
            torch.from_numpy(z[s:e]).to(device),
            torch.from_numpy(b[s:e]).to(device),
            torch.from_numpy(kv[s:e]).to(device),
            torch.from_numpy(rr[s:e]).to(device),
        ).cpu().numpy())
    return np.concatenate(out).reshape(C,M,R).astype(np.float64)


def calibrated_hidden_probs(raw_logits,log_posw,T):
    return sigmoid((raw_logits-log_posw)/T)


def detector_channel_probs(pY,tpr,fpr):
    # P(E_i=1 | Z,q), E is visible detector binary evidence
    return fpr[None,None,:] + (tpr-fpr)[None,None,:]*pY


def mi_k(P,W,Klabels,Kprior):
    # P [C,M,R] Bernoulli visible evidence probabilities
    w=W/W.sum()
    rbar=np.einsum("m,cmr->cr",w,P)
    Hm=hbin(rbar).sum(1)
    Hk=np.zeros(P.shape[0])
    for k,pk in Kprior.items():
        mask=(Klabels==int(k))
        wk=W[mask]; wk=wk/wk.sum()
        rk=np.einsum("m,cmr->cr",wk,P[:,mask,:])
        Hk += pk*hbin(rk).sum(1)
    return np.maximum(Hm-Hk,0)


def update_weights(Pe,evidence,W,beta):
    p=np.clip(Pe,1e-8,1-1e-8)
    e=evidence[None,:]
    ll=(e*np.log(p)+(1-e)*np.log(1-p)).sum(1)
    lw=np.log(np.clip(W,1e-300,None))+beta*ll
    lw-=lw.max()
    wp=np.exp(lw)
    return wp/wp.sum()


def k_metrics(W,Klabels,true_k):
    kvals=sorted(set(map(int,Klabels.tolist())))
    pk=np.array([W[Klabels==k].sum() for k in kvals],float)
    pk/=pk.sum()
    tgt=np.array([1 if k==true_k else 0 for k in kvals],float)
    i=kvals.index(true_k)
    return {
        "logprob":float(np.log(max(pk[i],1e-12))),
        "brier":float(((pk-tgt)**2).sum()),
        "abs":float(sum(pk[j]*abs(k-true_k) for j,k in enumerate(kvals))),
        "ptrue":float(pk[i]),
        "top1":int(kvals[int(np.argmax(pk))]==true_k),
    }


def mean(xs):
    return float(np.mean(xs)) if xs else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--vis-dir",default="artifacts/visible_output_fingerprint_v1")
    ap.add_argument("--cal-dir",default="artifacts/calibrated_bayes_diag_v1")
    ap.add_argument("--out",default="artifacts/visible_adaptive_bridge_v1")
    ap.add_argument("--particles-per-k",type=int,default=16)
    ap.add_argument("--batch-size",type=int,default=2048)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=37)
    args=ap.parse_args()

    seed_all(args.seed)
    om=load_obs()
    root=Path(args.root); obsdir=Path(args.obs_dir); visdir=Path(args.vis_dir)
    caldir=Path(args.cal_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    examples,roles,_,_=om.collect_examples(root)
    split=json.loads((obsdir/"split_systems.json").read_text())
    trsys,vasys,tesys=set(split["train"]),set(split["val"]),set(split["test"])
    tri=np.array([i for i,x in enumerate(examples) if x["system_id"] in trsys],int)
    vai=np.array([i for i,x in enumerate(examples) if x["system_id"] in vasys],int)
    tei=np.array([i for i,x in enumerate(examples) if x["system_id"] in tesys],int)

    Q=np.load(obsdir/"query_embeddings.npy").astype(np.float32)
    O=np.load(visdir/"output_embeddings.npy").astype(np.float32)
    X=np.concatenate([Q,O],axis=1)
    if len(X)!=len(examples): raise RuntimeError("embedding/example mismatch")

    Y=labels(examples,roles,"selected_roles")
    print("fitting visible fingerprint detector...")
    models,cals,thr,tpr,fpr,Pval=fit_visible_detector(X,Y,tri,vai)
    Ptest=visible_probs(models,cals,X[tei])
    det=detector_metrics(Y[tei],Ptest,thr)
    print("visible detector test:",det)
    print("mean TPR/FPR:",float(tpr.mean()),float(fpr.mean()))

    # Forward observation model + calibration constants
    ckpt=torch.load(obsdir/"model.pt",map_location="cpu")
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
        hidden=ckpt["hidden"],dropout=ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    role_to_i={r:i for i,r in enumerate(roles)}
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]["selected_roles"]:
            pos[role_to_i[r]]+=1
    neg=len(tri)-pos
    posw=np.clip(neg/np.maximum(pos,1.0),1.0,20.0)
    log_posw=np.log(posw)

    calsum=json.loads((caldir/"summary.json").read_text())
    T=float(calsum["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])
    beta=float(calsum["validation_posterior_update"]["selected_beta_by_validation_log_gain"])
    print("forward calibration T/beta:",T,beta)

    # Precompute visible binary evidence for every example (runtime-visible q+o only)
    Pall=visible_probs(models,cals,X)
    Eall=(Pall>=thr[None,:]).astype(np.float64)

    max_round=max(x["round"] for x in examples)
    traj=defaultdict(lambda:defaultdict(list))
    for i in tei:
        x=examples[i]
        traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))

    stats=defaultdict(list)
    per_group=[]
    total=sum(len(v) for v in traj.values()); done=0

    for (parent,sid),rm in traj.items():
        for rnd,rows in sorted(rm.items()):
            done+=1
            rows=sorted(rows,key=lambda ix:ix[1]["candidate_index"])
            if len(rows)<2: continue
            x0=rows[0][1]
            true_k=len(x0["truth_roles"])
            if true_k not in [int(k) for k in x0["k_probs"].keys()]:
                continue

            belief=np.array([x0["role_probs"][r] for r in roles],dtype=np.float32)
            kvec=np.array([x0["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
            rnorm=float(rnd/max(max_round,1))
            rng=np.random.default_rng(abs(hash((parent,sid,rnd,args.seed)))%(2**32))
            Z,W,Klabels,Kprior=stratified_particles(
                x0["k_probs"],x0["role_probs"],roles,args.particles_per_k,rng
            )

            qg=np.stack([Q[i] for i,_ in rows]).astype(np.float32)
            L=particle_logits(model,qg,belief,kvec,rnorm,Z,device,args.batch_size)
            pY=calibrated_hidden_probs(L,log_posw,T)
            pE=detector_channel_probs(pY,tpr,fpr)

            visible_scores=mi_k(pE,W,Klabels,Kprior)
            hidden_scores=mi_k(pY,W,Klabels,Kprior)

            prior=k_metrics(W,Klabels,true_k)
            cands=[]
            for j,(idx,x) in enumerate(rows):
                ev=Eall[idx]
                Wp=update_weights(pE[j],ev,W,beta)
                post=k_metrics(Wp,Klabels,true_k)
                gains={
                    "log":post["logprob"]-prior["logprob"],
                    "brier":prior["brier"]-post["brier"],
                    "abs":prior["abs"]-post["abs"],
                    "ptrue":post["ptrue"]-prior["ptrue"],
                    "top1_after":post["top1"],
                }
                cands.append({
                    "candidate_index":x["candidate_index"],
                    "visible_mi_k":float(visible_scores[j]),
                    "hidden_mi_k":float(hidden_scores[j]),
                    "old_score":float(x["selector_predicted_utility"]) if x["selector_predicted_utility"] is not None else -1e30,
                    "gains":gains,
                })

            oracle=max(cands,key=lambda c:(c["gains"]["log"],-c["candidate_index"]))
            vis=max(cands,key=lambda c:(c["visible_mi_k"],-c["candidate_index"]))
            hid=max(cands,key=lambda c:(c["hidden_mi_k"],-c["candidate_index"]))
            old=max(cands,key=lambda c:(c["old_score"],-c["candidate_index"]))

            for metric in ["log","brier","abs","ptrue"]:
                vals=[c["gains"][metric] for c in cands]
                stats[f"oracle_{metric}"].append(oracle["gains"][metric])
                stats[f"visible_mi_{metric}"].append(vis["gains"][metric])
                stats[f"hidden_mi_{metric}"].append(hid["gains"][metric])
                stats[f"old_{metric}"].append(old["gains"][metric])
                stats[f"random_{metric}"].append(float(np.mean(vals)))

            rho=safe_spearman(
                [c["visible_mi_k"] for c in cands],
                [c["gains"]["log"] for c in cands]
            )
            if rho is not None: stats["visible_mi_vs_log_gain_spearman"].append(rho)

            stats["visible_optimal_hit"].append(int(vis["candidate_index"]==oracle["candidate_index"]))
            stats["hidden_optimal_hit"].append(int(hid["candidate_index"]==oracle["candidate_index"]))
            stats["old_optimal_hit"].append(int(old["candidate_index"]==oracle["candidate_index"]))

            per_group.append({
                "system_id":sid,"round":rnd,"true_k":true_k,
                "prior":prior,"candidates":cands,
                "oracle_candidate":oracle["candidate_index"],
                "visible_mi_candidate":vis["candidate_index"],
                "hidden_mi_candidate":hid["candidate_index"],
                "old_candidate":old["candidate_index"],
            })

            if done%100==0: print(f"processed {done}/{total} groups")

    summary={
        "setting":{
            "test_systems":len(tesys),
            "particles_per_K":args.particles_per_k,
            "forward_temperature":T,
            "likelihood_beta":beta,
            "runtime_visible_inputs":["query","final_output"],
            "later_rounds":"one-step conditional on recorded history; not full counterfactual replay",
        },
        "visible_fingerprint_detector":{
            **det,
            "mean_tpr":float(tpr.mean()),
            "mean_fpr":float(fpr.mean()),
        },
        "one_step_visible_belief_update_and_selection":{
            k:mean(v) for k,v in stats.items()
        },
        "interpretation":{
            "visible_mi_k":"selector score based on predicted detector-visible evidence, not hidden selected_roles",
            "hidden_mi_k":"diagnostic comparator using latent activation channel",
            "oracle":"best candidate by realized visible-evidence K log-probability gain; evaluation only",
        }
    }

    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"per_group.jsonl").open("w",encoding="utf-8") as f:
        for g in per_group: f.write(json.dumps(g)+"\n")

    print("\n"+"="*100)
    print("VISIBLE ADAPTIVE BRIDGE V1")
    print("="*100)
    print("\nVISIBLE DETECTOR:",summary["visible_fingerprint_detector"])
    print("\nONE-STEP RESULTS")
    for k,v in summary["one_step_visible_belief_update_and_selection"].items():
        print(f"{k:42s}: {v}")
    print("\n[saved]",out/"summary.json")


if __name__=="__main__":
    main()
