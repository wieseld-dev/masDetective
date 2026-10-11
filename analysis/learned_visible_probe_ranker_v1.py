#!/usr/bin/env python3
"""
learned_visible_probe_ranker_v1.py

Learned next-probe utility ranker aligned with the actual black-box goal.

At SELECTION time it uses only deployable/pre-query information:
  - current belief P(K), P(role)
  - candidate query embedding
  - forward-model prediction P(hidden activation | Z,q,belief)
  - predicted VISIBLE fingerprint channel P(E_visible | Z,q,belief)
  - derived disagreement / MI features
  - old selector score and round

It NEVER uses the candidate's actual output, selected_roles, truth_roles,
or true K as an input feature.

Training/evaluation target:
  realized change in log P(K_true) after seeing the candidate's ACTUAL visible
  output, passing it through the visible fingerprint detector, and performing
  the calibrated/tempered Bayesian update.

Splits are by SYSTEM:
  train systems -> ranker training
  validation systems -> model/config selection
  test systems -> final untouched evaluation

This is still a one-step ranking experiment on recorded states, not a
counterfactual multi-step adaptive rollout.
"""

from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def mean(xs):
    return float(np.mean(xs)) if xs else None


def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    if spearmanr is None:
        return None
    v=spearmanr(a,b).correlation
    return None if np.isnan(v) else float(v)


def hbin(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def rolewise_k_mi(P,W,Klabels,Kprior):
    # P [C,M,R]
    w=W/W.sum()
    rbar=np.einsum("m,cmr->cr",w,P)
    out=hbin(rbar)
    for k,pk in Kprior.items():
        mask=(Klabels==int(k))
        if not np.any(mask): continue
        wk=W[mask]; wk=wk/wk.sum()
        rk=np.einsum("m,cmr->cr",wk,P[:,mask,:])
        out -= float(pk)*hbin(rk)
    return np.maximum(out,0.0)


def weighted_particle_var(P,W):
    # P [C,M,R]
    w=W/W.sum()
    mu=np.einsum("m,cmr->cr",w,P)
    second=np.einsum("m,cmr->cr",w,P*P)
    return np.maximum(second-mu*mu,0.0),mu


def k_conditional_range(P,W,Klabels,Kprior):
    mus=[]
    for k,_pk in Kprior.items():
        mask=(Klabels==int(k))
        if not np.any(mask): continue
        wk=W[mask]; wk=wk/wk.sum()
        mus.append(np.einsum("m,cmr->cr",wk,P[:,mask,:]))
    if not mus:
        return np.zeros((P.shape[0],P.shape[2]),dtype=np.float64)
    A=np.stack(mus,axis=0)  # K,C,R
    return A.max(0)-A.min(0)


def load_helpers():
    sys.path.insert(0,str(Path("analysis").resolve()))
    import observation_model_v1 as om
    import visible_adaptive_bridge_v1 as vb
    return om,vb


def build_feature_names(n_roles,pca_dim):
    names=[]
    names += [f"q_pca_{i}" for i in range(pca_dim)]
    names += [f"belief_role_{i}" for i in range(n_roles)]
    names += [f"kprob_{k}" for k in [3,4,5,6]]
    names += ["round_norm","k_entropy","old_selector_score","visible_mi_k","hidden_mi_k"]
    for prefix in ["visible_mean","visible_var","visible_role_mi","visible_k_range",
                   "hidden_mean","hidden_var"]:
        names += [f"{prefix}_{i}" for i in range(n_roles)]
    names += ["expected_visible_cardinality","visible_total_variance",
              "visible_total_role_mi","visible_total_k_range"]
    return names


def candidate_features(qpca,belief,kvec,rnorm,old_scores,visible_mi,hidden_mi,
                       pE,pY,W,Klabels,Kprior):
    C=pE.shape[0]
    vvar,vmean=weighted_particle_var(pE,W)
    yvar,ymean=weighted_particle_var(pY,W)
    vmi=rolewise_k_mi(pE,W,Klabels,Kprior)
    vrange=k_conditional_range(pE,W,Klabels,Kprior)

    kp=np.asarray(kvec,float)
    kpn=kp/(kp.sum()+1e-12)
    kent=float(-(kpn*np.log(np.clip(kpn,1e-12,1))).sum())

    common=np.concatenate([
        np.tile(np.asarray(belief,float)[None,:],(C,1)),
        np.tile(np.asarray(kvec,float)[None,:],(C,1)),
        np.full((C,1),float(rnorm)),
        np.full((C,1),kent),
        np.asarray(old_scores,float)[:,None],
        np.asarray(visible_mi,float)[:,None],
        np.asarray(hidden_mi,float)[:,None],
    ],axis=1)

    tail=np.concatenate([
        vmean,vvar,vmi,vrange,ymean,yvar,
        vmean.sum(1,keepdims=True),
        vvar.sum(1,keepdims=True),
        vmi.sum(1,keepdims=True),
        vrange.sum(1,keepdims=True),
    ],axis=1)
    return np.concatenate([qpca,common,tail],axis=1).astype(np.float32)


def collect_split(split_name,system_set,examples,Q,Qpca,Eall,roles,model,device,
                  log_posw,T,beta,tpr,fpr,vb,particles_per_k,batch_size,max_round,seed):
    traj=defaultdict(lambda:defaultdict(list))
    for i,x in enumerate(examples):
        if x["system_id"] in system_set:
            traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))

    Xrows=[]; yrows=[]; meta=[]; groups=[]
    gid=0
    total=sum(len(v) for v in traj.values())
    done=0

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
            rng=np.random.default_rng(abs(hash((split_name,parent,sid,rnd,seed)))%(2**32))
            Z,W,Klabels,Kprior=vb.stratified_particles(
                x0["k_probs"],x0["role_probs"],roles,particles_per_k,rng
            )

            qg=np.stack([Q[i] for i,_ in rows]).astype(np.float32)
            L=vb.particle_logits(model,qg,belief,kvec,rnorm,Z,device,batch_size)
            pY=vb.calibrated_hidden_probs(L,log_posw,T)
            pE=vb.detector_channel_probs(pY,tpr,fpr)
            visible_mi=vb.mi_k(pE,W,Klabels,Kprior)
            hidden_mi=vb.mi_k(pY,W,Klabels,Kprior)

            old_scores=[
                float(x["selector_predicted_utility"]) if x["selector_predicted_utility"] is not None else 0.0
                for _,x in rows
            ]
            F=candidate_features(
                np.stack([Qpca[i] for i,_ in rows]),
                belief,kvec,rnorm,old_scores,visible_mi,hidden_mi,
                pE,pY,W,Klabels,Kprior
            )

            prior=vb.k_metrics(W,Klabels,true_k)
            gains=[]
            for j,(idx,x) in enumerate(rows):
                Wp=vb.update_weights(pE[j],Eall[idx],W,beta)
                post=vb.k_metrics(Wp,Klabels,true_k)
                gains.append({
                    "log":post["logprob"]-prior["logprob"],
                    "brier":prior["brier"]-post["brier"],
                    "abs":prior["abs"]-post["abs"],
                    "ptrue":post["ptrue"]-prior["ptrue"],
                })

            raw=np.array([g["log"] for g in gains],dtype=np.float64)
            centered=raw-raw.mean()

            start=len(Xrows)
            for j,(idx,x) in enumerate(rows):
                Xrows.append(F[j])
                yrows.append(centered[j])
                meta.append({
                    "group_id":gid,
                    "system_id":sid,
                    "round":rnd,
                    "candidate_index":int(x["candidate_index"]),
                    "realized_log_gain":float(raw[j]),
                    "realized_brier_gain":float(gains[j]["brier"]),
                    "realized_abs_gain":float(gains[j]["abs"]),
                    "realized_ptrue_gain":float(gains[j]["ptrue"]),
                    "visible_mi":float(visible_mi[j]),
                    "hidden_mi":float(hidden_mi[j]),
                    "old_score":float(old_scores[j]),
                })
            groups.append(list(range(start,len(Xrows))))
            gid+=1

            if done%250==0:
                print(f"[{split_name}] processed {done}/{total} recorded states",flush=True)

    return np.asarray(Xrows,np.float32),np.asarray(yrows,np.float32),meta,groups


def evaluate_scores(pred,meta,groups):
    acc=defaultdict(list)
    for g in groups:
        ms=[meta[i] for i in g]
        pp=np.array([pred[i] for i in g],float)
        realized=np.array([m["realized_log_gain"] for m in ms],float)

        learned=int(np.argmax(pp))
        oracle=int(np.argmax(realized))
        vis=int(np.argmax([m["visible_mi"] for m in ms]))
        hid=int(np.argmax([m["hidden_mi"] for m in ms]))
        old=int(np.argmax([m["old_score"] for m in ms]))

        def add(prefix,j):
            m=ms[j]
            acc[f"{prefix}_log"].append(m["realized_log_gain"])
            acc[f"{prefix}_brier"].append(m["realized_brier_gain"])
            acc[f"{prefix}_abs"].append(m["realized_abs_gain"])
            acc[f"{prefix}_ptrue"].append(m["realized_ptrue_gain"])

        add("learned",learned); add("oracle",oracle); add("visible_mi",vis)
        add("hidden_mi",hid); add("old",old)

        for metric in ["log","brier","abs","ptrue"]:
            acc[f"random_{metric}"].append(float(np.mean([
                m[f"realized_{metric}_gain"] for m in ms
            ])))

        acc["learned_optimal_hit"].append(int(learned==oracle))
        acc["visible_mi_optimal_hit"].append(int(vis==oracle))
        acc["hidden_mi_optimal_hit"].append(int(hid==oracle))
        acc["old_optimal_hit"].append(int(old==oracle))

        rho=safe_spearman(pp,realized)
        if rho is not None: acc["learned_vs_log_gain_spearman"].append(rho)

    return {k:mean(v) for k,v in acc.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--vis-dir",default="artifacts/visible_output_fingerprint_v1")
    ap.add_argument("--cal-dir",default="artifacts/calibrated_bayes_diag_v1")
    ap.add_argument("--out",default="artifacts/learned_visible_probe_ranker_v1")
    ap.add_argument("--particles-per-k",type=int,default=8)
    ap.add_argument("--pca-dim",type=int,default=24)
    ap.add_argument("--batch-size",type=int,default=4096)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=51)
    args=ap.parse_args()

    seed_all(args.seed)
    om,vb=load_helpers()
    root=Path(args.root); obsdir=Path(args.obs_dir); visdir=Path(args.vis_dir)
    caldir=Path(args.cal_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    print("loading examples/embeddings...",flush=True)
    examples,roles,_,_=om.collect_examples(root)
    split=json.loads((obsdir/"split_systems.json").read_text())
    trsys,vasys,tesys=set(split["train"]),set(split["val"]),set(split["test"])

    Q=np.load(obsdir/"query_embeddings.npy").astype(np.float32)
    O=np.load(visdir/"output_embeddings.npy").astype(np.float32)
    Xtxt=np.concatenate([Q,O],axis=1)

    tri=np.array([i for i,x in enumerate(examples) if x["system_id"] in trsys],int)
    vai=np.array([i for i,x in enumerate(examples) if x["system_id"] in vasys],int)
    tei=np.array([i for i,x in enumerate(examples) if x["system_id"] in tesys],int)

    Y=vb.labels(examples,roles,"selected_roles")
    print("fitting visible fingerprint detector on TRAIN, calibrating on VAL...",flush=True)
    fp_models,fp_cals,thr,tpr,fpr,_=vb.fit_visible_detector(Xtxt,Y,tri,vai)
    Pall=vb.visible_probs(fp_models,fp_cals,Xtxt)
    Eall=(Pall>=thr[None,:]).astype(np.float64)

    print("fitting query PCA on TRAIN only...",flush=True)
    pca=PCA(n_components=args.pca_dim,svd_solver="randomized",random_state=args.seed)
    pca.fit(Q[tri])
    Qpca=pca.transform(Q).astype(np.float32)

    print("loading calibrated forward observation model...",flush=True)
    ckpt=torch.load(obsdir/"model.pt",map_location="cpu")
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
        hidden=ckpt["hidden"],dropout=ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    ri={r:i for i,r in enumerate(roles)}
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]["selected_roles"]: pos[ri[r]]+=1
    neg=len(tri)-pos
    posw=np.clip(neg/np.maximum(pos,1.0),1.0,20.0)
    log_posw=np.log(posw)

    cs=json.loads((caldir/"summary.json").read_text())
    T=float(cs["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])
    beta=float(cs["validation_posterior_update"]["selected_beta_by_validation_log_gain"])
    max_round=max(x["round"] for x in examples)

    print("building deployable pre-query features + realized visible-output K-gain labels...",flush=True)
    common=dict(
        examples=examples,Q=Q,Qpca=Qpca,Eall=Eall,roles=roles,model=model,
        device=device,log_posw=log_posw,T=T,beta=beta,tpr=tpr,fpr=fpr,vb=vb,
        particles_per_k=args.particles_per_k,batch_size=args.batch_size,
        max_round=max_round,seed=args.seed
    )
    Xtr,ytr,mtr,gtr=collect_split("train",trsys,**common)
    Xva,yva,mva,gva=collect_split("val",vasys,**common)
    Xte,yte,mte,gte=collect_split("test",tesys,**common)

    print("rows/groups:",
          {"train":(len(Xtr),len(gtr)),"val":(len(Xva),len(gva)),"test":(len(Xte),len(gte))},
          flush=True)

    feature_names=build_feature_names(len(roles),args.pca_dim)
    if Xtr.shape[1]!=len(feature_names):
        raise RuntimeError((Xtr.shape,len(feature_names)))

    configs=[
        {"name":"hgb_small","max_iter":120,"learning_rate":0.05,"max_leaf_nodes":15,"l2_regularization":2.0},
        {"name":"hgb_mid","max_iter":200,"learning_rate":0.05,"max_leaf_nodes":31,"l2_regularization":3.0},
        {"name":"hgb_fast","max_iter":140,"learning_rate":0.10,"max_leaf_nodes":15,"l2_regularization":3.0},
    ]
    val_results=[]
    fitted=[]

    print("training rankers...",flush=True)
    for c in configs:
        mdl=HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=c["max_iter"],
            learning_rate=c["learning_rate"],
            max_leaf_nodes=c["max_leaf_nodes"],
            l2_regularization=c["l2_regularization"],
            min_samples_leaf=30,
            random_state=args.seed,
        )
        mdl.fit(Xtr,ytr)
        pv=mdl.predict(Xva)
        ev=evaluate_scores(pv,mva,gva)
        val_results.append({"config":c,"metrics":ev})
        fitted.append(mdl)
        print(c["name"],"VAL learned_log=",ev.get("learned_log"),
              "random=",ev.get("random_log"),
              "oracle=",ev.get("oracle_log"),
              "rho=",ev.get("learned_vs_log_gain_spearman"),flush=True)

    # Linear deployable sanity baseline
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    ridge.fit(Xtr,ytr)
    ridge_val=evaluate_scores(ridge.predict(Xva),mva,gva)

    best_i=max(range(len(fitted)),key=lambda i:val_results[i]["metrics"].get("learned_log",-1e30))
    best=fitted[best_i]
    best_cfg=configs[best_i]

    # Refit best config on TRAIN+VAL; validation was only used to select config.
    Xtv=np.concatenate([Xtr,Xva],axis=0)
    ytv=np.concatenate([ytr,yva],axis=0)
    final=HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=best_cfg["max_iter"],
        learning_rate=best_cfg["learning_rate"],
        max_leaf_nodes=best_cfg["max_leaf_nodes"],
        l2_regularization=best_cfg["l2_regularization"],
        min_samples_leaf=30,
        random_state=args.seed,
    )
    final.fit(Xtv,ytv)
    pte=final.predict(Xte)
    test_metrics=evaluate_scores(pte,mte,gte)

    # Also evaluate the train-only selected model on test as a strict no-val-refit check.
    strict_test=evaluate_scores(best.predict(Xte),mte,gte)

    summary={
        "setting":{
            "runtime_selector_inputs_only":"belief + candidate query + forward predicted hidden/visible fingerprint distributions; NO actual candidate output",
            "training_label":"realized visible-output Bayesian delta log P(K_true), group-centered",
            "split_by_system":True,
            "train_systems":len(trsys),"val_systems":len(vasys),"test_systems":len(tesys),
            "particles_per_K":args.particles_per_k,
            "query_pca_dim":args.pca_dim,
            "forward_temperature":T,
            "likelihood_beta":beta,
            "later_rounds":"one-step conditional on recorded history; not full counterfactual rollout",
        },
        "dataset":{
            "train_rows":len(Xtr),"train_groups":len(gtr),
            "val_rows":len(Xva),"val_groups":len(gva),
            "test_rows":len(Xte),"test_groups":len(gte),
            "feature_dim":int(Xtr.shape[1]),
        },
        "validation_configs":val_results,
        "validation_ridge_baseline":ridge_val,
        "selected_config":best_cfg,
        "test_selected_model_train_only":strict_test,
        "test_final_refit_train_plus_val":test_metrics,
        "interpretation":{
            "learned_log":"realized visible-output K log-probability gain of the candidate chosen by learned ranker",
            "random_log":"mean realized gain over candidates in each group",
            "oracle_log":"best available candidate's realized gain; evaluation-only ceiling",
            "visible_mi_log":"previous analytic expected-visible-MI selector",
            "learned_vs_log_gain_spearman":"within-group correlation between learned score and realized visible K gain",
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    with (out/"test_predictions.jsonl").open("w",encoding="utf-8") as f:
        for i,m in enumerate(mte):
            z=dict(m); z["learned_score"]=float(pte[i]); f.write(json.dumps(z)+"\n")

    print("\n"+"="*100)
    print("LEARNED VISIBLE PROBE RANKER V1")
    print("="*100)
    print("selected config:",best_cfg)
    print("\nSTRICT TEST (train-only model):")
    for k,v in strict_test.items(): print(f"{k:44s}: {v}")
    print("\nFINAL TEST (refit train+val):")
    for k,v in test_metrics.items(): print(f"{k:44s}: {v}")
    print("\n[saved]",out/"summary.json")


if __name__=="__main__":
    main()
