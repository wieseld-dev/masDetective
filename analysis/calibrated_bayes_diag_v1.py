#!/usr/bin/env python3
"""
calibrated_bayes_diag_v1.py

Diagnose and repair the probability layer used by Bayesian belief updates.

Critical fact:
observation_model_v1 was trained with BCEWithLogitsLoss(pos_weight=w_i).
That is good for imbalanced classification, but raw sigmoid(logit) is NOT a
calibrated estimate of P(Y_i=1|Z,q,b). At the population optimum:

    logit(raw_output_i) ~= logit(true_probability_i) + log(w_i)

So the first analytical correction is:

    corrected_logit_i = raw_logit_i - log(w_i)

Then:
  1) fit one temperature T on held-out validation systems,
  2) optionally fit a likelihood-power beta on validation systems to temper
     the product of 20 approximately dependent Bernoulli outputs,
  3) evaluate posterior K updates on untouched test systems,
  4) recompute MI_K selection using calibrated probabilities.

Ground-truth Z and recorded selected_roles are used only for retrospective
calibration/evaluation, never as candidate-score inputs on test.
"""

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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_obs():
    sys.path.insert(0, str(Path("analysis").resolve()))
    import observation_model_v1 as om
    return om


def sigmoid_np(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def hbin(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def binary_nll(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())


def brier(y, p):
    return float(np.square(y - p).mean())


def ece_binary(y, p, bins=15):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        out += float(mask.sum()) / n * abs(conf - acc)
    return float(out)


def suffix_esym(weights, kmax):
    n = len(weights)
    E = np.zeros((n + 1, kmax + 1), dtype=np.float64)
    E[n, 0] = 1.0
    for i in range(n - 1, -1, -1):
        E[i, 0] = 1.0
        for k in range(1, kmax + 1):
            E[i, k] = E[i+1, k] + weights[i] * E[i+1, k-1]
    return E


def sample_fixed_k(weights, k, rng, E):
    n = len(weights)
    z = np.zeros(n, dtype=np.float32)
    rem = int(k)
    for i in range(n):
        if rem == 0:
            break
        if n - i == rem:
            z[i:] = 1.0
            break
        den = E[i, rem]
        if den <= 0:
            ids = rng.choice(np.arange(i, n), size=rem, replace=False)
            z[ids] = 1.0
            break
        pin = float(np.clip(weights[i] * E[i+1, rem-1] / den, 0, 1))
        if rng.random() < pin:
            z[i] = 1.0
            rem -= 1
    return z


def stratified_particles(k_probs, role_probs, roles, per_k, rng):
    p = np.array([float(role_probs[r]) for r in roles], dtype=np.float64)
    p = np.clip(p, 1e-5, 1-1e-5)
    odds = p / (1-p)

    kp = [(int(k), float(v)) for k, v in k_probs.items() if float(v) > 0]
    norm = sum(v for _, v in kp)
    kp = [(k, v/norm) for k, v in kp]

    E = suffix_esym(odds, max(k for k, _ in kp))
    Z, W, K = [], [], []
    for k, pk in kp:
        for _ in range(per_k):
            Z.append(sample_fixed_k(odds, k, rng, E))
            W.append(pk / per_k)
            K.append(k)
    return (
        np.stack(Z).astype(np.float32),
        np.asarray(W, dtype=np.float64),
        np.asarray(K, dtype=np.int64),
        dict(kp),
    )


def make_arrays(examples, indices, roles, max_round):
    role_to_i = {r:i for i,r in enumerate(roles)}
    qidx, Z, B, KV, RR, Y = [], [], [], [], [], []

    for i in indices:
        x = examples[i]
        z = np.zeros(len(roles), dtype=np.float32)
        y = np.zeros(len(roles), dtype=np.float32)
        for r in x["truth_roles"]:
            z[role_to_i[r]] = 1.0
        for r in x["selected_roles"]:
            y[role_to_i[r]] = 1.0

        qidx.append(i)
        Z.append(z)
        B.append(np.array([x["role_probs"][r] for r in roles], dtype=np.float32))
        KV.append(np.array([x["k_probs"].get(str(k), 0.0) for k in [3,4,5,6]], dtype=np.float32))
        RR.append(np.array([x["round"]/max(max_round,1)], dtype=np.float32))
        Y.append(y)

    return (
        np.asarray(qidx, dtype=np.int64),
        np.stack(Z), np.stack(B), np.stack(KV), np.stack(RR), np.stack(Y)
    )


@torch.no_grad()
def true_z_logits(model, qemb, examples, indices, roles, max_round, device, batch_size):
    qidx, Z, B, KV, RR, Y = make_arrays(examples, indices, roles, max_round)
    outs = []
    for s in range(0, len(indices), batch_size):
        e = min(len(indices), s + batch_size)
        outs.append(
            model(
                torch.from_numpy(qemb[qidx[s:e]]).to(device),
                torch.from_numpy(Z[s:e]).to(device),
                torch.from_numpy(B[s:e]).to(device),
                torch.from_numpy(KV[s:e]).to(device),
                torch.from_numpy(RR[s:e]).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(outs, axis=0).astype(np.float64), Y.astype(np.float64)


@torch.no_grad()
def particle_logits(model, qemb_group, belief, kvec, round_norm, Z, device, batch_size):
    C, M = qemb_group.shape[0], Z.shape[0]
    R = Z.shape[1]

    q = np.repeat(qemb_group[:,None,:], M, axis=1).reshape(C*M, -1)
    z = np.repeat(Z[None,:,:], C, axis=0).reshape(C*M, R)
    b = np.repeat(belief[None,:], C*M, axis=0)
    kv = np.repeat(kvec[None,:], C*M, axis=0)
    rr = np.full((C*M, 1), round_norm, dtype=np.float32)

    out = []
    for s in range(0, C*M, batch_size):
        e = min(C*M, s + batch_size)
        logits = model(
            torch.from_numpy(q[s:e]).to(device),
            torch.from_numpy(z[s:e]).to(device),
            torch.from_numpy(b[s:e]).to(device),
            torch.from_numpy(kv[s:e]).to(device),
            torch.from_numpy(rr[s:e]).to(device),
        )
        out.append(logits.cpu().numpy())

    return np.concatenate(out, axis=0).reshape(C, M, R).astype(np.float64)


def probs_from_logits(raw_logits, log_pos_weight, temperature=1.0, correct_posweight=True):
    L = raw_logits.copy()
    if correct_posweight:
        L = L - log_pos_weight
    L = L / float(temperature)
    return sigmoid_np(L)


def update_weights(Pcand, Y, W, beta=1.0):
    eps = 1e-8
    p = np.clip(Pcand, eps, 1-eps)
    y = Y[None,:]
    loglik = (y*np.log(p) + (1-y)*np.log(1-p)).sum(axis=1)
    lw = np.log(np.clip(W,1e-300,None)) + float(beta)*loglik
    lw -= np.max(lw)
    wp = np.exp(lw)
    s = wp.sum()
    if not np.isfinite(s) or s <= 0:
        return W / W.sum()
    return wp / s


def k_distribution(W, Klabels, kvals):
    W = W / W.sum()
    p = np.array([W[Klabels==k].sum() for k in kvals], dtype=np.float64)
    return p / max(p.sum(), 1e-12)


def k_metrics(pk, kvals, true_k):
    target = np.array([1.0 if k==true_k else 0.0 for k in kvals])
    true_i = kvals.index(true_k)
    return {
        "brier": float(np.square(pk-target).sum()),
        "logprob": float(np.log(max(pk[true_i],1e-12))),
        "abs_error": float(sum(pk[j]*abs(k-true_k) for j,k in enumerate(kvals))),
        "p_true": float(pk[true_i]),
        "top1_correct": int(kvals[int(np.argmax(pk))] == true_k),
    }


def mi_k_score(P, W, Klabels, Kprior):
    # P [C,M,R]
    C = P.shape[0]
    w = W/W.sum()
    rbar = np.einsum("m,cmr->cr", w, P)
    Hm = hbin(rbar).sum(axis=1)

    Hk = np.zeros(C, dtype=np.float64)
    for k, pk in Kprior.items():
        mask = (Klabels == int(k))
        wk = W[mask]
        wk = wk/wk.sum()
        Pk = P[:,mask,:]
        rk = np.einsum("m,cmr->cr", wk, Pk)
        Hk += pk * hbin(rk).sum(axis=1)

    return np.maximum(Hm-Hk, 0.0)


def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    if spearmanr is not None:
        v=spearmanr(a,b).correlation
        return None if np.isnan(v) else float(v)
    ra=np.empty(len(a)); rb=np.empty(len(b))
    ra[np.argsort(a)]=np.arange(len(a)); rb[np.argsort(b)]=np.arange(len(b))
    v=np.corrcoef(ra,rb)[0,1]
    return None if np.isnan(v) else float(v)


def group_examples(examples, indices):
    traj = defaultdict(lambda:defaultdict(list))
    for i in indices:
        x=examples[i]
        traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))
    return traj


def prior_support_diag(examples, indices, roles, particles_per_k, seed):
    groups = group_examples(examples, indices)
    role_to_i={r:i for i,r in enumerate(roles)}
    vals=defaultdict(list)

    for (parent,sid),rm in groups.items():
        for rnd,rows in rm.items():
            x0=rows[0][1]
            truth=set(x0["truth_roles"])
            true_k=len(truth)
            kp={int(k):float(v) for k,v in x0["k_probs"].items()}
            vals["p_k_true"].append(kp.get(true_k,0.0))
            kvals=sorted(kp)
            p=np.array([kp[k] for k in kvals],dtype=np.float64)
            p=p/max(p.sum(),1e-12)
            vals["k_top1"].append(int(kvals[int(np.argmax(p))]==true_k))
            vals["k_abs"].append(float(sum(p[j]*abs(k-true_k) for j,k in enumerate(kvals))))

            raw_role=np.array([x0["role_probs"][r] for r in roles],dtype=np.float64)
            true_z=np.zeros(len(roles))
            for r in truth: true_z[role_to_i[r]]=1.0
            vals["raw_role_brier"].append(float(np.square(raw_role-true_z).mean()))

            rng=np.random.default_rng(abs(hash((parent,sid,rnd,seed)))%(2**32))
            Z,W,Klabels,Kprior=stratified_particles(
                x0["k_probs"],x0["role_probs"],roles,particles_per_k,rng
            )
            d=np.abs(Z-true_z[None,:]).sum(axis=1)
            vals["min_particle_hamming"].append(float(d.min()))
            vals["exact_truth_particle"].append(int(np.any(d==0)))

    return {k:float(np.mean(v)) for k,v in vals.items()} | {"groups":len(vals["p_k_true"])}


def evaluate_config(
    model, qemb, examples, indices, roles, max_round, log_pos_weight,
    temperature, beta, correct_posweight, particles_per_k, device, batch_size, seed,
    collect_selector=False
):
    role_to_i={r:i for i,r in enumerate(roles)}
    traj=group_examples(examples,indices)

    all_candidate = defaultdict(list)
    selector = defaultdict(list)

    for (parent,sid),rm in traj.items():
        for rnd,rows in rm.items():
            rows=sorted(rows,key=lambda ix:ix[1]["candidate_index"])
            if len(rows)<2:
                continue
            x0=rows[0][1]
            truth=set(x0["truth_roles"])
            true_k=len(truth)
            if true_k not in [int(k) for k in x0["k_probs"].keys()]:
                continue

            belief=np.array([x0["role_probs"][r] for r in roles],dtype=np.float32)
            kvec=np.array([x0["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
            rnorm=float(rnd/max(max_round,1))

            rng=np.random.default_rng(abs(hash((parent,sid,rnd,seed)))%(2**32))
            Z,W,Klabels,Kprior=stratified_particles(
                x0["k_probs"],x0["role_probs"],roles,particles_per_k,rng
            )
            kvals=sorted(Kprior)

            qg=np.stack([qemb[i] for i,_ in rows]).astype(np.float32)
            L=particle_logits(model,qg,belief,kvec,rnorm,Z,device,batch_size)
            P=probs_from_logits(L,log_pos_weight,temperature,correct_posweight)

            pk0=k_distribution(W,Klabels,kvals)
            m0=k_metrics(pk0,kvals,true_k)

            gains=[]
            for j,(idx,x) in enumerate(rows):
                Y=np.zeros(len(roles),dtype=np.float64)
                for r in x["selected_roles"]:
                    Y[role_to_i[r]]=1.0
                Wp=update_weights(P[j],Y,W,beta)
                pk1=k_distribution(Wp,Klabels,kvals)
                m1=k_metrics(pk1,kvals,true_k)
                g={
                    "log_gain":m1["logprob"]-m0["logprob"],
                    "brier_gain":m0["brier"]-m1["brier"],
                    "abs_gain":m0["abs_error"]-m1["abs_error"],
                    "p_true_gain":m1["p_true"]-m0["p_true"],
                    "top1_after":m1["top1_correct"],
                }
                gains.append(g)
                for k,v in g.items():
                    all_candidate[k].append(v)

            if collect_selector:
                scores=mi_k_score(P,W,Klabels,Kprior)
                choose_mi=int(np.argmax(scores))

                old_scores=[
                    float(x["selector_predicted_utility"])
                    if x["selector_predicted_utility"] is not None else -1e30
                    for _,x in rows
                ]
                choose_old=int(np.argmax(old_scores))
                mean_random=float(np.mean([g["log_gain"] for g in gains]))

                selector["mi_log_gain"].append(gains[choose_mi]["log_gain"])
                selector["mi_brier_gain"].append(gains[choose_mi]["brier_gain"])
                selector["mi_abs_gain"].append(gains[choose_mi]["abs_gain"])
                selector["old_log_gain"].append(gains[choose_old]["log_gain"])
                selector["random_log_gain"].append(mean_random)

                rho=safe_spearman(scores,[g["log_gain"] for g in gains])
                if rho is not None:
                    selector["mi_vs_log_gain_spearman"].append(rho)

    def mean_dict(d):
        return {k:float(np.mean(v)) if v else None for k,v in d.items()}

    return {
        "all_candidate_updates": mean_dict(all_candidate),
        "selector_eval": mean_dict(selector) if collect_selector else None,
        "candidate_updates_n": len(all_candidate["log_gain"]),
        "selector_groups_n": len(selector["mi_log_gain"]) if collect_selector else 0,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--out",default="artifacts/calibrated_bayes_diag_v1")
    ap.add_argument("--particles-per-k",type=int,default=32)
    ap.add_argument("--batch-size",type=int,default=2048)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=29)
    args=ap.parse_args()

    seed_all(args.seed)
    om=load_obs()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    obs_dir=Path(args.obs_dir)

    ckpt=torch.load(obs_dir/"model.pt",map_location="cpu")
    roles=ckpt["role_order"]
    role_to_i={r:i for i,r in enumerate(roles)}
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device

    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    examples,roles2,_,_=om.collect_examples(Path(args.root))
    if roles2!=roles:
        raise RuntimeError("role order mismatch")

    split=json.loads((obs_dir/"split_systems.json").read_text())
    train_systems=set(split["train"])
    val_systems=set(split["val"])
    test_systems=set(split["test"])

    train_idx=[i for i,x in enumerate(examples) if x["system_id"] in train_systems]
    val_idx=[i for i,x in enumerate(examples) if x["system_id"] in val_systems]
    test_idx=[i for i,x in enumerate(examples) if x["system_id"] in test_systems]

    qemb=np.load(obs_dir/"query_embeddings.npy")
    max_round=max(x["round"] for x in examples)

    # Reconstruct the exact pos_weight used in V1 training.
    pos=np.zeros(len(roles),dtype=np.float64)
    for i in train_idx:
        for r in examples[i]["selected_roles"]:
            pos[role_to_i[r]]+=1
    neg=len(train_idx)-pos
    pos_weight=np.clip(neg/np.maximum(pos,1.0),1.0,20.0)
    log_pos_weight=np.log(pos_weight)

    print("rows train/val/test:",len(train_idx),len(val_idx),len(test_idx))
    print("pos_weight min/mean/max:",float(pos_weight.min()),float(pos_weight.mean()),float(pos_weight.max()))

    # True-Z calibration diagnostics.
    print("computing true-Z validation/test logits...")
    Lval,Yval=true_z_logits(model,qemb,examples,val_idx,roles,max_round,device,args.batch_size)
    Ltest,Ytest=true_z_logits(model,qemb,examples,test_idx,roles,max_round,device,args.batch_size)

    Pval_raw=sigmoid_np(Lval)
    Ptest_raw=sigmoid_np(Ltest)
    Pval_corr=sigmoid_np(Lval-log_pos_weight)
    Ptest_corr=sigmoid_np(Ltest-log_pos_weight)

    raw_cal={
        "val_nll":binary_nll(Yval,Pval_raw),
        "val_brier":brier(Yval,Pval_raw),
        "val_ece":ece_binary(Yval,Pval_raw),
        "test_nll":binary_nll(Ytest,Ptest_raw),
        "test_brier":brier(Ytest,Ptest_raw),
        "test_ece":ece_binary(Ytest,Ptest_raw),
        "test_true_cardinality":float(Ytest.sum(axis=1).mean()),
        "test_prob_cardinality":float(Ptest_raw.sum(axis=1).mean()),
    }
    corr_cal={
        "val_nll":binary_nll(Yval,Pval_corr),
        "val_brier":brier(Yval,Pval_corr),
        "val_ece":ece_binary(Yval,Pval_corr),
        "test_nll":binary_nll(Ytest,Ptest_corr),
        "test_brier":brier(Ytest,Ptest_corr),
        "test_ece":ece_binary(Ytest,Ptest_corr),
        "test_true_cardinality":float(Ytest.sum(axis=1).mean()),
        "test_prob_cardinality":float(Ptest_corr.sum(axis=1).mean()),
    }

    # Temperature fit on validation after analytical pos_weight correction.
    temperatures=[0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]
    temp_rows=[]
    for T in temperatures:
        p=sigmoid_np((Lval-log_pos_weight)/T)
        temp_rows.append({"T":T,"val_nll":binary_nll(Yval,p),"val_brier":brier(Yval,p),"val_ece":ece_binary(Yval,p)})
    bestT=min(temp_rows,key=lambda x:x["val_nll"])["T"]
    Ptest_temp=sigmoid_np((Ltest-log_pos_weight)/bestT)
    temp_cal={
        "best_T":bestT,
        "grid":temp_rows,
        "test_nll":binary_nll(Ytest,Ptest_temp),
        "test_brier":brier(Ytest,Ptest_temp),
        "test_ece":ece_binary(Ytest,Ptest_temp),
        "test_true_cardinality":float(Ytest.sum(axis=1).mean()),
        "test_prob_cardinality":float(Ptest_temp.sum(axis=1).mean()),
    }

    print("best temperature:",bestT)

    prior_diag_val=prior_support_diag(examples,val_idx,roles,args.particles_per_k,args.seed)
    prior_diag_test=prior_support_diag(examples,test_idx,roles,args.particles_per_k,args.seed)

    # First compare RAW vs corrected vs temp-corrected at beta=1 on validation.
    print("validation update diagnostics...")
    val_raw=evaluate_config(
        model,qemb,examples,val_idx,roles,max_round,log_pos_weight,
        1.0,1.0,False,args.particles_per_k,device,args.batch_size,args.seed,False
    )
    val_corr=evaluate_config(
        model,qemb,examples,val_idx,roles,max_round,log_pos_weight,
        1.0,1.0,True,args.particles_per_k,device,args.batch_size,args.seed,False
    )
    val_temp=evaluate_config(
        model,qemb,examples,val_idx,roles,max_round,log_pos_weight,
        bestT,1.0,True,args.particles_per_k,device,args.batch_size,args.seed,False
    )

    # Tune power-posterior beta on validation AFTER correction+temperature.
    betas=[0.05,0.1,0.2,0.35,0.5,0.75,1.0]
    beta_rows=[]
    for beta in betas:
        print("tuning beta",beta)
        x=evaluate_config(
            model,qemb,examples,val_idx,roles,max_round,log_pos_weight,
            bestT,beta,True,args.particles_per_k,device,args.batch_size,args.seed,False
        )
        beta_rows.append({
            "beta":beta,
            "mean_log_gain":x["all_candidate_updates"]["log_gain"],
            "mean_brier_gain":x["all_candidate_updates"]["brier_gain"],
            "mean_abs_gain":x["all_candidate_updates"]["abs_gain"],
        })

    # Choose beta by mean posterior log probability gain; no test leakage.
    bestBeta=max(beta_rows,key=lambda x:x["mean_log_gain"])["beta"]
    print("best beta:",bestBeta)

    print("test update diagnostics...")
    test_raw=evaluate_config(
        model,qemb,examples,test_idx,roles,max_round,log_pos_weight,
        1.0,1.0,False,args.particles_per_k,device,args.batch_size,args.seed,True
    )
    test_corr=evaluate_config(
        model,qemb,examples,test_idx,roles,max_round,log_pos_weight,
        1.0,1.0,True,args.particles_per_k,device,args.batch_size,args.seed,True
    )
    test_temp=evaluate_config(
        model,qemb,examples,test_idx,roles,max_round,log_pos_weight,
        bestT,1.0,True,args.particles_per_k,device,args.batch_size,args.seed,True
    )
    test_final=evaluate_config(
        model,qemb,examples,test_idx,roles,max_round,log_pos_weight,
        bestT,bestBeta,True,args.particles_per_k,device,args.batch_size,args.seed,True
    )

    summary={
        "setting":{
            "test_true_Z_not_used_for_candidate_scoring":True,
            "validation_used_for_temperature_and_beta_only":True,
            "particles_per_K":args.particles_per_k,
            "train_rows":len(train_idx),"val_rows":len(val_idx),"test_rows":len(test_idx),
        },
        "key_fact":{
            "training_loss":"BCEWithLogitsLoss(pos_weight=per_role_weight)",
            "analytical_correction":"corrected_logit = raw_logit - log(pos_weight)",
            "pos_weight_min":float(pos_weight.min()),
            "pos_weight_mean":float(pos_weight.mean()),
            "pos_weight_max":float(pos_weight.max()),
        },
        "prior_support":{
            "validation":prior_diag_val,
            "test":prior_diag_test,
        },
        "true_Z_probability_calibration":{
            "raw_sigmoid":raw_cal,
            "inverse_posweight_corrected":corr_cal,
            "corrected_plus_temperature":temp_cal,
        },
        "validation_posterior_update":{
            "raw":val_raw,
            "posweight_corrected":val_corr,
            "corrected_plus_temperature":val_temp,
            "beta_grid_after_correction_and_temperature":beta_rows,
            "selected_beta_by_validation_log_gain":bestBeta,
        },
        "test_posterior_update":{
            "raw":test_raw,
            "posweight_corrected":test_corr,
            "corrected_plus_temperature":test_temp,
            "final_corrected_temperature_power":test_final,
        },
        "interpretation_keys":{
            "if_posweight_correction_helps":"Raw sigmoid outputs were classification scores distorted by weighted BCE, not usable likelihood probabilities.",
            "if_beta_much_less_than_1_helps":"The factorized 20-bit likelihood is overcounting correlated evidence; a tempered/generalized Bayes update is safer.",
            "if_prior_support_is_bad":"The reconstructed joint posterior over Z is itself a bottleneck even with a calibrated observation model.",
        },
    }

    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"pos_weight.json").write_text(json.dumps({r:float(pos_weight[i]) for i,r in enumerate(roles)},indent=2),encoding="utf-8")

    print("\n"+"="*100)
    print("CALIBRATED BAYES DIAGNOSTIC V1")
    print("="*100)
    print("\nPRIOR TEST:",prior_diag_test)
    print("\nCALIBRATION")
    print("raw:",raw_cal)
    print("corrected:",corr_cal)
    print("temp:",{k:v for k,v in temp_cal.items() if k!="grid"})
    print("\nBETA GRID")
    for x in beta_rows: print(x)
    print("\nBEST T/BETA:",bestT,bestBeta)
    print("\nTEST UPDATE")
    for name,x in [
        ("raw",test_raw),
        ("corrected",test_corr),
        ("temp",test_temp),
        ("final",test_final),
    ]:
        print("\n",name)
        print("all candidates:",x["all_candidate_updates"])
        print("selector:",x["selector_eval"])
    print("\n[saved]",out/"summary.json")


if __name__=="__main__":
    main()
