#!/usr/bin/env python3
"""
repeat_probe_stability_analyze_v1.py

Analyzes fresh repeated executions to answer:

1) Is visible fingerprint evidence itself stable for the same probe/system?
2) Is realized K-information utility stable enough to be a learnable target?
3) Does averaging repeats make pre-query selectors correlate with expected utility?
4) Are we looking at aleatoric output noise, or a structurally weak utility model?

Uses the SAME visible fingerprint detector and calibrated/tempered Bayesian
machinery as visible_adaptive_bridge_v1.
"""

from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr=None

sys.path.insert(0,str(Path("analysis").resolve()))
import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb


def read_jsonl(p):
    out=[]
    with Path(p).open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out


def mean(xs): return float(np.mean(xs)) if xs else None


def safe_spear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12 or spearmanr is None: return None
    v=spearmanr(a,b).correlation
    return None if np.isnan(v) else float(v)


def jaccard(a,b):
    a=set(a); b=set(b)
    return len(a&b)/len(a|b) if (a|b) else 1.0


def pairwise_jaccard(sets):
    vals=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            vals.append(jaccard(sets[i],sets[j]))
    return mean(vals)


def reliability_from_candidates(candidate_repeat_values):
    """
    Approximate single-measure ICC/reliability within one state:
      between = max(Var(candidate means) - mean(within_var)/R, 0)
      reliability = between / (between + within)
    """
    means=[]; within=[]
    rs=[]
    for vals in candidate_repeat_values:
        vals=np.asarray(vals,float)
        if len(vals)<2: continue
        means.append(vals.mean())
        within.append(vals.var(ddof=1))
        rs.append(len(vals))
    if len(means)<2: return None
    avg_within=float(np.mean(within))
    R=float(np.mean(rs))
    obs_between=float(np.var(means,ddof=1))
    latent_between=max(obs_between-avg_within/max(R,1),0.0)
    den=latent_between+avg_within
    return {
        "candidate_mean_variance_observed":obs_between,
        "within_candidate_variance":avg_within,
        "latent_between_candidate_variance":latent_between,
        "single_repeat_reliability":latent_between/den if den>0 else 0.0,
        "mean_repeat_reliability":latent_between/(latent_between+avg_within/max(R,1)) if (latent_between+avg_within/max(R,1))>0 else 0.0,
    }


def detector_train(existing_X,Y,tri,vai):
    return vb.fit_visible_detector(existing_X,Y,tri,vai)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--vis-dir",default="artifacts/visible_output_fingerprint_v1")
    ap.add_argument("--cal-dir",default="artifacts/calibrated_bayes_diag_v1")
    ap.add_argument("--repeat-dir",default="artifacts/repeat_probe_stability_v1")
    ap.add_argument("--particles-per-k",type=int,default=32)
    ap.add_argument("--batch-size",type=int,default=4096)
    ap.add_argument("--seed",type=int,default=73)
    args=ap.parse_args()

    root=Path(args.root); obsdir=Path(args.obs_dir); visdir=Path(args.vis_dir)
    caldir=Path(args.cal_dir); repdir=Path(args.repeat_dir)
    outrows=[x for x in read_jsonl(repdir/"fresh_repeats.jsonl") if x.get("execution_status")=="ok"]
    if not outrows: raise RuntimeError("No successful fresh repeats found.")

    print("loading existing fingerprint training data...",flush=True)
    examples,roles,_,_=om.collect_examples(root)
    split=json.loads((obsdir/"split_systems.json").read_text())
    trsys,vasys=set(split["train"]),set(split["val"])
    tri=np.array([i for i,x in enumerate(examples) if x["system_id"] in trsys],int)
    vai=np.array([i for i,x in enumerate(examples) if x["system_id"] in vasys],int)

    Q=np.load(obsdir/"query_embeddings.npy").astype(np.float32)
    O=np.load(visdir/"output_embeddings.npy").astype(np.float32)
    Xbase=np.concatenate([Q,O],axis=1)
    Y=vb.labels(examples,roles,"selected_roles")
    models,cals,thr,tpr,fpr,_=detector_train(Xbase,Y,tri,vai)

    print(f"encoding {len(outrows)} fresh visible outputs...",flush=True)
    st=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
    texts=[str(x["final_output"]) for x in outrows]
    Onew=st.encode(texts,batch_size=64,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=False).astype(np.float32)
    Qnew=np.stack([Q[int(x["example_index"])] for x in outrows]).astype(np.float32)
    Pnew=vb.visible_probs(models,cals,np.concatenate([Qnew,Onew],axis=1))
    Enew=(Pnew>=thr[None,:]).astype(np.float64)

    print("loading calibrated forward model...",flush=True)
    ckpt=torch.load(obsdir/"model.pt",map_location="cpu")
    model=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
        hidden=ckpt["hidden"],dropout=ckpt["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    ri={r:i for i,r in enumerate(roles)}
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]["selected_roles"]: pos[ri[r]]+=1
    posw=np.clip((len(tri)-pos)/np.maximum(pos,1.0),1.0,20.0)
    log_posw=np.log(posw)
    cs=json.loads((caldir/"summary.json").read_text())
    T=float(cs["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])
    beta=float(cs["validation_posterior_update"]["selected_beta_by_validation_log_gain"])

    # example lookup / candidate repeat grouping
    ex_by_i={i:x for i,x in enumerate(examples)}
    cand_rows=defaultdict(list)
    for r_i,row in enumerate(outrows):
        cand_rows[(row["system_id"],int(row["round"]),int(row["candidate_index"]))].append((r_i,row))

    group_cands=defaultdict(list)
    for key in cand_rows:
        group_cands[(key[0],key[1])].append(key)

    cand_stats={}
    group_stats=[]
    all_within_std=[]
    fp_jaccards=[]
    internal_jaccards=[]
    all_candidate_means=[]
    all_scores={"visible_mi":[],"learned_score":[],"old_score":[]}
    all_mean_gains=[]

    for (sid,rnd),keys in group_cands.items():
        x0=ex_by_i[int(cand_rows[keys[0]][0][1]["example_index"])]
        true_k=len(x0["truth_roles"])
        belief=np.array([x0["role_probs"][r] for r in roles],dtype=np.float32)
        kvec=np.array([x0["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
        max_round=max(x["round"] for x in examples)
        rnorm=float(rnd/max(max_round,1))
        rng=np.random.default_rng(abs(hash((sid,rnd,args.seed)))%(2**32))
        Z,W,Klabels,Kprior=vb.stratified_particles(
            x0["k_probs"],x0["role_probs"],roles,args.particles_per_k,rng
        )
        prior=vb.k_metrics(W,Klabels,true_k)

        candidate_values=[]
        csummary=[]
        for key in sorted(keys,key=lambda k:k[2]):
            reps=cand_rows[key]
            exidx=int(reps[0][1]["example_index"])
            qg=Q[exidx:exidx+1].astype(np.float32)
            L=vb.particle_logits(model,qg,belief,kvec,rnorm,Z,"cpu",args.batch_size)
            pY=vb.calibrated_hidden_probs(L,log_posw,T)
            pE=vb.detector_channel_probs(pY,tpr,fpr)[0]

            gains=[]; fp_sets=[]; sel_sets=[]; softs=[]
            for new_i,row in reps:
                Wp=vb.update_weights(pE,Enew[new_i],W,beta)
                post=vb.k_metrics(Wp,Klabels,true_k)
                gains.append(post["logprob"]-prior["logprob"])
                fp_sets.append([roles[j] for j in np.where(Enew[new_i]>0.5)[0]])
                sel_sets.append(row.get("selected_roles_training_only",[]))
                softs.append(Pnew[new_i])

            gmean=float(np.mean(gains))
            gstd=float(np.std(gains,ddof=1)) if len(gains)>1 else 0.0
            fpj=pairwise_jaccard(fp_sets)
            inj=pairwise_jaccard(sel_sets)
            soft=np.stack(softs)
            soft_l1=float(np.mean(np.abs(soft-soft.mean(0,keepdims=True))))

            first=reps[0][1]
            c={
                "candidate_index":key[2],
                "n_repeats":len(gains),
                "mean_log_gain":gmean,
                "std_log_gain":gstd,
                "min_log_gain":float(np.min(gains)),
                "max_log_gain":float(np.max(gains)),
                "visible_fingerprint_pairwise_jaccard":fpj,
                "internal_selected_roles_pairwise_jaccard":inj,
                "visible_soft_prob_mean_abs_deviation":soft_l1,
                "visible_mi":float(first.get("visible_mi",0.0)),
                "learned_score":float(first.get("learned_score",0.0)),
                "old_score":float(first.get("old_score",0.0)),
                "repeat_log_gains":list(map(float,gains)),
            }
            cand_stats[f"{sid}|r{rnd}|c{key[2]}"]=c
            csummary.append(c)
            candidate_values.append(gains)
            all_within_std.append(gstd)
            if fpj is not None: fp_jaccards.append(fpj)
            if inj is not None: internal_jaccards.append(inj)
            all_candidate_means.append(gmean)
            all_mean_gains.append(gmean)
            for nm in all_scores: all_scores[nm].append(c[nm])

        rel=reliability_from_candidates(candidate_values)

        # How stable is the winner across repetition index?
        lengths=[len(v) for v in candidate_values]
        R=min(lengths)
        mean_winner=int(np.argmax([np.mean(v) for v in candidate_values]))
        winner_agree=[]
        pair_order=[]
        for rr in range(R):
            vals=[v[rr] for v in candidate_values]
            winner_agree.append(int(np.argmax(vals)==mean_winner))
        for a in range(len(candidate_values)):
            for b in range(a+1,len(candidate_values)):
                ma=np.mean(candidate_values[a]); mb=np.mean(candidate_values[b])
                if abs(ma-mb)<1e-12: continue
                s=1 if ma>mb else -1
                for rr in range(R):
                    d=candidate_values[a][rr]-candidate_values[b][rr]
                    if abs(d)>1e-12: pair_order.append(int((1 if d>0 else -1)==s))

        # selector choices vs repeated-MEAN expected utility within this panel
        means=np.array([c["mean_log_gain"] for c in csummary])
        idx_best=int(np.argmax(means))
        choices={
            "panel_oracle_mean":float(means[idx_best]),
            "random_panel_mean":float(means.mean()),
        }
        for nm in ["visible_mi","learned_score","old_score"]:
            j=int(np.argmax([c[nm] for c in csummary]))
            choices[f"{nm}_chosen_mean_gain"]=float(means[j])
            choices[f"{nm}_hits_panel_oracle"]=int(j==idx_best)

        group_stats.append({
            "system_id":sid,"round":rnd,"true_k":true_k,
            "candidate_count":len(csummary),
            "reliability":rel,
            "single_repeat_winner_matches_mean_winner":mean(winner_agree),
            "pairwise_order_matches_mean_order":mean(pair_order),
            "choices":choices,
            "candidates":csummary,
        })

    # Global correlations now use REPEATED-MEAN utility rather than one noisy realization.
    corrs={nm:safe_spear(vals,all_mean_gains) for nm,vals in all_scores.items()}

    # Average selector comparisons over groups
    selector_summary={}
    for key in [
        "panel_oracle_mean","random_panel_mean",
        "visible_mi_chosen_mean_gain","learned_score_chosen_mean_gain","old_score_chosen_mean_gain",
        "visible_mi_hits_panel_oracle","learned_score_hits_panel_oracle","old_score_hits_panel_oracle",
    ]:
        selector_summary[key]=mean([g["choices"][key] for g in group_stats])

    rels=[g["reliability"] for g in group_stats if g["reliability"]]
    summary={
        "setting":{
            "fresh_repetitions":True,
            "old_query_cache_bypassed":True,
            "groups":len(group_stats),
            "successful_fresh_outputs":len(outrows),
            "particles_per_K":args.particles_per_k,
            "forward_temperature":T,
            "likelihood_beta":beta,
            "visible_observation":"query + fresh final_output only",
        },
        "fingerprint_stability":{
            "mean_pairwise_jaccard_visible_binary_fingerprint":mean(fp_jaccards),
            "mean_pairwise_jaccard_internal_selected_roles_training_only":mean(internal_jaccards),
        },
        "utility_stability":{
            "mean_within_candidate_log_gain_std":mean(all_within_std),
            "global_std_of_candidate_mean_log_gain":float(np.std(all_candidate_means,ddof=1)) if len(all_candidate_means)>1 else None,
            "mean_single_repeat_reliability_across_groups":mean([r["single_repeat_reliability"] for r in rels]),
            "mean_repeated_mean_reliability_across_groups":mean([r["mean_repeat_reliability"] for r in rels]),
            "mean_single_repeat_winner_matches_mean_winner":mean([g["single_repeat_winner_matches_mean_winner"] for g in group_stats]),
            "mean_pairwise_order_matches_mean_order":mean([g["pairwise_order_matches_mean_order"] for g in group_stats]),
        },
        "prequery_score_vs_repeated_mean_utility_spearman":corrs,
        "selector_vs_repeated_mean_panel_utility":selector_summary,
        "decision_rule":{
            "if_reliability_high_but_correlations_low":"Utility is stable, but current pre-query feature/model is structurally missing the useful signal.",
            "if_reliability_low_and_repeated_mean_reliability_rises":"Single execution labels are noisy; train/select on expected utility using multi-sample or distributional targets.",
            "if_fingerprint_stable_but_utility_unstable":"Instability is mainly in Bayesian utility sensitivity, not output fingerprint extraction.",
            "if_internal_selected_roles_unstable":"The MAS routing itself is stochastic for identical probes; selector should model outcome distributions rather than deterministic winners."
        }
    }
    (repdir/"stability_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (repdir/"stability_groups.jsonl").open("w",encoding="utf-8") as f:
        for g in group_stats: f.write(json.dumps(g)+"\n")

    print("\n"+"="*100)
    print("REPEATED PROBE STABILITY V1")
    print("="*100)
    print("\nFINGERPRINT STABILITY")
    for k,v in summary["fingerprint_stability"].items(): print(f"{k:58s}: {v}")
    print("\nUTILITY STABILITY")
    for k,v in summary["utility_stability"].items(): print(f"{k:58s}: {v}")
    print("\nPRE-QUERY SCORE vs REPEATED-MEAN UTILITY SPEARMAN")
    for k,v in corrs.items(): print(f"{k:58s}: {v}")
    print("\nSELECTOR vs REPEATED-MEAN PANEL UTILITY")
    for k,v in selector_summary.items(): print(f"{k:58s}: {v}")
    print("\n[saved]",repdir/"stability_summary.json")


if __name__=="__main__":
    main()
