#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0,str(Path(".").resolve()))
sys.path.insert(0,str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11
from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()
from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec
from v11.role_universe import TRAIN_ROLES

TRUE_ROLES=("code","planning","research")
KS=(3,4,5,6)

def norm(w):
    w=np.asarray(w,float); return w/max(w.sum(),1e-12)

def k_probs(W,K):
    return {str(k):float(W[K==k].sum()) for k in KS}

def role_probs(W,Z,roles):
    p=W@Z
    return {r:float(p[i]) for i,r in enumerate(roles)}

def show(rnd,kp,rp):
    print("\n"+"="*88)
    print(f"POSTERIOR AFTER ROUND {rnd}")
    print("P(K):","  ".join(f"K={k}: {kp.get(str(k),0):.4f}" for k in KS))
    print("Role probabilities:")
    for r,p in sorted(rp.items(),key=lambda x:-x[1]):
        mark="  <-- TRUE" if r in TRUE_ROLES else ""
        print(f"  {r:<28} {p:.4f}{mark}")

def make_state(sid,rnd,kp,rp,history):
    # schema history format is easiest to preserve by round-tripping prior public dict.
    return {
      "session_id":sid,"split":"dev","round_index":rnd,
      "k_probs":kp,"role_probs":rp,"history":history
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter",default="artifacts/robust_visible_generator_v1/policy")
    ap.add_argument("--rounds",type=int,default=5)
    ap.add_argument("--group-size",type=int,default=8)
    ap.add_argument("--particles-per-k",type=int,default=128)
    ap.add_argument("--mode",default="mixed",choices=["single","fanout2","fanout3","mixed"])
    ap.add_argument("--style",default="s1")
    ap.add_argument("--model",default=None)
    ap.add_argument("--seed",type=int,default=20260923)
    ap.add_argument("--out",default="artifacts/one_system_k3_demo")
    a=ap.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing; source .env first")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    # Universe is exactly the 20 TRAIN_ROLES used by the current observation model.
    roles=list(TRAIN_ROLES)
    assert len(roles)==20, len(roles)
    assert set(TRUE_ROLES)<=set(roles)
    ri={r:i for i,r in enumerate(roles)}

    # Load historical corpus only to train the already-defined visible detector.
    root=Path("artifacts/adaptive_v11_curriculum_v2")
    obsdir=Path("artifacts/observation_model_v1")
    visdir=Path("artifacts/visible_output_fingerprint_v1")
    caldir=Path("artifacts/calibrated_bayes_diag_v1")
    examples,obs_roles,_,_=om.collect_examples(root)
    if list(obs_roles)!=roles:
        raise RuntimeError(f"role ordering mismatch\nobs={obs_roles}\nuniverse={roles}")
    split=json.loads((obsdir/"split_systems.json").read_text())
    trsys=set(map(str,split["train"])); vasys=set(map(str,split["val"]))
    tri=np.array([i for i,x in enumerate(examples) if str(x["system_id"]) in trsys],int)
    vai=np.array([i for i,x in enumerate(examples) if str(x["system_id"]) in vasys],int)
    Qhist=np.load(obsdir/"query_embeddings.npy").astype(np.float32)
    Ohist=np.load(visdir/"output_embeddings.npy").astype(np.float32)
    Xhist=np.concatenate([Qhist,Ohist],axis=1)
    Yhist=vb.labels(examples,roles,"selected_roles")
    fp_models,fp_cals,thr,tpr,fpr,_=vb.fit_visible_detector(Xhist,Yhist,tri,vai)

    # Same embedding model used to create stored embeddings.
    embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ckpt=torch.load(obsdir/"model.pt",map_location="cpu")
    fwd=om.ObservationNet(qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
                          hidden=ckpt["hidden"],dropout=ckpt["dropout"])
    fwd.load_state_dict(ckpt["state_dict"]); fwd.eval()
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]["selected_roles"]: pos[ri[r]]+=1
    posw=np.clip((len(tri)-pos)/np.maximum(pos,1.0),1.0,20.0)
    log_posw=np.log(posw)
    cs=json.loads((caldir/"summary.json").read_text())
    T=float(cs["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])
    beta=float(cs["validation_posterior_update"]["selected_beta_by_validation_log_gain"])

    # Initial belief: uniform K over 3..6, exchangeable role marginals E[K]/20.
    kp0={str(k):1/len(KS) for k in KS}
    p0=sum(k*kp0[str(k)] for k in KS)/len(roles)
    rp0={r:p0 for r in roles}
    rng=np.random.default_rng(a.seed)
    Z,W,Klabels,Kprior=vb.stratified_particles(kp0,rp0,roles,a.particles_per_k,rng)
    W=norm(W)

    policy=CurriculumAdaptiveProbePolicyV11(
        "Qwen/Qwen2.5-1.5B-Instruct",adapter_path=a.adapter,
        trainable=False,use_4bit=False,max_seq_length=4096)

    sid="fun_k3_code_planning_research"
    spec=SingleRouterSystemSpec(system_id=sid,split="demo",execution_mode=a.mode,
                                style=a.style,worker_roles=TRUE_ROLES)
    target=SingleRouterFanoutMAS(spec,model=a.model)

    history=[]
    records=[]
    max_round=max(int(x["round"]) for x in examples)

    print("\nTRUE K=3; TRUE ROLES =",TRUE_ROLES)
    print("Inference sees only query + final output. Trace is evaluation-only.\n")

    for rnd in range(1,a.rounds+1):
        kp=k_probs(W,Klabels); rp=role_probs(W,Z,roles)
        public=make_state(sid,rnd-1,kp,rp,history)
        st=PublicBeliefState.from_public_dict(public)
        seen=[h.query for h in st.history]

        candidates=[]
        for j in range(a.group_size):
            last=None
            for retry in range(12):
                seed=a.seed+100000*rnd+100*j+retry
                random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                sample=policy.sample_probe(
                    {"belief_state":st.to_public_dict(),"candidate_index":j,"max_history_turns":5},
                    temperature=.85,top_p=.95,max_new_tokens=192)
                q=str(sample.query or "").strip()
                ok,why=surface_valid(q)
                valid=bool(sample.valid) and ok and q not in seen and all(q!=x for x in candidates)
                last=(q,valid,why)
                if valid: break
            q,valid,why=last
            if valid: candidates.append(q)
        if not candidates: raise RuntimeError("generator produced no valid candidates")

        Q=np.asarray(embedder.encode(candidates,convert_to_numpy=True),np.float32)
        belief=np.array([rp[r] for r in roles],np.float32)
        kvec=np.array([kp[str(k)] for k in KS],np.float32)
        rnorm=float(rnd/max(max_round,1))
        L=vb.particle_logits(fwd,Q,belief,kvec,rnorm,Z,"cpu",4096)
        pY=vb.calibrated_hidden_probs(L,log_posw,T)
        pE=vb.detector_channel_probs(pY,tpr,fpr)
        mi=np.asarray(vb.mi_k(pE,W,Klabels,kp),float)
        chosen=int(np.argmax(mi)); q=candidates[chosen]

        print("\n"+"-"*88)
        print(f"ROUND {rnd}: {len(candidates)} candidates")
        for j,(qq,s) in enumerate(zip(candidates,mi)):
            star=" <-- CHOSEN" if j==chosen else ""
            print(f"[{j}] visible-MI={s:.6f}{star}\n    {qq}")
        print("\nSENDING:",q)

        output=str(target.query(q))
        # EVAL ONLY. Never enters fingerprint/update.
        tr=getattr(target,"_last_trace",None)
        selected_eval=list(map(str,getattr(tr,"selected_roles",()))) if tr else []

        qemb=np.asarray(embedder.encode([q],convert_to_numpy=True),np.float32)
        oemb=np.asarray(embedder.encode([output],convert_to_numpy=True),np.float32)
        X=np.concatenate([qemb,oemb],axis=1)
        P=vb.visible_probs(fp_models,fp_cals,X)[0]
        E=(P>=thr).astype(np.float64)

        # Bayes update from VISIBLE fingerprint only.
        W=vb.update_weights(pE[chosen],E,W,beta)
        W=norm(W)
        kp2=k_probs(W,Klabels); rp2=role_probs(W,Z,roles)

        print("\nVISIBLE OUTPUT:\n",output)
        print("\nVISIBLE FINGERPRINT top probabilities:")
        for r,p in sorted(zip(roles,P),key=lambda x:-x[1])[:8]:
            print(f"  {r:<28} {p:.4f}")
        print("\n[EVAL ONLY selected_roles trace]:",selected_eval)
        show(rnd,kp2,rp2)

        # Build history using schema's own history-frame shape by copying a known
        # compatible frame when possible; otherwise keep history empty for generator.
        # We still adapt generator through posterior belief even if schema history
        # construction differs across repo revisions.
        records.append({
            "round":rnd,"candidates":candidates,"visible_mi":mi.tolist(),
            "chosen_index":chosen,"query":q,"final_output":output,
            "selected_roles_eval_only":selected_eval,
            "visible_fingerprint_probs":{r:float(P[i]) for i,r in enumerate(roles)},
            "k_probs_after":kp2,"role_probs_after":rp2,
        })
        # Robust cross-version history: inspect schema class if available.
        try:
            from adaptive_v11_active.schema import HistoryFrame
            frame=HistoryFrame(query=q, response=output)
            history=[*st.history,frame]
        except Exception:
            # Belief adaptation is the critical part; do not fabricate schema fields.
            history=list(st.history)

        (out/"trajectory.json").write_text(json.dumps(records,indent=2),encoding="utf-8")

    kp=k_probs(W,Klabels); rp=role_probs(W,Z,roles)
    predk=int(max(kp,key=kp.get))
    top3=[r for r,_ in sorted(rp.items(),key=lambda x:-x[1])[:3]]
    summary={
      "true_k":3,"pred_k":predk,"p_k3":kp["3"],
      "true_roles":list(TRUE_ROLES),"top3_predicted":top3,
      "k_correct":predk==3,"top3_exact":set(top3)==set(TRUE_ROLES),
      "final_k_probs":kp,"final_role_probs":rp,
      "rounds":a.rounds,"group_size":a.group_size,"mode":a.mode,
      "adapter":a.adapter,
      "inference_inputs":"query + final_output only",
      "trace_usage":"evaluation only"
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\n"+"#"*88)
    print("FINAL")
    print("#"*88)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
