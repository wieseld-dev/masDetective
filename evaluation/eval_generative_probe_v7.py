from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

from detective.benign_gate_v7 import CombinedBenignGateV7, SemanticBenignGuardV7
from detective.generative_probe_policy_v7 import GenerativeProbePolicyV7
from estimator.candidate_presence_contrastive_v64 import CandidateRecord, load_profile
from rl.generative_probe_env_v7 import GenerativeProbeEnvV7, ResponseCacheV7, SeedStoreV7
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo_v65 import get_v65_split


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def _metrics(rows, probs, thresholds):
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    p = np.asarray(probs, dtype=float)
    pred = np.asarray([float(pp) >= float(thresholds[r["role"]]) for r, pp in zip(rows, p)], dtype=bool)
    out = {
        "auc": _auc(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    bysys = defaultdict(list)
    for r, pp, pr in zip(rows, p, pred): bysys[r["system_id"]].append((r, float(pp), bool(pr)))
    hard=[]; soft=[]; exact=[]; within=[]
    for vals in bysys.values():
        k=int(vals[0][0]["k_true"]); kh=sum(int(v[2]) for v in vals); ks=sum(v[1] for v in vals)
        hard.append(abs(kh-k)); soft.append(abs(ks-k)); exact.append(kh==k); within.append(abs(kh-k)<=1)
    out.update({
        "hard_K_MAE": float(np.mean(hard)), "soft_K_MAE": float(np.mean(soft)),
        "exact_K_accuracy": float(np.mean(exact)), "within_1_K_accuracy": float(np.mean(within)),
    })
    return out


def _seed_probability(estimator, seed_store, spec, role):
    rec = CandidateRecord(
        system_id="blackbox", role=role, mode="unknown", true_roles=tuple(),
        anchor_output=seed_store.anchor(spec.system_id), target_output=seed_store.output(spec.system_id, role),
    )
    return float(estimator.predict_proba_records([rec])[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="V7 generative benign probe evaluation on untouched V6.5 validation compositions")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/v7_generative_policy")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--estimator", default="artifacts/candidate_presence_v65.joblib")
    ap.add_argument("--seed-public", default="artifacts/v65_val_public.jsonl")
    ap.add_argument("--max-followups", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--limit-systems", type=int, default=0)
    ap.add_argument("--output-dir", default="artifacts/v7_eval")
    ap.add_argument("--response-public-cache", default="artifacts/v7_val_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v7_val_response_truth.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile); roles = tuple(profile["candidate_roles"])
    specs = get_v65_split("val")
    if args.limit_systems > 0: specs = specs[:args.limit_systems]
    interactions = len(specs) * len(roles) * args.max_followups
    upper_target_calls = interactions * max(s.estimated_calls_per_query for s in specs)
    upper_guard = interactions if args.gate_mode == "hybrid" else 0
    print("=== V7 UNSEEN-COMPOSITION EVALUATION PLAN ===")
    print("validation systems:", len(specs), "candidates/system:", len(roles))
    print("seed-only baseline uses already-paid V6.5 outputs")
    print("trained policy gets seed response, then generates free-text benign follow-ups or STOP")
    print("upper-bound NEW external follow-up interactions:", interactions)
    print("rough upper-bound target-model API calls:", upper_target_calls)
    print("rough upper-bound semantic-guard calls:", upper_guard)
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute after reviewing cost.")
        return
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY is not set")
    if not Path(args.adapter).exists(): raise SystemExit(f"missing policy adapter: {args.adapter}")

    estimator = joblib.load(args.estimator)
    if hasattr(estimator, "embedder"): estimator.embedder.device = "cpu"
    seed_store = SeedStoreV7.from_jsonl(args.seed_public)
    semantic = SemanticBenignGuardV7(model=args.guard_model) if args.gate_mode == "hybrid" else None
    gate = CombinedBenignGateV7(mode=args.gate_mode, semantic_guard=semantic)
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = GenerativeProbePolicyV7(args.model_name, adapter_path=args.adapter, trainable=False, use_4bit=True)
    thresholds = {r: float(estimator.threshold_for(r)) for r in roles}

    rows=[]; seed_probs=[]; final_probs=[]; histories=[]
    generated=blocked=accepted=external=0
    present_followup_selected=[]; present_episode_any=[]
    for si,spec in enumerate(specs,1):
        print(f"[{si}/{len(specs)}] {spec.system_id} mode={spec.execution_mode} K={spec.k_true}", flush=True)
        for role in roles:
            p0 = _seed_probability(estimator, seed_store, spec, role)
            target = SingleRouterFanoutMAS(spec)
            env = GenerativeProbeEnvV7(
                target=target, estimator=estimator, profile=profile, candidate_role=role,
                seed_store=seed_store, gate=gate, response_cache=cache,
                max_followups=args.max_followups, max_attempts=args.max_attempts,
            )
            obs=env.observation(); actions=[]; episode_selected=[]
            while not env.done and env.attempts < args.max_attempts:
                sample=policy.sample(obs, deterministic=True)
                generated += 1
                nxt, reward, done, info = env.step(sample.parsed.action, sample.parsed.query)
                if sample.parsed.action == "ask":
                    if info.allowed: accepted += 1
                    else: blocked += 1
                external += int(info.external_query_sent)
                if role in set(spec.worker_roles) and info.allowed and sample.parsed.action == "ask":
                    if info.selected_target_training_only is not None:
                        present_followup_selected.append(bool(info.selected_target_training_only))
                        episode_selected.append(bool(info.selected_target_training_only))
                actions.append({
                    "completion": sample.completion, "action": sample.parsed.action, "query": sample.parsed.query,
                    "allowed": info.allowed, "gate_reason": info.gate_reason,
                    "belief_before": info.belief_before, "belief_after": info.belief_after,
                    "external_query_sent": info.external_query_sent, "reward_eval_only": reward,
                    "selected_target_training_only": info.selected_target_training_only,
                })
                obs=nxt
                if done: break
            rows.append({
                "system_id":spec.system_id,"mode":spec.execution_mode,"role":role,
                "label":int(role in set(spec.worker_roles)),"k_true":spec.k_true,
            })
            seed_probs.append(p0); final_probs.append(float(env.belief))
            histories.append({"system_id":spec.system_id,"role":role,"actions":actions,"final_belief":float(env.belief)})
            if role in set(spec.worker_roles):
                present_episode_any.append(any(episode_selected) if episode_selected else False)

    seed_m=_metrics(rows,seed_probs,thresholds); final_m=_metrics(rows,final_probs,thresholds)
    print("\nSEED-ONLY BASELINE")
    print(json.dumps(seed_m, indent=2))
    print("\nGENERATIVE RL FINAL")
    print(json.dumps(final_m, indent=2))
    per_mode={}
    for mode in sorted({r["mode"] for r in rows}):
        idx=[i for i,r in enumerate(rows) if r["mode"]==mode]
        per_mode[mode] = {
            "seed_only": _metrics([rows[i] for i in idx],[seed_probs[i] for i in idx],thresholds),
            "generative_rl": _metrics([rows[i] for i in idx],[final_probs[i] for i in idx],thresholds),
        }
    per_role={}
    for role in roles:
        idx=[i for i,r in enumerate(rows) if r["role"]==role]
        per_role[role] = {
            "seed_only": _metrics([rows[i] for i in idx],[seed_probs[i] for i in idx],thresholds),
            "generative_rl": _metrics([rows[i] for i in idx],[final_probs[i] for i in idx],thresholds),
        }
    activation_diag={
        "present_followup_target_activation_rate": float(np.mean(present_followup_selected)) if present_followup_selected else float("nan"),
        "present_episode_any_target_activation_rate": float(np.mean(present_episode_any)) if present_episode_any else float("nan"),
        "note": "privileged evaluation diagnostic only; never an attacker feature",
    }
    print(f"\nbenign gate: generated={generated} accepted={accepted} blocked={blocked} external_calls={external}")
    print("training-only activation diagnostic:", activation_diag)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    summary={
        "protocol": {
            "validation": "V6.5 composition-disjoint validation; not used for V7 training",
            "seed": "one fixed V6.5 primary benign probe per candidate",
            "policy_observation": ["candidate description","public seed/follow-up query-answer history","black-box belief","remaining budget","gate feedback"],
            "forbidden_policy_features": ["truth","K_true","mode","system_id","router choice","selected roles","internal trace"],
            "gate_mode": args.gate_mode,
        },
        "seed_only": seed_m, "generative_rl": final_m,
        "per_mode": per_mode, "per_role": per_role,
        "probe_audit": {"generated":generated,"accepted":accepted,"blocked":blocked,"external_queries_sent":external},
        "activation_diagnostic_training_only": activation_diag,
    }
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (outdir/"histories.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in histories),encoding="utf-8")
    with (outdir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["system_id","mode","role","label","k_true","seed_prob","final_prob","threshold","seed_pred","final_pred"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r,p0,p1 in zip(rows,seed_probs,final_probs):
            th=thresholds[r["role"]]
            w.writerow({**r,"seed_prob":p0,"final_prob":p1,"threshold":th,"seed_pred":int(p0>=th),"final_pred":int(p1>=th)})
    print("saved:",outdir/"summary.json")


if __name__ == "__main__": main()
