#!/usr/bin/env python3
from __future__ import annotations

import argparse, itertools, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb
from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()
from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec
from v11.role_universe import TRAIN_ROLES

TRUE_ROLES = ("code", "planning", "research")
KS = (3, 4, 5, 6)

# One natural, task-oriented probe per role. No "do you have agent X?" questions.
PROBES = {
    "code": "Write a Python function that takes a list of integers and returns the three most frequent values with deterministic tie-breaking.",
    "writing": "Rewrite this short paragraph into a concise professional executive summary: The rollout slipped because several teams interpreted the requirements differently and rework accumulated.",
    "math": "Solve for x and show the key steps: 3(x - 4) + 7 = 2x + 11.",
    "planning": "Create a five-step implementation plan for migrating a small internal service to a new API without disrupting users.",
    "research": "Compare two plausible explanations for why retrieval-augmented generation can reduce hallucinations, and state what evidence would distinguish them.",
    "data": "Given columns user_id, signup_date, country, revenue, and churned, describe a compact analysis to identify the strongest churn patterns.",
    "design": "Propose a clean dashboard layout for monitoring service health, latency, errors, and recent incidents.",
    "operations": "A production job is intermittently missing its SLA. Give an operational triage sequence that minimizes time to isolate the bottleneck.",
    "qa_testing": "Design a focused test plan for a password-reset flow, including edge cases and failure conditions.",
    "translation": "Translate into clear English: 'המערכת צריכה להמשיך לעבוד גם אם אחד השירותים הזמניים אינו זמין.'",
    "financial_analysis": "A project costs $120,000 upfront and is expected to save $3,000 per month. Give a simple payback-period analysis and key caveats.",
    "database_design": "Design a normalized relational schema for customers, orders, products, and order items, including primary and foreign keys.",
    "accessibility_review": "Review a web form conceptually for accessibility risks involving labels, keyboard navigation, focus order, and error messages.",
    "fact_checking": "Assess the claim 'adding more parameters always makes a language model more accurate' and explain what would need verification.",
    "requirements_analysis": "Turn this vague request into testable requirements: 'Build a fast, easy-to-use internal search tool for company documents.'",
    "incident_management": "Draft the first ten minutes of an incident-response plan for a sudden production outage affecting all users.",
    "experiment_design": "Design an experiment to test whether a new ranking algorithm improves user task completion without increasing latency.",
    "risk_assessment": "Identify and prioritize the main risks in moving a critical internal workflow from manual review to an automated system.",
    "documentation": "Write a concise README section explaining how to configure, run, and verify a command-line data-processing tool.",
    "customer_support_triage": "Classify and prioritize these support issues: cannot log in, billing question, data loss report, feature request, slow page load.",
}

def normalize(w):
    w = np.asarray(w, dtype=np.float64)
    return w / max(float(w.sum()), 1e-300)

def exact_hypotheses(roles):
    zs, ks, ws = [], [], []
    for k in KS:
        combos = list(itertools.combinations(range(len(roles)), k))
        per = (1.0 / len(KS)) / len(combos)
        for c in combos:
            z = np.zeros(len(roles), dtype=np.float32)
            z[list(c)] = 1.0
            zs.append(z); ks.append(k); ws.append(per)
    return np.stack(zs), np.asarray(ks, dtype=np.int64), normalize(ws)

def k_probs(W, K):
    return {str(k): float(W[K == k].sum()) for k in KS}

def role_probs(W, Z, roles):
    p = W @ Z
    return {r: float(p[i]) for i, r in enumerate(roles)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--mode", default="mixed", choices=["single","fanout2","fanout3","mixed"])
    ap.add_argument("--style", default="s1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out", default="artifacts/oracle_capability_sweep_k3")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing; source .env first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    roles = list(TRAIN_ROLES)
    if len(roles) != 20:
        raise RuntimeError(f"expected 20 TRAIN_ROLES, got {len(roles)}")
    if set(PROBES) != set(roles):
        raise RuntimeError(
            "Probe map mismatch.\n"
            f"missing={sorted(set(roles)-set(PROBES))}\n"
            f"extra={sorted(set(PROBES)-set(roles))}"
        )

    # Exact hypothesis space.
    Z, Klabels, W = exact_hypotheses(roles)
    true_vec = np.asarray([1.0 if r in TRUE_ROLES else 0.0 for r in roles], np.float32)
    truth_mask = np.all(Z == true_vec[None, :], axis=1)

    print("EXACT HYPOTHESES:", len(Z))
    print("TRUE:", TRUE_ROLES)
    print("TRUE PRIOR MASS:", float(W[truth_mask].sum()))
    print("BETA:", args.beta)

    # Observation model + calibration.
    root = Path("artifacts/adaptive_v11_curriculum_v2")
    obsdir = Path("artifacts/observation_model_v1")
    caldir = Path("artifacts/calibrated_bayes_diag_v1")

    examples, obs_roles, _, _ = om.collect_examples(root)
    if list(obs_roles) != roles:
        raise RuntimeError("role ordering mismatch")

    split = json.loads((obsdir / "split_systems.json").read_text())
    train_systems = set(map(str, split["train"]))
    train_idx = np.asarray(
        [i for i,x in enumerate(examples) if str(x["system_id"]) in train_systems],
        dtype=int,
    )

    ckpt = torch.load(obsdir / "model.pt", map_location="cpu")
    fwd = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    )
    fwd.load_state_dict(ckpt["state_dict"])
    fwd.eval()

    ri = {r:i for i,r in enumerate(roles)}
    pos = np.zeros(len(roles), dtype=np.float64)
    for i in train_idx:
        for r in examples[i]["selected_roles"]:
            pos[ri[r]] += 1
    posw = np.clip((len(train_idx)-pos)/np.maximum(pos,1.0), 1.0, 20.0)
    log_posw = np.log(posw)

    cal = json.loads((caldir / "summary.json").read_text())
    T = float(cal["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])
    max_round = max(int(x["round"]) for x in examples)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sid = "oracle_sweep_k3_code_planning_research"
    rows = []

    # Sweep in universe order for reproducibility.
    for t, target_role in enumerate(roles, start=1):
        q = PROBES[target_role]

        # Fresh instance for each probe; same hidden composition.
        spec = SingleRouterSystemSpec(
            system_id=sid,
            split="demo",
            execution_mode=args.mode,
            style=args.style,
            worker_roles=TRUE_ROLES,
        )
        target = SingleRouterFanoutMAS(spec, model=args.model)
        output = str(target.query(q))
        tr = getattr(target, "_last_trace", None)
        if tr is None:
            raise RuntimeError("target did not expose _last_trace")
        selected = list(map(str, getattr(tr, "selected_roles", ())))

        kp_before = k_probs(W, Klabels)
        rp_before = role_probs(W, Z, roles)
        belief = np.asarray([rp_before[r] for r in roles], np.float32)
        kvec = np.asarray([kp_before[str(k)] for k in KS], np.float32)

        qemb = np.asarray(embedder.encode([q], convert_to_numpy=True), np.float32)
        # Keep normalized round in-range; this is a diagnostic sweep beyond normal 5 rounds.
        rnorm = float(min(t, max_round) / max(max_round, 1))

        logits = vb.particle_logits(
            fwd, qemb, belief, kvec, rnorm, Z, "cpu", 4096
        )
        pY = vb.calibrated_hidden_probs(logits, log_posw, T)[0]

        Y = np.asarray([1.0 if r in set(selected) else 0.0 for r in roles], np.float64)

        # ORACLE / evaluation-only update from true selected roles.
        W = vb.update_weights(pY, Y, W, args.beta)
        W = normalize(W)

        kp_after = k_probs(W, Klabels)
        rp_after = role_probs(W, Z, roles)
        pred_k = int(max(kp_after, key=kp_after.get))
        top5 = sorted(rp_after.items(), key=lambda x: -x[1])[:5]

        print("\n" + "="*100)
        print(f"SWEEP {t:02d}/20 | TARGET CAPABILITY: {target_role}")
        print("QUERY:", q)
        print("ACTUALLY SELECTED [EVAL ONLY]:", selected)
        print("P(K):", {k: round(v,6) for k,v in kp_after.items()}, " pred=", pred_k)
        print("TRUE COMPOSITION MASS:", float(W[truth_mask].sum()))
        print("TOP ROLES:")
        for r,p in top5:
            flag = " <-- TRUE" if r in TRUE_ROLES else ""
            print(f"  {r:<28} {p:.4f}{flag}")

        rows.append({
            "step": t,
            "target_capability": target_role,
            "query": q,
            "final_output": output,
            "selected_roles_eval_only": selected,
            "k_probs": kp_after,
            "pred_k": pred_k,
            "role_probs": rp_after,
            "true_composition_mass": float(W[truth_mask].sum()),
        })
        (out / "trajectory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

        if args.sleep > 0:
            time.sleep(args.sleep)

    kp = k_probs(W, Klabels)
    rp = role_probs(W, Z, roles)
    pred_k = int(max(kp, key=kp.get))
    top3 = [r for r,_ in sorted(rp.items(), key=lambda x:-x[1])[:3]]

    summary = {
        "true_k": 3,
        "true_roles": list(TRUE_ROLES),
        "pred_k": pred_k,
        "p_k3": kp["3"],
        "k_correct": pred_k == 3,
        "top3": top3,
        "top3_exact": set(top3) == set(TRUE_ROLES),
        "true_composition_mass": float(W[truth_mask].sum()),
        "final_k_probs": kp,
        "final_role_probs": rp,
        "beta": args.beta,
        "n_probes": len(roles),
        "mode": args.mode,
        "note": "ORACLE diagnostic: selected_roles is privileged evaluation-only evidence.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "#"*100)
    print("FINAL")
    print("#"*100)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
