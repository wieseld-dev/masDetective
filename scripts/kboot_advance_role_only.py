from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, write_json, write_jsonl
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES

EPS = 1e-6


def role_only_probs(bundle: FrozenBeliefBundle, history):
    if not history:
        _, rp = bundle.initial()
        return dict(rp)
    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    pcal = bundle._apply_platt(bundle.role_platt, raw)
    return {
        r: float(min(1.0 - EPS, max(EPS, pcal[i])))
        for i, r in enumerate(bundle.roles)
    }


def role_metrics(role_probs: dict[str, float], true_roles: tuple[str, ...]) -> dict:
    truth = set(true_roles)
    ys = np.asarray([int(r in truth) for r in TRAIN_ROLES], dtype=np.int64)
    ps = np.asarray([min(1-EPS, max(EPS, float(role_probs[r]))) for r in TRAIN_ROLES], dtype=np.float64)
    ll = float(np.mean(-(ys*np.log(ps) + (1-ys)*np.log1p(-ps))))
    brier = float(np.mean((ps-ys)**2))
    auc = float(roc_auc_score(ys, ps)) if len(set(ys.tolist())) == 2 else float("nan")
    pos = ps[ys == 1]
    neg = ps[ys == 0]
    margin = float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan")
    k = len(true_roles)
    top = set(np.asarray(TRAIN_ROLES)[np.argsort(-ps)[:k]].tolist())
    recall = len(top & truth) / max(1, k)
    return {
        "role_log_loss": ll,
        "role_brier": brier,
        "role_auc": auc,
        "true_false_margin": margin,
        "oracle_k_role_recall": float(recall),
    }


def main():
    ap = argparse.ArgumentParser(description="Advance K-bootstrap trajectories using role evidence only; K remains uniform")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    bundle = FrozenBeliefBundle.load(args.belief)
    uniform_k, _ = bundle.initial()

    states = {
        r["session_id"]: PublicBeliefState.from_public_dict(r)
        for r in read_jsonl(root / f"states_{args.split}_round_{args.round-1:02d}.jsonl")
    }
    candidates = read_jsonl(root / f"candidates_{args.split}_round_{args.round:02d}.jsonl")
    responses = read_jsonl(root / f"responses_{args.split}_round_{args.round:02d}.jsonl")
    scored = read_jsonl(root / f"scored_{args.split}_round_{args.round:02d}.jsonl")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {r["session_id"]: r for r in manifest["systems"] if r["split"] == args.split}

    cand_by = {str(r["session_id"]): r for r in candidates}
    resp_by = {str(r["session_id"]): r for r in responses}
    score_by = {}
    for r in scored:
        sid = str(r.get("system_id", r.get("session_id", "")))
        score_by[sid] = r

    next_states = []
    metrics = []
    for sid, state in sorted(states.items()):
        c = cand_by.get(sid)
        r = resp_by.get(sid)
        sc = score_by.get(sid)
        observed = bool(
            c is not None and r is not None and sc is not None
            and c.get("valid") and r.get("execution_status") == "ok"
        )

        history = list(state.history)
        status = "carry_forward"
        if observed:
            frame = EvidenceFrame(
                query=str(c["query"]),
                answer=str(r.get("final_output", "")),
                v135={str(k): float(v) for k, v in dict(sc.get("evidence", {})).items()},
                v134={str(k): float(v) for k, v in dict(sc.get("metadata", {}).get("evidence_v134", {})).items()},
            )
            history.append(frame)
            status = "observed"

        rp = role_only_probs(bundle, history)
        ns = PublicBeliefState(
            session_id=sid,
            split=args.split,
            round_index=args.round,
            k_probs=dict(uniform_k),  # deliberately do NOT use the old K head
            role_probs=dict(rp),
            history=history,
        )
        ns.validate(TRAIN_ROLES, bundle.ks)
        next_states.append(ns.to_public_dict())

        t = truth[sid]
        rm = role_metrics(rp, tuple(map(str, t["roles"])))
        metrics.append({
            "session_id": sid,
            "split": args.split,
            "round": args.round,
            "probe_status": status,
            "history_len": len(history),
            "true_k_eval_only": int(t["k"]),
            **rm,
        })

    write_jsonl(root / f"states_{args.split}_round_{args.round:02d}.jsonl", next_states)
    write_jsonl(root / f"kboot_metrics_{args.split}_round_{args.round:02d}.jsonl", metrics)

    def mean(k):
        xs = [float(x[k]) for x in metrics if math.isfinite(float(x[k]))]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "phase": "k_bootstrap_uniform_k_role_adaptive",
        "split": args.split,
        "round": args.round,
        "sessions": len(metrics),
        "observed": sum(x["probe_status"] == "observed" for x in metrics),
        "mean_history_len": float(np.mean([x["history_len"] for x in metrics])),
        "mean_role_log_loss": mean("role_log_loss"),
        "mean_role_brier": mean("role_brier"),
        "mean_role_auc": mean("role_auc"),
        "mean_true_false_margin": mean("true_false_margin"),
        "mean_oracle_k_role_recall": mean("oracle_k_role_recall"),
        "note": "K is held exactly uniform over 3..6 during acquisition; only role belief and public history adapt.",
    }
    write_json(root / f"kboot_advance_{args.split}_round_{args.round:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
