from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, stable_int, write_json, write_jsonl
from adaptive_v11_active.reward import group_relative_advantages
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.role_reward import adaptive_role_reward, role_log_score
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES

EPS = 1e-6


def role_metrics(role_probs: dict[str, float], true_roles: tuple[str, ...]) -> dict:
    truth = set(true_roles)
    ys = np.asarray([int(r in truth) for r in TRAIN_ROLES], dtype=np.int64)
    ps = np.asarray([min(1-EPS, max(EPS, float(role_probs[r]))) for r in TRAIN_ROLES], dtype=np.float64)
    ll = float(np.mean(-(ys*np.log(ps)+(1-ys)*np.log1p(-ps))))
    brier = float(np.mean((ps-ys)**2))
    auc = float(roc_auc_score(ys, ps)) if len(set(ys.tolist())) == 2 else float("nan")
    pos = float(ps[ys == 1].mean()) if np.any(ys == 1) else float("nan")
    neg = float(ps[ys == 0].mean()) if np.any(ys == 0) else float("nan")
    k = len(truth)
    topk = {r for r, _ in sorted(role_probs.items(), key=lambda kv: kv[1], reverse=True)[:k]}
    oracle_k_recall = len(topk & truth) / max(1, k)
    return {
        "role_log_loss": ll,
        "role_brier": brier,
        "role_auc": auc,
        "mean_true_role_prob": pos,
        "mean_false_role_prob": neg,
        "true_false_margin": pos-neg,
        "oracle_k_role_recall": float(oracle_k_recall),
    }


def main():
    ap = argparse.ArgumentParser(description="Role-only belief update + truth-only GRPO reward")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", default="artifacts/adaptive_v11_role_only_v1")
    ap.add_argument("--belief", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--manifest", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--seed", type=int, default=984311)
    args = ap.parse_args()

    root = Path(args.root)
    states = {
        r["session_id"]: PublicBeliefState.from_public_dict(r)
        for r in read_jsonl(root / f"states_{args.split}_round_{args.round-1:02d}.jsonl")
    }
    candidates = read_jsonl(root / f"candidates_{args.split}_round_{args.round:02d}.jsonl")
    responses = read_jsonl(root / f"responses_{args.split}_round_{args.round:02d}.jsonl")
    scored = read_jsonl(root / f"scored_{args.split}_round_{args.round:02d}.jsonl")
    manifest = json.loads(Path(args.manifest).read_text())
    truth = {r["session_id"]: r for r in manifest["systems"] if r["split"] == args.split}
    belief = FrozenBeliefBundle.load(args.belief)

    cand_by = {(r["session_id"], int(r["candidate_index"])): r for r in candidates}
    resp_by = {(r["session_id"], int(r["candidate_index"])): r for r in responses}
    score_by = {}
    for r in scored:
        sid = str(r.get("system_id", r.get("session_id", "")))
        var = int(dict(r.get("action", {})).get("variant", r.get("candidate_index", 0)))
        score_by[(sid, var)] = r

    grouped = defaultdict(list)
    for (sid, _), c in cand_by.items():
        grouped[sid].append(c)

    train_rows = []
    next_states = []
    metrics = []

    for sid, cs in sorted(grouped.items()):
        state = states[sid]
        t = truth[sid]
        true_roles = tuple(map(str, t["roles"]))
        evaluated = []

        for c in sorted(cs, key=lambda x: int(x["candidate_index"])):
            idx = int(c["candidate_index"])
            r = resp_by[(sid, idx)]
            sc = score_by.get((sid, idx))
            if c.get("valid") and r.get("execution_status") == "ok" and sc is None:
                raise RuntimeError(f"missing scorer output sid={sid} cand={idx}")

            if c.get("valid") and r.get("execution_status") == "ok" and sc is not None:
                frame = EvidenceFrame(
                    query=str(c["query"]),
                    answer=str(r.get("final_output", "")),
                    v135={str(k): float(v) for k, v in dict(sc.get("evidence", {})).items()},
                    v134={str(k): float(v) for k, v in dict(sc.get("metadata", {}).get("evidence_v134", {})).items()},
                )
                new_history = [*state.history, frame]
                new_k, new_roles = role_only_predict(belief, new_history)
            else:
                frame = None
                new_k, new_roles = dict(state.k_probs), dict(state.role_probs)

            rb = adaptive_role_reward(
                old_role_probs=state.role_probs,
                new_role_probs=new_roles,
                true_roles=true_roles,
                query=str(c.get("query", "")),
                previous_queries=[h.query for h in state.history],
                generator_valid=bool(c.get("valid")),
            )
            evaluated.append((c, frame, new_k, new_roles, rb))

        rewards = [x[4].total for x in evaluated]
        advs = group_relative_advantages(rewards)
        for (c, frame, new_k, new_roles, rb), adv in zip(evaluated, advs):
            if args.split == "train" and c.get("completion"):
                train_rows.append({
                    "session_id": sid,
                    "round": args.round,
                    "candidate_index": int(c["candidate_index"]),
                    "group_id": c["group_id"],
                    "prompt": c["prompt"],
                    "completion": c["completion"],
                    "old_logp": float(c["old_logp"]),
                    "ref_logp": float(c["ref_logp"]),
                    "reward": float(rb.total),
                    "advantage": float(adv),
                    "reward_breakdown_training_only": rb.to_dict(),
                })

        viable = [x for x in evaluated if x[1] is not None]
        if not viable:
            # No usable observation for this session in this round.
            # Preserve the previous public belief/history instead of
            # crashing the whole experiment or fabricating evidence.
            print(
                f"[advance-warning] no viable candidate in {sid}; "
                f"carrying previous belief forward",
                flush=True,
            )

            ns = PublicBeliefState(
                session_id=sid,
                split=args.split,
                round_index=args.round,
                k_probs=dict(state.k_probs),
                role_probs=dict(state.role_probs),
                history=list(state.history),
            )
            ns.validate(TRAIN_ROLES, belief.ks)
            next_states.append(ns.to_public_dict())

            rm = role_metrics(state.role_probs, true_roles)
            metrics.append({
                "session_id": sid,
                "split": args.split,
                "round": args.round,
                "chosen_candidate": -1,
                "chosen_reward_training_only": 0.0,
                "probe_status": "no_viable_candidate",
                **rm,
            })
            continue

        # Continue from a designated on-policy sample independent of reward.
        rng = random.Random(stable_int(args.seed, args.split, args.round, sid))
        chosen = viable[rng.randrange(len(viable))]
        c, frame, new_k, new_roles, rb = chosen
        ns = PublicBeliefState(
            session_id=sid,
            split=args.split,
            round_index=args.round,
            k_probs=dict(new_k),  # deliberately uniform in phase 1
            role_probs=dict(new_roles),
            history=[*state.history, frame],
        )
        ns.validate(TRAIN_ROLES, belief.ks)
        next_states.append(ns.to_public_dict())

        rm = role_metrics(new_roles, true_roles)
        metrics.append({
            "session_id": sid,
            "split": args.split,
            "round": args.round,
            "chosen_candidate": int(c["candidate_index"]),
            "chosen_reward_training_only": float(rb.total),
            **rm,
        })

    write_jsonl(root / f"states_{args.split}_round_{args.round:02d}.jsonl", next_states)
    write_jsonl(root / f"metrics_{args.split}_round_{args.round:02d}.jsonl", metrics)
    if args.split == "train":
        write_jsonl(root / f"grpo_train_round_{args.round:02d}.jsonl", train_rows)

    summary = {
        "phase": "role_only",
        "split": args.split,
        "round": args.round,
        "sessions": len(metrics),
        "mean_role_log_loss": float(np.mean([x["role_log_loss"] for x in metrics])),
        "mean_role_brier": float(np.mean([x["role_brier"] for x in metrics])),
        "mean_role_auc": float(np.nanmean([x["role_auc"] for x in metrics])),
        "mean_true_false_margin": float(np.mean([x["true_false_margin"] for x in metrics])),
        "mean_oracle_k_role_recall": float(np.mean([x["oracle_k_role_recall"] for x in metrics])),
        "note": "K is intentionally uniform and is not used in policy state updates or reward during phase 1.",
    }
    write_json(root / f"advance_{args.split}_round_{args.round:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
