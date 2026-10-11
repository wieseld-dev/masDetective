from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, stable_int, write_json, write_jsonl
from adaptive_v11_active.reward import adaptive_reward, group_relative_advantages, map_k, predicted_set
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


def f1(pred: set[str], truth: set[str]) -> float:
    if not pred and not truth: return 1.0
    if not pred or not truth: return 0.0
    tp = len(pred & truth); p = tp/len(pred); r=tp/len(truth)
    return 0.0 if p+r == 0 else 2*p*r/(p+r)


def main():
    ap = argparse.ArgumentParser(description="Update beliefs, compute truth-only rewards, and advance on-policy trajectories")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", default="artifacts/adaptive_v11_active_v1")
    ap.add_argument("--belief", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--manifest", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--seed", type=int, default=984311)
    args = ap.parse_args()

    root = Path(args.root)
    states = {r["session_id"]: PublicBeliefState.from_public_dict(r) for r in read_jsonl(root / f"states_{args.split}_round_{args.round-1:02d}.jsonl")}
    candidates = read_jsonl(root / f"candidates_{args.split}_round_{args.round:02d}.jsonl")
    responses = read_jsonl(root / f"responses_{args.split}_round_{args.round:02d}.jsonl")
    scored = read_jsonl(root / f"scored_{args.split}_round_{args.round:02d}.jsonl")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {r["session_id"]: r for r in manifest["systems"] if r["split"] == args.split}
    belief = FrozenBeliefBundle.load(args.belief)

    cand_by = {(r["session_id"], int(r["candidate_index"])): r for r in candidates}
    resp_by = {(r["session_id"], int(r["candidate_index"])): r for r in responses}
    # Existing scorer preserves system_id and action.variant; session_id may not survive.
    score_by = {}
    for r in scored:
        sid = str(r.get("system_id", r.get("session_id", "")))
        var = int(dict(r.get("action", {})).get("variant", r.get("candidate_index", 0)))
        score_by[(sid, var)] = r

    grouped = defaultdict(list)
    for key, c in cand_by.items(): grouped[key[0]].append(c)
    train_rows = []
    next_states = []
    metrics = []

    for sid, cs in sorted(grouped.items()):
        state = states[sid]
        t = truth[sid]
        true_roles = tuple(map(str, t["roles"]))
        true_k = int(t["k"])
        evaluated = []
        for c in sorted(cs, key=lambda x: int(x["candidate_index"])):
            idx = int(c["candidate_index"])
            r = resp_by[(sid, idx)]
            sc = score_by.get((sid, idx))
            if c.get("valid") and r.get("execution_status") == "ok" and sc is None:
                raise RuntimeError(f"missing scored observation sid={sid} cand={idx}")
            if c.get("valid") and r.get("execution_status") == "ok" and sc is not None:
                frame = EvidenceFrame(
                    query=str(c["query"]),
                    answer=str(r.get("final_output", "")),
                    v135={str(k): float(v) for k, v in dict(sc.get("evidence", {})).items()},
                    v134={str(k): float(v) for k, v in dict(sc.get("metadata", {}).get("evidence_v134", {})).items()},
                )
                new_history = [*state.history, frame]
                new_k, new_roles = belief.predict(new_history)
            else:
                frame = None
                new_history = list(state.history)
                new_k, new_roles = dict(state.k_probs), dict(state.role_probs)

            rb = adaptive_reward(
                old_k_probs=state.k_probs,
                new_k_probs=new_k,
                old_role_probs=state.role_probs,
                new_role_probs=new_roles,
                true_k=true_k,
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

        # Critical: continuation is independent of reward. Uniform over successfully observed
        # policy samples; best-of-G continuation would create optimistic off-policy states.
        viable = [x for x in evaluated if x[1] is not None]
        if not viable:
            raise RuntimeError(f"no viable candidate for trajectory continuation in {sid}")
        rng = random.Random(stable_int(args.seed, args.split, args.round, sid))
        chosen = viable[rng.randrange(len(viable))]
        c, frame, new_k, new_roles, rb = chosen
        ns = PublicBeliefState(
            session_id=sid,
            split=args.split,
            round_index=args.round,
            k_probs=dict(new_k),
            role_probs=dict(new_roles),
            history=[*state.history, frame],
        )
        ns.validate(TRAIN_ROLES, belief.ks)
        next_states.append(ns.to_public_dict())
        pred_set = predicted_set(new_roles, new_k)
        metrics.append({
            "session_id": sid,
            "split": args.split,
            "round": args.round,
            "chosen_candidate": int(c["candidate_index"]),
            "chosen_reward_training_only": float(rb.total),
            "true_k_eval_only": true_k,
            "pred_k": int(map_k(new_k)),
            "k_correct": int(map_k(new_k) == true_k),
            "role_f1": float(f1(pred_set, set(true_roles))),
            "exact_set": int(pred_set == set(true_roles)),
        })

    write_jsonl(root / f"states_{args.split}_round_{args.round:02d}.jsonl", next_states)
    write_jsonl(root / f"metrics_{args.split}_round_{args.round:02d}.jsonl", metrics)
    if args.split == "train":
        write_jsonl(root / f"grpo_train_round_{args.round:02d}.jsonl", train_rows)
    summary = {
        "split": args.split,
        "round": args.round,
        "sessions": len(metrics),
        "k_acc": sum(x["k_correct"] for x in metrics) / max(1, len(metrics)),
        "mean_role_f1": sum(x["role_f1"] for x in metrics) / max(1, len(metrics)),
        "exact_set": sum(x["exact_set"] for x in metrics) / max(1, len(metrics)),
        "note": "truth appears only in reward/metrics; next policy-visible state excludes it",
    }
    write_json(root / f"advance_{args.split}_round_{args.round:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
