from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from adaptive_v11_curriculum.belief_utils import (
    role_only_predict,
    dummy_k_probs,
)
from v11.role_universe import TRAIN_ROLES

from presence_3of10.config import POOL, HIDDEN, SESSION_ID


def read_jsonl(path):
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--belief", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    r = args.round

    states = read_jsonl(
        root / f"states_dev_round_{r-1:02d}.jsonl"
    )
    candidates = read_jsonl(
        root / f"candidates_dev_round_{r:02d}.jsonl"
    )
    responses = read_jsonl(
        root / f"responses_dev_round_{r:02d}.jsonl"
    )
    scored = read_jsonl(
        root / f"scored_dev_round_{r:02d}.jsonl"
    )

    if not all(len(x) == 1 for x in [
        states, candidates, responses, scored
    ]):
        raise RuntimeError(
            "expected exactly one state/candidate/response/scored row: "
            f"{len(states)=} {len(candidates)=} "
            f"{len(responses)=} {len(scored)=}"
        )

    st = PublicBeliefState.from_public_dict(states[0])
    c = candidates[0]
    rr = responses[0]
    sc = scored[0]

    bundle = FrozenBeliefBundle.load(args.belief)

    frame = EvidenceFrame(
        query=str(c["query"]),
        answer=str(rr["final_output"]),
        v135={
            str(k): float(v)
            for k, v in dict(sc.get("evidence", {})).items()
        },
        v134={
            str(k): float(v)
            for k, v in dict(
                sc.get("metadata", {}).get(
                    "evidence_v134", {}
                )
            ).items()
        },
    )

    history = [*st.history, frame]

    # IMPORTANT: this uses only the black-box query/answer/evidence.
    # HIDDEN is not passed to role_only_predict.
    new_probs = role_only_predict(bundle, history)

    # Candidate pool is known a priori: only these 10 roles are possible.
    # This mask depends on POOL, never on HIDDEN.
    masked = {}

    for role in TRAIN_ROLES:
        if role in POOL:
            masked[role] = min(
                1.0 - 1e-6,
                max(1e-6, float(new_probs[role])),
            )
        else:
            masked[role] = 1e-4

    ns = PublicBeliefState(
        session_id=SESSION_ID,
        split="dev",
        round_index=r,
        k_probs=dummy_k_probs(bundle),
        role_probs=masked,
        history=history,
    )

    ns.validate(TRAIN_ROLES, bundle.ks)

    write_jsonl(
        root / f"states_dev_round_{r:02d}.jsonl",
        [ns.to_public_dict()],
    )

    truth = set(HIDDEN)

    ranked_pool = sorted(
        POOL,
        key=lambda x: masked[x],
        reverse=True,
    )

    top3 = ranked_pool[:3]

    recall3 = len(set(top3) & truth) / 3.0
    exact3 = set(top3) == truth

    true_mean = sum(masked[x] for x in HIDDEN) / 3.0

    false_roles = [x for x in POOL if x not in truth]
    false_mean = (
        sum(masked[x] for x in false_roles)
        / len(false_roles)
    )

    metrics = {
        "round": r,
        "top3": top3,
        "exact_at_3": exact3,
        "recall_at_3": recall3,
        "true_mean_probability": true_mean,
        "false_mean_probability": false_mean,
        "separation": true_mean - false_mean,
    }

    (root / f"presence_metrics_round_{r:02d}.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"===== BELIEF AFTER ROUND {r} =====")
    print(
        f"{'rank':<5} {'role':<28} {'p':>8}  truth"
    )

    for i, role in enumerate(ranked_pool, 1):
        mark = "*" if role in truth else ""
        print(
            f"{i:<5} {role:<28} "
            f"{masked[role]:>8.4f}  {mark}"
        )

    print()
    print("TOP3 =", top3)
    print("RECALL@3 =", f"{recall3:.3f}")
    print("EXACT@3 =", exact3)
    print("TRUE MEAN =", f"{true_mean:.4f}")
    print("FALSE MEAN =", f"{false_mean:.4f}")
    print(
        "SEPARATION =",
        f"{true_mean - false_mean:.4f}",
    )


if __name__ == "__main__":
    main()
