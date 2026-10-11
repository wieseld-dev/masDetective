#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
BELIEF_PATH = Path(
    "artifacts/v14_adaptive_role_v1/frozen_belief.joblib"
)
OUT = Path("artifacts/llm_selector_v3_auc_data")

SEED = 38027
CHALLENGE_FRACTION = 0.20
HISTORY_TURNS = 2
ANSWER_CHARS = 320


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def auc_simple(probs, true_roles):
    truth = set(map(str, true_roles))

    pos = [
        float(probs[r])
        for r in TRAIN_ROLES
        if r in truth
    ]

    neg = [
        float(probs[r])
        for r in TRAIN_ROLES
        if r not in truth
    ]

    if not pos or not neg:
        return float("nan")

    wins = 0.0

    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5

    return wins / (len(pos) * len(neg))


def history_text(state):
    if not state.history:
        return "- no previous public interaction"

    blocks = []

    for h in state.history[-HISTORY_TURNS:]:
        q = " ".join(str(h.query).split())
        a = " ".join(str(h.answer).split())

        if len(a) > ANSWER_CHARS:
            a = a[:ANSWER_CHARS] + "..."

        blocks.append(
            f"Q: {q}\n"
            f"VISIBLE ANSWER: {a}"
        )

    return "\n\n".join(blocks)


def scored_key(x):
    sid = str(
        x.get("system_id")
        or x.get("session_id")
        or ""
    )

    action = dict(x.get("action") or {})

    idx = action.get(
        "variant",
        x.get("candidate_index")
    )

    if not sid or idx is None:
        return None

    try:
        idx = int(idx)
    except Exception:
        return None

    return sid, idx


def candidate_file(cycle, r):
    p = cycle / f"candidate_pool_train_round_{r:02d}.jsonl"

    if p.exists():
        return p

    p = cycle / f"candidates_train_round_{r:02d}.jsonl"

    if p.exists():
        return p

    return None


def parse_round(path):
    m = re.search(
        r"round_(\d+)\.jsonl$",
        path.name,
    )

    if not m:
        raise RuntimeError(
            f"cannot parse round from {path}"
        )

    return int(m.group(1))


def unpack_role_prediction(pred):
    # Current active implementation returns (k_probs, role_probs).
    # Keep compatibility with older helper variants that returned only roles.
    if isinstance(pred, tuple):
        if len(pred) != 2:
            raise RuntimeError(
                f"unexpected role_only_predict tuple length={len(pred)}"
            )
        return dict(pred[1])

    return dict(pred)


def main():
    if not ROOT.exists():
        raise RuntimeError(f"missing {ROOT}")

    if not BELIEF_PATH.exists():
        raise RuntimeError(f"missing {BELIEF_PATH}")

    OUT.mkdir(parents=True, exist_ok=True)

    bundle = FrozenBeliefBundle.load(
        BELIEF_PATH
    )

    rows = []

    stats = Counter()

    cycle_dirs = sorted(
        p for p in ROOT.glob("stage_*/cycle_*")
        if p.is_dir()
    )

    print("cycle_dirs =", len(cycle_dirs), flush=True)

    for ci, cycle in enumerate(cycle_dirs, 1):

        response_files = sorted(
            cycle.glob("responses_train_round_*.jsonl")
        )

        if not response_files:
            continue

        print(
            f"[cycle {ci}/{len(cycle_dirs)}] {cycle}",
            flush=True,
        )

        for response_path in response_files:

            r = parse_round(response_path)

            states_path = (
                cycle /
                f"states_train_round_{r-1:02d}.jsonl"
            )

            scored_path = (
                cycle /
                f"scored_train_round_{r:02d}.jsonl"
            )

            cand_path = candidate_file(
                cycle,
                r,
            )

            if (
                not states_path.exists()
                or not scored_path.exists()
                or cand_path is None
            ):
                stats["missing_round_artifacts"] += 1
                continue

            states = {
                str(x["session_id"]):
                    PublicBeliefState.from_public_dict(x)
                for x in read_jsonl(states_path)
            }

            candidates = read_jsonl(cand_path)
            responses = read_jsonl(response_path)
            scored = read_jsonl(scored_path)

            resp_by = {}

            for x in responses:
                try:
                    key = (
                        str(
                            x.get("session_id")
                            or x.get("system_id")
                        ),
                        int(x["candidate_index"]),
                    )
                except Exception:
                    continue

                resp_by[key] = x

            score_by = {}

            for x in scored:
                key = scored_key(x)

                if key is not None:
                    score_by[key] = x

            for c in candidates:

                stats["candidate_rows_seen"] += 1

                if not c.get("valid"):
                    stats["invalid_candidate"] += 1
                    continue

                sid = str(c["session_id"])
                idx = int(c["candidate_index"])

                st = states.get(sid)
                rr = resp_by.get((sid, idx))
                sc = score_by.get((sid, idx))

                if st is None:
                    stats["missing_state"] += 1
                    continue

                if rr is None:
                    stats["missing_response"] += 1
                    continue

                if sc is None:
                    stats["missing_score"] += 1
                    continue

                if rr.get("execution_status") != "ok":
                    stats["bad_execution"] += 1
                    continue

                v135 = {
                    str(k): float(v)
                    for k, v in dict(
                        sc.get("evidence", {})
                    ).items()
                }

                v134 = {
                    str(k): float(v)
                    for k, v in dict(
                        sc.get("metadata", {})
                        .get("evidence_v134", {})
                    ).items()
                }

                # We want V3 aligned to the CURRENT dual-channel sensor.
                if not v135 or not v134:
                    stats["not_dual_channel"] += 1
                    continue

                truth = (
                    rr.get("truth_roles_training_only")
                    or sc.get("truth_roles")
                    or []
                )

                truth = list(map(str, truth))

                if not truth:
                    stats["missing_truth"] += 1
                    continue

                answer = str(
                    rr.get("final_output")
                    or rr.get("output")
                    or rr.get("answer")
                    or ""
                )

                frame = EvidenceFrame(
                    query=str(c["query"]),
                    answer=answer,
                    v135=v135,
                    v134=v134,
                )

                new_history = [
                    *st.history,
                    frame,
                ]

                pred = role_only_predict(
                    bundle,
                    new_history,
                )

                new_roles = unpack_role_prediction(
                    pred
                )

                before_auc = auc_simple(
                    st.role_probs,
                    truth,
                )

                after_auc = auc_simple(
                    new_roles,
                    truth,
                )

                if (
                    not math.isfinite(before_auc)
                    or not math.isfinite(after_auc)
                ):
                    stats["nonfinite_auc"] += 1
                    continue

                gain = (
                    after_auc
                    - before_auc
                )

                k_true = len(truth)

                n_pairs = (
                    k_true
                    * (len(TRAIN_ROLES) - k_true)
                )

                pair_gain = (
                    gain * n_pairs
                )

                state_id = (
                    f"{cycle}::{sid}::r{r:02d}"
                )

                rows.append({
                    "state_id":
                        state_id,

                    "system_id":
                        sid,

                    "source_cycle":
                        str(cycle),

                    "round":
                        r,

                    "candidate_index":
                        idx,

                    "query":
                        str(c["query"]).strip(),

                    "history_text":
                        history_text(st),

                    "role_probs_before": {
                        str(k): float(v)
                        for k, v
                        in st.role_probs.items()
                    },

                    "k_probs_before": {
                        str(k): float(v)
                        for k, v
                        in st.k_probs.items()
                    },

                    # PRIMARY V3 TARGET
                    "role_auc_gain_training_only":
                        float(gain),

                    "role_auc_before_training_only":
                        float(before_auc),

                    "role_auc_after_training_only":
                        float(after_auc),

                    # Equivalent within-state ranking signal.
                    "role_pairwise_gain_training_only":
                        float(pair_gain),

                    "role_pairs_training_only":
                        int(n_pairs),

                    "truth_roles_training_only":
                        truth,

                    # Preserve old trainer schema.
                    # lambda_k=0 so this is never optimized.
                    "k_brier_gain_training_only":
                        0.0,
                })

                stats["kept"] += 1

    # ---------------------------------------------------------
    # Require >=2 usable candidates per decision state.
    # ---------------------------------------------------------

    by_state = defaultdict(list)

    for x in rows:
        by_state[x["state_id"]].append(x)

    good_states = {
        sid
        for sid, xs in by_state.items()
        if len(xs) >= 2
    }

    rows = [
        x for x in rows
        if x["state_id"] in good_states
    ]

    by_state = defaultdict(list)

    for x in rows:
        by_state[x["state_id"]].append(x)

    systems = sorted({
        str(x["system_id"])
        for x in rows
    })

    rng = random.Random(SEED)
    rng.shuffle(systems)

    n_challenge = max(
        1,
        int(round(
            len(systems)
            * CHALLENGE_FRACTION
        )),
    )

    challenge_systems = set(
        systems[:n_challenge]
    )

    train_rows = [
        x for x in rows
        if x["system_id"]
        not in challenge_systems
    ]

    challenge_rows = [
        x for x in rows
        if x["system_id"]
        in challenge_systems
    ]

    # ---------------------------------------------------------
    # Dataset diagnostics.
    # ---------------------------------------------------------

    gains = [
        float(x["role_auc_gain_training_only"])
        for x in rows
    ]

    group_spreads = []
    random_gain = []
    oracle_gain = []
    positive_pool = []

    for sid, xs in by_state.items():
        ys = [
            float(x["role_auc_gain_training_only"])
            for x in xs
        ]

        group_spreads.append(
            max(ys) - min(ys)
        )

        random_gain.append(
            sum(ys) / len(ys)
        )

        oracle_gain.append(
            max(ys)
        )

        positive_pool.append(
            float(any(y > 0 for y in ys))
        )

    mean = (
        sum(gains) / len(gains)
        if gains else float("nan")
    )

    oracle_mean = (
        sum(oracle_gain) / len(oracle_gain)
        if oracle_gain else float("nan")
    )

    random_mean = (
        sum(random_gain) / len(random_gain)
        if random_gain else float("nan")
    )

    headroom = (
        oracle_mean - random_mean
    )

    summary = {
        "version":
            "llm_selector_v3_auc_dataset_v1",

        "source":
            str(ROOT),

        "belief_bundle":
            str(BELIEF_PATH),

        "target":
            "role_auc_gain_training_only",

        "sensor_requirement":
            "V13.5 evidence + V13.4 evidence_v134",

        "rows":
            len(rows),

        "states":
            len(by_state),

        "systems":
            len(systems),

        "train_rows":
            len(train_rows),

        "challenge_rows":
            len(challenge_rows),

        "train_systems":
            len({
                x["system_id"]
                for x in train_rows
            }),

        "challenge_systems":
            len(challenge_systems),

        "group_size_counts":
            dict(
                Counter(
                    len(xs)
                    for xs in by_state.values()
                )
            ),

        "gain": {
            "min":
                min(gains) if gains else None,

            "max":
                max(gains) if gains else None,

            "mean":
                mean,

            "positive_fraction":
                (
                    sum(g > 0 for g in gains)
                    / len(gains)
                    if gains else None
                ),
        },

        "within_state": {
            "mean_spread":
                (
                    sum(group_spreads)
                    / len(group_spreads)
                    if group_spreads else None
                ),

            "random_mean_gain":
                random_mean,

            "oracle_mean_gain":
                oracle_mean,

            "oracle_minus_random":
                headroom,

            "pool_has_positive_fraction":
                (
                    sum(positive_pool)
                    / len(positive_pool)
                    if positive_pool else None
                ),
        },

        "skip_stats":
            dict(stats),

        "challenge_system_ids":
            sorted(challenge_systems),
    }

    write_jsonl(
        OUT / "train.jsonl",
        train_rows,
    )

    write_jsonl(
        OUT / "challenge.jsonl",
        challenge_rows,
    )

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SELECTOR V3 AUC DATASET")
    print("=" * 100)
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    # Safety gates: do not launch 3B training on junk.
    if len(rows) < 500:
        raise RuntimeError(
            f"Too few aligned rows: {len(rows)}"
        )

    if len(by_state) < 100:
        raise RuntimeError(
            f"Too few decision states: {len(by_state)}"
        )

    if not math.isfinite(headroom):
        raise RuntimeError(
            "Non-finite selector headroom"
        )

    if headroom <= 0.01:
        raise RuntimeError(
            f"Insufficient oracle-vs-random headroom: {headroom:.6f}"
        )

    print()
    print("DATASET PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
