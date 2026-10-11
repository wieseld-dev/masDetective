#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, write_json, write_jsonl
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


EPS = 1e-8


def role_brier(probs, truth_roles):
    truth = set(map(str, truth_roles))

    y = np.asarray(
        [int(r in truth) for r in TRAIN_ROLES],
        dtype=np.float64,
    )

    p = np.asarray(
        [
            min(1.0 - EPS, max(EPS, float(probs[r])))
            for r in TRAIN_ROLES
        ],
        dtype=np.float64,
    )

    return float(np.mean((p - y) ** 2))


def role_auc(probs, truth_roles):
    truth = set(map(str, truth_roles))

    y = np.asarray(
        [int(r in truth) for r in TRAIN_ROLES],
        dtype=np.int64,
    )

    p = np.asarray(
        [float(probs[r]) for r in TRAIN_ROLES],
        dtype=np.float64,
    )

    if len(set(y.tolist())) < 2:
        return float("nan")

    return float(roc_auc_score(y, p))


def kprob(d, k):
    if str(k) in d:
        return float(d[str(k)])
    if k in d:
        return float(d[k])
    return 0.0


def k_brier(probs, true_k, ks):
    y = np.asarray(
        [int(int(k) == int(true_k)) for k in ks],
        dtype=np.float64,
    )

    p = np.asarray(
        [kprob(probs, k) for k in ks],
        dtype=np.float64,
    )

    return float(np.mean((p - y) ** 2))


def scored_index(rows):
    out = {}

    for x in rows:
        sid = str(
            x.get(
                "system_id",
                x.get("session_id", "")
            )
        )

        action = dict(x.get("action", {}))

        idx = int(
            action.get(
                "variant",
                x.get("candidate_index", 0)
            )
        )

        key = (sid, idx)

        if key in out:
            raise RuntimeError(
                f"duplicate scored key {key}"
            )

        out[key] = x

    return out


def response_index(rows):
    out = {}

    for x in rows:
        key = (
            str(x["session_id"]),
            int(x["candidate_index"]),
        )

        if key in out:
            raise RuntimeError(
                f"duplicate response key {key}"
            )

        out[key] = x

    return out


def candidate_index(rows):
    out = {}

    for x in rows:
        key = (
            str(x["session_id"]),
            int(x["candidate_index"]),
        )

        if key in out:
            raise RuntimeError(
                f"duplicate candidate key {key}"
            )

        out[key] = x

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base",
        default="artifacts/adaptive_v11_curriculum_v2",
    )

    ap.add_argument(
        "--belief",
        default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib",
    )

    ap.add_argument(
        "--out-dir",
        default="artifacts/selector_v3_multiround_g8_dataset_v1",
    )

    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    belief = FrozenBeliefBundle.load(args.belief)

    output_rows = []

    group_sizes_before = Counter()
    group_sizes_after = Counter()
    round_counts = Counter()
    parent_counts = Counter()

    skipped_invalid = 0
    skipped_failed = 0
    skipped_missing_score = 0
    missing_join = 0

    roots = sorted(
        base.glob("stage_*/cycle_*")
    )

    print(
        f"roots={len(roots)}",
        flush=True,
    )

    for root_i, root in enumerate(roots, 1):

        pool_files = sorted(
            root.glob(
                "candidate_pool_train_round_*.jsonl"
            )
        )

        if not pool_files:
            continue

        print(
            f"[{root_i:02d}/{len(roots):02d}] "
            f"{root}",
            flush=True,
        )

        for pool_path in pool_files:

            rd = int(
                pool_path.stem.split("_")[-1]
            )

            state_path = (
                root /
                f"states_train_round_{rd-1:02d}.jsonl"
            )

            response_path = (
                root /
                f"responses_train_round_{rd:02d}.jsonl"
            )

            scored_path = (
                root /
                f"scored_train_round_{rd:02d}.jsonl"
            )

            for p in [
                state_path,
                response_path,
                scored_path,
            ]:
                if not p.exists():
                    raise FileNotFoundError(p)

            states = {
                str(x["session_id"]):
                    PublicBeliefState.from_public_dict(x)
                for x in read_jsonl(state_path)
            }

            candidates = candidate_index(
                read_jsonl(pool_path)
            )

            responses = response_index(
                read_jsonl(response_path)
            )

            scored = scored_index(
                read_jsonl(scored_path)
            )

            by_sid = defaultdict(list)

            for key, c in candidates.items():
                by_sid[key[0]].append(c)

            for sid, cs in by_sid.items():
                group_sizes_before[len(cs)] += 1

                if sid not in states:
                    raise RuntimeError(
                        f"missing state {root} R{rd} {sid}"
                    )

                state = states[sid]
                kept = 0

                for c in sorted(
                    cs,
                    key=lambda x:
                        int(x["candidate_index"])
                ):
                    idx = int(c["candidate_index"])
                    key = (sid, idx)

                    if key not in responses:
                        missing_join += 1
                        continue

                    r = responses[key]

                    if not bool(c.get("valid", False)):
                        skipped_invalid += 1
                        continue

                    if str(
                        r.get("execution_status", "")
                    ) != "ok":
                        skipped_failed += 1
                        continue

                    sc = scored.get(key)

                    if sc is None:
                        skipped_missing_score += 1
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

                    if not v134 or not v135:
                        raise RuntimeError(
                            f"missing dual evidence "
                            f"{root} R{rd} {key}"
                        )

                    answer = str(
                        r.get(
                            "final_output",
                            r.get(
                                "answer",
                                r.get("output", "")
                            )
                        )
                    )

                    frame = EvidenceFrame(
                        query=str(c["query"]),
                        answer=answer,
                        v135=v135,
                        v134=v134,
                    )

                    new_history = [
                        *state.history,
                        frame,
                    ]

                    new_k, new_roles = (
                        role_only_predict(
                            belief,
                            new_history,
                        )
                    )

                    truth_roles = tuple(
                        map(
                            str,
                            sc.get(
                                "truth_roles",
                                r.get(
                                    "truth_roles_training_only",
                                    [],
                                ),
                            ),
                        )
                    )

                    if not truth_roles:
                        raise RuntimeError(
                            f"missing truth roles {key}"
                        )

                    true_k = len(truth_roles)

                    rb0 = role_brier(
                        state.role_probs,
                        truth_roles,
                    )

                    rb1 = role_brier(
                        new_roles,
                        truth_roles,
                    )

                    kb0 = k_brier(
                        state.k_probs,
                        true_k,
                        belief.ks,
                    )

                    kb1 = k_brier(
                        new_k,
                        true_k,
                        belief.ks,
                    )

                    auc0 = role_auc(
                        state.role_probs,
                        truth_roles,
                    )

                    auc1 = role_auc(
                        new_roles,
                        truth_roles,
                    )

                    state_id = (
                        f"{root.as_posix()}::"
                        f"r{rd:02d}::{sid}"
                    )

                    output_rows.append({
                        "state_id": state_id,
                        "system_id": sid,
                        "parent": root.as_posix(),

                        "round": rd,
                        "candidate_index": idx,

                        "query":
                            str(c["query"]).strip(),

                        "role_probs_before":
                            dict(state.role_probs),

                        "k_probs_before":
                            dict(state.k_probs),

                        "truth_roles_training_only":
                            list(truth_roles),

                        "true_k_training_only":
                            true_k,

                        "prior_role_brier_training_only":
                            rb0,

                        "post_role_brier_training_only":
                            rb1,

                        "role_brier_gain_training_only":
                            rb0 - rb1,

                        "prior_k_brier_training_only":
                            kb0,

                        "post_k_brier_training_only":
                            kb1,

                        "k_brier_gain_training_only":
                            kb0 - kb1,

                        "role_auc_before_training_only":
                            auc0,

                        "role_auc_after_training_only":
                            auc1,

                        "role_auc_gain_training_only":
                            auc1 - auc0,

                        "sensor_method":
                            str(sc.get("method")),

                        "sensor_encoding": {
                            "v134":
                                "absolute_composition_raw_logit",
                            "v135":
                                "query_prior_cancelled_causal_evidence",
                        },

                        "selected_roles_not_used_for_update":
                            r.get(
                                "selected_roles_training_only",
                                [],
                            ),
                    })

                    kept += 1

                group_sizes_after[kept] += 1
                round_counts[rd] += kept
                parent_counts[root.as_posix()] += kept

    if not output_rows:
        raise RuntimeError("zero output rows")

    write_jsonl(
        out / "all.jsonl",
        output_rows,
    )

    gains = np.asarray(
        [
            float(
                x["role_brier_gain_training_only"]
            )
            for x in output_rows
        ],
        dtype=np.float64,
    )

    auc_gains = np.asarray(
        [
            float(
                x["role_auc_gain_training_only"]
            )
            for x in output_rows
            if math.isfinite(
                float(
                    x["role_auc_gain_training_only"]
                )
            )
        ],
        dtype=np.float64,
    )

    state_ids = {
        x["state_id"]
        for x in output_rows
    }

    systems = {
        x["system_id"]
        for x in output_rows
    }

    summary = {
        "rows": len(output_rows),
        "states": len(state_ids),
        "systems": len(systems),

        "candidate_count_before_filter":
            dict(sorted(group_sizes_before.items())),

        "candidate_count_after_filter":
            dict(sorted(group_sizes_after.items())),

        "round_row_counts":
            dict(sorted(round_counts.items())),

        "parents": len(parent_counts),

        "role_brier_gain": {
            "mean": float(gains.mean()),
            "std": float(gains.std()),
            "min": float(gains.min()),
            "median": float(
                np.median(gains)
            ),
            "max": float(gains.max()),
            "positive_fraction": float(
                np.mean(gains > 0)
            ),
        },

        "role_auc_gain": {
            "mean": float(auc_gains.mean()),
            "median": float(
                np.median(auc_gains)
            ),
            "positive_fraction": float(
                np.mean(auc_gains > 0)
            ),
        },

        "skipped": {
            "invalid": skipped_invalid,
            "failed_execution": skipped_failed,
            "missing_score":
                skipped_missing_score,
            "missing_join": missing_join,
        },

        "belief":
            str(Path(args.belief)),

        "protocol":
            "dual_v13_4_v13_5 + current frozen belief",
    }

    write_json(
        out / "summary.json",
        summary,
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
