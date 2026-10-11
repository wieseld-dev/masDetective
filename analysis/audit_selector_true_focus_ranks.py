#!/usr/bin/env python3

import json
import re
import statistics
from pathlib import Path

from sklearn.metrics import roc_auc_score
from v11.role_universe import TRAIN_ROLES

ROOT = Path(
    "artifacts/v14_cleanR7_selectorV2_simple5_v1"
)

TARGET_RE = re.compile(
    r"TARGET CANDIDATE CAPABILITY:\s*([^\n\r]+)"
)


def read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def get_focus(row):
    prompt = str(row.get("prompt", ""))
    m = TARGET_RE.search(prompt)
    if not m:
        return None
    return m.group(1).strip()


def auc_state(st, true_roles):
    rp = {
        str(k): float(v)
        for k, v in st["role_probs"].items()
    }

    y = [
        int(r in true_roles)
        for r in TRAIN_ROLES
    ]

    s = [
        rp.get(r, 0.0)
        for r in TRAIN_ROLES
    ]

    return float(roc_auc_score(y, s))


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x["split"] == "dev"
}

records = []

print("=" * 118)
print("SELECTOR TRUE-FOCUS RANK AUDIT")
print("=" * 118)

for rd in range(1, 9):

    diag = read_jsonl(
        ROOT /
        f"selector_diag_dev_round_{rd:02d}.jsonl"
    )

    before = {
        str(x["session_id"]): x
        for x in read_jsonl(
            ROOT /
            f"states_dev_round_{rd-1:02d}.jsonl"
        )
    }

    after = {
        str(x["session_id"]): x
        for x in read_jsonl(
            ROOT /
            f"states_dev_round_{rd:02d}.jsonl"
        )
    }

    groups = {}

    for x in diag:
        groups.setdefault(
            str(x["session_id"]),
            []
        ).append(x)

    print()
    print("#" * 118)
    print(f"ROUND {rd}")
    print("#" * 118)

    for sid in sorted(groups):

        rows = groups[sid]
        tr = truth[sid]

        rows = sorted(
            rows,
            key=lambda x: int(x["selector_rank"])
        )

        parsed = []

        for x in rows:
            focus = get_focus(x)

            parsed.append({
                "rank": int(x["selector_rank"]),
                "score": float(x["selector_score"]),
                "candidate_index": int(x["candidate_index"]),
                "focus": focus,
                "is_true_focus": focus in tr,
                "selected": bool(x["selector_selected"]),
                "query": str(x.get("query", "")),
            })

        selected = [
            x for x in parsed
            if x["selected"]
        ]

        if len(selected) != 1:
            raise RuntimeError(
                f"{sid} R{rd}: expected exactly one selected, "
                f"got {len(selected)}"
            )

        selected = selected[0]

        true_candidates = [
            x for x in parsed
            if x["is_true_focus"]
        ]

        auc0 = auc_state(
            before[sid],
            tr
        )

        auc1 = auc_state(
            after[sid],
            tr
        )

        delta = auc1 - auc0

        print()
        print("-" * 118)
        print(f"{sid}")
        print(f"TRUE ROLES       : {sorted(tr)}")
        print(
            f"SELECTED         : rank={selected['rank']} "
            f"cand={selected['candidate_index']:02d} "
            f"focus={selected['focus']} "
            f"score={selected['score']:+.6f} "
            f"{'<<< TRUE-FOCUS' if selected['is_true_focus'] else '<<< FALSE-FOCUS'}"
        )

        if true_candidates:

            best_true = min(
                true_candidates,
                key=lambda x: x["rank"]
            )

            gap = (
                selected["score"]
                - best_true["score"]
            )

            print(
                f"BEST TRUE-FOCUS  : rank={best_true['rank']} "
                f"cand={best_true['candidate_index']:02d} "
                f"focus={best_true['focus']} "
                f"score={best_true['score']:+.6f}"
            )

            print(
                f"SCORE GAP         : "
                f"selected - best_true = {gap:+.6f}"
            )

        else:
            best_true = None
            gap = None
            print(
                "BEST TRUE-FOCUS  : NONE IN POOL"
            )

        print(
            f"AUC              : "
            f"{auc0:.4f} -> {auc1:.4f} "
            f"({delta:+.4f})"
        )

        print()
        print("SELECTOR RANKING:")

        for x in parsed:

            marks = []

            if x["selected"]:
                marks.append("SELECTED")

            if x["is_true_focus"]:
                marks.append("TRUE-FOCUS")

            marker = (
                " [" + ",".join(marks) + "]"
                if marks
                else ""
            )

            print(
                f"  #{x['rank']:02d} "
                f"score={x['score']:+.6f} "
                f"cand={x['candidate_index']:02d} "
                f"focus={str(x['focus']):<28}"
                f"{marker}"
            )

        records.append({
            "round": rd,
            "session_id": sid,
            "n_valid": len(parsed),
            "has_true": bool(true_candidates),
            "n_true": len(true_candidates),
            "selected_true": selected["is_true_focus"],
            "selected_score": selected["score"],
            "best_true_rank":
                best_true["rank"]
                if best_true else None,
            "best_true_score":
                best_true["score"]
                if best_true else None,
            "score_gap": gap,
            "auc_before": auc0,
            "auc_after": auc1,
            "auc_delta": delta,
        })


opps = [
    x for x in records
    if x["has_true"]
]

no_opps = [
    x for x in records
    if not x["has_true"]
]

picked_true = [
    x for x in opps
    if x["selected_true"]
]

missed_true = [
    x for x in opps
    if not x["selected_true"]
]

ranks = [
    x["best_true_rank"]
    for x in opps
]

gaps = [
    x["score_gap"]
    for x in missed_true
    if x["score_gap"] is not None
]

print()
print("=" * 118)
print("GLOBAL SUMMARY")
print("=" * 118)

print(
    f"Total decisions                         = "
    f"{len(records)}"
)

print(
    f"Rounds with >=1 true-focus candidate    = "
    f"{len(opps)}/{len(records)}"
)

print(
    f"Rounds with NO true-focus candidate     = "
    f"{len(no_opps)}/{len(records)}"
)

print(
    f"Selector picked true-focus when possible= "
    f"{len(picked_true)}/{len(opps)} "
    f"({100*len(picked_true)/len(opps):.1f}%)"
)

print(
    f"Selector picked true-focus overall      = "
    f"{sum(x['selected_true'] for x in records)}/{len(records)}"
)

print()
print("TRUE-FOCUS RANK DISTRIBUTION")

print(
    f"Mean best true-focus rank               = "
    f"{statistics.mean(ranks):.3f}"
)

print(
    f"Median best true-focus rank             = "
    f"{statistics.median(ranks):.3f}"
)

for k in [1, 2, 3, 4]:
    n = sum(r <= k for r in ranks)

    print(
        f"Best true-focus in Top-{k:<2}             = "
        f"{n}/{len(ranks)} "
        f"({100*n/len(ranks):.1f}%)"
    )

bottom = sum(
    x["best_true_rank"] > (x["n_valid"] / 2)
    for x in opps
)

print(
    f"Best true-focus in bottom half          = "
    f"{bottom}/{len(opps)} "
    f"({100*bottom/len(opps):.1f}%)"
)

print()
print("SCORE SEPARATION")

if gaps:
    print(
        f"Mean selected-minus-best-true gap       = "
        f"{statistics.mean(gaps):+.6f}"
    )

    print(
        f"Median selected-minus-best-true gap     = "
        f"{statistics.median(gaps):+.6f}"
    )

print()
print("AUC EFFECT")

true_deltas = [
    x["auc_delta"]
    for x in records
    if x["selected_true"]
]

false_deltas = [
    x["auc_delta"]
    for x in records
    if not x["selected_true"]
]

if true_deltas:
    print(
        f"Mean ΔAUC when true-focus selected      = "
        f"{statistics.mean(true_deltas):+.4f} "
        f"(n={len(true_deltas)})"
    )

if false_deltas:
    print(
        f"Mean ΔAUC when false-focus selected     = "
        f"{statistics.mean(false_deltas):+.4f} "
        f"(n={len(false_deltas)})"
    )

print()
print("=" * 118)
