#!/usr/bin/env python3

import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from adaptive_v11_active.io import stable_int
from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/v14_cleanR7_selectorV2_simple5_v1"
)


def read_jsonl(p):
    if not p.exists():
        return []

    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x["split"] == "dev"
}


def auc_for_state(st, true_roles):
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

    return float(
        roc_auc_score(y, s)
    )


def focus_role(state, candidate_index):
    """
    Reconstruct EXACT V14 target_role from
    v14/adaptive_belief_policy.py +
    scripts/v14_adaptive_generate_round_v1.py
    """

    sid = str(state["session_id"])

    rp = {
        str(k): float(v)
        for k, v in state["role_probs"].items()
    }

    roles = [
        r
        for r in TRAIN_ROLES
        if r in rp
    ]

    uncertain = sorted(
        roles,
        key=lambda r: (
            abs(rp[r] - 0.5),
            -rp[r],
            r,
        ),
    )

    offset = (
        stable_int("v14a_focus", sid)
        % max(1, len(rp))
    )

    policy_candidate_index = (
        int(candidate_index) + offset
    )

    return uncertain[
        (policy_candidate_index * 7)
        % len(uncertain)
    ]


print("=" * 120)
print("SIMPLE-5 FAILURE AUDIT")
print("=" * 120)

summary = {}

for rd in range(1, 9):

    states_before = {
        str(x["session_id"]): x
        for x in read_jsonl(
            ROOT /
            f"states_dev_round_{rd-1:02d}.jsonl"
        )
    }

    states_after = {
        str(x["session_id"]): x
        for x in read_jsonl(
            ROOT /
            f"states_dev_round_{rd:02d}.jsonl"
        )
    }

    pool = read_jsonl(
        ROOT /
        f"candidate_pool_dev_round_{rd:02d}.jsonl"
    )

    selected = read_jsonl(
        ROOT /
        f"candidates_dev_round_{rd:02d}.jsonl"
    )

    responses = read_jsonl(
        ROOT /
        f"responses_dev_round_{rd:02d}.jsonl"
    )

    pool_by = {}

    for x in pool:
        pool_by.setdefault(
            str(x["session_id"]),
            []
        ).append(x)

    sel_by = {
        str(x["session_id"]): x
        for x in selected
    }

    resp_by = {
        str(x["session_id"]): x
        for x in responses
    }

    print()
    print("#" * 120)
    print(f"ROUND {rd}")
    print("#" * 120)

    for sid in sorted(states_before):

        st0 = states_before[sid]
        st1 = states_after.get(sid)

        tr = truth[sid]

        xs = sorted(
            pool_by.get(sid, []),
            key=lambda z:
                int(z["candidate_index"])
        )

        candidate_focus = []

        for x in xs:
            fr = focus_role(
                st0,
                int(x["candidate_index"])
            )

            candidate_focus.append(
                (
                    int(x["candidate_index"]),
                    fr,
                    bool(x.get("valid")),
                    str(x.get("query", "")),
                )
            )

        available_true = [
            fr
            for _, fr, valid, _ in candidate_focus
            if valid and fr in tr
        ]

        sel = sel_by.get(sid)

        if sel is None:
            print(
                f"\n{sid}: NO SELECTED CANDIDATE"
            )
            continue

        ci = int(sel["candidate_index"])

        selected_focus = focus_role(
            st0,
            ci
        )

        resp = resp_by.get(sid, {})

        activated = set(
            map(
                str,
                resp.get(
                    "selected_roles_training_only",
                    []
                )
            )
        )

        auc0 = auc_for_state(
            st0,
            tr
        )

        auc1 = (
            auc_for_state(st1, tr)
            if st1 is not None
            else float("nan")
        )

        print()
        print("-" * 120)
        print(sid)
        print(
            "TRUE ROLES          :",
            sorted(tr)
        )
        print(
            "TRUE FOCUS IN POOL  :",
            available_true,
            f"({len(available_true)}/8)"
        )
        print(
            "SELECTED CANDIDATE  :",
            ci
        )
        print(
            "SELECTED FOCUS      :",
            selected_focus,
            "<<< TRUE"
            if selected_focus in tr
            else "<<< FALSE"
        )
        print(
            "ACTIVATED EVAL-ONLY :",
            sorted(activated)
        )
        print(
            f"AUC                 : "
            f"{auc0:.4f} -> {auc1:.4f} "
            f"({auc1-auc0:+.4f})"
        )
        print(
            "SELECTED QUERY      :",
            str(sel.get("query", ""))
        )

        print()
        print("POOL:")

        for idx, fr, valid, q in candidate_focus:

            marks = []

            if fr in tr:
                marks.append("TRUE-FOCUS")

            if idx == ci:
                marks.append("SELECTED")

            tag = (
                " [" + ",".join(marks) + "]"
                if marks
                else ""
            )

            print(
                f"  cand={idx:02d} "
                f"focus={fr:<28} "
                f"valid={str(valid):<5}"
                f"{tag}"
            )

        ss = summary.setdefault(
            sid,
            {
                "rounds": 0,
                "pool_true": 0,
                "selected_true": 0,
                "activated_true": 0,
                "auc_start": None,
                "auc_end": None,
            },
        )

        ss["rounds"] += 1
        ss["pool_true"] += len(
            set(available_true)
        )

        ss["selected_true"] += int(
            selected_focus in tr
        )

        ss["activated_true"] += len(
            activated & tr
        )

        if ss["auc_start"] is None:
            ss["auc_start"] = auc0

        ss["auc_end"] = auc1


print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

for sid, x in sorted(summary.items()):

    print()
    print(sid)
    print(
        "  true-role appearances in pools =",
        x["pool_true"],
    )
    print(
        "  selector chose true-focus      =",
        f'{x["selected_true"]}/{x["rounds"]}',
    )
    print(
        "  true activations across rounds =",
        x["activated_true"],
    )
    print(
        "  AUC                            =",
        f'{x["auc_start"]:.4f}'
        f' -> '
        f'{x["auc_end"]:.4f}',
    )
