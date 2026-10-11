#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/v14_genericact_simple5_v1")
MAX_ROUND = 8
REFRESH = 20


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
}


def auc_for(role_probs, true_roles):
    y = np.asarray(
        [int(r in true_roles) for r in TRAIN_ROLES],
        dtype=int,
    )

    p = np.asarray(
        [float(role_probs[r]) for r in TRAIN_ROLES],
        dtype=float,
    )

    return float(roc_auc_score(y, p))


def completed_rounds():
    out = []

    for r in range(0, MAX_ROUND + 1):
        p = ROOT / f"states_dev_round_{r:02d}.jsonl"

        if p.exists() and p.stat().st_size > 0:
            out.append(r)

    return out


def round_summary(r):
    states = read_jsonl(
        ROOT / f"states_dev_round_{r:02d}.jsonl"
    )

    vals = []

    for s in states:
        sid = str(s["session_id"])
        vals.append(
            auc_for(
                s["role_probs"],
                truth[sid],
            )
        )

    return float(np.mean(vals))


def print_system(s):
    sid = str(s["session_id"])
    probs = {
        str(k): float(v)
        for k, v in s["role_probs"].items()
    }

    true_roles = truth[sid]

    ranking = sorted(
        probs.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    auc = auc_for(
        probs,
        true_roles,
    )

    top3 = [
        role
        for role, _ in ranking[:3]
    ]

    exact = set(top3) == true_roles

    print()
    print("=" * 92)
    print(sid)
    print("=" * 92)

    print(
        "TRUE ROLES :",
        ", ".join(sorted(true_roles))
    )

    print(
        f"AUC        : {auc:.4f}"
    )

    print(
        "TOP-3      :",
        ", ".join(top3)
    )

    print(
        "TOP3 EXACT :",
        exact
    )

    print()
    print(
        f"{'RANK':>4}  "
        f"{'ROLE':<30} "
        f"{'P(INSTALLED)':>14}  "
        f"{'TRUTH':<8}"
    )

    print("-" * 66)

    for rank, (role, p) in enumerate(
        ranking,
        1,
    ):
        marker = "✓ TRUE" if role in true_roles else ""

        print(
            f"{rank:4d}  "
            f"{role:<30} "
            f"{p:14.6f}  "
            f"{marker:<8}"
        )


def render():
    rounds = completed_rounds()

    print(
        "SIMPLE-5 — LIVE INSTALLED-ROLE BELIEF"
    )
    print("=" * 92)

    if not rounds:
        print("No state files yet.")
        return

    print()
    print("ROUND PROGRESSION")
    print("-" * 40)

    previous = None

    for r in rounds:
        auc = round_summary(r)

        if previous is None:
            delta = ""
        else:
            delta = f"  Δ={auc-previous:+.4f}"

        print(
            f"Round {r:>2}: "
            f"mean AUC = {auc:.4f}"
            f"{delta}"
        )

        previous = auc

    latest = max(rounds)

    print()
    print("#" * 92)
    print(
        f"LATEST COMPLETED ROUND = {latest}"
    )
    print("#" * 92)

    states = read_jsonl(
        ROOT /
        f"states_dev_round_{latest:02d}.jsonl"
    )

    states = sorted(
        states,
        key=lambda x: str(x["session_id"]),
    )

    for s in states:
        print_system(s)

    print()
    print("=" * 92)
    print(
        f"LATEST MEAN AUC = "
        f"{round_summary(latest):.4f}"
    )
    print(
        "HISTORICAL SIMPLE-5 = 0.8471"
    )
    print(
        f"DELTA VS HISTORICAL = "
        f"{round_summary(latest)-0.8471:+.4f}"
    )
    print("=" * 92)


while True:
    os.system("clear")
    render()

    print()
    print(
        f"Refreshing every {REFRESH}s "
        f"(Ctrl+C to stop)"
    )

    time.sleep(REFRESH)
