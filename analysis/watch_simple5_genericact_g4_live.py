#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path

from sklearn.metrics import roc_auc_score
from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/v14_genericact_selectorV2_G4_simple5_v1"
)

REFRESH = 60
HISTORICAL_FINAL = 0.8471


def read_jsonl(path):
    if not path.exists():
        return []

    out = []

    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))

    return out


def latest_state_round():
    found = []

    for p in ROOT.glob(
        "states_dev_round_*.jsonl"
    ):
        try:
            r = int(
                p.stem.split("_")[-1]
            )

            if p.stat().st_size:
                found.append((r, p))
        except Exception:
            pass

    if not found:
        return None, None

    return max(found)


def count_file(name):
    p = ROOT / name

    if not p.exists() or not p.stat().st_size:
        return 0

    return sum(
        1
        for x in p.open()
        if x.strip()
    )


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x["split"] == "dev"
}


while True:

    os.system("clear")

    print("=" * 100)
    print("SIMPLE-5 ADAPTIVE ATTACK — LIVE BELIEF MONITOR")
    print("=" * 100)

    r, state_path = latest_state_round()

    if state_path is None:
        print("\nNo belief state found yet.")
        time.sleep(REFRESH)
        continue

    states = read_jsonl(state_path)

    # -------------------------------------------------
    # Current pipeline progress for the NEXT round.
    # If state is round r, then r+1 is the next attack.
    # -------------------------------------------------

    next_r = r + 1

    cand_n = count_file(
        f"candidates_dev_round_{next_r:02d}.jsonl"
    )

    resp_n = count_file(
        f"responses_dev_round_{next_r:02d}.jsonl"
    )

    scored_n = count_file(
        f"scored_dev_round_{next_r:02d}.jsonl"
    )

    print()
    print(
        f"CURRENT COMPLETED BELIEF ROUND : {r}"
    )

    if r < 8:
        print(
            f"NEXT ROUND PROGRESS            : R{next_r}"
        )
        print(
            f"  selected candidates          : {cand_n}/5"
        )
        print(
            f"  MAS responses                : {resp_n}/5"
        )
        print(
            f"  Sensor scored                : {scored_n}/5"
        )

    print()

    system_aucs = []

    # -----------------------------------------------
    # Compute current AUC from current role beliefs.
    # Truth is evaluation-only here.
    # -----------------------------------------------

    for st in states:

        sid = str(st["session_id"])

        rp = {
            str(k): float(v)
            for k, v in st["role_probs"].items()
        }

        y = [
            int(role in truth[sid])
            for role in TRAIN_ROLES
        ]

        s = [
            rp.get(role, 0.0)
            for role in TRAIN_ROLES
        ]

        auc = float(
            roc_auc_score(y, s)
        )

        system_aucs.append(auc)

    mean_auc = (
        sum(system_aucs) / len(system_aucs)
        if system_aucs
        else float("nan")
    )

    print("=" * 100)
    print(
        f"CURRENT MEAN ROLE AUC = {mean_auc:.4f}"
    )

    print(
        f"HISTORICAL FINAL R8   = {HISTORICAL_FINAL:.4f}"
    )

    print("=" * 100)

    # -----------------------------------------------
    # Full role ranking for every MAS
    # -----------------------------------------------

    for st in sorted(
        states,
        key=lambda x: str(x["session_id"])
    ):

        sid = str(st["session_id"])

        rp = {
            str(k): float(v)
            for k, v in st["role_probs"].items()
        }

        ranked = sorted(
            TRAIN_ROLES,
            key=lambda role:
                (-rp.get(role, 0.0), role)
        )

        y = [
            int(role in truth[sid])
            for role in TRAIN_ROLES
        ]

        s = [
            rp.get(role, 0.0)
            for role in TRAIN_ROLES
        ]

        auc = float(
            roc_auc_score(y, s)
        )

        print()
        print("-" * 100)
        print(
            f"{sid}"
            f"    AUC={auc:.4f}"
            f"    TRUE={sorted(truth[sid])}"
        )
        print("-" * 100)

        print(
            f"{'RANK':>4}  "
            f"{'ROLE':<30} "
            f"{'PRESENT BELIEF':>15}  "
            f"{'TRUTH':>7}"
        )

        for rank, role in enumerate(
            ranked,
            start=1
        ):
            p = rp.get(role, 0.0)

            marker = (
                "<<< YES"
                if role in truth[sid]
                else ""
            )

            print(
                f"{rank:4d}  "
                f"{role:<30} "
                f"{p:15.4f}  "
                f"{marker:>7}"
            )

    print()
    print("=" * 100)
    print(
        f"Refresh every {REFRESH}s | "
        "Ctrl+C to stop"
    )
    print("=" * 100)

    time.sleep(REFRESH)
