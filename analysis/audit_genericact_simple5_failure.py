from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from v11.role_universe import TRAIN_ROLES


TEST = Path(
    "artifacts/v14_genericact_selectorV2_G4_simple5_v1"
)

CTRL = Path(
    "artifacts/v14_cleanR7_selectorV2_simple5_v1"
)


def read_jsonl(p):
    if not p.exists():
        return []

    out = []

    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            out.append(
                json.loads(line)
            )
        except Exception:
            pass

    return out


def state_auc(st, truth):
    probs = {
        str(k): float(v)
        for k,v in st["role_probs"].items()
    }

    y = [
        int(r in truth)
        for r in TRAIN_ROLES
    ]

    s = [
        probs.get(r, 0.0)
        for r in TRAIN_ROLES
    ]

    return float(
        roc_auc_score(y, s)
    )


manifest = json.loads(
    (TEST / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}

sids = sorted(truth)


print("=" * 110)
print("GENERICACT SIMPLE5 FAILURE AUDIT")
print("=" * 110)


# ============================================================
# ROUND-LEVEL COMPARISON
# ============================================================

print()
print("ROUND-BY-ROUND MEAN AUC")
print("-" * 110)

print(
    f"{'ROUND':<8}"
    f"{'GENERICACT':>15}"
    f"{'CONTROL':>15}"
    f"{'DELTA':>15}"
)

for r in range(0,9):

    if r == 0:
        t = c = 0.5

    else:
        tp = TEST / f"advance_dev_round_{r:02d}_summary.json"
        cp = CTRL / f"advance_dev_round_{r:02d}_summary.json"

        if not tp.exists():
            break

        t = float(
            json.loads(
                tp.read_text()
            )["mean_role_auc"]
        )

        c = (
            float(
                json.loads(
                    cp.read_text()
                )["mean_role_auc"]
            )
            if cp.exists()
            else float("nan")
        )

    print(
        f"R{r:<7}"
        f"{t:>15.6f}"
        f"{c:>15.6f}"
        f"{t-c:>+15.6f}"
    )


# ============================================================
# PER-MAS TRAJECTORY + EXECUTED QUERIES
# ============================================================

print()
print("=" * 110)
print("PER-MAS TRAJECTORIES")
print("=" * 110)

states = {}

for r in range(0,9):
    p = TEST / f"states_dev_round_{r:02d}.jsonl"

    xs = read_jsonl(p)

    if xs:
        states[r] = {
            str(x["session_id"]): x
            for x in xs
        }


for sid in sids:

    print()
    print("#" * 110)
    print("SYSTEM:", sid)
    print("TRUE:", sorted(truth[sid]))
    print("#" * 110)

    prev_auc = None

    for r in sorted(states):

        st = states[r].get(sid)

        if not st:
            continue

        auc = state_auc(
            st,
            truth[sid],
        )

        delta = (
            auc - prev_auc
            if prev_auc is not None
            else 0.0
        )

        probs = {
            str(k): float(v)
            for k,v in st["role_probs"].items()
        }

        top = sorted(
            probs.items(),
            key=lambda z: -z[1],
        )[:6]

        top_txt = " | ".join(
            f"{role}={p:.3f}"
            + ("*" if role in truth[sid] else "")
            for role,p in top
        )

        print(
            f"R{r}: "
            f"AUC={auc:.4f} "
            f"dAUC={delta:+.4f} | "
            f"{top_txt}"
        )

        if r > 0:

            cp = (
                TEST /
                f"candidates_dev_round_{r:02d}.jsonl"
            )

            cs = [
                x for x in read_jsonl(cp)
                if str(x["session_id"]) == sid
            ]

            if cs:
                q = str(
                    cs[0].get(
                        "query",
                        ""
                    )
                )

                print(
                    "    EXECUTED:",
                    " ".join(q.split())
                )

            rp = (
                TEST /
                f"responses_dev_round_{r:02d}.jsonl"
            )

            rs = [
                x for x in read_jsonl(rp)
                if str(
                    x.get(
                        "system_id",
                        x.get(
                            "session_id",
                            ""
                        )
                    )
                ) == sid
            ]

            if rs:
                ans = str(
                    rs[-1].get(
                        "final_output",
                        ""
                    )
                )

                ans = " ".join(
                    ans.split()
                )

                print(
                    "    RESPONSE:",
                    ans[:240]
                )

        prev_auc = auc


# ============================================================
# SELECTOR CANDIDATE AUDIT
# ============================================================

print()
print("=" * 110)
print("GENERATOR + SELECTOR CANDIDATES")
print("=" * 110)

for r in range(1,9):

    p = (
        TEST /
        f"selector_diag_dev_round_{r:02d}.jsonl"
    )

    rows = read_jsonl(p)

    if not rows:
        continue

    print()
    print(
        f"================ ROUND {r} ================"
    )

    for sid in sids:

        xs = [
            x for x in rows
            if str(x["session_id"]) == sid
        ]

        if not xs:
            continue

        xs.sort(
            key=lambda x: (
                int(
                    x.get(
                        "selector_rank",
                        999
                    )
                ),
                int(
                    x.get(
                        "candidate_index",
                        999
                    )
                )
            )
        )

        print()
        print(sid)

        for x in xs:

            q = " ".join(
                str(
                    x.get(
                        "query",
                        ""
                    )
                ).split()
            )

            selected = bool(
                x.get(
                    "selector_selected",
                    False
                )
            )

            score = x.get(
                "selector_score"
            )

            try:
                score_txt = (
                    f"{float(score):+.4f}"
                )
            except Exception:
                score_txt = str(score)

            print(
                f"{'>>>' if selected else '   '} "
                f"rank={x.get('selector_rank')} "
                f"score={score_txt:>8} | "
                f"{q}"
            )


# ============================================================
# ROUNDS THAT HURT
# ============================================================

print()
print("=" * 110)
print("NEGATIVE BELIEF UPDATES")
print("=" * 110)

bad = []

for sid in sids:

    prev = None

    for r in sorted(states):

        st = states[r].get(sid)

        if not st:
            continue

        auc = state_auc(
            st,
            truth[sid],
        )

        if (
            prev is not None
            and auc < prev
        ):
            bad.append(
                (
                    auc-prev,
                    sid,
                    r,
                    prev,
                    auc,
                )
            )

        prev = auc


for delta,sid,r,before,after in sorted(bad):

    print(
        f"{sid} R{r}: "
        f"{before:.4f} -> {after:.4f} "
        f"delta={delta:+.4f}"
    )

    cp = (
        TEST /
        f"candidates_dev_round_{r:02d}.jsonl"
    )

    cs = [
        x for x in read_jsonl(cp)
        if str(x["session_id"]) == sid
    ]

    if cs:
        print(
            "   QUERY:",
            " ".join(
                str(
                    cs[0].get(
                        "query",
                        ""
                    )
                ).split()
            )
        )


print()
print("=" * 110)
print("END")
print("=" * 110)
