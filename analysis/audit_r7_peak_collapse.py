from __future__ import annotations

import json
from pathlib import Path

from v11.role_universe import TRAIN_ROLES

ROOT = Path(
    "artifacts/v14_R7_selectorV2_G4_simple5_retest_v1"
)


def jl(p):
    if not p.exists():
        return []

    out = []

    for line in p.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue

        try:
            out.append(json.loads(line))
        except Exception:
            pass

    return out


def auc_pairwise(probs, truth):
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

    wins = 0.0

    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5

    return wins / (len(pos) * len(neg))


def rank_roles(probs):
    return sorted(
        TRAIN_ROLES,
        key=lambda r: (
            -float(probs.get(r, 0.0)),
            r,
        ),
    )


def fmt_top(probs, truth, n=10):
    ranked = rank_roles(probs)

    return " | ".join(
        f"{i+1}:{r}={float(probs.get(r,0)):.3f}"
        + ("*" if r in truth else "")
        for i,r in enumerate(ranked[:n])
    )


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}

states = {}

for r in range(0, 9):
    p = ROOT / f"states_dev_round_{r:02d}.jsonl"

    rows = jl(p)

    if rows:
        states[r] = {
            str(x["session_id"]): x
            for x in rows
        }


print("=" * 125)
print("R7 SIMPLE5 — PEAK / COLLAPSE AUDIT")
print("=" * 125)

print()
print("PER-SYSTEM AUC TRAJECTORY")
print("-" * 125)

header = (
    f"{'SYSTEM':38s}"
    + "".join(f"{'R'+str(r):>9s}" for r in range(0,9))
)

print(header)

system_auc = {}

for sid in sorted(truth):

    vals = {}

    for r in sorted(states):

        if sid not in states[r]:
            continue

        vals[r] = auc_pairwise(
            states[r][sid]["role_probs"],
            truth[sid],
        )

    system_auc[sid] = vals

    print(
        f"{sid:38s}"
        + "".join(
            (
                f"{vals[r]:9.4f}"
                if r in vals
                else f"{'--':>9s}"
            )
            for r in range(0,9)
        )
    )


print()
print("=" * 125)
print("WHO CAUSED R5 -> R6 -> R7 -> R8?")
print("=" * 125)

for sid in sorted(truth):

    v = system_auc[sid]

    if not all(r in v for r in [5,6,7,8]):
        continue

    print(
        f"{sid:38s} "
        f"R5→R6={v[6]-v[5]:+.4f}   "
        f"R6→R7={v[7]-v[6]:+.4f}   "
        f"R7→R8={v[8]-v[7]:+.4f}"
    )


for r in [6,7,8]:

    print()
    print("#" * 125)
    print(f"ROUND {r}")
    print("#" * 125)

    candidates = {}

    for x in jl(
        ROOT / f"selector_diag_dev_round_{r:02d}.jsonl"
    ):
        candidates.setdefault(
            str(x["session_id"]),
            []
        ).append(x)

    selected = {
        str(x["session_id"]): x
        for x in jl(
            ROOT / f"candidates_dev_round_{r:02d}.jsonl"
        )
    }

    responses = {
        str(
            x.get(
                "system_id",
                x.get("session_id", "")
            )
        ): x
        for x in jl(
            ROOT / f"responses_dev_round_{r:02d}.jsonl"
        )
    }

    scored = {
        str(x["system_id"]): x
        for x in jl(
            ROOT / f"scored_dev_round_{r:02d}.jsonl"
        )
    }

    for sid in sorted(truth):

        if (
            r not in states
            or r-1 not in states
            or sid not in states[r]
            or sid not in states[r-1]
        ):
            continue

        tr = truth[sid]

        before = states[r-1][sid]
        after = states[r][sid]

        a0 = auc_pairwise(
            before["role_probs"],
            tr,
        )

        a1 = auc_pairwise(
            after["role_probs"],
            tr,
        )

        print()
        print("=" * 125)
        print(
            f"{sid} | "
            f"AUC {a0:.4f} -> {a1:.4f} "
            f"delta={a1-a0:+.4f}"
        )
        print(
            "TRUE:",
            ", ".join(sorted(tr))
        )

        print(
            "BEFORE:",
            fmt_top(
                before["role_probs"],
                tr,
            )
        )

        print()
        print("CANDIDATES:")

        xs = sorted(
            candidates.get(sid, []),
            key=lambda x: int(
                x.get(
                    "selector_rank",
                    999,
                )
            ),
        )

        for x in xs:

            mark = (
                ">>>"
                if x.get("selector_selected")
                else "   "
            )

            score = x.get("selector_score")

            try:
                score = f"{float(score):+.4f}"
            except Exception:
                score = str(score)

            q = " ".join(
                str(x.get("query", "")).split()
            )

            print(
                f"{mark} "
                f"rank={x.get('selector_rank')} "
                f"score={score:>8} | "
                f"{q}"
            )

        sel = selected.get(sid, {})

        print()
        print(
            "EXECUTED:",
            " ".join(
                str(
                    sel.get("query", "")
                ).split()
            )
        )

        rr = responses.get(sid, {})

        answer = " ".join(
            str(
                rr.get(
                    "final_output",
                    rr.get("output", "")
                )
            ).split()
        )

        print(
            "RESPONSE:",
            answer[:450]
        )

        sc = scored.get(sid)

        if sc:

            v135 = {
                str(k): float(v)
                for k,v in dict(
                    sc.get("evidence", {})
                ).items()
            }

            v134 = {
                str(k): float(v)
                for k,v in dict(
                    sc.get(
                        "metadata",
                        {}
                    ).get(
                        "evidence_v134",
                        {},
                    )
                ).items()
            }

            def evfmt(d):
                order = sorted(
                    d,
                    key=lambda x: -d[x],
                )[:8]

                return " | ".join(
                    f"{x}={d[x]:+.3f}"
                    + ("*" if x in tr else "")
                    for x in order
                )

            print(
                "V13.5:",
                evfmt(v135)
            )

            print(
                "V13.4:",
                evfmt(v134)
            )

        print(
            "AFTER :",
            fmt_top(
                after["role_probs"],
                tr,
            )
        )


print()
print("=" * 125)
print("DONE")
print("=" * 125)
