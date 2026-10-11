from __future__ import annotations

import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from v11.role_universe import TRAIN_ROLES

ROOT = Path("artifacts/v14_genericact_selectorV2_G4_simple5_v1")


def jl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}


def auc(st, tr):
    p = st["role_probs"]
    y = [int(r in tr) for r in TRAIN_ROLES]
    s = [float(p.get(r, 0.0)) for r in TRAIN_ROLES]
    return float(roc_auc_score(y, s))


def rank(d, n=7):
    return sorted(
        [(str(k), float(v)) for k,v in d.items()],
        key=lambda x: -x[1]
    )[:n]


def fmt(xs, tr=None):
    z = []
    for r,p in xs:
        star = "*" if tr and r in tr else ""
        z.append(f"{r}={p:+.3f}{star}")
    return " | ".join(z)


states = {}

for r in range(0, 9):
    rows = jl(ROOT / f"states_dev_round_{r:02d}.jsonl")
    if rows:
        states[r] = {
            str(x["session_id"]): x
            for x in rows
        }


for r in range(1, 9):

    if r not in states or r-1 not in states:
        continue

    selected = {
        str(x["session_id"]): x
        for x in jl(
            ROOT / f"candidates_dev_round_{r:02d}.jsonl"
        )
    }

    diag = {}
    for x in jl(
        ROOT / f"selector_diag_dev_round_{r:02d}.jsonl"
    ):
        diag.setdefault(
            str(x["session_id"]), []
        ).append(x)

    scored = {
        str(x["system_id"]): x
        for x in jl(
            ROOT / f"scored_dev_round_{r:02d}.jsonl"
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

    for sid,tr in truth.items():

        if sid not in states[r-1] or sid not in states[r]:
            continue

        before = states[r-1][sid]
        after = states[r][sid]

        a0 = auc(before, tr)
        a1 = auc(after, tr)

        if a1 >= a0:
            continue

        print()
        print("=" * 130)
        print(
            f"{sid} | ROUND {r} | "
            f"AUC {a0:.4f} -> {a1:.4f} "
            f"({a1-a0:+.4f})"
        )
        print("TRUE:", sorted(tr))
        print("=" * 130)

        print()
        print(
            "BELIEF BEFORE:",
            fmt(rank(before["role_probs"]), tr)
        )

        print()
        print("CANDIDATES:")

        xs = sorted(
            diag.get(sid, []),
            key=lambda x: int(
                x.get("selector_rank", 999)
            )
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
                f"{mark} rank={x.get('selector_rank')} "
                f"score={score:>8} | {q}"
            )

        c = selected.get(sid, {})
        print()
        print(
            "SELECTED:",
            " ".join(
                str(c.get("query", "")).split()
            )
        )

        rr = responses.get(sid, {})
        ans = " ".join(
            str(
                rr.get(
                    "final_output",
                    rr.get("output", "")
                )
            ).split()
        )

        print()
        print(
            "RESPONSE:",
            ans[:500]
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
                    sc.get("metadata", {}).get(
                        "evidence_v134", {}
                    )
                ).items()
            }

            print()
            print(
                "V13.5 CAUSAL:",
                fmt(rank(v135), tr)
            )

            print(
                "V13.4 PRESENCE:",
                fmt(rank(v134), tr)
            )

        print()
        print(
            "BELIEF AFTER :",
            fmt(rank(after["role_probs"]), tr)
        )

print()
print("=" * 130)
print("DONE")
print("=" * 130)
