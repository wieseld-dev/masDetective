from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("artifacts/v14_genericact_selectorV2_G4_simple5_v1")


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def short(s, n=110):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


manifest_p = ROOT / "manifest_privileged.json"

if not manifest_p.exists():
    print("Waiting for experiment root...")
    raise SystemExit

manifest = json.loads(manifest_p.read_text())

systems = [
    x for x in manifest["systems"]
    if x.get("split") == "dev"
]

truth = {
    str(x["session_id"]): list(map(str, x["roles"]))
    for x in systems
}

sids = sorted(truth)

# ------------------------------------------------------------
# Latest available state per system
# ------------------------------------------------------------

state_by_round = {}

for r in range(0, 9):
    p = ROOT / f"states_dev_round_{r:02d}.jsonl"
    xs = read_jsonl(p)
    if xs:
        state_by_round[r] = {
            str(x["session_id"]): x
            for x in xs
        }

latest_state_round = max(state_by_round, default=0)

# ------------------------------------------------------------
# Queries / selector diagnostics
# ------------------------------------------------------------

selected_by_round = {}
diag_by_round = {}

for r in range(1, 9):

    # final selected query sent to MAS
    p = ROOT / f"candidates_dev_round_{r:02d}.jsonl"
    xs = read_jsonl(p)

    if xs:
        selected_by_round[r] = {
            str(x["session_id"]): x
            for x in xs
        }

    # all G=4 candidates + selector scores
    dp = ROOT / f"selector_diag_dev_round_{r:02d}.jsonl"
    ds = read_jsonl(dp)

    if ds:
        tmp = {}
        for x in ds:
            sid = str(x["session_id"])
            tmp.setdefault(sid, []).append(x)

        for sid in tmp:
            tmp[sid].sort(
                key=lambda z: (
                    int(z.get("selector_rank", 999)),
                    int(z.get("candidate_index", 999)),
                )
            )

        diag_by_round[r] = tmp


print("=" * 118)
print("GENERICACT + SELECTOR V2 + FROZEN SENSOR — SIMPLE5 LIVE")
print("=" * 118)
print(
    f"Latest completed belief state: R{latest_state_round} / 8"
)
print()


# ------------------------------------------------------------
# Global AUC progression
# ------------------------------------------------------------

print("AUC PROGRESSION")
print("-" * 118)

vals = ["R0=0.5000"]

for r in range(1, 9):
    p = ROOT / f"advance_dev_round_{r:02d}_summary.json"

    if p.exists():
        try:
            x = json.loads(p.read_text())
            vals.append(
                f"R{r}={float(x['mean_role_auc']):.4f}"
            )
        except Exception:
            vals.append(f"R{r}=?")
    else:
        vals.append(f"R{r}=...")

print("   ".join(vals))
print()


# ------------------------------------------------------------
# Each MAS
# ------------------------------------------------------------

for sid in sids:

    print("=" * 118)
    print(sid)
    print("TRUE [evaluation only]:", ", ".join(truth[sid]))
    print("=" * 118)

    # ---- belief trajectory
    print("BELIEF TRAJECTORY — TOP 7")
    print()

    for r in sorted(state_by_round):

        st = state_by_round[r].get(sid)

        if not st:
            continue

        probs = {
            str(k): float(v)
            for k, v in st.get("role_probs", {}).items()
        }

        ranked = sorted(
            probs.items(),
            key=lambda z: (-z[1], z[0]),
        )[:7]

        txt = " | ".join(
            f"{role}={p:.3f}"
            + ("*" if role in truth[sid] else "")
            for role, p in ranked
        )

        print(f"R{r}: {txt}")

    print()
    print("LATEST / CURRENT PROBES")
    print()

    # Show last generated round and one previous round for context.
    available = [
        r for r in range(1, 9)
        if sid in selected_by_round.get(r, {})
        or sid in diag_by_round.get(r, {})
    ]

    for r in available[-2:]:

        print(f"--- ROUND {r} ---")

        diags = diag_by_round.get(r, {}).get(sid, [])

        if diags:
            print("Generator candidates:")

            for x in diags:
                q = str(x.get("query", ""))
                rank = x.get("selector_rank", "?")
                score = x.get("selector_score", None)
                chosen = bool(x.get("selector_selected", False))

                mark = ">>> SELECTED" if chosen else "   "

                if score is None:
                    score_txt = "?"
                else:
                    try:
                        score_txt = f"{float(score):+.4f}"
                    except Exception:
                        score_txt = str(score)

                print(
                    f" {mark:12s} "
                    f"rank={rank} "
                    f"score={score_txt:>8s} | "
                    f"{short(q)}"
                )

        sel = selected_by_round.get(r, {}).get(sid)

        if sel:
            print()
            print(
                "EXECUTED QUERY:",
                short(sel.get("query", ""), 160)
            )

        # response if already executed
        rp = ROOT / f"responses_dev_round_{r:02d}.jsonl"

        responses = read_jsonl(rp)

        match = [
            x for x in responses
            if str(x.get("system_id", x.get("session_id", ""))) == sid
        ]

        if match:
            rr = match[-1]
            print(
                "MAS RESPONSE:",
                short(rr.get("final_output", ""), 180)
            )

        print()

    if not available:
        print("Waiting for first generated round...")

    print()

print("=" * 118)

# ------------------------------------------------------------
# Pipeline state
# ------------------------------------------------------------

jobs = ROOT / "jobs.tsv"

if jobs.exists():
    print("PIPELINE JOB IDS")
    print(jobs.read_text().strip())

print("=" * 118)
