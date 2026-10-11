import json
import math
import re
from pathlib import Path

from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

ROLES = list(map(str, TRAIN_ROLES))
ROUNDS = range(1, 26)


def load(path):
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def sid(x):
    return str(
        x.get("session_id", x.get("system_id", ""))
    )


def target_of(row):
    p = str(row.get("prompt", ""))
    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        p
    )
    return m.group(1).strip() if m else None


def role_probs(row):
    p = row.get("role_probs")
    if isinstance(p, dict):
        return {
            str(k): float(v)
            for k, v in p.items()
        }

    for key in ["belief", "public_belief", "state"]:
        d = row.get(key)
        if isinstance(d, dict):
            p = d.get("role_probs")
            if isinstance(p, dict):
                return {
                    str(k): float(v)
                    for k, v in p.items()
                }

    raise RuntimeError(
        f"role_probs not found; keys={sorted(row.keys())}"
    )


def auc(scores, truth):
    truth = set(truth)

    pos = [
        scores[r]
        for r in ROLES
        if r in truth
    ]

    neg = [
        scores[r]
        for r in ROLES
        if r not in truth
    ]

    wins = 0.0

    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / (len(pos) * len(neg))


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}

systems = sorted(truth)


# These are ranking scores, so zero initialization is enough.
target_raw = {
    s: {r: 0.0 for r in ROLES}
    for s in systems
}

target_sign = {
    s: {r: 0.0 for r in ROLES}
    for s in systems
}

target_clip4 = {
    s: {r: 0.0 for r in ROLES}
    for s in systems
}

all_raw = {
    s: {r: 0.0 for r in ROLES}
    for s in systems
}

target_counts = {
    s: {r: 0 for r in ROLES}
    for s in systems
}


rows = []


for rnd in ROUNDS:

    candidates = {
        sid(x): x
        for x in load(
            ROOT / f"candidates_dev_round_{rnd:02d}.jsonl"
        )
    }

    scored = {
        sid(x): x
        for x in load(
            ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
        )
    }

    frozen_states = {
        sid(x): x
        for x in load(
            ROOT / f"states_dev_round_{rnd:02d}.jsonl"
        )
    }

    for s in systems:

        cand = candidates[s]
        sc = scored[s]

        target = target_of(cand)

        if target is None:
            raise RuntimeError(
                f"Missing target R{rnd} {s}"
            )

        evidence = {
            str(k): float(v)
            for k, v in dict(
                sc.get("evidence", {})
            ).items()
        }

        if target not in evidence:
            raise RuntimeError(
                f"Target {target} missing evidence "
                f"R{rnd} {s}"
            )

        e = float(evidence[target])

        # 1) User's proposed hypothesis-test accumulator:
        # only the role actually probed gets updated.
        target_raw[s][target] += e

        # Diagnostic: only direction, discard magnitude.
        target_sign[s][target] += (
            1.0 if e > 0
            else -1.0 if e < 0
            else 0.0
        )

        # Diagnostic robustness against extreme scores.
        target_clip4[s][target] += max(
            -4.0,
            min(4.0, e)
        )

        target_counts[s][target] += 1

        # 2) Alternative:
        # treat the complete V13.5 vector as evidence
        # on every role each round.
        for r in ROLES:
            all_raw[s][r] += float(
                evidence.get(r, 0.0)
            )


    frozen_aucs = []
    target_raw_aucs = []
    target_sign_aucs = []
    target_clip4_aucs = []
    all_raw_aucs = []

    for s in systems:

        frozen = role_probs(
            frozen_states[s]
        )

        frozen_aucs.append(
            auc(frozen, truth[s])
        )

        target_raw_aucs.append(
            auc(target_raw[s], truth[s])
        )

        target_sign_aucs.append(
            auc(target_sign[s], truth[s])
        )

        target_clip4_aucs.append(
            auc(target_clip4[s], truth[s])
        )

        all_raw_aucs.append(
            auc(all_raw[s], truth[s])
        )


    rec = {
        "round": rnd,
        "frozen": sum(frozen_aucs) / len(frozen_aucs),
        "target_raw": sum(target_raw_aucs) / len(target_raw_aucs),
        "target_sign": sum(target_sign_aucs) / len(target_sign_aucs),
        "target_clip4": sum(target_clip4_aucs) / len(target_clip4_aucs),
        "all_raw": sum(all_raw_aucs) / len(all_raw_aucs),
    }

    rows.append(rec)


print("=" * 110)
print("OFFLINE EVIDENCE-ACCUMULATOR REPLAY")
print("=" * 110)
print()
print(
    f"{'R':>3s} "
    f"{'FROZEN':>10s} "
    f"{'TARGET_RAW':>12s} "
    f"{'TARGET_SIGN':>12s} "
    f"{'TARGET_CLIP4':>13s} "
    f"{'ALL_RAW':>10s}"
)

for x in rows:
    print(
        f"{x['round']:3d} "
        f"{x['frozen']:10.6f} "
        f"{x['target_raw']:12.6f} "
        f"{x['target_sign']:12.6f} "
        f"{x['target_clip4']:13.6f} "
        f"{x['all_raw']:10.6f}"
    )


print()
print("=" * 110)
print("FINAL / PEAK")
print("=" * 110)

for key in [
    "frozen",
    "target_raw",
    "target_sign",
    "target_clip4",
    "all_raw",
]:
    peak = max(
        rows,
        key=lambda x: x[key]
    )

    final = rows[-1][key]

    print(
        f"{key:14s} "
        f"final={final:.6f} "
        f"peak={peak[key]:.6f} "
        f"peak_round=R{peak['round']:02d}"
    )


print()
print("=" * 110)
print("PER-SYSTEM FINAL AUC")
print("=" * 110)

for s in systems:

    frozen = role_probs(
        {
            sid(x): x
            for x in load(
                ROOT / "states_dev_round_25.jsonl"
            )
        }[s]
    )

    print(
        f"{s:42s} "
        f"FROZEN={auc(frozen, truth[s]):.6f} "
        f"TARGET_RAW={auc(target_raw[s], truth[s]):.6f} "
        f"TARGET_SIGN={auc(target_sign[s], truth[s]):.6f} "
        f"TARGET_CLIP4={auc(target_clip4[s], truth[s]):.6f} "
        f"ALL_RAW={auc(all_raw[s], truth[s]):.6f}"
    )


print()
print("=" * 110)
print("TARGET PROBE COUNTS AT R25")
print("=" * 110)

for s in systems:

    print()
    print(s)

    ranked = sorted(
        target_counts[s].items(),
        key=lambda z: (-z[1], z[0])
    )

    print(
        " ".join(
            f"{r}:{n}"
            for r, n in ranked
            if n > 0
        )
    )
