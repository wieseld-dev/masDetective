from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import visible_adaptive_bridge_v1 as vb


ROOT = Path("artifacts/v6_fp_survival_multisys_v1")

SEEDS = [
    20261011,
    20261012,
    20261013,
    20261014,
    20261015,
]

# EXACT frozen visible reader used by the previous pipeline.
BRIDGE = Path(
    "artifacts/v3_coverage_elimination_pilot/visible_bridge.joblib"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

N_PERM = 1000
N_BOOT = 1000
RNG_SEED = 20260925


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def metrics(Y, P):
    yf = Y.reshape(-1)
    pf = P.reshape(-1)

    out = {
        "micro_auc": safe_auc(yf, pf),
        "average_precision": float(
            average_precision_score(yf, pf)
        ),
        "positive_events": int(yf.sum()),
        "negative_events": int((1-yf).sum()),
        "active_mean_score": float(
            pf[yf == 1].mean()
        ) if np.any(yf == 1) else None,
        "inactive_mean_score": float(
            pf[yf == 0].mean()
        ) if np.any(yf == 0) else None,
    }

    r1, r3, r5, rr = [], [], [], []
    margins = []
    top1_round = []

    for y, p in zip(Y, P):
        act = np.where(y == 1)[0]

        if len(act) == 0:
            continue

        ranked = np.argsort(-p)
        ranks = {
            int(role_idx): int(
                np.where(ranked == role_idx)[0][0] + 1
            )
            for role_idx in act
        }

        for idx in act:
            rank = ranks[int(idx)]
            r1.append(int(rank <= 1))
            r3.append(int(rank <= 3))
            r5.append(int(rank <= 5))
            rr.append(1.0 / rank)

        top1_round.append(
            int(ranked[0] in set(act.tolist()))
        )

        inactive = np.where(y == 0)[0]
        margins.append(
            float(
                np.min(p[act])
                - np.max(p[inactive])
            )
        )

    out.update({
        "recall_at_1": float(np.mean(r1)) if r1 else None,
        "recall_at_3": float(np.mean(r3)) if r3 else None,
        "recall_at_5": float(np.mean(r5)) if r5 else None,
        "mean_reciprocal_rank":
            float(np.mean(rr)) if rr else None,
        "top1_execution_hit_rate":
            float(np.mean(top1_round))
            if top1_round else None,
        "mean_execution_margin":
            float(np.mean(margins))
            if margins else None,
    })

    return out


# ------------------------------------------------------------
# LOAD 40 NEW V6-DPO EXECUTIONS
# ------------------------------------------------------------

rows = []

for seed in SEEDS:
    r = ROOT / f"seed_{seed}"
    manifest = json.loads(
        (r / "manifest_privileged.json").read_text()
    )

    meta = {
        x["session_id"]: x
        for x in manifest["systems"]
        if x["split"] == "dev"
    }

    candidates = {
        (
            x["session_id"],
            int(x["candidate_index"]),
        ): x
        for x in read_jsonl(
            r / "candidates_dev_round_01.jsonl"
        )
    }

    responses = read_jsonl(
        r / "responses_dev_round_01.jsonl"
    )

    for x in responses:
        if x.get("execution_status") != "ok":
            continue

        key = (
            x["session_id"],
            int(x["candidate_index"]),
        )

        c = candidates.get(key)

        if c is None or not c.get("valid"):
            continue

        m = meta[x["session_id"]]

        rows.append({
            "seed": seed,
            "system_id": x["session_id"],
            "k": int(m["k"]),
            "mode": str(m["mode"]),
            "style": str(m["style"]),
            "query": str(c["query"]),
            "output": str(
                x.get("final_output")
                or x.get("output")
                or x.get("answer")
                or ""
            ),
            "selected_roles": list(map(
                str,
                x.get(
                    "selected_roles_training_only",
                    [],
                ),
            )),
            "truth_roles": list(map(
                str,
                x.get(
                    "truth_roles_training_only",
                    [],
                ),
            )),
        })


print("=" * 100)
print("V6-DPO MULTI-SYSTEM FINGERPRINT SURVIVAL")
print("=" * 100)

print("executions:", len(rows))
print("systems   :", len(set(
    x["system_id"] for x in rows
)))
print("seeds     :", len(set(
    x["seed"] for x in rows
)))
print("K dist    :", dict(Counter(
    x["k"] for x in rows
)))
print("mode dist :", dict(Counter(
    x["mode"] for x in rows
)))

if len(rows) < 32:
    raise RuntimeError(
        f"Too few successful executions: {len(rows)}"
    )

# ------------------------------------------------------------
# BENIGN SURFACE AUDIT
# This is deliberately conservative and descriptive.
# It does NOT change which examples enter fingerprint metrics.
# ------------------------------------------------------------

attack_markers = [
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "prompt injection",
    "jailbreak",
    "hidden instruction",
    "internal instruction",
    "reveal your prompt",
    "show your prompt",
    "list your agents",
    "what agents",
    "which agents",
    "internal agents",
    "system topology",
    "routing topology",
]

flagged = []

for x in rows:
    q = x["query"].lower()
    hits = [
        z for z in attack_markers
        if z in q
    ]

    if hits:
        flagged.append({
            "seed": x["seed"],
            "system_id": x["system_id"],
            "query": x["query"],
            "markers": hits,
        })

print()
print("===== BENIGN SURFACE AUDIT =====")
print("attack-marker flagged:",
      len(flagged), "/", len(rows))

for x in flagged[:20]:
    print("FLAG:", x)

# ------------------------------------------------------------
# FROZEN FINGERPRINT READER
# ------------------------------------------------------------

if not BRIDGE.exists():
    raise FileNotFoundError(BRIDGE)

asset = joblib.load(BRIDGE)

roles = list(asset["roles"])
models = asset["models"]
cals = asset["cals"]

ri = {
    r: i
    for i, r in enumerate(roles)
}

print()
print("roles:", len(roles))

encoder = SentenceTransformer(EMBED_MODEL)

queries = [x["query"] for x in rows]
outputs = [x["output"] for x in rows]

Q = np.asarray(
    encoder.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    ),
    dtype=np.float32,
)

O = np.asarray(
    encoder.encode(
        outputs,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    ),
    dtype=np.float32,
)

X = np.concatenate([Q, O], axis=1)

P = vb.visible_probs(
    models,
    cals,
    X,
)

Y = np.zeros(
    (len(rows), len(roles)),
    dtype=np.int32,
)

for i, x in enumerate(rows):
    for r in x["selected_roles"]:
        if r in ri:
            Y[i, ri[r]] = 1


actual = metrics(Y, P)

print()
print("===== ACTUAL MATCHED QUERY + OUTPUT =====")
for k, v in actual.items():
    print(f"{k:30s}: {v}")


# ------------------------------------------------------------
# PER-SEED
# ------------------------------------------------------------

per_seed = {}

for seed in SEEDS:
    idx = np.asarray([
        i for i, x in enumerate(rows)
        if x["seed"] == seed
    ])

    if len(idx):
        per_seed[str(seed)] = metrics(
            Y[idx],
            P[idx],
        )

print()
print("===== PER SEED =====")

for seed, m in per_seed.items():
    print(
        seed,
        "n=",
        sum(x["seed"] == int(seed) for x in rows),
        "AUC=",
        m["micro_auc"],
        "R@3=",
        m["recall_at_3"],
        "AP=",
        m["average_precision"],
    )


# ------------------------------------------------------------
# PER K
# ------------------------------------------------------------

per_k = {}

for kval in sorted(set(x["k"] for x in rows)):
    idx = np.asarray([
        i for i, x in enumerate(rows)
        if x["k"] == kval
    ])

    per_k[str(kval)] = metrics(
        Y[idx],
        P[idx],
    )

print()
print("===== PER K =====")

for kval, m in per_k.items():
    print(
        "K=" + kval,
        "AUC=",
        m["micro_auc"],
        "R@3=",
        m["recall_at_3"],
        "AP=",
        m["average_precision"],
    )


# ------------------------------------------------------------
# OUTPUT-MATCH PERMUTATION TEST
#
# Keep queries fixed.
# Shuffle outputs WITHIN EACH GENERATOR SEED.
#
# If fingerprint comes only from query semantics,
# breaking the query/output match should not hurt much.
#
# If matching final output contains activation fingerprint,
# REAL AUC should beat this null distribution.
# ------------------------------------------------------------

rng = np.random.default_rng(RNG_SEED)

seed_groups = defaultdict(list)

for i, x in enumerate(rows):
    seed_groups[x["seed"]].append(i)

null_auc = []

for rep in range(N_PERM):
    Operm = O.copy()

    for seed, ids in seed_groups.items():
        ids = np.asarray(ids)
        perm = rng.permutation(ids)
        Operm[ids] = O[perm]

    Pperm = vb.visible_probs(
        models,
        cals,
        np.concatenate([Q, Operm], axis=1),
    )

    a = safe_auc(
        Y.reshape(-1),
        Pperm.reshape(-1),
    )

    if a is not None:
        null_auc.append(a)

null_auc = np.asarray(
    null_auc,
    dtype=float,
)

actual_auc = actual["micro_auc"]

perm_p = float(
    (
        1
        + np.sum(null_auc >= actual_auc)
    )
    /
    (len(null_auc) + 1)
)

print()
print("===== OUTPUT-MATCH PERMUTATION TEST =====")
print("actual AUC       :", actual_auc)
print("shuffle mean AUC :", float(null_auc.mean()))
print("shuffle std AUC  :", float(null_auc.std()))
print("AUC lift         :", float(
    actual_auc - null_auc.mean()
))
print("permutation p    :", perm_p)
print("null 95%         :", [
    float(np.quantile(null_auc, .025)),
    float(np.quantile(null_auc, .975)),
])


# ------------------------------------------------------------
# SYSTEM-CLUSTER BOOTSTRAP
#
# Resample SYSTEMS, not individual role labels.
# Each sampled system contributes all five generator seeds.
# ------------------------------------------------------------

systems = sorted(set(
    x["system_id"]
    for x in rows
))

by_system = {
    sid: np.asarray([
        i for i, x in enumerate(rows)
        if x["system_id"] == sid
    ])
    for sid in systems
}

boot_auc = []

for _ in range(N_BOOT):
    sampled = rng.choice(
        systems,
        size=len(systems),
        replace=True,
    )

    idx = np.concatenate([
        by_system[sid]
        for sid in sampled
    ])

    a = safe_auc(
        Y[idx].reshape(-1),
        P[idx].reshape(-1),
    )

    if a is not None:
        boot_auc.append(a)

boot_auc = np.asarray(
    boot_auc,
    dtype=float,
)

bootstrap = {
    "mean": float(boot_auc.mean()),
    "ci95": [
        float(np.quantile(boot_auc, .025)),
        float(np.quantile(boot_auc, .975)),
    ],
}

print()
print("===== SYSTEM-CLUSTER BOOTSTRAP =====")
print("AUC mean :", bootstrap["mean"])
print("AUC 95%  :", bootstrap["ci95"])


# ------------------------------------------------------------
# ROLE DIAGNOSTICS
# ------------------------------------------------------------

per_role = {}

for j, role in enumerate(roles):
    auc = safe_auc(
        Y[:, j],
        P[:, j],
    )

    pos = Y[:, j] == 1
    neg = ~pos

    per_role[role] = {
        "auc": auc,
        "positive_n": int(pos.sum()),
        "negative_n": int(neg.sum()),
        "positive_mean":
            float(P[pos, j].mean())
            if pos.any() else None,
        "negative_mean":
            float(P[neg, j].mean())
            if neg.any() else None,
    }

print()
print("===== ACTIVE ROLE DIAGNOSTICS =====")

for role, m in sorted(
    per_role.items(),
    key=lambda kv: (
        kv[1]["auc"] is None,
        -(
            kv[1]["auc"]
            if kv[1]["auc"] is not None
            else -1
        ),
    ),
):
    if m["positive_n"]:
        print(
            f"{role:25s} "
            f"n+={m['positive_n']:3d} "
            f"auc={m['auc']} "
            f"pos={m['positive_mean']:.3f} "
            f"neg={m['negative_mean']:.3f}"
        )


# ------------------------------------------------------------
# WORST FALSE POSITIVES
# ------------------------------------------------------------

false_rows = []

for i, x in enumerate(rows):
    active = set(x["selected_roles"])

    for j, role in enumerate(roles):
        if role not in active:
            false_rows.append({
                "score": float(P[i, j]),
                "role": role,
                "seed": x["seed"],
                "system_id": x["system_id"],
                "query": x["query"],
                "actual": sorted(active),
            })

false_rows.sort(
    key=lambda x: -x["score"]
)

print()
print("===== TOP 15 FALSE-POSITIVE SCORES =====")

for x in false_rows[:15]:
    print(
        f"{x['score']:.4f} "
        f"false={x['role']:24s} "
        f"actual={x['actual']} "
        f"Q={x['query'][:160]!r}"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

summary = {
    "experiment":
        "V6-DPO multi-system fingerprint survival",
    "generator":
        "artifacts/v6_matched_sft_dpo/policy",
    "reader":
        str(BRIDGE),
    "systems": len(systems),
    "executions": len(rows),
    "generator_seeds": SEEDS,
    "actual": actual,
    "per_seed": per_seed,
    "per_k": per_k,
    "output_match_permutation": {
        "n": int(len(null_auc)),
        "actual_auc": actual_auc,
        "shuffle_mean_auc":
            float(null_auc.mean()),
        "shuffle_std_auc":
            float(null_auc.std()),
        "auc_lift":
            float(actual_auc-null_auc.mean()),
        "p_value": perm_p,
        "null_95": [
            float(np.quantile(null_auc, .025)),
            float(np.quantile(null_auc, .975)),
        ],
    },
    "system_cluster_bootstrap": bootstrap,
    "per_role": per_role,
    "benign_surface_audit": {
        "attack_marker_flagged": len(flagged),
        "total": len(rows),
        "flagged": flagged,
    },
    "note": (
        "selected_roles are used only as evaluation labels. "
        "The frozen fingerprint reader receives query + final output."
    ),
}

(ROOT / "summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

(ROOT / "executions_scored.jsonl").write_text(
    "".join(
        json.dumps({
            **x,
            "fingerprint_probs": {
                role: float(P[i, j])
                for j, role in enumerate(roles)
            },
        }, ensure_ascii=False) + "\n"
        for i, x in enumerate(rows)
    ),
    encoding="utf-8",
)

print()
print("=" * 100)
print("PRIMARY RESULT")
print("=" * 100)
print(
    "REAL MATCHED AUC      =",
    actual_auc,
)
print(
    "SHUFFLED OUTPUT AUC   =",
    float(null_auc.mean()),
)
print(
    "OUTPUT-MATCH AUC LIFT =",
    float(actual_auc-null_auc.mean()),
)
print(
    "PERMUTATION p         =",
    perm_p,
)
print(
    "Recall@3              =",
    actual["recall_at_3"],
)
print(
    "Recall@5              =",
    actual["recall_at_5"],
)
print(
    "Cluster bootstrap 95% =",
    bootstrap["ci95"],
)
print(
    "Benign marker flags   =",
    len(flagged),
    "/",
    len(rows),
)
print()
print("wrote:", ROOT / "summary.json")
