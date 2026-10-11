from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("artifacts/llm_selector_v3_auc_data")
OUT = Path("artifacts/q38_selector_v41_data")

SEED = 1397
VAL_FRAC = 0.15


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def norm_query(q):
    q = str(q).lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


def build_groups(rows):
    raw = defaultdict(list)

    for x in rows:
        raw[str(x["state_id"])].append(x)

    out = []

    for sid, xs in raw.items():

        xs = sorted(
            xs,
            key=lambda z:
                int(z["candidate_index"]),
        )

        if len(xs) < 2:
            continue

        if len(xs) > 8:
            raise RuntimeError(
                f"{sid}: {len(xs)} candidates > 8"
            )

        # -----------------------------------------------------
        # DENOISE EXACT DUPLICATES
        #
        # Same query executed twice from same state can receive
        # different realized utility because downstream MAS /
        # sensor behavior is stochastic.
        #
        # Do NOT teach the selector that identical text has
        # different expected value. Average the utility.
        # -----------------------------------------------------

        dup = defaultdict(list)

        for i, x in enumerate(xs):
            dup[norm_query(x["query"])].append(i)

        auc_gain = [
            float(
                x["role_auc_gain_training_only"]
            )
            for x in xs
        ]

        pair_gain = [
            float(
                x[
                    "role_pairwise_gain_training_only"
                ]
            )
            for x in xs
        ]

        for _, inds in dup.items():
            if len(inds) <= 1:
                continue

            ma = sum(
                auc_gain[i]
                for i in inds
            ) / len(inds)

            mp = sum(
                pair_gain[i]
                for i in inds
            ) / len(inds)

            for i in inds:
                auc_gain[i] = ma
                pair_gain[i] = mp

        x0 = xs[0]

        out.append({
            "state_id":
                sid,

            "system_id":
                str(x0["system_id"]),

            "round":
                int(x0.get("round", 0)),

            "role_probs_before": {
                str(k): float(v)
                for k, v
                in x0[
                    "role_probs_before"
                ].items()
            },

            "history_text":
                str(
                    x0.get(
                        "history_text",
                        "- no previous public interaction",
                    )
                ),

            "queries": [
                str(x["query"]).strip()
                for x in xs
            ],

            "candidate_indices": [
                int(x["candidate_index"])
                for x in xs
            ],

            "auc_gains":
                auc_gain,

            "pair_gains":
                pair_gain,
        })

    return out


def metrics(groups):
    randoms = []
    oracles = []
    spreads = []

    for g in groups:
        y = list(
            map(float, g["auc_gains"])
        )

        p = list(
            map(float, g["pair_gains"])
        )

        randoms.append(
            sum(y) / len(y)
        )

        oracles.append(
            max(y)
        )

        spreads.append(
            max(p) - min(p)
        )

    mr = (
        sum(randoms) / len(randoms)
        if randoms else None
    )

    mo = (
        sum(oracles) / len(oracles)
        if oracles else None
    )

    return {
        "states":
            len(groups),

        "systems":
            len({
                g["system_id"]
                for g in groups
            }),

        "mean_random_auc_gain":
            mr,

        "mean_oracle_auc_gain":
            mo,

        "oracle_minus_random":
            (
                mo - mr
                if mo is not None
                and mr is not None
                else None
            ),

        "mean_pair_spread":
            (
                sum(spreads)
                / len(spreads)
                if spreads else None
            ),

        "candidate_count": dict(
            Counter(
                len(g["queries"])
                for g in groups
            )
        ),
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = build_groups(
        read_jsonl(
            SRC / "train.jsonl"
        )
    )

    challenge = build_groups(
        read_jsonl(
            SRC / "challenge.jsonl"
        )
    )

    systems = sorted({
        g["system_id"]
        for g in source
    })

    rng = random.Random(SEED)
    rng.shuffle(systems)

    n_val = max(
        1,
        round(
            len(systems)
            * VAL_FRAC
        ),
    )

    val_systems = set(
        systems[:n_val]
    )

    train = [
        g
        for g in source
        if g["system_id"]
        not in val_systems
    ]

    val = [
        g
        for g in source
        if g["system_id"]
        in val_systems
    ]

    tr_sys = {
        g["system_id"]
        for g in train
    }

    va_sys = {
        g["system_id"]
        for g in val
    }

    ch_sys = {
        g["system_id"]
        for g in challenge
    }

    if tr_sys & va_sys:
        raise RuntimeError(
            "TRAIN / VAL SYSTEM LEAKAGE"
        )

    if tr_sys & ch_sys:
        raise RuntimeError(
            "TRAIN / CHALLENGE SYSTEM LEAKAGE"
        )

    if va_sys & ch_sys:
        raise RuntimeError(
            "VAL / CHALLENGE SYSTEM LEAKAGE"
        )

    write_jsonl(
        OUT / "train_groups.jsonl",
        train,
    )

    write_jsonl(
        OUT / "val_groups.jsonl",
        val,
    )

    write_jsonl(
        OUT / "challenge_groups.jsonl",
        challenge,
    )

    summary = {
        "version":
            "q38_selector_v41_data",

        "source":
            str(SRC),

        "train":
            metrics(train),

        "val":
            metrics(val),

        "challenge":
            metrics(challenge),

        "notes": [
            "split by system",
            "exact duplicate probes share averaged realized utility",
            "all decision states retained",
            "pair gain used as soft ranking supervision",
            "challenge never used for model selection",
        ],
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if len(train) < 1000:
        raise RuntimeError(
            f"Too few TRAIN states: {len(train)}"
        )

    if len(val) < 200:
        raise RuntimeError(
            f"Too few VAL states: {len(val)}"
        )

    if (
        summary["train"][
            "oracle_minus_random"
        ]
        <= 0.01
    ):
        raise RuntimeError(
            "Insufficient selector headroom"
        )

    print()
    print(
        "V4.1 DATASET PREFLIGHT: PASS"
    )


if __name__ == "__main__":
    main()
