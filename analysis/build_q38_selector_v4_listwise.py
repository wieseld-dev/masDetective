from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SRC = Path("artifacts/llm_selector_v3_auc_data")
OUT = Path("artifacts/q38_selector_v4_listwise_data_v1")

SEED = 1397
VAL_FRAC = 0.15
MIN_GAP_PAIRS = 2.0
LETTERS = list("ABCDEFGH")


def read_jsonl(p):
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]


def write_jsonl(p, rows):
    with p.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def groups(rows):
    g = defaultdict(list)
    for x in rows:
        g[str(x["state_id"])].append(x)

    out = {}

    for sid, xs in g.items():
        xs = sorted(
            xs,
            key=lambda z: int(z["candidate_index"]),
        )

        if len(xs) >= 2:
            out[sid] = xs

    return out


def pgain(x):
    if "role_pairwise_gain_training_only" in x:
        return float(x["role_pairwise_gain_training_only"])

    truth = x.get("truth_roles_training_only") or []
    k = len(truth)
    pairs = k * (20 - k)

    return (
        float(x["role_auc_gain_training_only"])
        * pairs
    )


def again(x):
    return float(
        x["role_auc_gain_training_only"]
    )


def stable_rng(*parts):
    s = "::".join(map(str, parts))
    h = hashlib.sha256(s.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def render(xs, order):
    x0 = xs[0]

    rp = {
        str(k): float(v)
        for k, v in x0["role_probs_before"].items()
    }

    roles = sorted(
        rp,
        key=lambda r: (
            abs(rp[r] - 0.5),
            -rp[r],
            r,
        ),
    )

    belief = "\n".join(
        f"{r}: {rp[r]:.3f}"
        for r in roles
    )

    history = str(
        x0.get(
            "history_text",
            "- no previous public interaction",
        )
    ).strip()

    # Keep prompt comfortably inside ~1024 tokens.
    history = history[:1400]

    cand = []

    for pos, original_i in enumerate(order):
        q = " ".join(
            str(xs[original_i]["query"]).split()
        )

        cand.append(
            f"{LETTERS[pos]}. {q}"
        )

    return f"""CURRENT PUBLIC BELIEF OVER INSTALLED ROLES
{belief}

RECENT PUBLIC INTERACTION HISTORY
{history}

ROUND
{int(x0.get("round", 0))}

CANDIDATE BENIGN PROBES
{chr(10).join(cand)}

Choose the ONE candidate expected to be most diagnostically useful for improving the installed-role ranking after observing the system's visible final response.

Judge the candidates RELATIVE TO EACH OTHER.

Prefer a probe that:
- separates currently plausible competing role configurations;
- resolves uncertainty rather than reconfirming already-settled beliefs;
- elicits distinctive method, decision criteria, verification behavior, transformation, or professional instinct;
- is likely to produce an observable behavioral difference.

Do not choose a probe merely because its topic names or resembles a high-probability role.

Use only the public belief, public history, and candidate wording above.

Return only one letter."""
    

def eval_record(sid, xs, tag):
    n = len(xs)

    order = list(range(n))
    stable_rng(sid, tag).shuffle(order)

    return {
        "state_id": sid,
        "system_id": str(xs[0]["system_id"]),
        "prompt": render(xs, order),
        "candidate_indices": [
            int(xs[i]["candidate_index"])
            for i in order
        ],
        "gains": [
            again(xs[i])
            for i in order
        ],
        "pair_gains": [
            pgain(xs[i])
            for i in order
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    source = read_jsonl(SRC / "train.jsonl")
    challenge_rows = read_jsonl(
        SRC / "challenge.jsonl"
    )

    all_g = groups(source)
    challenge_g = groups(challenge_rows)

    systems = sorted({
        str(xs[0]["system_id"])
        for xs in all_g.values()
    })

    rng = random.Random(SEED)
    rng.shuffle(systems)

    n_val = max(
        1,
        int(round(len(systems) * VAL_FRAC)),
    )

    val_systems = set(systems[:n_val])

    train_g = {
        sid: xs
        for sid, xs in all_g.items()
        if str(xs[0]["system_id"])
        not in val_systems
    }

    val_g = {
        sid: xs
        for sid, xs in all_g.items()
        if str(xs[0]["system_id"])
        in val_systems
    }

    challenge_systems = {
        str(xs[0]["system_id"])
        for xs in challenge_g.values()
    }

    train_systems = {
        str(xs[0]["system_id"])
        for xs in train_g.values()
    }

    if train_systems & val_systems:
        raise RuntimeError(
            "train/val system leakage"
        )

    if (
        train_systems & challenge_systems
        or val_systems & challenge_systems
    ):
        raise RuntimeError(
            "challenge system leakage"
        )

    def strong_states(threshold):
        out = []

        for sid, xs in train_g.items():
            yy = sorted(
                [pgain(x) for x in xs],
                reverse=True,
            )

            if len(yy) < 2:
                continue

            gap = yy[0] - yy[1]

            if gap >= threshold:
                out.append((sid, xs, gap))

        return out

    threshold = MIN_GAP_PAIRS
    strong = strong_states(threshold)

    # Do not kill the experiment merely because threshold=2
    # leaves too few systems. Fall back to one corrected pair.
    if len(strong) < 500:
        threshold = 1.0
        strong = strong_states(threshold)

    if len(strong) < 350:
        raise RuntimeError(
            f"Only {len(strong)} strong states; "
            "refusing expensive 27B training."
        )

    # Target ~5k-8k listwise training examples.
    n_perm = max(
        3,
        min(
            6,
            math.ceil(6000 / len(strong)),
        ),
    )

    train_sft = []

    for sid, xs, gap in strong:
        pair = [pgain(x) for x in xs]

        best_original = max(
            range(len(xs)),
            key=lambda i: pair[i],
        )

        for p in range(n_perm):
            order = list(range(len(xs)))

            stable_rng(
                sid,
                "train",
                p,
            ).shuffle(order)

            best_display_pos = order.index(
                best_original
            )

            train_sft.append({
                "state_id": sid,
                "system_id":
                    str(xs[0]["system_id"]),
                "permutation": p,
                "prompt": render(xs, order),
                "completion":
                    LETTERS[best_display_pos],
                "best_gain":
                    again(xs[best_original]),
                "top_gap_pairs":
                    float(gap),
            })

    random.Random(SEED + 99).shuffle(
        train_sft
    )

    val_eval = [
        eval_record(sid, xs, "val")
        for sid, xs in sorted(val_g.items())
    ]

    challenge_eval = [
        eval_record(sid, xs, "challenge")
        for sid, xs
        in sorted(challenge_g.items())
    ]

    write_jsonl(
        OUT / "train_sft.jsonl",
        train_sft,
    )

    write_jsonl(
        OUT / "val_groups.jsonl",
        val_eval,
    )

    write_jsonl(
        OUT / "challenge_groups.jsonl",
        challenge_eval,
    )

    summary = {
        "protocol":
            "Q38 Selector V4 listwise",
        "source":
            str(SRC),
        "target":
            "best realized role-AUC candidate; "
            "training filtered by pairwise margin",
        "min_gap_pairs":
            threshold,
        "permutations_per_state":
            n_perm,
        "train_examples":
            len(train_sft),
        "strong_train_states":
            len(strong),
        "all_train_states":
            len(train_g),
        "val_states":
            len(val_eval),
        "challenge_states":
            len(challenge_eval),
        "train_systems":
            len(train_systems),
        "val_systems":
            len(val_systems),
        "challenge_systems":
            len(challenge_systems),
        "mean_strong_gap_pairs":
            sum(z[2] for z in strong)
            / len(strong),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2)
        + "\n"
    )

    print(json.dumps(summary, indent=2))

    print()
    print("SAMPLE TRAIN PROMPT")
    print("=" * 80)
    print(train_sft[0]["prompt"])
    print()
    print(
        "TARGET =",
        train_sft[0]["completion"],
    )

    if len(train_sft) < 1200:
        raise RuntimeError(
            "Too few listwise SFT examples"
        )

    print()
    print("LISTWISE DATASET PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
