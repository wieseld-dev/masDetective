import json
import math
import re
from collections import defaultdict
from pathlib import Path

from adaptive_v11_active.reward import group_relative_advantages


SRC = Path("artifacts/v14_adaptive_role_v1")

DST = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/16_CLEAN_R7_TARGETAUX_V1"
)

TAU = 1.0
LAMBDA = 0.10


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def sid(row):
    return str(
        row.get(
            "session_id",
            row.get("system_id", "")
        )
    )


def cidx(row):
    if "candidate_index" in row:
        return int(row["candidate_index"])

    if "variant" in row:
        return int(row["variant"])

    action = row.get("action", {})
    if isinstance(action, dict):
        if "candidate_index" in action:
            return int(action["candidate_index"])
        if "variant" in action:
            return int(action["variant"])

    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        if "candidate_index" in metadata:
            return int(metadata["candidate_index"])
        if "variant" in metadata:
            return int(metadata["variant"])

    raise KeyError(
        "No candidate index found. "
        f"Top-level keys={sorted(row.keys())}; "
        f"action={row.get('action')!r}"
    )


def target_from_prompt(prompt):
    prompt = str(prompt)

    for pat in [
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        r"TARGET ROLE:\s*([^\n]+)",
        r"TARGET CAPABILITY:\s*([^\n]+)",
    ]:
        m = re.search(
            pat,
            prompt,
            re.I
        )

        if m:
            return m.group(1).strip()

    return None


def sign_consistent_advantages(
    relative_advs,
    sign_rewards,
):
    out = []

    for a, r in zip(
        relative_advs,
        sign_rewards
    ):
        a = float(a)
        r = float(r)

        if r > 0:
            out.append(
                max(0.0, a)
            )

        elif r < 0:
            out.append(
                min(0.0, a)
            )

        else:
            out.append(0.0)

    return out


# ================================================================
# Load
# ================================================================

original = read_jsonl(
    SRC /
    "grpo_train_round_07.jsonl"
)

candidates = {
    (sid(x), cidx(x)): x
    for x in read_jsonl(
        SRC /
        "candidates_train_round_07.jsonl"
    )
}

scored = {
    (sid(x), cidx(x)): x
    for x in read_jsonl(
        SRC /
        "scored_train_round_07.jsonl"
    )
}

manifest = json.loads(
    (
        SRC /
        "manifest_privileged.json"
    ).read_text()
)

truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "train"
}


groups = defaultdict(list)

for i, row in enumerate(original):
    groups[
        str(row["group_id"])
    ].append(
        (i, row)
    )


# ================================================================
# STEP 1:
# Exact reproduction of ORIGINAL R7 advantage rule
# ================================================================

bad = 0
max_err = 0.0
reconstructed_nonzero = 0

for gid, items in groups.items():

    rewards = [
        float(r["reward"])
        for _, r in items
    ]

    relative = (
        group_relative_advantages(
            rewards
        )
    )

    final = (
        sign_consistent_advantages(
            relative,
            rewards
        )
    )

    for (_, row), a in zip(
        items,
        final
    ):
        stored = float(
            row["advantage"]
        )

        err = abs(
            stored - a
        )

        max_err = max(
            max_err,
            err
        )

        if err > 1e-8:
            bad += 1

        if abs(a) > 1e-10:
            reconstructed_nonzero += 1


print("=" * 120)
print("EXACT ORIGINAL R7 RECONSTRUCTION")
print("=" * 120)

print(
    "rows                  =",
    len(original)
)

print(
    "groups                =",
    len(groups)
)

print(
    "max advantage error   =",
    f"{max_err:.12g}"
)

print(
    "mismatching rows      =",
    f"{bad}/{len(original)}"
)

print(
    "nonzero advantages    =",
    reconstructed_nonzero
)

if bad:
    raise SystemExit(
        "STOP: exact R7 reconstruction still fails."
    )


# ================================================================
# Add target metadata + auxiliary
# ================================================================

base_rows = []

joined = 0
missing = 0

for row in original:

    x = dict(row)

    s = str(
        row["session_id"]
    )

    idx = int(
        row["candidate_index"]
    )

    target = target_from_prompt(
        row["prompt"]
    )

    sc = scored.get(
        (s, idx)
    )

    signed = None
    target_e = None
    present = None
    aux = 0.0
    status = "ok"

    if (
        target is None
        or s not in truth
        or sc is None
    ):
        missing += 1
        status = "missing"

    else:
        evidence = {
            str(k): float(v)
            for k, v in dict(
                sc.get(
                    "evidence",
                    {}
                )
            ).items()
        }

        if target not in evidence:
            missing += 1
            status = (
                "target_missing_evidence"
            )

        else:
            joined += 1

            target_e = float(
                evidence[target]
            )

            present = (
                target in truth[s]
            )

            signed = (
                target_e
                if present
                else -target_e
            )

            aux = math.tanh(
                signed / TAU
            )

    global_reward = float(
        row["reward"]
    )

    composite_reward = (
        global_reward
        + LAMBDA * aux
    )

    x[
        "global_reward_training_only"
    ] = global_reward

    x[
        "target_aux_reward_training_only"
    ] = float(aux)

    x[
        "target_signed_evidence_training_only"
    ] = signed

    x[
        "target_evidence_training_only"
    ] = target_e

    x[
        "target_present_training_only"
    ] = present

    x[
        "target_role_training_only"
    ] = target

    x[
        "target_aux_status_training_only"
    ] = status

    x[
        "target_aux_tau_training_only"
    ] = TAU

    x[
        "target_aux_lambda_training_only"
    ] = LAMBDA

    x["reward"] = float(
        composite_reward
    )

    base_rows.append(x)


print()
print("=" * 120)
print("TARGET AUX COVERAGE")
print("=" * 120)

print(
    "joined =",
    joined
)

print(
    "missing =",
    missing
)


# ================================================================
# Build two variants
# ================================================================

def build_variant(
    name,
    sign_source,
):
    rows = [
        dict(x)
        for x in base_rows
    ]

    grouped = defaultdict(list)

    for i, row in enumerate(rows):
        grouped[
            str(row["group_id"])
        ].append(
            (i, row)
        )

    for gid, items in grouped.items():

        composite_rewards = [
            float(r["reward"])
            for _, r in items
        ]

        relative = (
            group_relative_advantages(
                composite_rewards
            )
        )

        if sign_source == "composite":

            signs = (
                composite_rewards
            )

        elif sign_source == "global":

            signs = [
                float(
                    r[
                        "global_reward_training_only"
                    ]
                )
                for _, r in items
            ]

        else:
            raise ValueError(
                sign_source
            )

        final = (
            sign_consistent_advantages(
                relative,
                signs
            )
        )

        for (i, _), a in zip(
            items,
            final
        ):
            rows[i][
                "advantage"
            ] = float(a)

            rows[i][
                "targetaux_sign_rule_training_only"
            ] = sign_source


    old_lookup = {
        (
            str(x["session_id"]),
            int(x["candidate_index"])
        ): x
        for x in original
    }

    old_pos = 0
    old_good = 0

    new_pos = 0
    new_good = 0

    pos_to_zero = 0
    zero_to_pos = 0
    neg_to_pos = 0

    nonzero = 0

    for x in rows:

        k = (
            str(x["session_id"]),
            int(x["candidate_index"])
        )

        old_a = float(
            old_lookup[k]["advantage"]
        )

        new_a = float(
            x["advantage"]
        )

        signed = (
            x.get(
                "target_signed_evidence_training_only"
            )
        )

        good = (
            signed is not None
            and float(signed) > 0
        )

        if old_a > 0:
            old_pos += 1
            old_good += int(good)

        if new_a > 0:
            new_pos += 1
            new_good += int(good)

        if abs(new_a) > 1e-10:
            nonzero += 1

        if (
            old_a > 0
            and new_a <= 0
        ):
            pos_to_zero += 1

        if (
            abs(old_a) <= 1e-10
            and new_a > 0
        ):
            zero_to_pos += 1

        if (
            old_a < 0
            and new_a > 0
        ):
            neg_to_pos += 1


    out = (
        DST /
        name /
        "grpo_train_round_07.jsonl"
    )

    write_jsonl(
        out,
        rows
    )


    print()
    print("=" * 120)
    print(name)
    print("=" * 120)

    print(
        "sign rule            =",
        sign_source
    )

    print(
        "rows                 =",
        len(rows)
    )

    print(
        "nonzero advantages   =",
        nonzero
    )

    print(
        "OLD ADV+ correct     =",
        f"{old_good}/{old_pos}",
        f"({old_good/old_pos:.1%})"
    )

    print(
        "NEW ADV+ correct     =",
        f"{new_good}/{new_pos}",
        f"({new_good/new_pos:.1%})"
    )

    print(
        "old + -> new <=0     =",
        pos_to_zero
    )

    print(
        "old 0 -> new +       =",
        zero_to_pos
    )

    print(
        "old - -> new +       =",
        neg_to_pos
    )

    print(
        "output               =",
        out
    )

    return {
        "name": name,
        "sign_source": sign_source,
        "rows": len(rows),
        "nonzero_advantages": nonzero,
        "old_positive_correct":
            old_good / old_pos,
        "new_positive_correct":
            new_good / new_pos,
        "old_positive_count":
            old_pos,
        "new_positive_count":
            new_pos,
        "old_positive_to_nonpositive":
            pos_to_zero,
        "old_zero_to_positive":
            zero_to_pos,
        "old_negative_to_positive":
            neg_to_pos,
    }


a = build_variant(
    "composite_sign",
    "composite",
)

b = build_variant(
    "global_sign",
    "global",
)


config = {
    "source": str(SRC),
    "round": 7,
    "parent_checkpoint": str(
        SRC /
        "checkpoints/round_06/policy"
    ),
    "tau": TAU,
    "lambda": LAMBDA,
    "exact_r7_reconstruction": {
        "max_error": max_err,
        "mismatches": bad,
        "nonzero_advantages":
            reconstructed_nonzero,
    },
    "target_aux_coverage": {
        "joined": joined,
        "missing": missing,
    },
    "variants": [
        a,
        b,
    ],
    "original_training": {
        "epochs": 1,
        "lr": 1.5e-6,
        "clip": 0.20,
        "kl_beta": 0.05,
        "samples": 1020,
    },
}

(
    DST /
    "targetaux_r7_config.json"
).write_text(
    json.dumps(
        config,
        indent=2
    )
)

print()
print("=" * 120)
print("CONFIG")
print("=" * 120)

print(
    json.dumps(
        config,
        indent=2
    )
)
