from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from adaptive_v11_active.reward import group_relative_advantages


SRC = Path("artifacts/v14_adaptive_g50_fresh_v3")
TEACH = Path("artifacts/generic_activation_teacher_v1")
DST = Path("artifacts/v14_adaptive_g50_genericact_r1_v1")

LAMBDA = 0.20
MIN_GROUP_COVERAGE = 0.80


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


src_rows = read_jsonl(
    SRC / "grpo_train_round_01.jsonl"
)

teacher_rows = read_jsonl(
    TEACH / "train_teacher_scores.jsonl"
)

teacher = {
    (
        str(x["session_id"]),
        int(x["candidate_index"]),
    ): x
    for x in teacher_rows
}


print("=" * 100)
print("GENERIC ACTIVATION R1 BUILDER")
print("=" * 100)

print("source rows  =", len(src_rows))
print("teacher rows =", len(teacher_rows))


groups = defaultdict(list)

for x in src_rows:
    groups[str(x["group_id"])].append(x)


sizes = [
    len(xs)
    for xs in groups.values()
]

print("groups =", len(groups))
print(
    "group size min/max =",
    min(sizes),
    "/",
    max(sizes),
)

if min(sizes) != 50 or max(sizes) != 50:
    raise RuntimeError(
        "Expected every G50 group to contain exactly 50 candidates"
    )


# ============================================================
# COVERAGE AUDIT
# ============================================================

coverage_by_group = {}

for gid, xs in groups.items():
    covered = sum(
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        )
        in teacher
        for x in xs
    )

    coverage_by_group[gid] = (
        covered / len(xs)
    )


overall_covered = sum(
    (
        str(x["session_id"]),
        int(x["candidate_index"]),
    )
    in teacher
    for x in src_rows
)

overall_coverage = (
    overall_covered
    / len(src_rows)
)

print(
    "overall teacher coverage =",
    f"{overall_coverage:.2%}",
)

print(
    "group coverage min/mean/max =",
    f"{min(coverage_by_group.values()):.2%}",
    f"{np.mean(list(coverage_by_group.values())):.2%}",
    f"{max(coverage_by_group.values()):.2%}",
)


if min(coverage_by_group.values()) < MIN_GROUP_COVERAGE:
    raise RuntimeError(
        "At least one G50 group has insufficient activation-teacher coverage"
    )


# ============================================================
# BUILD SHAPED ADVANTAGES
#
# IMPORTANT:
#   reward field remains ORIGINAL visible reward.
#
#   Activation only modifies within-group ranking used to
#   compute advantage.
#
#   Final advantage sign is gated by ORIGINAL visible reward:
#
#       visible > 0 -> advantage >= 0
#       visible < 0 -> advantage <= 0
#
# Thus privileged activation can never positively reinforce
# a probe that harmed visible inference utility.
# ============================================================

out = []

all_visible = []
all_shaped = []
all_bonus = []

teacher_used = 0
missing_teacher = 0

sign_violations = 0


for gid, xs in groups.items():

    enriched = []

    for x in xs:

        key = (
            str(x["session_id"]),
            int(x["candidate_index"]),
        )

        visible = float(
            x["reward"]
        )

        t = teacher.get(key)

        if t is None:
            # Neutral bonus for rows without a usable
            # privileged activation trace.
            rank = None
            bonus = 0.0
            missing_teacher += 1

        else:
            rank = float(
                t["agreement_rank"]
            )

            # rank [0,1] -> bonus [-1,+1]
            bonus = (
                2.0 * rank
                - 1.0
            )

            teacher_used += 1

        shaped = (
            visible
            + LAMBDA * bonus
        )

        enriched.append(
            (
                x,
                visible,
                shaped,
                rank,
                bonus,
            )
        )

        all_visible.append(
            visible
        )

        all_shaped.append(
            shaped
        )

        all_bonus.append(
            bonus
        )


    shaped_rewards = [
        z[2]
        for z in enriched
    ]

    relative = group_relative_advantages(
        shaped_rewards
    )


    for (
        x,
        visible,
        shaped,
        rank,
        bonus,
    ), rel_adv in zip(
        enriched,
        relative,
    ):

        # Preserve ABSOLUTE visible-utility direction.
        if visible > 0.0:
            advantage = max(
                0.0,
                float(rel_adv),
            )

        elif visible < 0.0:
            advantage = min(
                0.0,
                float(rel_adv),
            )

        else:
            advantage = 0.0


        if (
            visible > 0
            and advantage < -1e-12
        ) or (
            visible < 0
            and advantage > 1e-12
        ):
            sign_violations += 1


        y = dict(x)

        # Keep original visible reward untouched.
        y["reward"] = visible

        # Only advantage is activation-shaped.
        y["advantage"] = advantage

        y[
            "generic_activation_teacher_training_only"
        ] = {
            "teacher":
                "qwen2.5-1.5b_L24+mistral7b_cross_family_agreement",

            "agreement_rank":
                rank,

            "activation_bonus":
                float(bonus),

            "lambda":
                LAMBDA,

            "visible_reward":
                visible,

            "shaped_score":
                float(shaped),

            "teacher_available":
                rank is not None,
        }

        out.append(y)


if sign_violations:
    raise RuntimeError(
        f"Found {sign_violations} visible-sign violations"
    )


DST.mkdir(
    parents=True,
    exist_ok=True,
)

write_jsonl(
    DST / "grpo_train_round_01.jsonl",
    out,
)


# Copy only useful static artifacts if available.
for name in [
    "manifest_privileged.json",
    "frozen_belief.joblib",
    "states_train_round_00.jsonl",
    "states_dev_round_00.jsonl",
]:
    src = SRC / name

    if src.exists():
        shutil.copy2(
            src,
            DST / name,
        )


visible = np.asarray(
    all_visible,
    dtype=float,
)

shaped = np.asarray(
    all_shaped,
    dtype=float,
)

bonus = np.asarray(
    all_bonus,
    dtype=float,
)

advantages = np.asarray(
    [
        float(x["advantage"])
        for x in out
    ],
    dtype=float,
)


summary = {
    "source":
        str(SRC),

    "teacher":
        str(TEACH),

    "rows":
        len(out),

    "groups":
        len(groups),

    "group_size":
        50,

    "lambda":
        LAMBDA,

    "teacher_used":
        teacher_used,

    "teacher_missing":
        missing_teacher,

    "teacher_coverage":
        overall_coverage,

    "group_coverage_min":
        float(
            min(
                coverage_by_group.values()
            )
        ),

    "visible_reward_mean":
        float(visible.mean()),

    "visible_reward_std":
        float(visible.std()),

    "activation_bonus_mean":
        float(bonus.mean()),

    "activation_bonus_std":
        float(bonus.std()),

    "shaped_score_mean":
        float(shaped.mean()),

    "shaped_score_std":
        float(shaped.std()),

    "advantage_positive_fraction":
        float(
            np.mean(
                advantages > 0
            )
        ),

    "advantage_negative_fraction":
        float(
            np.mean(
                advantages < 0
            )
        ),

    "advantage_zero_fraction":
        float(
            np.mean(
                advantages == 0
            )
        ),

    "sign_violations":
        sign_violations,

    "principle": (
        "Cross-family activation consensus changes only "
        "within-group ranking/advantage magnitude. "
        "Original visible reward controls gradient sign."
    ),
}


(
    DST / "generic_activation_build_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 100)
print("BUILD SUMMARY")
print("=" * 100)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print(
    "WROTE =",
    DST / "grpo_train_round_01.jsonl",
)
