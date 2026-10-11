from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from adaptive_v11_active.io import (
    stable_int,
)

from adaptive_v12_benign.gate_v12_1 import (
    strict_rich_benign_gate,
)

from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)

from v10.policy import (
    SeededProbePolicyV10,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
]


def clean_generated_text(x):
    s = str(x or "").strip()

    if not s:
        return ""

    s = re.sub(
        r"^\s*(?:question|query|probe|task)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in {"'", '"'}
    ):
        s = s[1:-1].strip()

    return s


def recover(sample):
    for source in (
        "query",
        "completion",
        "raw_completion",
    ):
        text = clean_generated_text(
            getattr(
                sample,
                source,
                "",
            )
        )

        if text:
            return text, source

    return "", "empty"


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def generate_one(
    policy,
    role,
    arm,
    retries,
    temperature,
    top_p,
    max_new_tokens,
):
    desc = ROLE_DESCRIPTIONS[
        role
    ]

    anchor = anchor_probe(
        role
    )

    last = None

    for attempt in range(
        retries + 1
    ):
        seed = stable_int(
            20260927,
            "targeted_bank_v12_2",
            arm,
            role,
            attempt,
        )

        seed_all(seed)

        obs = {
            "candidate_role":
                role,
            "candidate_description":
                desc,
            "anchor_probe":
                anchor,
            "avoid_probes":
                [anchor],
        }

        sample = policy.sample_probe(
            obs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        q, recovered_from = recover(
            sample
        )

        ok, reason, meta = (
            strict_rich_benign_gate(q)
        )

        row = {
            "role":
                role,
            "arm":
                arm,
            "query":
                q,
            "recovered_from":
                recovered_from,
            "legacy_sample_valid":
                bool(
                    getattr(
                        sample,
                        "valid",
                        False,
                    )
                ),
            "legacy_invalid_reason":
                str(
                    getattr(
                        sample,
                        "invalid_reason",
                        "",
                    )
                    or ""
                ),
            "hard_benign":
                bool(ok),
            "gate_reason":
                reason,
            "quality":
                meta,
            "attempt":
                attempt,
            "seed":
                int(seed),
        }

        last = row

        # For the bank, accept first hard-benign generation.
        if ok:
            return row

    return last


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
        ),
    )

    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/"
            "robust_visible_generator_v1_epoch2/"
            "policy"
        ),
    )

    ap.add_argument(
        "--out",
        default=(
            "artifacts/"
            "generator_v12_2/"
            "targeted_probe_bank.json"
        ),
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )

    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=224,
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    a = ap.parse_args()

    old = SeededProbePolicyV10(
        a.model,
        adapter_path=a.adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    rich = TargetedRichProbePolicyV12_2(
        a.model,
        adapter_path=a.adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    rows = []

    for role in ROLES:
        anchor = anchor_probe(
            role
        )

        ok, reason, meta = (
            strict_rich_benign_gate(
                anchor
            )
        )

        rows.append({
            "role":
                role,
            "arm":
                "canonical",
            "query":
                anchor,
            "recovered_from":
                "human_anchor",
            "legacy_sample_valid":
                True,
            "legacy_invalid_reason":
                "",
            "hard_benign":
                bool(ok),
            "gate_reason":
                reason,
            "quality":
                meta,
            "attempt":
                0,
            "seed":
                None,
        })

        rows.append(
            generate_one(
                old,
                role,
                "old_v10",
                a.retries,
                a.temperature,
                a.top_p,
                a.max_new_tokens,
            )
        )

        rows.append(
            generate_one(
                rich,
                role,
                "rich_v12_2",
                a.retries,
                a.temperature,
                a.top_p,
                a.max_new_tokens,
            )
        )

    out = Path(
        a.out
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment":
            "targeted_probe_bank_v12_2",
        "roles":
            ROLES,
        "arms": [
            "canonical",
            "old_v10",
            "rich_v12_2",
        ],
        "rows":
            rows,
    }

    out.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("TARGETED PROBE BANK")
    print("=" * 120)

    for role in ROLES:
        print()
        print("#" * 120)
        print("ROLE:", role)
        print(
            "DESC:",
            ROLE_DESCRIPTIONS[
                role
            ],
        )
        print("#" * 120)

        for row in rows:
            if row["role"] != role:
                continue

            print()
            print(
                f"[{row['arm']}] "
                f"benign={row['hard_benign']} "
                f"from={row['recovered_from']} "
                f"quality={row['quality']}"
            )

            if row[
                "gate_reason"
            ]:
                print(
                    "GATE:",
                    row[
                        "gate_reason"
                    ],
                )

            print(
                row[
                    "query"
                ]
            )

    print()
    print("SAVED:", out)


if __name__ == "__main__":
    main()
