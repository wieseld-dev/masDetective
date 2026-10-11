from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from adaptive_v11_active.io import stable_int
from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)
from adaptive_v12_benign.gate_v12_1 import (
    strict_rich_benign_gate,
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


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(x):
    s = str(x or "").strip()

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
    for field in (
        "query",
        "completion",
        "raw_completion",
    ):
        text = clean_text(
            getattr(sample, field, "")
        )

        if text:
            return text, field

    return "", "empty"


def run_arm(
    arm,
    adapter_path,
    args,
):
    print(
        "\n" + "=" * 100,
        flush=True,
    )
    print(
        "ARM:",
        arm,
        "adapter=",
        adapter_path,
        flush=True,
    )
    print(
        "=" * 100,
        flush=True,
    )

    policy = TargetedRichProbePolicyV12_2(
        args.model,
        adapter_path=adapter_path,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    rows = []

    for role in ROLES:
        for rep in range(
            args.replicates
        ):
            seed = stable_int(
                20260927,
                "v12_3",
                role,
                rep,
            )

            seed_all(seed)

            obs = {
                "candidate_role":
                    role,
                "candidate_description":
                    ROLE_DESCRIPTIONS[role],
                "anchor_probe":
                    anchor_probe(role),
                "avoid_probes": [
                    anchor_probe(role)
                ],
            }

            sample = policy.sample_probe(
                obs,
                temperature=
                    args.temperature,
                top_p=
                    args.top_p,
                max_new_tokens=
                    args.max_new_tokens,
            )

            q, source = recover(
                sample
            )

            ok, reason, meta = (
                strict_rich_benign_gate(
                    q
                )
            )

            words = len(
                re.findall(
                    r"\b[\w'-]+\b",
                    q,
                )
            )

            row = {
                "arm":
                    arm,
                "role":
                    role,
                "rep":
                    rep,
                "query":
                    q,
                "words":
                    words,
                "source":
                    source,
                "gate_valid":
                    bool(ok),
                "gate_reason":
                    reason,
                "gate_meta":
                    meta,
                "legacy_valid":
                    bool(
                        getattr(
                            sample,
                            "valid",
                            False,
                        )
                    ),
                "legacy_reason":
                    str(
                        getattr(
                            sample,
                            "invalid_reason",
                            "",
                        )
                        or ""
                    ),
            }

            rows.append(row)

            print(
                f"{role:<10} "
                f"rep={rep} "
                f"words={words:<3} "
                f"gate={ok} "
                f"reason={reason}",
                flush=True,
            )

    del policy

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def summarize(rows):
    out = {}

    for arm in sorted(
        set(x["arm"] for x in rows)
    ):
        xs = [
            x for x in rows
            if x["arm"] == arm
        ]

        words = [
            x["words"]
            for x in xs
        ]

        bands = Counter(
            x.get(
                "gate_meta",
                {},
            ).get(
                "quality_band",
                "unknown",
            )
            for x in xs
        )

        out[arm] = {
            "n":
                len(xs),
            "mean_words":
                sum(words)
                / len(words),
            "median_words":
                sorted(words)[
                    len(words) // 2
                ],
            "min_words":
                min(words),
            "max_words":
                max(words),
            "gate_pass_rate":
                sum(
                    x["gate_valid"]
                    for x in xs
                )
                / len(xs),
            "quality_bands":
                dict(bands),
        }

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=
            "Qwen/Qwen2.5-1.5B-Instruct",
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
        "--replicates",
        type=int,
        default=2,
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
        "--out",
        default=(
            "artifacts/"
            "generator_v12_3/"
            "base_vs_adapter"
        ),
    )

    args = ap.parse_args()

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    rows += run_arm(
        "old_adapter",
        args.adapter,
        args,
    )

    rows += run_arm(
        "base_no_adapter",
        None,
        args,
    )

    summary = summarize(
        rows
    )

    with (
        out / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    for role in ROLES:
        print(
            "\n"
            + "#" * 100
        )
        print(
            "ROLE:",
            role,
        )
        print(
            "#" * 100
        )

        for arm in (
            "old_adapter",
            "base_no_adapter",
        ):
            for row in rows:
                if (
                    row["role"]
                    == role
                    and
                    row["arm"]
                    == arm
                ):
                    print(
                        f"\n[{arm} "
                        f"rep={row['rep']} "
                        f"words={row['words']}]"
                    )
                    print(
                        row["query"]
                    )


if __name__ == "__main__":
    main()
