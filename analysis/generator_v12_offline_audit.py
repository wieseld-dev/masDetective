from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import re
from collections import Counter
import numpy as np
import torch

from adaptive_v11_active.io import (
    read_jsonl,
    stable_int,
)

from adaptive_v11_active.schema import (
    PublicBeliefState,
    assert_policy_safe,
)

from adaptive_v11_active.reward import (
    normalized_word_jaccard,
)

from adaptive_v12_benign.policy import (
    DiagnosticBenignProbePolicyV12,
)

from adaptive_v12_benign.gate import (
    strict_rich_benign_gate,
)


def norm(q):
    return re.sub(
        r"\s+",
        " ",
        str(q).lower().strip(),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--states",
        required=True,
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
        "--model",
        default=(
            "Qwen/Qwen2.5-1.5B-Instruct"
        ),
    )

    ap.add_argument(
        "--n",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--group-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=0.90,
    )

    ap.add_argument(
        "--top-p",
        type=float,
        default=0.96,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--out",
        default=(
            "artifacts/generator_v12/"
            "offline_audit"
        ),
    )

    a = ap.parse_args()

    out = Path(
        a.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    states = read_jsonl(
        a.states
    )

    if not states:
        raise RuntimeError(
            "no states"
        )

    policy = (
        DiagnosticBenignProbePolicyV12(
            a.model,
            adapter_path=a.adapter,
            trainable=False,
            use_4bit=False,
            max_seq_length=4096,
        )
    )

    rows = []

    sample_i = 0

    while (
        len(rows) < a.n
    ):
        raw = states[
            sample_i
            % len(states)
        ]

        assert_policy_safe(
            raw
        )

        st = PublicBeliefState.from_public_dict(
            raw
        )

        j = (
            sample_i
            % a.group_size
        )

        seed = stable_int(
            20260927,
            "v12audit",
            st.session_id,
            sample_i,
        )

        random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        obs = {
            "belief_state":
                st.to_public_dict(),
            "candidate_index":
                j,
            "max_history_turns":
                5,
        }

        sample = policy.sample_probe(
            obs,
            temperature=a.temperature,
            top_p=a.top_p,
            max_new_tokens=a.max_new_tokens,
        )

        q = str(
            sample.query or ""
        ).strip()

        ok, reason, meta = (
            strict_rich_benign_gate(
                q
            )
        )

        rows.append({
            "i":
                sample_i,
            "session_id":
                st.session_id,
            "candidate_index":
                j,
            "family":
                j % 4,
            "query":
                q,
            "sample_valid":
                bool(
                    sample.valid
                ),
            "gate_valid":
                bool(
                    ok
                ),
            "reason":
                reason,
            "meta":
                meta,
        })

        sample_i += 1

        if (
            len(rows) % 50
            == 0
        ):
            print(
                f"[audit] {len(rows)}/{a.n}",
                flush=True,
            )

    accepted = [
        x
        for x in rows
        if x[
            "gate_valid"
        ]
    ]

    rejected = [
        x
        for x in rows
        if not x[
            "gate_valid"
        ]
    ]

    # Exact duplicate rate.
    n_unique = len(
        {
            norm(
                x[
                    "query"
                ]
            )
            for x in accepted
        }
    )

    exact_dup_rate = (
        1.0
        - n_unique
        / max(
            1,
            len(
                accepted
            ),
        )
    )

    # Semantic-ish lexical near duplicate audit.
    near_pairs = 0
    comparisons = 0

    for i in range(
        len(
            accepted
        )
    ):
        for j in range(
            max(
                0,
                i - 40,
            ),
            i,
        ):
            comparisons += 1

            sim = normalized_word_jaccard(
                accepted[
                    i
                ][
                    "query"
                ],
                accepted[
                    j
                ][
                    "query"
                ],
            )

            if sim >= 0.72:
                near_pairs += 1

    word_counts = [
        x.get(
            "meta",
            {},
        ).get(
            "words",
            0,
        )
        for x in accepted
    ]

    richness = [
        x.get(
            "meta",
            {},
        ).get(
            "richness",
            0,
        )
        for x in accepted
    ]

    reasons = Counter(
        x[
            "reason"
        ]
        for x in rejected
    )

    summary = {
        "raw_candidates":
            len(
                rows
            ),

        "accepted":
            len(
                accepted
            ),

        "rejected":
            len(
                rejected
            ),

        "strict_pass_rate":
            len(
                accepted
            )
            / max(
                len(
                    rows
                ),
                1,
            ),

        "median_words":
            float(
                np.median(
                    word_counts
                )
            )
            if word_counts
            else None,

        "p10_words":
            float(
                np.quantile(
                    word_counts,
                    0.10,
                )
            )
            if word_counts
            else None,

        "p90_words":
            float(
                np.quantile(
                    word_counts,
                    0.90,
                )
            )
            if word_counts
            else None,

        "mean_richness":
            float(
                np.mean(
                    richness
                )
            )
            if richness
            else None,

        "exact_duplicate_rate":
            exact_dup_rate,

        "near_duplicate_pairs":
            near_pairs,

        "near_duplicate_comparisons":
            comparisons,

        "rejection_reasons":
            dict(
                reasons
            ),
    }

    with (
        out
        / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Manual audit set:
    # richest accepted + random accepted + rejected.
    accepted_sorted = sorted(
        accepted,
        key=lambda x: (
            x.get(
                "meta",
                {},
            ).get(
                "richness",
                0,
            ),
            x.get(
                "meta",
                {},
            ).get(
                "words",
                0,
            ),
        ),
        reverse=True,
    )

    rng = random.Random(
        20260927
    )

    manual = (
        accepted_sorted[:30]
        + rng.sample(
            accepted,
            min(
                30,
                len(
                    accepted
                ),
            ),
        )
        + rejected[:40]
    )

    with (
        out
        / "manual_audit.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in manual:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
