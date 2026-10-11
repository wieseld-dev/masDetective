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

from adaptive_v12_benign.gate_v12_1 import (
    strict_rich_benign_gate,
)


def norm(q):
    return re.sub(
        r"\s+",
        " ",
        str(q).lower().strip(),
    )


def clean_generated_text(x):
    s = str(x or "").strip()

    if not s:
        return ""

    # Remove common model wrappers, but do not rewrite content.
    s = re.sub(
        r"^\s*(?:question|query|probe)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    # Strip simple surrounding quotation marks.
    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in {"'", '"'}
    ):
        s = s[1:-1].strip()

    return s


def recover_query(sample):
    """
    Legacy sample_probe can mark a rich generation invalid under the
    old one-question normalizer. In that case sample.query may be empty
    even though raw_completion contains usable text.

    V12.1 recovers text, then applies its OWN hard benign gate.
    """
    q = clean_generated_text(
        getattr(sample, "query", "")
    )

    if q:
        return q, "sample.query"

    completion = clean_generated_text(
        getattr(sample, "completion", "")
    )

    if completion:
        return completion, "sample.completion"

    raw = clean_generated_text(
        getattr(sample, "raw_completion", "")
    )

    if raw:
        return raw, "sample.raw_completion"

    return "", "empty"


def write_jsonl(path, rows):
    with Path(path).open(
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
        default=100,
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
            "artifacts/"
            "generator_v12_1/"
            "smoke100"
        ),
    )

    a = ap.parse_args()

    out = Path(a.out)

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

    for sample_i in range(a.n):
        raw_state = states[
            sample_i
            % len(states)
        ]

        assert_policy_safe(
            raw_state
        )

        st = (
            PublicBeliefState
            .from_public_dict(
                raw_state
            )
        )

        j = (
            sample_i
            % a.group_size
        )

        seed = stable_int(
            20260927,
            "v12_1_audit",
            st.session_id,
            sample_i,
        )

        random.seed(seed)
        torch.manual_seed(seed)

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

        q, recovered_from = (
            recover_query(sample)
        )

        ok, reason, meta = (
            strict_rich_benign_gate(q)
        )

        row = {
            "i": sample_i,
            "session_id":
                st.session_id,
            "candidate_index": j,
            "family": j % 4,

            "query": q,

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

            "completion":
                str(
                    getattr(
                        sample,
                        "completion",
                        "",
                    )
                    or ""
                ),

            "raw_completion":
                str(
                    getattr(
                        sample,
                        "raw_completion",
                        "",
                    )
                    or ""
                ),

            "gate_valid":
                bool(ok),

            "reason":
                reason,

            "meta":
                meta,
        }

        rows.append(row)

        if (
            len(rows) % 20
            == 0
        ):
            print(
                f"[audit-v12.1] "
                f"{len(rows)}/{a.n}",
                flush=True,
            )

    accepted = [
        x for x in rows
        if x["gate_valid"]
    ]

    rejected = [
        x for x in rows
        if not x["gate_valid"]
    ]

    bands = Counter(
        x.get(
            "meta",
            {},
        ).get(
            "quality_band",
            "n/a",
        )
        for x in accepted
    )

    recovery = Counter(
        x["recovered_from"]
        for x in rows
    )

    reasons = Counter(
        x["reason"]
        for x in rejected
    )

    legacy_invalid = [
        x for x in rows
        if not x["legacy_sample_valid"]
    ]

    salvaged_legacy_invalid = [
        x for x in legacy_invalid
        if x["gate_valid"]
    ]

    word_counts = [
        x["meta"].get(
            "words",
            0,
        )
        for x in accepted
    ]

    richness_values = [
        x["meta"].get(
            "richness",
            0,
        )
        for x in accepted
    ]

    # Duplicate diagnostics over hard-valid candidates.
    normalized = [
        norm(x["query"])
        for x in accepted
    ]

    exact_dup_rate = (
        1.0
        - len(set(normalized))
        / max(
            1,
            len(normalized),
        )
    )

    near_pairs = 0
    comparisons = 0

    for i in range(
        len(accepted)
    ):
        for j in range(
            max(
                0,
                i - 40,
            ),
            i,
        ):
            comparisons += 1

            sim = (
                normalized_word_jaccard(
                    accepted[i]["query"],
                    accepted[j]["query"],
                )
            )

            if sim >= 0.72:
                near_pairs += 1

    summary = {
        "experiment":
            "generator_v12_1_offline_audit",

        "raw_candidates":
            len(rows),

        "hard_benign_accepted":
            len(accepted),

        "hard_benign_rejected":
            len(rejected),

        "hard_benign_pass_rate":
            len(accepted)
            / max(
                1,
                len(rows),
            ),

        "quality_bands":
            dict(bands),

        "recovered_from":
            dict(recovery),

        "legacy_invalid_count":
            len(legacy_invalid),

        "legacy_invalid_but_v12_1_salvaged":
            len(
                salvaged_legacy_invalid
            ),

        "rejection_reasons":
            dict(reasons),

        "median_words":
            float(
                np.median(
                    word_counts
                )
            )
            if word_counts
            else 0.0,

        "mean_richness":
            float(
                np.mean(
                    richness_values
                )
            )
            if richness_values
            else 0.0,

        "exact_duplicate_rate":
            float(
                exact_dup_rate
            ),

        "near_duplicate_pairs":
            near_pairs,

        "near_duplicate_comparisons":
            comparisons,
    }

    write_jsonl(
        out / "rows.jsonl",
        rows,
    )

    write_jsonl(
        out / "accepted.jsonl",
        accepted,
    )

    write_jsonl(
        out / "rejected.jsonl",
        rejected,
    )

    with (
        out / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
