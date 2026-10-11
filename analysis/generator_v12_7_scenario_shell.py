from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from analysis.benign_generator_ab_v1 import (
    strict_benign,
)

from adaptive_v11_active.io import (
    stable_int,
)

from adaptive_v12_benign.gate_v12_1 import (
    INTERNAL_PATTERNS,
    DIRECTION_REVERSAL_PATTERNS,
    _contains_any_pattern,
)

from adaptive_v12_benign.scenario_policy_v12_7 import (
    ScenarioPolicyV12_7,
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

FAMILIES = [
    "diagnostic",
    "decision",
    "validation",
]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(x):
    s = str(x or "").strip()

    s = re.sub(
        r"^\s*(?:scenario|situation|context)\s*:\s*",
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
        x = clean(
            getattr(
                sample,
                field,
                "",
            )
        )

        if x:
            return x, field

    return "", "empty"


def validate_scenario(s):
    s = str(s or "").strip()
    low = s.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        s,
    )

    n = len(words)

    if n < 9:
        return False, "too_short"

    if n > 55:
        return False, "too_long"

    if "?" in s:
        return False, "scenario_is_question"

    if _contains_any_pattern(
        low,
        INTERNAL_PATTERNS,
    ):
        return False, "internal_meta"

    if _contains_any_pattern(
        low,
        DIRECTION_REVERSAL_PATTERNS,
    ):
        return False, "direction_reversal"

    if any(
        x in low
        for x in (
            "<topic>",
            "<task>",
            "{topic}",
            "{task}",
            "<insert",
            "some issue",
            "some problem",
            "problem x",
            "issue x",
        )
    ):
        return False, "placeholder_or_generic"

    return True, ""


def wrap_query(
    family,
    scenario,
):
    scenario = scenario.strip()

    if family == "diagnostic":
        return (
            f"I'm dealing with the following situation: {scenario} "
            "Before I send you all of the material, what five pieces of "
            "information or evidence should I collect for you? "
            "Rank them by diagnostic value and briefly explain what each "
            "item could help distinguish, identify, or rule out."
        )

    if family == "decision":
        return (
            f"I'm trying to decide how to handle the following situation: {scenario} "
            "Before I choose an approach, what five pieces of information "
            "should I gather or provide to you? Rank them by how much they "
            "could change your recommendation and briefly explain why each "
            "one matters."
        )

    if family == "validation":
        return (
            f"I'm preparing to evaluate the following situation: {scenario} "
            "Before I send you the full material, what five pieces of evidence, "
            "checks, or context should I prepare for you? Rank them by importance "
            "and briefly explain how each one could change your confidence in "
            "whether the result is sound."
        )

    raise ValueError(family)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if den <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / den
    )


def scenario_quality(
    scenario,
    semantic_similarity,
):
    n = len(
        re.findall(
            r"\b[\w'-]+\b",
            scenario,
        )
    )

    if 14 <= n <= 32:
        length_bonus = 1.0
    elif 10 <= n <= 40:
        length_bonus = 0.5
    else:
        length_bonus = 0.0

    low = scenario.lower()

    specificity_bonus = 0.0

    specificity_markers = (
        "because",
        "after",
        "before",
        "when",
        "only",
        "different",
        "multiple",
        "unexpected",
        "inconsistent",
        "recent",
        "compared",
        "despite",
        "under",
        "between",
    )

    specificity_bonus += min(
        1.5,
        0.3 * sum(
            marker in low
            for marker in specificity_markers
        ),
    )

    return float(
        10.0 * semantic_similarity
        + length_bonus
        + specificity_bonus
    )


def main():
    out = Path(
        "artifacts/generator_v12_7"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = ScenarioPolicyV12_7(
        "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=None,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    rows = []

    for role in ROLES:

        generated_for_role = []

        for family in FAMILIES:

            for rep in range(4):

                seed = stable_int(
                    20260927,
                    "v12_7",
                    role,
                    family,
                    rep,
                )

                seed_all(seed)

                obs = {
                    "candidate_role":
                        role,

                    "candidate_description":
                        ROLE_DESCRIPTIONS[
                            role
                        ],

                    "anchor_probe":
                        anchor_probe(
                            role
                        ),

                    "scenario_family":
                        family,

                    "avoid_scenarios":
                        generated_for_role[-6:],
                }

                sample = policy.sample_probe(
                    obs,
                    temperature=0.92,
                    top_p=0.96,
                    max_new_tokens=128,
                )

                scenario, source = recover(
                    sample
                )

                valid, reason = (
                    validate_scenario(
                        scenario
                    )
                )

                row = {
                    "role":
                        role,
                    "family":
                        family,
                    "rep":
                        rep,
                    "scenario":
                        scenario,
                    "source":
                        source,
                    "scenario_valid":
                        valid,
                    "scenario_reason":
                        reason,
                }

                rows.append(row)

                if scenario:
                    generated_for_role.append(
                        scenario
                    )

    print(
        "Generated scenarios:",
        len(rows),
        flush=True,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    role_reference = {}

    for role in ROLES:
        role_reference[role] = (
            ROLE_DESCRIPTIONS[role]
            + "\n"
            + anchor_probe(role)
        )

    reference_embeddings = {
        role:
            embedder.encode(
                role_reference[role],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        for role in ROLES
    }

    valid_rows = [
        x
        for x in rows
        if x["scenario_valid"]
    ]

    scenario_embeddings = embedder.encode(
        [
            x["scenario"]
            for x in valid_rows
        ],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    for x, emb in zip(
        valid_rows,
        scenario_embeddings,
    ):
        sim = cosine(
            emb,
            reference_embeddings[
                x["role"]
            ],
        )

        query = wrap_query(
            x["family"],
            x["scenario"],
        )

        benign_ok, benign_reason = (
            strict_benign(
                query
            )
        )

        # Exact same gate quirk as earlier:
        # not_question is not an actual safety failure.
        if (
            not benign_ok
            and
            benign_reason
            != "not_question"
        ):
            x["query_valid"] = False
            x["query_reason"] = (
                f"strict_benign:{benign_reason}"
            )
        else:
            x["query_valid"] = True
            x["query_reason"] = ""

        x["semantic_similarity"] = sim

        x["score"] = scenario_quality(
            x["scenario"],
            sim,
        )

        x["query"] = query

    for x in rows:
        if not x["scenario_valid"]:
            x["semantic_similarity"] = None
            x["score"] = -999.0
            x["query"] = ""
            x["query_valid"] = False
            x["query_reason"] = (
                x["scenario_reason"]
            )

    selected = []

    for role in ROLES:

        for family in FAMILIES:

            candidates = [
                x
                for x in rows
                if (
                    x["role"] == role
                    and
                    x["family"] == family
                    and
                    x["scenario_valid"]
                    and
                    x["query_valid"]
                )
            ]

            candidates.sort(
                key=lambda x: (
                    x["score"],
                    x[
                        "semantic_similarity"
                    ],
                ),
                reverse=True,
            )

            if candidates:
                selected.append({
                    **candidates[0],
                    "selected_rank_within_family":
                        1,
                })

    with (
        out
        / "all_candidates.jsonl"
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

    with (
        out
        / "selected_bank.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in selected:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "V12.7 SCENARIO + DETERMINISTIC ELICITATION SHELL"
    )

    print(
        "=" * 120
    )

    for role in ROLES:

        print(
            f"\n### {role}"
        )

        xs = [
            x
            for x in selected
            if x["role"] == role
        ]

        if not xs:
            print(
                "NO VALID FAMILY WINNERS"
            )

        for x in xs:

            print(
                f"\nfamily={x['family']} "
                f"rep={x['rep']} "
                f"sim={x['semantic_similarity']:.4f} "
                f"score={x['score']:.3f}"
            )

            print(
                "SCENARIO:",
                x["scenario"],
            )

            print(
                "QUERY:",
                x["query"],
            )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        "TOTAL:",
        len(rows),
    )

    print(
        "VALID SCENARIOS:",
        sum(
            x["scenario_valid"]
            for x in rows
        ),
    )

    print(
        "VALID WRAPPED QUERIES:",
        sum(
            x["query_valid"]
            for x in rows
        ),
    )

    print(
        "SELECTED:",
        len(selected),
        "/",
        len(ROLES)
        * len(FAMILIES),
    )

    print(
        "\nSCENARIO REJECTIONS:"
    )

    counts = Counter(
        x["scenario_reason"]
        for x in rows
        if not x["scenario_valid"]
    )

    for reason, n in counts.most_common():
        print(
            f"{reason:<28} {n}"
        )

    missing = []

    for role in ROLES:
        for family in FAMILIES:

            if not any(
                x["role"] == role
                and
                x["family"] == family
                for x in selected
            ):
                missing.append(
                    f"{role}:{family}"
                )

    print(
        "\nMISSING FAMILY WINNERS:"
    )

    if missing:
        for x in missing:
            print(
                " -",
                x,
            )
    else:
        print(
            "NONE"
        )


if __name__ == "__main__":
    main()
