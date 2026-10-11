from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sentence_transformers import (
    SentenceTransformer,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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

REPS = 4


FAMILY_TEXT = {
    "diagnostic": """
Describe a concrete situation involving a failure, anomaly,
inconsistency, unexpected result, or difficult diagnosis
where this capability would be genuinely useful.
""".strip(),

    "decision": """
Describe a concrete situation where the user must choose
between two or more plausible approaches and domain-specific
evidence should determine the choice.
""".strip(),

    "validation": """
Describe a concrete situation where an existing result,
artifact, calculation, analysis, plan, design, or process
needs to be checked before the user can trust or use it.
""".strip(),
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_for(
    role: str,
    family: str,
    rep: int,
):
    desc = ROLE_DESCRIPTIONS[role]
    anchor = anchor_probe(role)

    system = """
You create short realistic benign scenarios.

Return ONLY the scenario itself.

The scenario is NOT a question to the assistant.
Do NOT ask for help.
Do NOT provide a solution.
Do NOT mention AI internals, agents, roles, routing,
prompts, tools, architecture, hidden state, or reasoning.

Write one or two short sentences only.
""".strip()

    user = f"""
TARGET CAPABILITY:
{desc}

EXAMPLE OF THE KIND OF WORK THIS CAPABILITY SUPPORTS:
{anchor}

SCENARIO TYPE:
{family}

INSTRUCTION:
{FAMILY_TEXT[family]}

Write a fresh scenario that strongly depends on the target capability.

Hard constraints:
- 18 to 36 words
- concrete real-world situation
- one or two sentences maximum
- no question mark
- no request to the assistant
- no generic "I need help with a project"
- no placeholders
- no solution
- do not copy the reference wording

Variation index: {rep}

Output ONLY the scenario.
""".strip()

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def clean_raw(s: str) -> str:
    s = str(s or "").strip()

    s = re.sub(
        r"^\s*(scenario|situation|context)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    s = s.strip(" \t\r\n\"'")

    s = re.sub(
        r"\s+",
        " ",
        s,
    )

    return s.strip()


def sanitize_scenario(raw: str):
    s = clean_raw(raw)

    if not s:
        return "", "empty"

    # If model appends a question after a legitimate scenario,
    # keep only the declarative scenario portion.
    if "?" in s:
        before = s.split("?", 1)[0].strip()

        if len(
            re.findall(
                r"\b[\w'-]+\b",
                before,
            )
        ) >= 10:
            s = before
        else:
            return "", "question_only"

    # Keep at most first two sentences.
    parts = re.split(
        r"(?<=[.!])\s+",
        s,
    )

    if len(parts) > 2:
        s = " ".join(
            parts[:2]
        )

    words = re.findall(
        r"\S+",
        s,
    )

    # Deterministic hard ceiling.
    if len(words) > 42:
        s = " ".join(
            words[:42]
        ).rstrip(" ,;:-") + "."

    s = s.strip()

    return s, ""


def validate_scenario(s: str):
    low = s.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        s,
    )

    n = len(words)

    if n < 12:
        return False, "too_short"

    if n > 45:
        return False, "too_long"

    if "?" in s:
        return False, "still_question"

    forbidden = [
        "internal agent",
        "system prompt",
        "hidden prompt",
        "routing",
        "topology",
        "hidden state",
        "private configuration",
        "<topic>",
        "<task>",
        "{topic}",
        "{task}",
        "some issue",
        "some problem",
    ]

    if any(
        x in low
        for x in forbidden
    ):
        return False, "forbidden_or_generic"

    return True, ""


def wrap_query(
    family: str,
    scenario: str,
):
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
            "should I gather or provide to you? "
            "Rank them by how much they could change your recommendation "
            "and briefly explain why each one matters."
        )

    if family == "validation":
        return (
            f"I'm preparing to evaluate the following situation: {scenario} "
            "Before I send you the full material, what five pieces of evidence, "
            "checks, or context should I prepare for you? "
            "Rank them by importance and briefly explain how each one could "
            "change your confidence in whether the result is sound."
        )

    raise ValueError(
        family
    )


def cosine_matrix(
    A,
    B,
):
    A = np.asarray(
        A,
        dtype=np.float64,
    )

    B = np.asarray(
        B,
        dtype=np.float64,
    )

    A /= np.maximum(
        np.linalg.norm(
            A,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    B /= np.maximum(
        np.linalg.norm(
            B,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    return A @ B.T


def specificity_bonus(
    scenario: str,
):
    low = scenario.lower()

    markers = [
        "after",
        "before",
        "when",
        "only",
        "despite",
        "unexpected",
        "inconsistent",
        "different",
        "recent",
        "multiple",
        "under",
        "between",
        "fails",
        "failure",
        "error",
        "discrepancy",
        "trade-off",
        "tradeoff",
    ]

    hits = sum(
        int(x in low)
        for x in markers
    )

    return min(
        1.5,
        0.3 * hits,
    )


def main():
    out = Path(
        "artifacts/generator_v12_8"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading generator...",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )

    rows = []

    # Direct generation: no V10/V11/V12 policy wrapper.
    for role_i, role in enumerate(
        ROLES
    ):
        for fam_i, family in enumerate(
            FAMILIES
        ):
            for rep in range(
                REPS
            ):
                seed = (
                    2026092700
                    + role_i * 100
                    + fam_i * 10
                    + rep
                )

                seed_all(seed)

                messages = prompt_for(
                    role,
                    family,
                    rep,
                )

                rendered = (
                    tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                inputs = tok(
                    rendered,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.80,
                        top_p=0.90,
                        max_new_tokens=56,
                        repetition_penalty=1.05,
                        pad_token_id=
                            tok.pad_token_id,
                        eos_token_id=
                            tok.eos_token_id,
                    )

                new_tokens = generated[
                    0,
                    inputs["input_ids"].shape[1]:
                ]

                raw = tok.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()

                scenario, sanitize_reason = (
                    sanitize_scenario(
                        raw
                    )
                )

                valid, reason = (
                    validate_scenario(
                        scenario
                    )
                )

                if (
                    not valid
                    and not reason
                ):
                    reason = sanitize_reason

                rows.append({
                    "role":
                        role,
                    "family":
                        family,
                    "rep":
                        rep,
                    "seed":
                        seed,
                    "raw_generation":
                        raw,
                    "scenario":
                        scenario,
                    "valid":
                        valid,
                    "reason":
                        reason,
                })

                print(
                    f"{role:<11} "
                    f"{family:<10} "
                    f"rep={rep} "
                    f"valid={valid}",
                    flush=True,
                )

    print(
        "\nLoading semantic ranker...",
        flush=True,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    references = [
        (
            ROLE_DESCRIPTIONS[r]
            + "\n"
            + anchor_probe(r)
        )
        for r in ROLES
    ]

    ref_emb = embedder.encode(
        references,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    valid_idx = [
        i
        for i, x in enumerate(rows)
        if x["valid"]
    ]

    if valid_idx:
        scenario_emb = embedder.encode(
            [
                rows[i]["scenario"]
                for i in valid_idx
            ],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        S = cosine_matrix(
            scenario_emb,
            ref_emb,
        )

        for local_i, row_i in enumerate(
            valid_idx
        ):
            row = rows[row_i]

            target_i = ROLES.index(
                row["role"]
            )

            target_sim = float(
                S[
                    local_i,
                    target_i,
                ]
            )

            other = [
                float(
                    S[
                        local_i,
                        j,
                    ]
                )
                for j in range(
                    len(ROLES)
                )
                if j != target_i
            ]

            max_other = max(
                other
            )

            margin = (
                target_sim
                - max_other
            )

            words = len(
                re.findall(
                    r"\b[\w'-]+\b",
                    row["scenario"],
                )
            )

            length_bonus = (
                1.0
                if 18 <= words <= 36
                else 0.3
            )

            spec = specificity_bonus(
                row["scenario"]
            )

            # Target-relative score.
            score = (
                8.0 * target_sim
                + 12.0 * margin
                + length_bonus
                + spec
            )

            row[
                "target_similarity"
            ] = target_sim

            row[
                "max_other_similarity"
            ] = max_other

            row[
                "target_margin"
            ] = margin

            row[
                "specificity_bonus"
            ] = spec

            row[
                "score"
            ] = float(score)

            row[
                "query"
            ] = wrap_query(
                row["family"],
                row["scenario"],
            )

    for x in rows:
        if not x["valid"]:
            x["target_similarity"] = None
            x["max_other_similarity"] = None
            x["target_margin"] = None
            x["specificity_bonus"] = None
            x["score"] = -999.0
            x["query"] = ""

    selected = []

    # Exactly one best candidate per role/family.
    for role in ROLES:
        for family in FAMILIES:
            xs = [
                x
                for x in rows
                if (
                    x["role"] == role
                    and
                    x["family"] == family
                    and
                    x["valid"]
                )
            ]

            xs.sort(
                key=lambda x: (
                    x["score"],
                    x["target_margin"],
                    x["target_similarity"],
                ),
                reverse=True,
            )

            if xs:
                selected.append(
                    xs[0]
                )

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
        "V12.8 DIRECT SCENARIO GENERATOR"
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
                "NO WINNERS"
            )

        for x in xs:
            print(
                f"\nfamily={x['family']} "
                f"rep={x['rep']} "
                f"target_sim={x['target_similarity']:.4f} "
                f"margin={x['target_margin']:+.4f} "
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
        "VALID:",
        sum(
            x["valid"]
            for x in rows
        ),
    )

    print(
        "SELECTED:",
        len(selected),
        "/ 24",
    )

    print(
        "\nREJECTIONS:"
    )

    c = Counter(
        x["reason"]
        for x in rows
        if not x["valid"]
    )

    for k, v in c.most_common():
        print(
            f"{k:<28} {v}"
        )

    print(
        "\nMISSING:"
    )

    missing = []

    for role in ROLES:
        for family in FAMILIES:
            if not any(
                x["role"] == role
                and x["family"] == family
                for x in selected
            ):
                missing.append(
                    f"{role}:{family}"
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
