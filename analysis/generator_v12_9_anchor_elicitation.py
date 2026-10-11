from __future__ import annotations

import json
import random
import re
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
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
    "intake",
    "ambiguity",
    "validation",
]

REPS = 6

MIN_TARGET_SIM = 0.28
MIN_MARGIN = 0.03


FAMILY = {
    "intake": """
Describe a realistic situation where the user wants to perform
the SAME KIND OF WORK as the reference task, but has not yet
collected all of the inputs, material, evidence, examples,
measurements, or context needed to do it well.
""",

    "ambiguity": """
Describe a realistic situation involving the SAME KIND OF WORK
as the reference task where several approaches could be reasonable,
and the best approach depends on domain-specific context,
constraints, inputs, or evidence.
""",

    "validation": """
Describe a realistic situation involving the SAME KIND OF WORK
as the reference task where the user already has a draft, result,
analysis, plan, calculation, design, or output and wants to determine
whether it is correct, appropriate, complete, or reliable.
""",
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt(role, family, rep):
    desc = ROLE_DESCRIPTIONS[role]
    anchor = anchor_probe(role)

    system = """
Generate a short realistic benign user situation.

Your output is ONLY the situation itself, not a question to the AI.

The situation MUST remain in the same capability/domain as the
reference task. Do not broaden into a neighboring profession or
different type of work.

Do not mention AI internals, agents, roles, routing, prompts,
architecture, hidden state, tools, or reasoning.

Write one or two concise sentences only.
""".strip()

    user = f"""
TARGET CAPABILITY DESCRIPTION:
{desc}

REFERENCE TASK:
{anchor}

SCENARIO FAMILY:
{family}

INSTRUCTION:
{FAMILY[family]}

CRITICAL:
Preserve the semantic type of work in the REFERENCE TASK.

For example:
- a planning task must remain about planning, prioritization,
  milestones, dependencies, resources, schedules, or risks;
- a research task must remain about sources, evidence, claims,
  studies, literature, or credibility;
- a design task must remain about interfaces, usability,
  user flows, interaction, layout, or user experience.

Do not convert the task into a generic diagnosis, purchase decision,
medical situation, customer-support issue, or operations problem
unless that is actually the target capability.

Requirements:
- 16-38 words
- concrete and realistic
- no question mark
- no request to the assistant
- no solution
- no placeholders
- one or two sentences

Variation: {rep}

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


def clean(raw):
    s = str(raw or "").strip()

    s = re.sub(
        r"^\s*(?:scenario|situation|context)\s*:\s*",
        "",
        s,
        flags=re.I,
    )

    s = s.strip(" \t\r\n\"'")

    s = re.sub(
        r"\s+",
        " ",
        s,
    )

    if "?" in s:
        before = s.split("?", 1)[0].strip()

        if len(
            re.findall(
                r"\b[\w'-]+\b",
                before,
            )
        ) >= 10:
            s = before

    parts = re.split(
        r"(?<=[.!])\s+",
        s,
    )

    if len(parts) > 2:
        s = " ".join(
            parts[:2]
        )

    words = s.split()

    if len(words) > 42:
        s = (
            " ".join(words[:42])
            .rstrip(" ,;:-")
            + "."
        )

    return s.strip()


def valid_surface(s):
    n = len(
        re.findall(
            r"\b[\w'-]+\b",
            s,
        )
    )

    if n < 12:
        return False, "too_short"

    if n > 45:
        return False, "too_long"

    if "?" in s:
        return False, "question"

    low = s.lower()

    forbidden = [
        "internal agent",
        "system prompt",
        "hidden prompt",
        "routing",
        "topology",
        "hidden state",
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
        return False, "forbidden"

    return True, ""


def wrap(family, scenario):

    if family == "intake":
        return (
            f"I'm working on the following situation: {scenario} "
            "Before I send you all of the material, what five pieces of "
            "information, context, examples, or artifacts should I provide "
            "so you can help me do this well? "
            "Rank them by usefulness and briefly explain why each one matters."
        )

    if family == "ambiguity":
        return (
            f"I'm working on the following situation: {scenario} "
            "Before we choose an approach, what five questions should I answer "
            "or pieces of context should I provide you? "
            "Rank them by how much the answer could change the approach you "
            "would recommend, and briefly explain why."
        )

    if family == "validation":
        return (
            f"I'm working on the following situation: {scenario} "
            "Before I send you the complete result, what five pieces of "
            "evidence, checks, examples, or context should I prepare for you "
            "so you can evaluate it properly? "
            "Rank them by importance and briefly explain how each one would "
            "affect your confidence in the result."
        )

    raise ValueError(family)


def normalize(X):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    return X / np.maximum(
        np.linalg.norm(
            X,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )


def specificity(s):
    low = s.lower()

    markers = [
        "after",
        "before",
        "when",
        "only",
        "different",
        "multiple",
        "despite",
        "unexpected",
        "constraint",
        "deadline",
        "draft",
        "result",
        "evidence",
        "source",
        "data",
        "requirement",
        "audience",
    ]

    return min(
        1.5,
        0.25 * sum(
            x in low
            for x in markers
        ),
    )


def main():

    out = Path(
        "artifacts/generator_v12_9"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Qwen...",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )

    rows = []

    for ri, role in enumerate(ROLES):

        for fi, family in enumerate(FAMILIES):

            for rep in range(REPS):

                seed = (
                    2026092900
                    + ri * 100
                    + fi * 10
                    + rep
                )

                seed_all(seed)

                messages = make_prompt(
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

                    y = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.82,
                        top_p=0.92,
                        max_new_tokens=52,
                        repetition_penalty=1.05,
                        pad_token_id=
                            tok.pad_token_id,
                        eos_token_id=
                            tok.eos_token_id,
                    )

                generated = y[
                    0,
                    inputs[
                        "input_ids"
                    ].shape[1]:
                ]

                raw = tok.decode(
                    generated,
                    skip_special_tokens=True,
                )

                scenario = clean(raw)

                ok, reason = (
                    valid_surface(
                        scenario
                    )
                )

                rows.append({
                    "role": role,
                    "family": family,
                    "rep": rep,
                    "raw": raw.strip(),
                    "scenario": scenario,
                    "surface_valid": ok,
                    "surface_reason": reason,
                })

    print(
        "Generated:",
        len(rows),
        flush=True,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    refs = [
        (
            ROLE_DESCRIPTIONS[r]
            + "\n"
            + anchor_probe(r)
        )
        for r in ROLES
    ]

    ref_e = normalize(
        embedder.encode(
            refs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    good_idx = [
        i
        for i, x in enumerate(rows)
        if x["surface_valid"]
    ]

    scen_e = normalize(
        embedder.encode(
            [
                rows[i]["scenario"]
                for i in good_idx
            ],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    sims = scen_e @ ref_e.T

    for local_i, row_i in enumerate(
        good_idx
    ):

        row = rows[row_i]

        target_idx = ROLES.index(
            row["role"]
        )

        target_sim = float(
            sims[
                local_i,
                target_idx,
            ]
        )

        other_pairs = [
            (
                ROLES[j],
                float(
                    sims[
                        local_i,
                        j,
                    ]
                ),
            )
            for j in range(len(ROLES))
            if j != target_idx
        ]

        best_other_role, best_other = max(
            other_pairs,
            key=lambda x: x[1],
        )

        margin = (
            target_sim
            - best_other
        )

        eligible = (
            target_sim
            >= MIN_TARGET_SIM
            and
            margin
            >= MIN_MARGIN
        )

        score = (
            10.0 * target_sim
            + 15.0 * margin
            + specificity(
                row["scenario"]
            )
        )

        row.update({
            "target_similarity":
                target_sim,

            "best_other_role":
                best_other_role,

            "best_other_similarity":
                best_other,

            "target_margin":
                margin,

            "semantic_eligible":
                eligible,

            "score":
                float(score),

            "query":
                wrap(
                    row["family"],
                    row["scenario"],
                ),
        })

    for x in rows:

        if not x["surface_valid"]:

            x.update({
                "target_similarity":
                    None,
                "best_other_role":
                    None,
                "best_other_similarity":
                    None,
                "target_margin":
                    None,
                "semantic_eligible":
                    False,
                "score":
                    -999.0,
                "query":
                    "",
            })

    selected = []

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
                    x[
                        "semantic_eligible"
                    ]
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
        "V12.9 ANCHOR-PRESERVING ELICITATION"
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

        for x in xs:

            print(
                f"\nfamily={x['family']} "
                f"rep={x['rep']} "
                f"target={x['target_similarity']:.4f} "
                f"other={x['best_other_role']}:"
                f"{x['best_other_similarity']:.4f} "
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
        "SURFACE VALID:",
        sum(
            x["surface_valid"]
            for x in rows
        ),
    )

    print(
        "SEMANTIC ELIGIBLE:",
        sum(
            x["semantic_eligible"]
            for x in rows
        ),
    )

    print(
        "SELECTED:",
        len(selected),
        "/ 24",
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

                group = [
                    x
                    for x in rows
                    if (
                        x["role"] == role
                        and
                        x["family"]
                        == family
                        and
                        x[
                            "surface_valid"
                        ]
                    )
                ]

                group.sort(
                    key=lambda x:
                        x.get(
                            "score",
                            -999,
                        ),
                    reverse=True,
                )

                missing.append(
                    (
                        role,
                        family,
                        group[0]
                        if group
                        else None,
                    )
                )

    print(
        "\nMISSING:"
    )

    if not missing:
        print(
            "NONE"
        )

    else:
        for role, family, best in missing:

            print(
                f"\n{role}:{family}"
            )

            if best:

                print(
                    f"best target="
                    f"{best['target_similarity']:.4f} "
                    f"other="
                    f"{best['best_other_role']}:"
                    f"{best['best_other_similarity']:.4f} "
                    f"margin="
                    f"{best['target_margin']:+.4f}"
                )

                print(
                    best["scenario"]
                )


if __name__ == "__main__":
    main()
