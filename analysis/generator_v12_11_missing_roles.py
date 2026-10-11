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
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


MODEL = "Qwen/Qwen2.5-3B-Instruct"

ROLES = [
    "code",
    "operations",
]

FAMILIES = [
    "intake",
    "validation",
]

REPS = 8

MIN_TARGET_SIM = 0.30
MIN_MARGIN = 0.05


FAMILY_TEXT = {
    "intake": """
Describe a concrete situation where the user has a legitimate problem
or goal in this exact domain but has not yet provided enough material
for someone to work on it well.

Describe ONLY the situation and goal.
Do not say what information should be provided.
""",

    "validation": """
Describe a concrete situation where the user already has a result,
implementation, process, or change in this exact domain and wants to
determine whether it is correct, reliable, or ready to use.

Describe ONLY the situation and goal.
Do not say which checks or evidence should be used.
""",
}


FORBIDDEN_COMMON = [
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bhelp me\b",
    r"\bhelp us\b",
    r"\byour help\b",
    r"\bguide me\b",
    r"\byour guidance\b",
    r"\btell me\b",
    r"\bshow me\b",

    r"\bwhat should\b",
    r"\bwhat information\b",
    r"\bwhat details\b",
    r"\bwhat steps\b",
    r"\bhow should\b",
    r"\bhow can\b",

    r"\bneed to collect\b",
    r"\bneeds to collect\b",
    r"\bneed to gather\b",
    r"\bneeds to gather\b",
    r"\bshould collect\b",
    r"\bshould gather\b",
]


FORBIDDEN_BY_ROLE = {
    "code": [
        r"\blogs?\b",
        r"\berror messages?\b",
        r"\bstack traces?\b",
        r"\breproduction steps?\b",
        r"\benvironment details?\b",
        r"\bdebugger\b",
        r"\bbreakpoints?\b",
    ],

    "operations": [
        r"\blogs?\b",
        r"\bmaintenance records?\b",
        r"\bfailure history\b",
        r"\bincident records?\b",
        r"\bmonitoring metrics?\b",
        r"\bthroughput metrics?\b",
        r"\bdowntime records?\b",
    ],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_messages(
    role,
    family,
    rep,
):
    desc = ROLE_DESCRIPTIONS[role]
    anchor = anchor_probe(role)

    system = """
You generate short realistic benign situations.

Output ONLY the situation itself.

Do not ask the assistant anything.
Do not address an assistant.
Do not provide a solution.
Do not state what evidence, logs, measurements, records,
checks, or inputs should be collected.

The situation must stay tightly within the target capability.
Write one or two concise declarative sentences.
""".strip()

    user = f"""
TARGET CAPABILITY:
{desc}

REFERENCE TASK:
{anchor}

SCENARIO TYPE:
{family}

{FAMILY_TEXT[family]}

Hard requirements:
- 16-34 words
- one or two declarative sentences
- concrete and realistic
- no question mark
- no request for assistance
- no recommended next step
- no list of evidence or information to collect
- no placeholders
- preserve the type of work in the reference task

Variation: {rep}

Output ONLY the situation.
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
        r"^\s*(scenario|situation|context)\s*:\s*",
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

    parts = re.split(
        r"(?<=[.!])\s+",
        s,
    )

    if len(parts) > 2:
        s = " ".join(
            parts[:2]
        )

    words = s.split()

    if len(words) > 38:
        s = (
            " ".join(words[:38])
            .rstrip(" ,;:-")
            + "."
        )

    return s.strip()


def matches(patterns, text):
    return any(
        re.search(
            p,
            text,
            flags=re.I,
        )
        for p in patterns
    )


def surface_gate(
    role,
    scenario,
):
    low = scenario.lower()

    n = len(
        re.findall(
            r"\b[\w'-]+\b",
            scenario,
        )
    )

    if n < 12:
        return False, "too_short"

    if n > 42:
        return False, "too_long"

    if "?" in scenario:
        return False, "question"

    if matches(
        FORBIDDEN_COMMON,
        low,
    ):
        return False, "assistant_or_leakage"

    if matches(
        FORBIDDEN_BY_ROLE[role],
        low,
    ):
        return False, "evidence_leakage"

    forbidden = [
        "system prompt",
        "hidden prompt",
        "internal agent",
        "routing",
        "topology",
        "hidden state",
        "<topic>",
        "<task>",
        "{topic}",
        "{task}",
    ]

    if any(
        x in low
        for x in forbidden
    ):
        return False, "forbidden_meta"

    return True, ""


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


def wrap(
    family,
    scenario,
):
    if family == "intake":
        return (
            f"I'm working on the following situation: {scenario} "
            "Before I send you all of the material, what five pieces of "
            "information, context, examples, or artifacts should I provide "
            "so you can help me do this well? "
            "Rank them by usefulness and briefly explain why each one matters."
        )

    return (
        f"I'm working on the following situation: {scenario} "
        "Before I send you the complete result, what five pieces of evidence, "
        "checks, examples, or context should I prepare for you so you can "
        "evaluate it properly? Rank them by importance and briefly explain "
        "how each one would affect your confidence in the result."
    )


def main():

    out = Path(
        "artifacts/generator_v12_11"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Qwen 3B...",
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
                    2026093110
                    + ri * 100
                    + fi * 10
                    + rep
                )

                seed_all(seed)

                messages = make_messages(
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
                        temperature=0.78,
                        top_p=0.90,
                        max_new_tokens=48,
                        repetition_penalty=1.05,
                        pad_token_id=
                            tok.pad_token_id,
                        eos_token_id=
                            tok.eos_token_id,
                    )

                new_tokens = y[
                    0,
                    inputs[
                        "input_ids"
                    ].shape[1]:
                ]

                raw = tok.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )

                scenario = clean(
                    raw
                )

                ok, reason = surface_gate(
                    role,
                    scenario,
                )

                rows.append({
                    "role":
                        role,
                    "family":
                        family,
                    "rep":
                        rep,
                    "raw":
                        raw.strip(),
                    "scenario":
                        scenario,
                    "surface_valid":
                        ok,
                    "surface_reason":
                        reason,
                })

                print(
                    role,
                    family,
                    rep,
                    "surface=",
                    ok,
                    flush=True,
                )

    print(
        "\nSemantic scoring...",
        flush=True,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    # Compare against ALL 8 current target roles,
    # not merely code vs operations.
    all_roles = [
        "code",
        "writing",
        "math",
        "planning",
        "research",
        "data",
        "design",
        "operations",
    ]

    refs = [
        (
            ROLE_DESCRIPTIONS[r]
            + "\n"
            + anchor_probe(r)
        )
        for r in all_roles
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

    for li, row_i in enumerate(
        good_idx
    ):

        row = rows[row_i]

        ti = all_roles.index(
            row["role"]
        )

        target = float(
            sims[
                li,
                ti,
            ]
        )

        others = [
            (
                all_roles[j],
                float(
                    sims[
                        li,
                        j,
                    ]
                ),
            )
            for j in range(
                len(all_roles)
            )
            if j != ti
        ]

        other_role, other_sim = max(
            others,
            key=lambda x: x[1],
        )

        margin = (
            target
            - other_sim
        )

        eligible = (
            target
            >= MIN_TARGET_SIM
            and
            margin
            >= MIN_MARGIN
        )

        score = (
            10.0 * target
            + 15.0 * margin
        )

        row.update({
            "target_similarity":
                target,
            "best_other_role":
                other_role,
            "best_other_similarity":
                other_sim,
            "target_margin":
                margin,
            "eligible":
                eligible,
            "score":
                float(score),
            "query":
                wrap(
                    row["family"],
                    row["scenario"],
                ),
        })

    for row in rows:

        if not row["surface_valid"]:

            row.update({
                "target_similarity":
                    None,
                "best_other_role":
                    None,
                "best_other_similarity":
                    None,
                "target_margin":
                    None,
                "eligible":
                    False,
                "score":
                    -999.0,
                "query":
                    "",
            })

    selected = []

    for role in ROLES:

        xs = [
            x
            for x in rows
            if (
                x["role"] == role
                and
                x["eligible"]
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
        / "selected_missing.jsonl"
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
        "V12.11 TARGETED MISSING ROLES"
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
                "NO CLEAN WINNER"
            )

        for x in xs:

            print(
                f"family={x['family']} "
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
        "\nSUMMARY"
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
            x["eligible"]
            for x in rows
        ),
    )

    print(
        "SELECTED:",
        len(selected),
        "/ 2",
    )


if __name__ == "__main__":
    main()
