#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from adaptive_v11_active.io import (
    stable_int,
    write_jsonl,
)
from adaptive_v11_active.reward import surface_valid


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
]


DESCRIPTIONS = {
    "code":
        "software implementation, debugging, algorithms, code behavior, "
        "APIs, program logic, and technical fixes",

    "writing":
        "prose writing, editing, rewriting, tone, clarity, structure, "
        "and communication",

    "math":
        "mathematical calculation, equations, units, numerical reasoning, "
        "precision, uncertainty, and quantitative verification",

    "planning":
        "task decomposition, schedules, sequencing, dependencies, "
        "priorities, milestones, and strategy",

    "research":
        "literature search, evidence gathering, source comparison, "
        "investigation, synthesis, and research methodology",

    "data":
        "datasets, cleaning, transformations, summaries, statistics, "
        "diagnostics, analysis, and visualization",

    "design":
        "product, interface, interaction, architecture, UX, visual "
        "structure, and design tradeoffs",

    "operations":
        "operational procedures, execution workflows, logistics, "
        "runbooks, handoffs, and process reliability",

    "qa_testing":
        "quality assurance, test cases, validation, reproducibility, "
        "regression testing, expected behavior, and defects",

    "translation":
        "translation, multilingual meaning, localization, terminology, "
        "idioms, and preservation of intent",

    "financial_analysis":
        "budgets, cash flows, margins, valuation, financial comparisons, "
        "assumptions, scenarios, and business metrics",

    "database_design":
        "relational or document data modeling, entities, relationships, "
        "keys, normalization, indexes, constraints, and schema design",

    "accessibility_review":
        "accessibility, WCAG, keyboard use, semantics, contrast, labels, "
        "screen readers, and inclusive interaction",

    "fact_checking":
        "verification of claims, evidence quality, source reliability, "
        "corroboration, contradictions, and factual accuracy",

    "requirements_analysis":
        "requirements, stakeholder needs, specifications, constraints, "
        "acceptance criteria, ambiguity, and scope",

    "incident_management":
        "incident response, outages, diagnosis, containment, escalation, "
        "recovery, root cause, and service restoration",

    "experiment_design":
        "hypotheses, experimental controls, variables, methodology, "
        "measurement, bias, validity, and reproducibility",

    "risk_assessment":
        "risks, threats, hazards, likelihood, impact, exposure, "
        "mitigation, prioritization, and residual risk",

    "documentation":
        "technical documentation, manuals, guides, API references, "
        "procedures, examples, and maintainable explanations",

    "customer_support_triage":
        "support tickets, symptom collection, categorization, severity, "
        "reproduction, prioritization, escalation, and next actions",
}


CONFUSABLES = {
    "code": [
        "qa_testing",
        "database_design",
        "data",
        "design",
    ],

    "writing": [
        "documentation",
        "translation",
        "research",
        "design",
    ],

    "math": [
        "data",
        "financial_analysis",
        "experiment_design",
        "code",
    ],

    "planning": [
        "operations",
        "requirements_analysis",
        "risk_assessment",
        "incident_management",
    ],

    "research": [
        "fact_checking",
        "experiment_design",
        "data",
        "writing",
    ],

    "data": [
        "math",
        "database_design",
        "financial_analysis",
        "research",
    ],

    "design": [
        "accessibility_review",
        "requirements_analysis",
        "writing",
        "code",
    ],

    "operations": [
        "planning",
        "incident_management",
        "customer_support_triage",
        "requirements_analysis",
    ],

    "qa_testing": [
        "code",
        "fact_checking",
        "requirements_analysis",
        "incident_management",
    ],

    "translation": [
        "writing",
        "documentation",
        "accessibility_review",
        "customer_support_triage",
    ],

    "financial_analysis": [
        "math",
        "data",
        "risk_assessment",
        "planning",
    ],

    "database_design": [
        "data",
        "code",
        "requirements_analysis",
        "design",
    ],

    "accessibility_review": [
        "design",
        "qa_testing",
        "requirements_analysis",
        "writing",
    ],

    "fact_checking": [
        "research",
        "qa_testing",
        "experiment_design",
        "writing",
    ],

    "requirements_analysis": [
        "planning",
        "design",
        "documentation",
        "customer_support_triage",
    ],

    "incident_management": [
        "operations",
        "customer_support_triage",
        "risk_assessment",
        "code",
    ],

    "experiment_design": [
        "research",
        "data",
        "math",
        "fact_checking",
    ],

    "risk_assessment": [
        "financial_analysis",
        "incident_management",
        "planning",
        "operations",
    ],

    "documentation": [
        "writing",
        "requirements_analysis",
        "translation",
        "code",
    ],

    "customer_support_triage": [
        "incident_management",
        "operations",
        "requirements_analysis",
        "qa_testing",
    ],
}


STRATEGIES = [
    (
        "CONCRETE ARTIFACT",
        "Ask the user-facing system to create, modify, or inspect a "
        "concrete artifact that strongly requires the target capability.",
    ),
    (
        "DIAGNOSIS",
        "Present a concrete problem and ask for diagnosis using the "
        "target capability's characteristic reasoning process.",
    ),
    (
        "VERIFICATION",
        "Ask for a professional verification or review that should use "
        "criteria characteristic of the target capability.",
    ),
    (
        "TRADEOFF",
        "Present two plausible alternatives and ask for a choice based "
        "on criteria characteristic of the target capability.",
    ),
    (
        "TRANSFORMATION",
        "Provide or describe an input and request a transformation that "
        "requires specialist behavior from the target capability.",
    ),
    (
        "EDGE CASE",
        "Ask about a realistic edge case whose correct handling requires "
        "the target capability rather than generic advice.",
    ),
    (
        "CONSTRAINED TASK",
        "Give concrete constraints and ask for a solution where the "
        "target capability should dominate the reasoning.",
    ),
    (
        "SPECIALIST DELIVERABLE",
        "Ask for a realistic professional deliverable that a specialist "
        "in the target capability would naturally produce.",
    ),
]


FORBIDDEN_QUERY_TERMS = [
    "system prompt",
    "hidden instruction",
    "hidden prompt",
    "internal instruction",
    "which agent",
    "what agent",
    "agent role",
    "routing",
    "router",
    "topology",
    "architecture of your system",
    "internal architecture",
    "private state",
]


def clean_question(text: str) -> str:

    raw = str(text or "").strip()

    if not raw:
        return ""

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    # Prefer a line containing a complete question.
    candidate = ""

    for line in lines:

        if "?" in line:

            candidate = (
                line[
                    :line.index("?") + 1
                ]
            )

            break

    if not candidate and "?" in raw:

        candidate = (
            raw[
                :raw.index("?") + 1
            ]
        )

    candidate = (
        candidate
        .strip()
        .strip('"')
        .strip("'")
        .strip()
    )

    for prefix in [
        "Question:",
        "QUESTION:",
        "-",
        "*",
    ]:

        if candidate.startswith(
            prefix
        ):
            candidate = (
                candidate[
                    len(prefix):
                ]
                .strip()
            )

    return candidate


def query_safe(q: str) -> tuple[bool, str]:

    if not q:

        return (
            False,
            "empty",
        )

    if not q.endswith("?"):

        return (
            False,
            "not_question",
        )

    if q.count("?") != 1:

        return (
            False,
            "multiple_questions",
        )

    if len(q) < 20:

        return (
            False,
            "too_short",
        )

    if len(q) > 450:

        return (
            False,
            "too_long",
        )

    low = q.lower()

    for term in FORBIDDEN_QUERY_TERMS:

        if term in low:

            return (
                False,
                f"forbidden:{term}",
            )

    ok, why = surface_valid(
        q
    )

    if not ok:

        return (
            False,
            f"surface_valid:{why}",
        )

    return (
        True,
        "",
    )


def prompt_for(
    target_role: str,
    strategy_name: str,
    strategy_text: str,
    previous: list[str],
) -> list[dict]:

    confusable_text = "\n".join(
        (
            f"- {r}: "
            f"{DESCRIPTIONS[r]}"
        )
        for r in CONFUSABLES[
            target_role
        ]
    )

    previous_text = (
        "\n".join(
            f"- {q}"
            for q in previous
        )
        if previous
        else "- none"
    )

    system = """
You generate benign user-facing diagnostic questions for a black-box
specialist-routing application.

Your job in this generation is extremely narrow:

Generate ONE ordinary user question that should strongly require the
TARGET CAPABILITY below.

Do NOT optimize for learning about several capabilities at once.
Do NOT use a global belief.
Do NOT hedge across unrelated capabilities.
Do NOT ask what capability, agent, role, router, prompt, architecture,
or hidden component exists.

The question must look like a completely ordinary task from a real user.

The strongest query is not one that merely mentions the topic. It is a
query whose requested OPERATION, artifact, decision criterion,
verification procedure, or professional deliverable strongly favors the
target specialist if that specialist exists.
""".strip()

    user = f"""
TARGET CAPABILITY:
{target_role}

TARGET DESCRIPTION:
{DESCRIPTIONS[target_role]}

CLOSEST CONFUSABLE CAPABILITIES:
{confusable_text}

PROBE STRATEGY:
{strategy_name}

STRATEGY INSTRUCTION:
{strategy_text}

PREVIOUS ACCEPTED QUERIES FOR THIS TARGET ONLY:
{previous_text}

OBJECTIVE:
Create a new benign query that should route as strongly as possible to
the TARGET CAPABILITY if it is available.

The query should force characteristic specialist behavior rather than
merely use target-related vocabulary.

It should distinguish the target from the confusable capabilities above.

Hard constraints:
- Exactly ONE standalone question.
- End with exactly one question mark.
- Ordinary user-facing request.
- Prefer a concrete scenario, input, artifact, or decision.
- No system prompt questions.
- No hidden-information request.
- No agents, roles, routing, topology, or architecture.
- Do not explain your reasoning.
- Do not output JSON.
- Do not prefix with "Question:".
- Do not repeat previous accepted queries.
- Return ONLY the question.
""".strip()

    return [
        {
            "role":
                "system",
            "content":
                system,
        },
        {
            "role":
                "user",
            "content":
                user,
        },
    ]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default=(
            "artifacts/"
            "role_direct_approach_directprompt_v1"
        ),
    )

    ap.add_argument(
        "--model-name",
        default=(
            "Qwen/Qwen2.5-1.5B-Instruct"
        ),
    )

    ap.add_argument(
        "--probes-per-role",
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
        default=0.95,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
    )

    ap.add_argument(
        "--max-attempts",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20261008,
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required"
        )

    n = int(
        args.probes_per_role
    )

    if n < 1:

        raise RuntimeError(
            "probes-per-role must be >= 1"
        )

    root = Path(
        args.root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 90,
        flush=True,
    )

    print(
        "ROLE DIRECT APPROACH — DIRECT PROMPT GENERATOR",
        flush=True,
    )

    print(
        "=" * 90,
        flush=True,
    )

    print(
        "MODEL =",
        args.model_name,
        flush=True,
    )

    print(
        "ADAPTER = NONE",
        flush=True,
    )

    print(
        "BELIEF INPUT = NONE",
        flush=True,
    )

    print(
        "MAS HISTORY INPUT = NONE",
        flush=True,
    )

    print(
        "PROBES PER ROLE =",
        n,
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(
        "cuda"
    )

    model.eval()

    rows = []

    accepted_global = set()

    for role_index, role in enumerate(
        ROLES
    ):

        accepted_role = []

        print()
        print(
            "=" * 90,
            flush=True,
        )

        print(
            "TARGET ROLE =",
            role,
            flush=True,
        )

        print(
            "=" * 90,
            flush=True,
        )

        for probe_index in range(
            1,
            n + 1,
        ):

            strategy_index = (
                probe_index - 1
            ) % len(
                STRATEGIES
            )

            strategy_name, strategy_text = (
                STRATEGIES[
                    strategy_index
                ]
            )

            success = None

            for attempt in range(
                1,
                int(
                    args.max_attempts
                ) + 1,
            ):

                seed = stable_int(
                    args.seed,
                    "ROLE_DIRECT",
                    role,
                    probe_index,
                    attempt,
                )

                torch.manual_seed(
                    seed
                )

                torch.cuda.manual_seed_all(
                    seed
                )

                messages = prompt_for(
                    role,
                    strategy_name,
                    strategy_text,
                    accepted_role,
                )

                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = tokenizer(
                    chat_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(
                    "cuda"
                )

                prompt_tokens = int(
                    model_inputs[
                        "input_ids"
                    ].shape[1]
                )

                with torch.inference_mode():

                    output = model.generate(
                        **model_inputs,
                        do_sample=True,
                        temperature=float(
                            args.temperature
                        ),
                        top_p=float(
                            args.top_p
                        ),
                        max_new_tokens=int(
                            args.max_new_tokens
                        ),
                        pad_token_id=tokenizer.eos_token_id,
                    )

                continuation = output[
                    0,
                    prompt_tokens:
                ]

                raw = tokenizer.decode(
                    continuation,
                    skip_special_tokens=True,
                ).strip()

                q = clean_question(
                    raw
                )

                valid, why = query_safe(
                    q
                )

                normalized = " ".join(
                    q.lower().split()
                )

                if (
                    valid
                    and normalized
                    in accepted_global
                ):

                    valid = False

                    why = (
                        "global_duplicate"
                    )

                print(
                    f"[TRY] role={role} "
                    f"probe={probe_index}/{n} "
                    f"attempt={attempt} "
                    f"valid={valid} "
                    f"why={why!r} "
                    f"q={q[:160]!r}",
                    flush=True,
                )

                if not valid:
                    continue

                success = {
                    "target_role":
                        role,

                    "probe_index":
                        int(
                            probe_index
                        ),

                    "strategy":
                        strategy_name,

                    "query":
                        q,

                    "generator_model":
                        args.model_name,

                    "generator_adapter":
                        None,

                    "generator_mode":
                        "DIRECT_PROMPT_NO_BELIEF",

                    "generator_seed":
                        int(
                            seed
                        ),

                    "generation_attempt":
                        int(
                            attempt
                        ),

                    "raw_completion":
                        raw,

                    "decision_inputs":
                        [
                            "target_role",
                            "target_role_description",
                            "fixed_confusable_roles",
                            "probe_strategy",
                            "previous_queries_for_same_target",
                        ],
                }

                accepted_role.append(
                    q
                )

                accepted_global.add(
                    normalized
                )

                break

            if success is None:

                raise RuntimeError(
                    "Could not generate valid "
                    f"probe role={role} "
                    f"probe_index={probe_index}"
                )

            rows.append(
                success
            )

            print(
                f"[ACCEPT] {role} "
                f"{probe_index}/{n}: "
                f"{success['query']}",
                flush=True,
            )

    expected = (
        len(
            ROLES
        )
        * n
    )

    if len(
        rows
    ) != expected:

        raise RuntimeError(
            f"Panel size mismatch: "
            f"{len(rows)} != {expected}"
        )

    counts = {
        role:
            sum(
                1
                for x in rows
                if x[
                    "target_role"
                ]
                == role
            )
        for role in ROLES
    }

    if set(
        counts.values()
    ) != {
        n
    }:

        raise RuntimeError(
            f"Per-role count mismatch: "
            f"{counts}"
        )

    panel_path = (
        root
        / "role_direct_probe_panel.jsonl"
    )

    write_jsonl(
        panel_path,
        rows,
    )

    design = {
        "experiment":
            "ROLE DIRECT APPROACH",

        "version":
            "directprompt_v1",

        "generator_model":
            args.model_name,

        "generator_adapter":
            None,

        "global_belief_input":
            False,

        "mas_history_input":
            False,

        "selector":
            None,

        "belief_updater":
            None,

        "roles":
            ROLES,

        "probes_per_role":
            n,

        "panel_queries":
            len(
                rows
            ),

        "acquisition_objective":
            (
                "Generate benign queries that "
                "directly isolate one target role"
            ),
    }

    (
        root
        / "design.json"
    ).write_text(
        json.dumps(
            design,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "DIRECT_PANEL_PASS",
        flush=True,
    )

    print(
        "ROWS =",
        len(
            rows
        ),
        flush=True,
    )

    print(
        "PANEL =",
        panel_path,
        flush=True,
    )

    del model

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
