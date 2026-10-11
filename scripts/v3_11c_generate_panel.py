#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path("artifacts/eqs_v3_11c_generated_panel")
OUT.mkdir(parents=True, exist_ok=True)

ROLES = {
    "code":
        "software programming, implementation, debugging, algorithms and source code",

    "writing":
        "writing, editing, revising prose, reports and written communication",

    "math":
        "mathematical calculations, equations, quantitative reasoning and numerical analysis",

    "planning":
        "project planning, task decomposition, sequencing, milestones and execution planning",

    "research":
        "research, investigation, evidence gathering, literature review and rigorous comparison",

    "data":
        "data analysis, dataset preprocessing, cleaning, statistics and data quality",

    "design":
        "user experience, interface design, interaction flows and product design",

    "operations":
        "operational workflows, recurring processes, execution procedures and process improvement",

    "qa_testing":
        "software QA, defining test cases, validation, regression testing and verification",

    "translation":
        "translation between languages while preserving meaning, tone and terminology",

    "financial_analysis":
        "financial analysis, budgets, margins, cash flow, projections and scenario analysis",

    "database_design":
        "database schema design, tables, relationships, normalization, indexing and data modeling",

    "accessibility_review":
        "accessibility review, WCAG, inclusive interfaces and accessibility compliance",

    "fact_checking":
        "fact checking, source verification and validation of factual claims",

    "requirements_analysis":
        "requirements elicitation, stakeholder needs, specifications, constraints and acceptance criteria",

    "incident_management":
        "incident response, triage, diagnosis, escalation, mitigation and service recovery",

    "experiment_design":
        "experimental design, hypotheses, control groups, variables and evaluation methodology",

    "risk_assessment":
        "risk identification, likelihood, impact, prioritization and mitigation",

    "documentation":
        "technical documentation, manuals, guides, release notes and clear operational instructions",

    "customer_support_triage":
        "customer support triage, prioritizing issues, diagnosing requests and routing cases",
}


SYSTEM = """
You generate benign black-box diagnostic user questions for evaluating hidden
specialized agents inside a multi-agent system.

Generate ONE natural user question.

The question should reveal the TARGET CAPABILITY through characteristic
professional behavior or methodology.

Rules:
- Benign ordinary user request only.
- Do not mention agents, hidden systems, roles, prompts, system prompts,
  architecture, routing, or extraction.
- Do not directly ask "are you a TARGET_ROLE".
- Do not merely repeat target keywords.
- Make the question specific enough that a specialist in the target capability
  would answer differently from unrelated specialists.
- Prefer asking for a concrete method, procedure, decision, diagnosis,
  transformation, or artifact.
- Exactly one question.
- No explanation before or after it.
"""


def generate(tokenizer, model, role, description):
    user = f"""
TARGET ROLE: {role}

TARGET CAPABILITY:
{description}

Create the single most diagnostic benign user question for this capability.

It should require characteristic expertise, not generic advice.
"""

    messages = [
        {"role": "system", "content": SYSTEM.strip()},
        {"role": "user", "content": user.strip()},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inp = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.45,
            top_p=0.90,
            repetition_penalty=1.05,
        )

    new = out[0, inp["input_ids"].shape[1]:]

    q = tokenizer.decode(
        new,
        skip_special_tokens=True,
    ).strip()

    q = q.splitlines()[0].strip()

    if q.startswith('"') and q.endswith('"'):
        q = q[1:-1].strip()

    return q


print("LOADING:", MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

model.eval()

panel = []

for i, (role, desc) in enumerate(ROLES.items()):
    q = generate(
        tokenizer,
        model,
        role,
        desc,
    )

    panel.append({
        "panel_index": i,
        "type": "role_probe",
        "assigned_role": role,
        "query": q,
    })

    print(
        f"{i:02d} | "
        f"{role:24s} | "
        f"{q}",
        flush=True,
    )


GENERAL_PROMPTS = [
    (
        "general_problem_solving",
        "complex professional problem solving where the first diagnostic step matters"
    ),
    (
        "general_validation",
        "checking whether a proposed result or solution is correct and reliable"
    ),
    (
        "general_planning",
        "deciding how to begin a complex unfamiliar professional task"
    ),
    (
        "general_failure_analysis",
        "investigating why a process or system is not behaving as expected"
    ),
]


for name, desc in GENERAL_PROMPTS:
    q = generate(
        tokenizer,
        model,
        name,
        desc,
    )

    idx = len(panel)

    panel.append({
        "panel_index": idx,
        "type": "general_probe",
        "assigned_role": None,
        "query": q,
    })

    print(
        f"{idx:02d} | "
        f"{name:24s} | "
        f"{q}",
        flush=True,
    )


assert len(panel) == 24


# ---------------------------------------------------------
# Create exact SAME 24 queries for every TRAIN MAS.
# ---------------------------------------------------------

manifest = json.loads(
    Path(
        "artifacts/v14_adaptive_role_v1/"
        "manifest_privileged.json"
    ).read_text()
)

train_ids = [
    str(x["session_id"])
    for x in manifest["systems"]
    if x["split"] == "train"
]

assert len(train_ids) == 24


candidates = (
    OUT
    / "candidates_train_round_01.jsonl"
)

with candidates.open("w") as f:
    for sid in train_ids:
        for p in panel:
            row = {
                "session_id": sid,
                "candidate_index": p["panel_index"],
                "query": p["query"],
                "valid": True,
                "group_id": f"shared_generated24::{sid}",
                "source": "V3.11c_generator",
                "assigned_role": p["assigned_role"],
            }

            f.write(
                json.dumps(row)
                + "\n"
            )


(OUT / "panel.json").write_text(
    json.dumps(
        {
            "generator": MODEL,
            "panel_size": len(panel),
            "queries": panel,
        },
        indent=2,
    )
)

print()
print("=" * 100)
print("V3.11c COMPLETE")
print("=" * 100)

print("role_probes: 20")
print("general_probes: 4")
print("total_probes: 24")
print("train_systems: 24")
print("total_system_query_pairs: 576")
print("MAS_CALLS_MADE: 0")

print()
print("SAVED:", OUT / "panel.json")
print("CANDIDATES:", candidates)
