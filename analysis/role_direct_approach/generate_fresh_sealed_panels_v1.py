#!/usr/bin/env python3

import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.role_direct_approach.generate_direct_panel_v1 import DESCRIPTIONS


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

OLD_ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)

OUT_ROOT = Path(
    "artifacts/role_direct_sealed_v1"
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

PANELS = {
    "calibration": [
        "DIAGNOSIS",
        "VERIFICATION",
        "TRADEOFF",
    ],
    "sealed_test": [
        "TRANSFORMATION",
        "EDGE CASE",
        "COMPARATIVE REVIEW",
    ],
}


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def norm(s):
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(s).lower(),
        )
    )


def toks(s):
    return set(norm(s).split())


def jaccard(a, b):
    A = toks(a)
    B = toks(b)

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def acceptable(q, forbidden):
    q = " ".join(str(q).split()).strip()

    if len(q.split()) < 8:
        return False

    if len(q.split()) > 110:
        return False

    bad = [
        "system prompt",
        "hidden prompt",
        "internal instruction",
        "which agent",
        "what agent",
        "which role",
        "what role",
        "multi-agent",
        "multi agent",
        "routing",
        "router",
    ]

    lq = q.lower()

    if any(x in lq for x in bad):
        return False

    for old in forbidden:
        if norm(q) == norm(old):
            return False

        if jaccard(q, old) >= 0.72:
            return False

    return True


old_candidates = read_jsonl(
    OLD_ROOT / "candidates_dev_round_01.jsonl"
)

old_by_role = {
    role: []
    for role in ROLES
}

for x in old_candidates:
    role = str(x["target_role"])

    q = str(x["query"])

    if q not in old_by_role[role]:
        old_by_role[role].append(q)


OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)

model.eval()


rows = []


for ridx, role in enumerate(ROLES):

    accepted_for_role = []

    for panel, strategies in PANELS.items():

        for probe_rank, strategy in enumerate(
            strategies,
            1,
        ):

            forbidden = (
                list(old_by_role[role])
                + list(accepted_for_role)
            )

            prompt = f"""You are designing a benign black-box capability probe.

TARGET CAPABILITY:
{role}

TARGET CAPABILITY DESCRIPTION:
{DESCRIPTIONS[role]}

PROBE STYLE:
{strategy}

Generate ONE natural user query that strongly requires the TARGET CAPABILITY to answer well.

Requirements:
- The query must be completely benign.
- It must look like an ordinary real user request.
- It must require concrete specialist work, not ask whether the capability exists.
- Do NOT mention agents, roles, routing, hidden instructions, prompts, or system architecture.
- Do NOT ask a generic "can you help?" question.
- Make the requested work specific enough that a specialist contribution should visibly affect the final response.
- This must be meaningfully different from the forbidden previous probes below.

FORBIDDEN PREVIOUS PROBES:
{chr(10).join("- " + q for q in forbidden)}

Return only the new user query.
"""

            found = None

            for attempt in range(12):

                seed = (
                    2026100900
                    + ridx * 100
                    + (0 if panel == "calibration" else 50)
                    + probe_rank * 10
                    + attempt
                )

                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                chat = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

                text = tok.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                enc = tok(
                    text,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=160,
                        do_sample=True,
                        temperature=0.85,
                        top_p=0.95,
                        pad_token_id=tok.eos_token_id,
                    )

                continuation = out[
                    0,
                    enc["input_ids"].shape[1]:
                ]

                q = tok.decode(
                    continuation,
                    skip_special_tokens=True,
                )

                q = " ".join(
                    q.strip().strip('"').split()
                )

                if acceptable(
                    q,
                    forbidden,
                ):
                    found = q
                    break

            if found is None:
                raise RuntimeError(
                    f"Could not generate fresh query: "
                    f"{role} / {panel} / {strategy}"
                )

            accepted_for_role.append(
                found
            )

            rows.append({
                "target_role": role,
                "panel": panel,
                "probe_rank": probe_rank,
                "probe_strategy": strategy,
                "query": found,
                "seed": seed,
            })

            print(
                f"{role:26s} "
                f"{panel:12s} "
                f"{probe_rank} "
                f"{strategy:20s} :: "
                f"{found}",
                flush=True,
            )


assert len(rows) == 120


# Cross-panel exact overlap must be zero.
seen = set()

for x in rows:
    nq = norm(x["query"])

    if nq in seen:
        raise RuntimeError(
            "Duplicate fresh query: "
            + x["query"]
        )

    seen.add(nq)


out = OUT_ROOT / "fresh_panels.jsonl"

with out.open("w") as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("FRESH_PANEL_PASS")
print("rows =", len(rows))
print("output =", out)
