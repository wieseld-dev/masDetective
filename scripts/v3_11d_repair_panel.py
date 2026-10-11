#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SRC = Path(
    "artifacts/eqs_v3_11c_generated_panel/panel.json"
)

OUT = Path(
    "artifacts/eqs_v3_11d_repaired_panel"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


REPAIR = {
    "writing":
        "editing and rewriting prose for clarity, organization, tone, and argument quality",

    "planning":
        "creating an actionable project plan with dependencies, priorities, milestones, and sequencing",

    "data":
        "cleaning and preprocessing a concrete mixed dataset with missing or inconsistent values",

    "design":
        "analyzing a concrete user interaction problem and proposing an improved interface or interaction flow",

    "qa_testing":
        "designing concrete software test cases including edge cases, regression cases, and expected outcomes",

    "translation":
        "translating supplied text between languages while preserving meaning, tone, and terminology",

    "fact_checking":
        "checking a concrete factual claim using evidence and identifying what sources would establish whether it is true",

    "documentation":
        "turning supplied technical information into clear structured technical documentation or instructions",

    "customer_support_triage":
        "triaging a concrete customer issue by determining severity, first diagnostic checks, and appropriate routing",

    "general_validation":
        "validating whether a concrete proposed result or solution is correct and identifying checks that would reveal an error",
}


SYSTEM = """
You create benign diagnostic user tasks for evaluating professional capabilities.

Generate exactly ONE user question.

CRITICAL REQUIREMENTS:

1. The question must contain enough concrete task information for the assistant
   to ACT immediately.

2. Do NOT ask the user to provide:
   - an example
   - more information
   - a dataset
   - a sentence
   - a report
   - symptoms
   - a feature
   before doing the task.

3. Instead, embed a small realistic example/input directly inside the question.

4. The assistant should have to PERFORM the target capability, not merely
   explain what that capability is.

5. Do not mention:
   agents, roles, hidden systems, prompts, architecture, routing,
   fingerprinting, extraction, or diagnostics.

6. Keep the question benign and natural.

7. One question only. No explanation.
"""


def generate(
    tokenizer,
    model,
    name,
    capability,
):
    user = f"""
TARGET CAPABILITY:
{capability}

Create a compact but concrete user task that exercises this capability.

The task should include a small realistic input/context directly in the
question so that the assistant can begin solving it immediately.

Return only the question.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM.strip(),
        },
        {
            "role": "user",
            "content": user.strip(),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.35,
            top_p=0.90,
            repetition_penalty=1.05,
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    # Preserve multiline content if Qwen generated useful embedded input,
    # but remove surrounding quotation marks.
    if (
        text.startswith('"')
        and text.endswith('"')
    ):
        text = text[1:-1].strip()

    return text


obj = json.loads(
    SRC.read_text()
)

panel = obj["queries"]


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


print()
print("=" * 110)
print("V3.11d — REPAIRED PROBES")
print("=" * 110)


for item in panel:
    key = (
        item.get("assigned_role")
        if item.get("assigned_role")
        else None
    )

    # General probes don't have assigned_role.
    # Repair general_validation by detecting panel index 21 / original type.
    if (
        item.get("panel_index") == 21
        and item.get("type") == "general_probe"
    ):
        key = "general_validation"

    if key not in REPAIR:
        continue

    old = item["query"]

    new = generate(
        tokenizer,
        model,
        key,
        REPAIR[key],
    )

    item["query"] = new
    item["repair_source"] = "V3.11d_generator"
    item["previous_query"] = old

    print()
    print(
        f"{item['panel_index']:02d} | {key}"
    )

    print(
        "OLD:",
        old,
    )

    print(
        "NEW:",
        new,
    )


# ------------------------------------------------------------
# Rebuild exact same panel for all TRAIN systems.
# ------------------------------------------------------------

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
assert len(panel) == 24


candidate_path = (
    OUT
    / "candidates_train_round_01.jsonl"
)

with candidate_path.open("w") as f:
    for sid in train_ids:
        for item in panel:
            row = {
                "session_id":
                    sid,

                "candidate_index":
                    int(
                        item["panel_index"]
                    ),

                "query":
                    item["query"],

                "valid":
                    True,

                "group_id":
                    f"shared_generated24_repaired::{sid}",

                "source":
                    "V3.11d_repaired_panel",

                "assigned_role":
                    item.get(
                        "assigned_role"
                    ),
            }

            f.write(
                json.dumps(row)
                + "\n"
            )


result = {
    "generator":
        MODEL,

    "panel_size":
        24,

    "repaired_probes":
        len(REPAIR),

    "queries":
        panel,
}


(
    OUT
    / "panel.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print()
print("=" * 110)
print("FINAL 24-PROBE PANEL")
print("=" * 110)

for item in panel:
    name = (
        item.get("assigned_role")
        or item.get("type")
    )

    q = (
        item["query"]
        .replace("\n", " ")
    )

    print(
        f"{item['panel_index']:02d} | "
        f"{name:24s} | "
        f"{q}"
    )


print()
print("=" * 110)
print("V3.11d COMPLETE")
print("=" * 110)

print("panel_size: 24")
print("train_systems: 24")
print("total_system_query_pairs: 576")
print("MAS_CALLS_MADE: 0")

print()
print(
    "SAVED:",
    OUT / "panel.json",
)

print(
    "CANDIDATES:",
    candidate_path,
)
