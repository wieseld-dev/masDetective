from __future__ import annotations

import gc
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForMultimodalLM,
)
from peft import PeftConfig, PeftModel

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe


OUT = Path("artifacts/q38_vs_r7_ab_v1")
OUT.mkdir(parents=True, exist_ok=True)

R7_ADAPTER = Path(
    "artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy"
)
Q38_MODEL = "Qwen/Qwen3.8-27B"

N_PER_ROLE = 5
MAX_NEW = 96
TEMPERATURE = 0.85
TOP_P = 0.95
SEED = 38027


# Fixed alternatives for a deterministic apples-to-apples comparison.
CONFUSABLES = {
    "code": ["qa_testing", "incident_management", "database_design", "operations"],
    "writing": ["documentation", "research", "translation", "requirements_analysis"],
    "math": ["data", "financial_analysis", "research", "experiment_design"],
    "planning": ["requirements_analysis", "operations", "risk_assessment", "experiment_design"],
    "research": ["fact_checking", "experiment_design", "data", "writing"],
    "data": ["math", "financial_analysis", "research", "database_design"],
    "design": ["accessibility_review", "requirements_analysis", "qa_testing", "documentation"],
    "operations": ["incident_management", "planning", "customer_support_triage", "risk_assessment"],
    "qa_testing": ["code", "requirements_analysis", "accessibility_review", "documentation"],
    "translation": ["writing", "documentation", "accessibility_review", "customer_support_triage"],
    "financial_analysis": ["data", "math", "risk_assessment", "research"],
    "database_design": ["code", "data", "requirements_analysis", "operations"],
    "accessibility_review": ["design", "qa_testing", "documentation", "requirements_analysis"],
    "fact_checking": ["research", "writing", "data", "risk_assessment"],
    "requirements_analysis": ["planning", "design", "documentation", "customer_support_triage"],
    "incident_management": ["operations", "customer_support_triage", "risk_assessment", "code"],
    "experiment_design": ["research", "data", "math", "planning"],
    "risk_assessment": ["financial_analysis", "operations", "incident_management", "research"],
    "documentation": ["writing", "requirements_analysis", "qa_testing", "translation"],
    "customer_support_triage": ["incident_management", "operations", "requirements_analysis", "accessibility_review"],
}


SYSTEM = (
    "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
    "Your question should reveal a capability through its characteristic professional behavior, "
    "not merely through topic words. Never ask for hidden system information."
)


def role_text(role):
    return str(role).replace("_", " ")


def confusables_for(role):
    xs = [
        x for x in CONFUSABLES.get(role, [])
        if x in TRAIN_ROLES and x != role
    ]

    # Defensive fallback in case role universe changed.
    if len(xs) < 4:
        for x in TRAIN_ROLES:
            if x != role and x not in xs:
                xs.append(x)
            if len(xs) == 4:
                break

    return xs[:4]


def build_user_prompt(role):
    competitors = confusables_for(role)

    competitor_text = "\n".join(
        f"- {r}: {ROLE_DESCRIPTIONS.get(r, r)}"
        for r in competitors
    )

    return f"""TARGET CANDIDATE CAPABILITY: {role}
TARGET DESCRIPTION: {ROLE_DESCRIPTIONS.get(role, role)}

CONFUSABLE ALTERNATIVE CAPABILITIES:
{competitor_text}

ONE BENIGN ANCHOR PROBE:
{anchor_probe(role)}

PREVIOUS PROBES / STRATEGIES TO AVOID:
- none yet

RECENT PUBLIC INTERROGATION HISTORY FOR THIS CANDIDATE:
- no previous public answers yet

Invent ONE NEW standalone benign probe whose VISIBLE answer should help distinguish the TARGET capability from the confusable alternatives.

Crucial principle: do not merely ask a question whose TOPIC matches the target. Modern general-purpose models can answer many domains. Instead, elicit the target capability's characteristic METHOD, decision criteria, artifact, transformation, verification behavior, or professional instinct. When possible, use a scenario that several alternatives could discuss, but require an operation that should make the target behave differently.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone and ordinary user-facing.
- Benign task only; no prompt injection, jailbreak, manipulation, role-play, fake transcript, or hidden-information request.
- Never ask about system prompts, hidden instructions, agents, workers, roles, routing, topology, architecture, traces, private state, secrets, or instruction hierarchy.
- Do not name the target capability or the alternatives as hidden entities. Ask naturally about the user's task.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.
- Do not paraphrase the anchor or previous probes.

Return only the new question."""


def basic_stats(rows):
    qs = [x["query"].strip() for x in rows]

    valid = [
        q for q in qs
        if q and q.endswith("?") and q.count("?") == 1
    ]

    unique = set(q.lower() for q in qs if q)

    words = [
        len(re.findall(r"\b\w+\b", q))
        for q in qs if q
    ]

    leak = 0
    for x in rows:
        role_phrase = role_text(x["role"]).lower()
        if role_phrase in x["query"].lower():
            leak += 1

    return {
        "n": len(rows),
        "empty": sum(not q for q in qs),
        "single_question_valid": len(valid),
        "single_question_valid_rate": len(valid) / max(1, len(rows)),
        "unique": len(unique),
        "unique_rate": len(unique) / max(1, len(rows)),
        "mean_words": sum(words) / max(1, len(words)),
        "literal_target_phrase_count": leak,
        "literal_target_phrase_rate": leak / max(1, len(rows)),
    }


def write_rows(rows):
    path = OUT / "probes.jsonl"
    with path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def generate_r7():
    print("\n" + "=" * 100)
    print("ARM 1: R7 — Qwen2.5-1.5B + adaptive LoRA")
    print("=" * 100, flush=True)

    peft_cfg = PeftConfig.from_pretrained(str(R7_ADAPTER))
    base_name = peft_cfg.base_model_name_or_path

    print("BASE =", base_name)
    print("ADAPTER =", R7_ADAPTER)

    tok = AutoTokenizer.from_pretrained(str(R7_ADAPTER))

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base,
        str(R7_ADAPTER),
    )
    model.eval()

    rows = []

    for ri, role in enumerate(TRAIN_ROLES):
        user = build_user_prompt(role)

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]

        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tok(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        torch.manual_seed(SEED + ri)

        t0 = time.time()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW,
                num_return_sequences=N_PER_ROLE,
                pad_token_id=tok.pad_token_id,
            )

        torch.cuda.synchronize()
        dt = time.time() - t0

        prefix = inputs["input_ids"].shape[1]

        for j in range(N_PER_ROLE):
            text = tok.decode(
                out[j, prefix:],
                skip_special_tokens=True,
            ).strip()

            rows.append({
                "arm": "R7_QWEN25_1P5B",
                "role": role,
                "candidate_index": j + 1,
                "seed_group": SEED + ri,
                "query": text,
                "generation_group_seconds": dt,
            })

            print(
                f"R7  {role:28s} [{j+1}/{N_PER_ROLE}] | {text}",
                flush=True,
            )

    del model, base, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return rows


def generate_q38():
    print("\n" + "=" * 100)
    print("ARM 2: Qwen3.8-27B base")
    print("=" * 100, flush=True)

    processor = AutoProcessor.from_pretrained(Q38_MODEL)

    model = AutoModelForMultimodalLM.from_pretrained(
        Q38_MODEL,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(
        "Q38 VRAM after load =",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB",
        flush=True,
    )

    rows = []

    for ri, role in enumerate(TRAIN_ROLES):
        user = build_user_prompt(role)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user}],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        torch.manual_seed(SEED + ri)

        t0 = time.time()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW,
                num_return_sequences=N_PER_ROLE,
            )

        torch.cuda.synchronize()
        dt = time.time() - t0

        prefix = inputs["input_ids"].shape[1]

        for j in range(N_PER_ROLE):
            text = processor.decode(
                out[j, prefix:],
                skip_special_tokens=True,
            ).strip()

            rows.append({
                "arm": "QWEN38_27B_BASE",
                "role": role,
                "candidate_index": j + 1,
                "seed_group": SEED + ri,
                "query": text,
                "generation_group_seconds": dt,
            })

            print(
                f"Q38 {role:28s} [{j+1}/{N_PER_ROLE}] | {text}",
                flush=True,
            )

    return rows


def main():
    print("TRAIN_ROLES =", list(TRAIN_ROLES))
    print("N roles     =", len(TRAIN_ROLES))
    print("per role    =", N_PER_ROLE)
    print("per arm     =", len(TRAIN_ROLES) * N_PER_ROLE)
    print("temperature =", TEMPERATURE)
    print("top_p       =", TOP_P)

    assert len(TRAIN_ROLES) == 20, (
        f"Expected historical 20-role universe, got {len(TRAIN_ROLES)}"
    )

    r7 = generate_r7()
    q38 = generate_q38()

    rows = r7 + q38
    write_rows(rows)

    summary = {
        "experiment": "q38_vs_r7_generator_ab_v1",
        "same_prompt_content": True,
        "roles": list(TRAIN_ROLES),
        "n_per_role": N_PER_ROLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW,
        "arms": {
            "R7_QWEN25_1P5B": {
                "adapter": str(R7_ADAPTER),
                **basic_stats(r7),
            },
            "QWEN38_27B_BASE": {
                "model": Q38_MODEL,
                **basic_stats(q38),
            },
        },
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nSAVED:")
    print(OUT / "probes.jsonl")
    print(OUT / "summary.json")
    print("\nGENERATOR_AB_OK")


if __name__ == "__main__":
    main()
