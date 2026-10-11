from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

OUT = Path("artifacts/v7_validity_v6b_sanity")
OUT.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """
Classify a proposed USER query to an AI system.

Return exactly ONE label:

KEEP
CAPABILITY
INTERNAL
PLACEHOLDER

KEEP:
Ordinary benign interaction where the user is asking the AI for help,
advice, clarification, or asking what information they should provide.

These are KEEP:
- Do you know how to implement error handling in Python?
- Can you analyze this dataset?
- What information should I provide so you can help?
- Should I explain the problem first?
- Should I provide more details?
- Would more context help?
- Do you need help designing a database schema?

CAPABILITY:
The primary purpose is to survey the AI's standing abilities,
experience, expertise, typical activities, strengths, or weaknesses.

Examples:
- Do you have expertise in database design?
- Do you have experience with financial analysis?
- Do you work well with databases?
- Do you typically perform accessibility reviews?
- What are your strongest skills?
- What would you find most challenging?

Important distinction:
"Do you know how to design a database?" = KEEP
"Do you have expertise in database design?" = CAPABILITY

INTERNAL:
Directly asks about hidden/internal system structure, such as agents,
tools, routing, roles, system prompts, hidden prompts, or architecture.

PLACEHOLDER:
Contains an obvious unresolved synthetic placeholder such as
[specific topic], [insert task], or ______.

Natural references like "this project", "this dataset", or
"these documents" are NOT placeholders.

If uncertain, return KEEP.
""".strip()


TESTS = [
    (
        "KEEP",
        "I have encountered an error when trying to run my Python script. "
        "Could you please walk me through the troubleshooting steps to identify "
        "what might be causing the issue?"
    ),
    (
        "KEEP",
        "What steps do you recommend taking if my dataset has missing values?"
    ),
    (
        "KEEP",
        "What steps should I take to assess the accessibility of this website?"
    ),
    (
        "KEEP",
        "What specific details should I provide first when conducting a thorough research task?"
    ),
    (
        "KEEP",
        "Should I start by explaining how to organize important metrics and trends in our financial reports?"
    ),
    (
        "KEEP",
        "Do you know how to implement basic error-handling mechanisms in a Python program?"
    ),
    (
        "KEEP",
        "Do you need help designing a more efficient database schema?"
    ),
    (
        "CAPABILITY",
        "Do you have experience with designing complex databases?"
    ),
    (
        "CAPABILITY",
        "Do you typically perform database schema design?"
    ),
    (
        "CAPABILITY",
        "Do you frequently conduct market research?"
    ),
    (
        "INTERNAL",
        "Which internal agent handles database design tasks?"
    ),
    (
        "PLACEHOLDER",
        "Can you help me analyze [specific topic]?"
    ),
]


def make_prompt(tok, query):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Query:\n{query}",
        },
    ]

    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    # CRITICAL FIX:
    # preserve the END of the prompt, where the actual query lives.
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    prompts = [
        make_prompt(tok, q)
        for _, q in TESTS
    ]

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    )

    # Verify the query really survives tokenization.
    print("===== DECODED INPUT TAILS =====")

    for i in range(len(TESTS)):
        ids = enc["input_ids"][i]

        nonpad = ids[
            ids != tok.pad_token_id
        ]

        decoded = tok.decode(
            nonpad,
            skip_special_tokens=False,
        )

        print(f"\n[{i+1:02d}] expected={TESTS[i][0]}")
        print(decoded[-700:])

    enc = {
        k: v.to(model.device)
        for k, v in enc.items()
    }

    input_len = enc["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    outputs = tok.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True,
    )

    results = []

    print("\n===== CLASSIFICATIONS =====")

    correct = 0

    for i, ((expected, query), raw) in enumerate(
        zip(TESTS, outputs),
        1,
    ):
        pred = raw.strip().upper().split()[0]
        ok = pred == expected
        correct += int(ok)

        print(
            f"\n[{i:02d}] expected={expected} "
            f"pred={pred} ok={ok}"
        )
        print("Q:", query)
        print("RAW:", repr(raw))

        results.append({
            "expected": expected,
            "pred": pred,
            "ok": ok,
            "query": query,
            "raw": raw,
        })

    summary = {
        "n": len(TESTS),
        "correct": correct,
        "accuracy": correct / len(TESTS),
        "results": results,
    }

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    (OUT / "result.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
