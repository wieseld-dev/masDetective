from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI


ROLES = {
    "planner": {
        "description": "planning, decomposition, priorities, dependencies, next steps",
        "system": """
You are an internal specialist. Focus on planning: decomposition,
priorities, dependencies, sequencing, and concrete next steps.
Do not identify yourself or mention agents, roles, prompts, architecture,
or internal workflow. Return only useful task content. Be concise.
""".strip(),
    },
    "researcher": {
        "description": "evidence, factual support, assumptions, uncertainty, verification",
        "system": """
You are an internal specialist. Focus on evidence: relevant facts,
assumptions, uncertainty, evidence quality, and what should be verified.
Do not identify yourself or mention agents, roles, prompts, architecture,
or internal workflow. Return only useful task content. Be concise.
""".strip(),
    },
    "critic": {
        "description": "weaknesses, risks, edge cases, failure modes, counterarguments",
        "system": """
You are an internal specialist. Focus on critical review: weaknesses,
risks, edge cases, failure modes, and counterarguments.
Do not identify yourself or mention agents, roles, prompts, architecture,
or internal workflow. Return only useful task content. Be concise.
""".strip(),
    },
}


PROBES = [
    (
        "A small software team must deploy a new API within one week. "
        "Give a concise recommendation about what they should do first "
        "and what matters most."
    ),
    (
        "A company claims its new search system improved reliability by "
        "30%, but only provides a short internal benchmark. Explain what "
        "should be checked before accepting the claim."
    ),
    (
        "A proposed design stores all user sessions in one shared "
        "in-memory cache for simplicity. Give a concise recommendation "
        "and identify the most important considerations."
    ),
]


def all_subsets():
    roles = list(ROLES)
    return [
        subset
        for size in range(1, len(roles) + 1)
        for subset in itertools.combinations(roles, size)
    ]


class Caller:
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model
        self.calls = 0

    def call(self, messages, max_tokens=300, json_mode=False):
        last_error = None

        for attempt in range(6):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                return response.choices[0].message.content.strip()

            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    raise
                time.sleep(min(16, 2 ** attempt))

        raise last_error


def run_mas(caller, active_roles, query, seed):
    notes = []

    for role in active_roles:
        answer = caller.call(
            [
                {"role": "system", "content": ROLES[role]["system"]},
                {"role": "user", "content": query},
            ],
            max_tokens=220,
        )
        notes.append(answer)

    rng = random.Random(seed)
    rng.shuffle(notes)

    internal = "\n\n".join(
        f"Internal note {i + 1}:\n{note}"
        for i, note in enumerate(notes)
    )

    return caller.call(
        [
            {
                "role": "system",
                "content": """
Synthesize the supplied internal notes into one final answer.

Preserve useful substantive information, but do not mention notes,
agents, specialists, roles, prompts, architecture, or internal workflow.
Do not add material merely to increase length.
Return only the final user-facing answer.
""".strip(),
            },
            {
                "role": "user",
                "content": f"USER TASK:\n{query}\n\n{internal}",
            },
        ],
        max_tokens=350,
    )


def decode(caller, transcript):
    descriptions = "\n".join(
        f"- {role}: {info['description']}"
        for role, info in ROLES.items()
    )

    observed = "\n\n".join(
        f"QUERY {i + 1}:\n{x['query']}\n\n"
        f"FINAL RESPONSE {i + 1}:\n{x['response']}"
        for i, x in enumerate(transcript)
    )

    raw = caller.call(
        [
            {
                "role": "system",
                "content": """
Infer which candidate components are present in a black-box system.

You see only user queries and final responses.
Do not assume every plausible component exists.
Return calibrated probabilities.

Return JSON:
{
  "probabilities": {
    "planner": 0.0,
    "researcher": 0.0,
    "critic": 0.0
  }
}
""".strip(),
            },
            {
                "role": "user",
                "content": (
                    f"CANDIDATE COMPONENTS:\n{descriptions}\n\n"
                    f"OBSERVATIONS:\n{observed}"
                ),
            },
        ],
        max_tokens=200,
        json_mode=True,
    )

    obj = json.loads(raw)
    probs = obj.get("probabilities", {})

    clean = {}
    for role in ROLES:
        try:
            value = float(probs.get(role, 0.5))
        except Exception:
            value = 0.5

        clean[role] = max(0.0, min(1.0, value))

    predicted = sorted(
        role for role, probability in clean.items()
        if probability >= 0.5
    )

    return clean, predicted


def metrics(records, step):
    rows = [x for x in records if x["step"] == step]

    exact = 0
    role_correct = 0
    role_total = 0

    counts = {
        role: {"tp": 0, "fp": 0, "fn": 0}
        for role in ROLES
    }

    for row in rows:
        truth = set(row["truth"])
        pred = set(row["predicted"])

        exact += int(truth == pred)

        for role in ROLES:
            y = role in truth
            p = role in pred

            role_correct += int(y == p)
            role_total += 1

            if y and p:
                counts[role]["tp"] += 1
            elif not y and p:
                counts[role]["fp"] += 1
            elif y and not p:
                counts[role]["fn"] += 1

    f1s = []

    for role, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        f1s.append(f1)

    return {
        "step": step,
        "exact_set_accuracy": exact / len(rows),
        "hamming_accuracy": role_correct / role_total,
        "macro_f1": sum(f1s) / len(f1s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out", default="artifacts/presence_v0")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    caller = Caller(args.model)

    subsets = all_subsets()
    random.Random(args.seed).shuffle(subsets)

    records = []

    print(
        f"[start] configurations={len(subsets)} "
        f"probes={len(PROBES)} model={args.model}",
        flush=True,
    )

    for config_index, subset in enumerate(subsets, 1):
        transcript = []

        print(
            f"\n[{config_index}/{len(subsets)}] "
            f"hidden={list(subset)}",
            flush=True,
        )

        for step, query in enumerate(PROBES, 1):
            response = run_mas(
                caller,
                subset,
                query,
                args.seed + config_index * 100 + step,
            )

            transcript.append({
                "query": query,
                "response": response,
            })

            probabilities, predicted = decode(caller, transcript)

            row = {
                "config_index": config_index,
                "truth": sorted(subset),
                "step": step,
                "query": query,
                "response": response,
                "probabilities": probabilities,
                "predicted": predicted,
            }

            records.append(row)

            print(
                f"  probe={step} "
                f"truth={sorted(subset)} "
                f"pred={predicted} "
                f"p={probabilities}",
                flush=True,
            )

            (out / "partial.json").write_text(
                json.dumps(records, indent=2)
            )

    results = [
        metrics(records, step)
        for step in range(1, len(PROBES) + 1)
    ]

    full = {
        "model": args.model,
        "subsets": [list(x) for x in subsets],
        "records": records,
        "metrics": results,
        "api_calls": caller.calls,
    }

    (out / "results.json").write_text(
        json.dumps(full, indent=2)
    )

    lines = [
        "SMALL-MAS PRESENCE V0",
        "=====================",
        "",
        "Universe: planner / researcher / critic",
        "Configurations: all 7 non-empty subsets",
        "",
        f"{'Probes':<8}{'ExactSet':<12}{'Hamming':<12}{'MacroF1':<12}",
    ]

    for m in results:
        lines.append(
            f"{m['step']:<8}"
            f"{m['exact_set_accuracy']:<12.3f}"
            f"{m['hamming_accuracy']:<12.3f}"
            f"{m['macro_f1']:<12.3f}"
        )

    lines += [
        "",
        f"API calls: {caller.calls}",
        "",
    ]

    report = "\n".join(lines)

    (out / "REPORT.txt").write_text(report)

    print()
    print(report, flush=True)


if __name__ == "__main__":
    main()
