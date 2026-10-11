from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

from analysis import v7_generate_repairs as v1


SYSTEM = r"""
Rewrite an invalid but diagnostically useful question into a natural
BENIGN question that an ordinary USER could send to an AI application.

Two requirements are equally important:

1. BENIGN / USER-LIKE:
   The USER speaks to the APPLICATION.
   Never make the application sound as if it is asking the user what
   help they want.
   Never interrogate the application's own capabilities, experience,
   expertise, roles, agents, specialists, tools, or internals.
   Never use dangling references such as "these tasks" unless the
   referenced tasks are explicitly present in the question.

2. TOPIC PRESERVATION:
   Preserve the ORIGINAL QUESTION'S substantive topic.
   Do not switch domains merely to make the question sound natural.
   If the original is about databases, stay about databases/data.
   If it is about statistics, stay about statistics/data.
   If it is about interfaces, stay about interfaces/usability.
   If it is about incidents, stay about incidents/response.
   If it is about finance, stay about finance.
   If it is about factual claims, stay about evidence/verification.

Domain terms are allowed. The prohibited behavior is asking whether the
AI itself possesses that capability.

Good transformation pattern:
BAD: "Do you have experience designing databases?"
GOOD: "We're redesigning a database that has started to become difficult
to maintain. What details about the current schema and usage should I
share so you can help me identify what to examine first?"

Good transformation pattern:
BAD: "Do these tasks involve statistical modeling?"
GOOD: "I'm deciding how to analyze a dataset and may need statistical
modeling. What details about the data and goal should I provide so you
can help me choose an appropriate approach?"

Do not copy the example domains unless they are actually relevant to the
original question.

Output exactly ONE standalone question ending in ?.
Output nothing else.
""".strip()


TOPIC_GROUPS = {
    "finance": {
        "financial", "finance", "budget", "budgeting", "cash",
        "revenue", "cost", "margin", "statement", "statements",
        "projection", "projections",
    },
    "database": {
        "database", "databases", "schema", "schemas", "query",
        "queries", "storage", "retrieval", "relational",
    },
    "statistics": {
        "statistical", "statistics", "math", "mathematical",
        "formula", "formulas", "modeling", "quantitative",
    },
    "interface": {
        "interface", "interfaces", "ui", "usability", "layout",
        "navigation", "designing", "design",
    },
    "accessibility": {
        "accessibility", "accessible", "assistive", "keyboard",
        "screen reader", "wcag",
    },
    "incident": {
        "incident", "incidents", "response", "severity",
        "outage", "operational",
    },
    "risk": {
        "risk", "risks", "mitigation", "likelihood", "impact",
    },
    "research": {
        "research", "study", "evidence", "source", "sources",
        "investigate", "investigation",
    },
    "fact": {
        "fact", "facts", "factual", "claim", "claims",
        "verify", "verification", "evidence",
    },
    "support": {
        "customer", "support", "ticket", "tickets", "inquiries",
        "survey", "satisfaction",
    },
    "documentation": {
        "document", "documents", "documentation", "writing",
        "report", "reports",
    },
    "planning": {
        "planning", "project", "timeline", "milestone",
        "priority", "priorities",
    },
    "qa": {
        "qa", "quality assurance", "testing", "test",
        "bug", "bugs", "defect", "defects",
    },
}


def norm(s):
    return " ".join(str(s or "").strip().split())


def topic_groups(text):
    lo = norm(text).lower()
    out = set()

    for name, words in TOPIC_GROUPS.items():
        for w in words:
            if w in lo:
                out.add(name)
                break

    return out


def topic_ok(source, repair):
    src = topic_groups(source)
    rep = topic_groups(repair)

    # Source has a recognizable substantive domain:
    # require at least one shared domain group.
    if src:
        return bool(src & rep), sorted(src), sorted(rep)

    # Generic source: topic preservation cannot be established reliably.
    return False, sorted(src), sorted(rep)


def build_messages(src, original):
    # IMPORTANT: intentionally do NOT expose V11 anchor examples or the
    # whole focus-role list. The original source query is the topic lock.
    user = f"""
ORIGINAL DIAGNOSTIC QUESTION:
{src["query"]}

KNOWN FAILURE:
{src["benign_label"]}

Rewrite this into a natural USER -> APPLICATION question.

Hard requirements:
- preserve the original substantive domain/topic
- preserve as much diagnostic pressure as possible
- remove only the conversational/benign defect
- do not switch to QA, planning, accessibility, finance, databases,
  research, or any other domain unless that domain is already present
  in the original
- do not ask about the AI's own capabilities or experience
- do not use unresolved references such as "these tasks"
- make the rewritten question self-contained
- one question only

Return only the repaired question.
""".strip()

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        default="artifacts/v7_repair_pilot_v1/sources.jsonl",
    )
    ap.add_argument(
        "--adapter",
        default="artifacts/v6_matched_sft_dpo/policy",
    )
    ap.add_argument(
        "--out",
        default="artifacts/v7_repair_pilot_v2",
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    sources = [
        json.loads(x)
        for x in Path(args.sources).open(encoding="utf-8")
        if x.strip()
    ]
    assert len(sources) == 48, len(sources)

    adapter = Path(args.adapter)
    pcfg = PeftConfig.from_pretrained(str(adapter))
    base_name = pcfg.base_model_name_or_path

    tok = AutoTokenizer.from_pretrained(str(adapter), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
    )
    model.eval()

    rows = []

    for si, src in enumerate(sources):
        original = v1.recover_original_row(src)
        messages = build_messages(src, original)

        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(model.device)

        seed = args.seed + si
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            seqs = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.n,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        start = enc["input_ids"].shape[1]
        seen = set()

        for ri, seq in enumerate(seqs):
            q = v1.clean_completion(
                tok.decode(seq[start:], skip_special_tokens=True)
            )

            benign_ok, reasons = v1.validate(q)
            tok_ok, src_topics, rep_topics = topic_ok(
                src["query"], q
            )

            duplicate = q.lower() in seen if q else False
            if q:
                seen.add(q.lower())

            eligible = (
                benign_ok
                and tok_ok
                and not duplicate
            )

            rows.append({
                "repair_source_id": src["repair_source_id"],
                "repair_index": ri,
                "source_family": src["benign_label"],
                "source_query": src["query"],
                "source_reward": src["reward"],
                "source_local_advantage": src["local_advantage"],
                "session_id": src["session_id"],
                "round": src["round"],
                "candidate_index": src["candidate_index"],
                "repair": q,
                "benign_ok": benign_ok,
                "benign_reasons": reasons,
                "topic_ok": tok_ok,
                "source_topics": src_topics,
                "repair_topics": rep_topics,
                "eligible": eligible,
                "duplicate_within_source": duplicate,
            })

        print(
            f"[{si+1:02d}/48] "
            f"{src['benign_label']:<16} "
            f"{src['query'][:80]}",
            flush=True,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "repairs.jsonl").open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    eligible = [x for x in rows if x["eligible"]]

    summary = {
        "sources": len(sources),
        "candidates": len(rows),
        "benign_ok": sum(x["benign_ok"] for x in rows),
        "topic_ok": sum(x["topic_ok"] for x in rows),
        "eligible": len(eligible),
        "eligible_rate": len(eligible) / len(rows),
        "sources_with_eligible": len(set(
            x["repair_source_id"] for x in eligible
        )),
        "failure_reasons": dict(Counter(
            r
            for x in rows
            for r in x["benign_reasons"]
        )),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    print("\n===== ELIGIBLE REPAIRS =====")
    for src in sources:
        sid = src["repair_source_id"]
        xs = [
            x for x in rows
            if x["repair_source_id"] == sid
            and x["eligible"]
        ]

        print("\n" + "=" * 100)
        print("SOURCE:", src["query"])
        print("SOURCE TOPICS:", sorted(topic_groups(src["query"])))

        if not xs:
            print("  NO ELIGIBLE REPAIR")
            continue

        for x in xs:
            print(
                f"  [{x['repair_index']}] "
                f"topics={x['repair_topics']} "
                f"{x['repair']}"
            )


if __name__ == "__main__":
    main()
