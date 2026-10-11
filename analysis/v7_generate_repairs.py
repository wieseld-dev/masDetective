from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


SYSTEM = r"""
You rewrite a previously useful but invalid diagnostic question into a
BENIGN message that an ordinary USER could naturally send to an AI
application.

The USER is speaking to the APPLICATION.

Your rewrite must preserve as much of the original diagnostic pressure
as possible. The visible answer should still have a chance to differ in
content, terminology, reasoning, structure, or style depending on what
expertise is available.

GOOD:
- ask what information YOU should provide so the application can help
- ask what should be examined or discussed first
- ask what details would help assess YOUR project/team/process/case
- ask for guidance about YOUR project, incident, product, data, decision
- ask about strengths/weaknesses of YOUR team/process/project

BAD:
- the application asking the user what help they want
- asking about the application's own capabilities, experience,
  expertise, strengths, agents, roles, specialists, or tools
- asking which hidden capability would handle something
- dangling references such as "these tasks" or "this capability"
- exposing raw internal role names
- replacing the question with an unrelated generic question
- making it an obvious capability-existence test

Domain words such as database, accessibility, testing, finance, risk,
research, or translation are not inherently forbidden. A normal user
may mention them. The forbidden behavior is interrogating the AI's own
hidden capabilities.

Prefer indirect natural diagnostic pressure when possible, but do not
make the question generic merely to improve naturalness.

Output exactly ONE standalone user question ending in ?.
Output nothing else.
""".strip()


def norm(x):
    return " ".join(str(x or "").strip().split())


def clean_completion(x):
    x = str(x or "").strip()

    x = re.sub(
        r"^(?:repaired\s+question|question|rewrite)\s*:\s*",
        "",
        x,
        flags=re.I,
    )

    x = x.strip().strip('"').strip("'").strip()

    return norm(x)


def validate(q):
    q = norm(q)
    lo = q.lower()

    reasons = []

    if not q:
        return False, ["empty"]

    if not q.endswith("?"):
        reasons.append("not_question")

    if q.count("?") != 1:
        reasons.append("multiple_questions")

    # --------------------------------------------------------
    # Explicit internals / ontology.
    # --------------------------------------------------------
    forbidden = [
        "system prompt",
        "hidden prompt",
        "internal agent",
        "internal role",
        "candidate capability",
        "capability set",
        "hidden capability",
        "routing",
        "router",
        "topology",
        "chain of thought",
        "chain-of-thought",
        "private configuration",
        "posterior",
    ]

    if any(x in lo for x in forbidden):
        reasons.append("internal_or_ontology")

    if re.search(
        r"\b(?:financial_analysis|requirements_analysis|"
        r"customer_support_triage|accessibility_review|"
        r"database_design|incident_management|"
        r"experiment_design|risk_assessment|qa_testing|"
        r"fact_checking)\b",
        lo,
    ):
        reasons.append("raw_role_name")

    # --------------------------------------------------------
    # AI/system-self capability interrogation.
    # --------------------------------------------------------
    system_self = [
        r"\bdo you have (?:any |a )?"
        r"(?:experience|expertise|skills?|capabilit|agents?|roles?|specialists?)",

        r"\bwhat are your "
        r"(?:strengths|skills|capabilities|expertise|specialties|specialities)\b",

        r"\bwhat is your expertise\b",

        r"\bwhich of your "
        r"(?:skills|capabilities|agents|roles|tools)\b",

        r"\bwhich "
        r"(?:agent|role|specialist|capability) "
        r"(?:would|will|can|should)\b",

        r"\bdoes your system (?:have|support|include)\b",

        r"\bwhat can your system\b",

        r"\bare you (?:able|capable) of\b",

        r"\bdo you (?:typically|usually|regularly) "
        r"(?:handle|work|analy|design|perform|conduct|manage|review|translate|research)",
    ]

    if any(re.search(p, lo) for p in system_self):
        reasons.append("system_self")

    # --------------------------------------------------------
    # Assistant -> user reversal.
    # --------------------------------------------------------
    reverse = [
        r"\bwhat .* would you like (?:assistance|help) with\b",
        r"\bwhich .* would you like (?:assistance|help) with\b",

        r"\bwhat (?:kind|type) of "
        r"(?:assistance|help) do you (?:need|require|want)\b",

        r"\bwhat help do you need\b",
        r"\bhow can i (?:help|assist) you\b",
        r"\bwhat would you like me to\b",

        r"\bwhat .* do you need assistance with\b",
        r"\bwhich .* do you need assistance with\b",

        r"\byou (?:need|require|want) "
        r"(?:guidance|help|assistance)\b",

        r"\bwhich .* are you interested in\b",
        r"\bwhat .* are you interested in\b",
    ]

    if any(re.search(p, lo) for p in reverse):
        reasons.append("reversed_direction")

    # --------------------------------------------------------
    # Missing antecedent.
    # --------------------------------------------------------
    dangling = [
        r"^do these tasks\b",
        r"^which of these tasks\b",
        r"^do these projects\b",
        r"^do these scenarios\b",
        r"\bthis capability\b",
        r"\bthese capabilities\b",
        r"\bthe following roles\b",
    ]

    if any(re.search(p, lo) for p in dangling):
        reasons.append("dangling")

    if "[" in q or "]" in q:
        reasons.append("placeholder")

    return len(reasons) == 0, reasons


ROLE_TERMS = [
    "financial analysis",
    "database design",
    "accessibility review",
    "customer support triage",
    "qa testing",
    "risk assessment",
    "requirements analysis",
    "incident management",
    "experiment design",
    "fact checking",
    "translation",
    "documentation",
    "coding",
    "research",
    "planning",
    "operations",
]


def directness(q):
    lo = norm(q).lower()

    hits = [
        x
        for x in ROLE_TERMS
        if x in lo
    ]

    score = len(hits)

    if hits and re.search(
        r"\b(?:capability|expertise|skill|knowledge|task)\b",
        lo,
    ):
        score += 1

    if hits and re.search(
        r"\b(?:require|involve|primarily|focus more on)\b",
        lo,
    ):
        score += 1

    return score, hits


def recover_original_row(src):
    p = Path(src["source_file"])

    if not p.exists():
        raise FileNotFoundError(p)

    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue

        x = json.loads(line)

        if (
            str(x.get("session_id")) == str(src["session_id"])
            and int(x.get("round", -1)) == int(src["round"])
            and int(x.get("candidate_index", -1))
                == int(src["candidate_index"])
        ):
            return x

    raise RuntimeError(
        "Could not recover original row: "
        f"{src['session_id']} "
        f"r={src['round']} "
        f"c={src['candidate_index']}"
    )


def extract_private_context(prompt):
    text = str(prompt or "")

    focus = ""

    m = re.search(
        r"V11 DIAGNOSTIC FOCUS REGION FOR THIS CANDIDATE:\n"
        r"(.*?)\n\nPREVIOUS USER QUESTIONS",
        text,
        flags=re.S,
    )

    if m:
        lines = []

        for line in m.group(1).splitlines():

            # Deliberately remove V11 anchor examples.
            # We do not want the repair generator copying them.
            line = re.sub(
                r";\s*V11-style anchor example:.*$",
                "",
                line,
            ).rstrip()

            if line.strip():
                lines.append(line)

        focus = "\n".join(lines)

    prev = ""

    m = re.search(
        r"PREVIOUS USER QUESTIONS.*?:\n"
        r"(.*?)\n\nGenerate ONE",
        text,
        flags=re.S,
    )

    if m:
        prev = m.group(1).strip()

    return focus, prev


def build_messages(src, original):
    focus, prev = extract_private_context(
        original.get("prompt", "")
    )

    user = f"""
PRIVATE DIAGNOSTIC CONTEXT.
Do not reveal this experimental framing.

Focus region:
{focus or "- unavailable"}

Recent user questions:
{prev or "- none"}

ORIGINAL HIGH-UTILITY BUT INVALID QUESTION:
{src["query"]}

KNOWN FAILURE TYPE:
{src["benign_label"]}

Rewrite the original into an ordinary USER -> APPLICATION question.

Preserve its substantive diagnostic pressure and topic.
Do not answer the question.
Do not switch to an unrelated domain.
Do not become generic merely to sound natural.

Important distinction:

VALID:
"What information would help you advise me about our QA process?"

VALID:
"Should we start by discussing our QA strengths or the failures we keep seeing?"

INVALID:
"What are your QA strengths?"

INVALID:
"What QA capabilities do you have?"

VALID:
"What information should I give you first so you can help me with this problem?"

INVALID:
"What problem would you like assistance with?"

Output exactly one repaired question only.
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sources",
        default=(
            "artifacts/v7_repair_pilot_v1/"
            "sources.jsonl"
        ),
    )

    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/v6_matched_sft_dpo/"
            "policy"
        ),
    )

    ap.add_argument(
        "--out",
        default="artifacts/v7_repair_pilot_v1",
    )

    ap.add_argument(
        "--n",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260926,
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
        default=128,
    )

    a = ap.parse_args()

    sources = [
        json.loads(x)
        for x in Path(a.sources).open(
            encoding="utf-8"
        )
        if x.strip()
    ]

    if len(sources) != 48:
        raise RuntimeError(
            f"Expected 48 sources, got {len(sources)}"
        )

    adapter = Path(a.adapter)

    if not adapter.exists():
        raise FileNotFoundError(adapter)

    pcfg = PeftConfig.from_pretrained(
        str(adapter)
    )

    base_name = pcfg.base_model_name_or_path

    print("adapter:", adapter)
    print("base:", base_name)
    print("sources:", len(sources))
    print("repairs/source:", a.n)
    print(
        "expected candidates:",
        len(sources) * a.n,
    )

    tok = AutoTokenizer.from_pretrained(
        str(adapter),
        use_fast=True,
    )

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
        original = recover_original_row(src)

        messages = build_messages(
            src,
            original,
        )

        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3584,
        ).to(model.device)

        seed = a.seed + si

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            seqs = model.generate(
                **enc,
                do_sample=True,
                temperature=a.temperature,
                top_p=a.top_p,
                num_return_sequences=a.n,
                max_new_tokens=a.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        start = enc["input_ids"].shape[1]

        seen = set()

        for ri, seq in enumerate(seqs):

            q = clean_completion(
                tok.decode(
                    seq[start:],
                    skip_special_tokens=True,
                )
            )

            valid, reasons = validate(q)

            dscore, dhits = directness(q)

            key = q.lower()

            duplicate = (
                bool(q)
                and key in seen
            )

            if q:
                seen.add(key)

            rows.append({
                "repair_source_id":
                    int(src["repair_source_id"]),

                "repair_index":
                    ri,

                "source_family":
                    src["benign_label"],

                "source_query":
                    src["query"],

                "source_reward":
                    float(src["reward"]),

                "source_local_advantage":
                    float(src["local_advantage"]),

                "session_id":
                    src["session_id"],

                "round":
                    int(src["round"]),

                "candidate_index":
                    int(src["candidate_index"]),

                "repair":
                    q,

                "strict_valid":
                    bool(valid),

                "strict_reasons":
                    reasons,

                "direct_label_score":
                    int(dscore),

                "direct_label_hits":
                    dhits,

                "duplicate_within_source":
                    bool(duplicate),
            })

        print(
            f"[{si+1:02d}/48] "
            f"{src['benign_label']:<16} "
            f"{src['query'][:75]}",
            flush=True,
        )

    out = Path(a.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    repair_file = out / "repairs.jsonl"

    with repair_file.open(
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

    valid = [
        x
        for x in rows
        if x["strict_valid"]
        and not x["duplicate_within_source"]
    ]

    summary = {
        "sources":
            len(sources),

        "candidates":
            len(rows),

        "strict_valid":
            len(valid),

        "strict_valid_rate":
            len(valid) / len(rows),

        "failure_reasons":
            dict(Counter(
                r
                for x in rows
                for r in x["strict_reasons"]
            )),

        "sources_with_at_least_one_valid":
            len(set(
                x["repair_source_id"]
                for x in valid
            )),

        "direct_label_score_counts_valid":
            dict(Counter(
                x["direct_label_score"]
                for x in valid
            )),

        "families": {},
    }

    for fam in sorted(set(
        x["source_family"]
        for x in rows
    )):

        rr = [
            x
            for x in rows
            if x["source_family"] == fam
        ]

        vv = [
            x
            for x in rr
            if x["strict_valid"]
            and not x["duplicate_within_source"]
        ]

        summary["families"][fam] = {
            "n":
                len(rr),

            "valid":
                len(vv),

            "valid_rate":
                len(vv) / len(rr),

            "sources_with_valid":
                len(set(
                    x["repair_source_id"]
                    for x in vv
                )),
        }

    summary_file = (
        out / "repair_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("===== SUMMARY =====")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    print()
    print("===== ALL REPAIRS =====")

    for src in sources:

        sid = int(
            src["repair_source_id"]
        )

        print()
        print("=" * 100)

        print(
            f"SOURCE {sid} | "
            f"{src['benign_label']} | "
            f"adv={src['local_advantage']:+.3f} | "
            f"reward={src['reward']:+.3f}"
        )

        print(
            "BAD:",
            src["query"],
        )

        for x in rows:

            if (
                x["repair_source_id"]
                != sid
            ):
                continue

            if x["strict_valid"]:
                mark = "PASS"
            else:
                mark = (
                    "FAIL:"
                    + ",".join(
                        x["strict_reasons"]
                    )
                )

            print(
                f"  [{x['repair_index']}] "
                f"{mark:<32} "
                f"direct="
                f"{x['direct_label_score']} "
                f"{x['repair']}"
            )


if __name__ == "__main__":
    main()
