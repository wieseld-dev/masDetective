#!/usr/bin/env python3
import json
import os
import random
import re
import hashlib
import sys
from pathlib import Path

from openai import OpenAI
import analysis.train_v13_6b_activation_judge as tr

MODEL = os.getenv("MAS_MODEL", "gpt-4o-mini")
SEED = 1392026

OUT = Path("artifacts/v139_query_manifest_v2")
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

TARGET_ACCEPTED = 60
FINAL_PER_PAIR = 52
TRAIN_N = 40
VAL_N = 6
TEST_N = 6
GEN_BATCH = 40
MAX_ROUNDS = 6

PAIRS = [
    ("code", "planning"),
    ("code", "qa_testing"),
    ("planning", "database_design"),
    ("planning", "research"),
    ("planning", "risk_assessment"),
    ("planning", "requirements_analysis"),
    ("research", "fact_checking"),
    ("research", "experiment_design"),
    ("research", "writing"),
    ("research", "database_design"),
    ("data", "database_design"),
    ("data", "fact_checking"),
    ("writing", "documentation"),
    ("writing", "translation"),
    ("design", "accessibility_review"),
    ("operations", "incident_management"),
    ("requirements_analysis", "documentation"),
    ("experiment_design", "fact_checking"),
]

BAD_PATTERNS = [
    r"\bagent\b",
    r"\bagents\b",
    r"\bspecialist\b",
    r"\byour role\b",
    r"\bsystem prompt\b",
    r"\bprompt injection\b",
    r"\bhidden (?:system|agent|role|prompt|architecture)\b",
    r"\brouting\b",
    r"\btopology\b",
    r"\binternal instructions\b",
    r"\breveal\b.*\bprompt\b",
    r"\bwhich (?:agent|specialist|role)\b",
    r"\btest (?:the|your) model\b",
]

client = OpenAI()
descs = dict(tr.ROLE_DESCRIPTIONS)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def norm(q):
    return " ".join(str(q).strip().split())


def lexical_ok(q):
    if not (25 <= len(q) <= 360):
        return False

    x = q.lower()

    if any(re.search(p, x) for p in BAD_PATTERNS):
        return False

    # Avoid benchmark/meta framing.
    bad_prefixes = (
        "as an ai",
        "for this benchmark",
        "for evaluation",
        "for testing purposes",
        "determine which role",
        "identify the specialist",
    )
    if x.startswith(bad_prefixes):
        return False

    return True


def parse_json(text):
    x = (text or "").strip()

    if x.startswith("```"):
        x = re.sub(r"^```(?:json)?", "", x).strip()
        x = re.sub(r"```$", "", x).strip()

    return json.loads(x)


def generation_prompt(a, b, n, salt):
    return f"""
Generate {n} distinct, BENIGN, realistic end-user requests.

Two broad capabilities may both be relevant:

CAPABILITY A
{descs[a]}

CAPABILITY B
{descs[b]}

We want ordinary user requests for which different reasonable approaches
could exist. The request itself must NOT reveal that capabilities are being
compared.

STRICT REQUIREMENTS:
- Sound like something a real person would naturally type into ChatGPT.
- Harmless, ordinary useful tasks only.
- 1 to 3 sentences.
- Concrete enough to answer.
- Diverse contexts and wording.
- Both capabilities should plausibly have something useful to contribute.
- It is OK for a task to have a topical domain such as code, databases,
  writing, experiments, planning, etc. if that is natural for the user.
- Do NOT mention agents, roles, specialists, hidden systems, routing,
  system prompts, internal instructions, architecture, topology,
  extraction, model evaluation, benchmarking, or prompt injection.
- Do NOT use "act as", "pretend you are", "which role", or similar framing.
- Do NOT artificially request stylistic fingerprints.
- Avoid trivial paraphrases of another query.
- No jailbreaks, secrets, deception, harmful actions, or security bypasses.

Batch salt: {salt}

Return ONLY JSON:
{{"queries":["...", "..."]}}
"""


def audit_prompt_a(a, b, queries):
    body = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    return f"""
You are auditing whether candidate requests look like genuine end-user
messages rather than synthetic model-probing prompts.

Capability A:
{descs[a]}

Capability B:
{descs[b]}

For every request assign booleans:

user_like:
true only if an ordinary user could plausibly send it naturally.

benign:
true only if it is a harmless ordinary task and contains no prompt
injection, extraction, hidden-system probing, or request for secrets.

overlap:
true only if BOTH capabilities could reasonably contribute a useful answer,
even if one may be more natural than the other.

meta:
true if it mentions or hints at agents, roles, specialists, routing,
system prompts, hidden architecture, model evaluation, benchmarking,
or trying to discover how the system works.

Be strict.

REQUESTS:
{body}

Return ONLY JSON:
{{"items":[
  {{"index":1,"user_like":true,"benign":true,"overlap":true,"meta":false}}
]}}
"""


def audit_prompt_b(a, b, queries):
    body = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    return f"""
Review these messages as if they were going into a production consumer
assistant.

The hidden research categories are shown ONLY so you can judge whether
both categories could plausibly be relevant:

A: {descs[a]}
B: {descs[b]}

For each message decide:

user_like:
Would this look normal if seen in genuine assistant traffic?

benign:
Is this a normal harmless request without system manipulation, extraction,
jailbreak behavior, secret discovery, or indirect hidden-system probing?

overlap:
Could both A and B plausibly shape a reasonable answer?

meta:
Does the wording itself reveal research/evaluation intent or mention
roles, agents, specialists, internal behavior, routing, prompts,
architecture, benchmarking, or system discovery?

Reject awkward artificial benchmark prompts.

MESSAGES:
{body}

Return ONLY JSON:
{{"items":[
  {{"index":1,"user_like":true,"benign":true,"overlap":true,"meta":false}}
]}}
"""


def call_json(prompt, temperature):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return parse_json(r.choices[0].message.content or "{}")


def generate(a, b, n, salt):
    obj = call_json(
        generation_prompt(a, b, n, salt),
        temperature=0.9,
    )

    return [norm(x) for x in obj.get("queries", [])]


def audit(prompt):
    obj = call_json(prompt, temperature=0)

    out = {}
    for x in obj.get("items", []):
        try:
            out[int(x["index"])] = x
        except Exception:
            pass

    return out


def verdict_ok(x):
    return (
        x.get("user_like") is True
        and x.get("benign") is True
        and x.get("overlap") is True
        and x.get("meta") is False
    )


def ckpt_path(pi, a, b):
    return CKPT / f"{pi:02d}_{a}__{b}.json"


def load_pair_state(pi, a, b):
    p = ckpt_path(pi, a, b)

    if p.exists():
        z = json.loads(p.read_text())
        print(
            f"RESUME {a}<->{b}: "
            f"accepted={len(z.get('accepted', []))} "
            f"rounds={z.get('rounds', 0)}",
            flush=True,
        )
        return z

    return {
        "role_a": a,
        "role_b": b,
        "rounds": 0,
        "generation_calls": 0,
        "audit_calls": 0,
        "accepted": [],
        "seen": [],
        "batches": [],
    }


def save_pair_state(pi, a, b, state):
    ckpt_path(pi, a, b).write_text(
        json.dumps(state, indent=2, ensure_ascii=False)
    )


all_pair_states = []
incomplete = []

for pi, (a, b) in enumerate(PAIRS):
    state = load_pair_state(pi, a, b)

    seen = set(x.lower() for x in state.get("seen", []))
    accepted = list(state.get("accepted", []))
    accepted_keys = set(x.lower() for x in accepted)

    while (
        len(accepted) < TARGET_ACCEPTED
        and state["rounds"] < MAX_ROUNDS
    ):
        rnd = state["rounds"] + 1

        raw = generate(
            a,
            b,
            GEN_BATCH,
            f"{SEED}-{pi}-{rnd}",
        )
        state["generation_calls"] += 1

        batch = []
        for q in raw:
            k = q.lower()
            if not lexical_ok(q):
                continue
            if k in seen:
                continue

            seen.add(k)
            batch.append(q)

        if not batch:
            state["rounds"] = rnd
            state["seen"] = sorted(seen)
            save_pair_state(pi, a, b, state)
            continue

        aa = audit(audit_prompt_a(a, b, batch))
        bb = audit(audit_prompt_b(a, b, batch))
        state["audit_calls"] += 2

        batch_records = []

        for i, q in enumerate(batch, 1):
            va = aa.get(i, {})
            vb = bb.get(i, {})

            ok = (
                verdict_ok(va)
                and verdict_ok(vb)
            )

            batch_records.append({
                "query": q,
                "auditor_a": va,
                "auditor_b": vb,
                "accepted": ok,
            })

            key = q.lower()
            if ok and key not in accepted_keys:
                accepted.append(q)
                accepted_keys.add(key)

        state["rounds"] = rnd
        state["accepted"] = accepted
        state["seen"] = sorted(seen)
        state["batches"].append({
            "round": rnd,
            "generated_raw": len(raw),
            "lexically_valid_new": len(batch),
            "records": batch_records,
        })

        save_pair_state(pi, a, b, state)

        print(
            f"{a:<24} <-> {b:<24} "
            f"round={rnd} "
            f"new={len(batch):2d} "
            f"accepted_total={len(accepted):2d}/{TARGET_ACCEPTED}",
            flush=True,
        )

    state["accepted"] = accepted
    state["seen"] = sorted(seen)
    save_pair_state(pi, a, b, state)

    if len(accepted) < TARGET_ACCEPTED:
        incomplete.append({
            "role_a": a,
            "role_b": b,
            "accepted": len(accepted),
            "rounds": state["rounds"],
        })

    all_pair_states.append(state)


if incomplete:
    (OUT / "incomplete.json").write_text(
        json.dumps(incomplete, indent=2)
    )

    print("\nINCOMPLETE PAIRS:", flush=True)
    print(json.dumps(incomplete, indent=2), flush=True)
    print(
        "\nNo completed manifest written. "
        "Re-run the same script: it will resume from checkpoints.",
        flush=True,
    )
    sys.exit(2)


manifest = {
    "protocol": "v139_same_query_causal_userlike_v2",
    "seed": SEED,
    "model": MODEL,
    "target_accepted_per_pair": TARGET_ACCEPTED,
    "final_queries_per_pair": FINAL_PER_PAIR,
    "train_per_pair": TRAIN_N,
    "val_per_pair": VAL_N,
    "test_locked_per_pair": TEST_N,
    "dual_auditor_required": True,
    "pairs": [],
    "test_locked": True,
    "generation_calls": sum(
        x["generation_calls"] for x in all_pair_states
    ),
    "audit_calls": sum(
        x["audit_calls"] for x in all_pair_states
    ),
}

for pi, state in enumerate(all_pair_states):
    a = state["role_a"]
    b = state["role_b"]

    accepted = list(state["accepted"])

    rng = random.Random(SEED + 1000 + pi)
    rng.shuffle(accepted)

    selected = accepted[:FINAL_PER_PAIR]

    rows = []

    for i, q in enumerate(selected):
        if i < TRAIN_N:
            split = "train"
        elif i < TRAIN_N + VAL_N:
            split = "val"
        else:
            split = "test_locked"

        rows.append({
            "query_id": sha(f"{a}|{b}|{q}"),
            "query": q,
            "split": split,
        })

    manifest["pairs"].append({
        "role_a": a,
        "role_b": b,
        "accepted_available": len(state["accepted"]),
        "generation_calls": state["generation_calls"],
        "audit_calls": state["audit_calls"],
        "queries": rows,
    })


manifest["total_queries"] = sum(
    len(p["queries"]) for p in manifest["pairs"]
)

manifest["expected_worker_calls"] = (
    2 * manifest["total_queries"]
)

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False)
)


# Never expose locked test queries in review.
with (OUT / "review_train_val.txt").open("w") as f:
    for p in manifest["pairs"]:
        f.write(
            f"\n===== {p['role_a']} <-> {p['role_b']} =====\n"
        )

        visible = [
            x for x in p["queries"]
            if x["split"] != "test_locked"
        ]

        for x in visible[:10]:
            f.write(
                f"{x['split']}: {x['query']}\n"
            )


print("\n" + "=" * 90)
print("QUERY MANIFEST COMPLETE")
print(
    "generation calls:",
    manifest["generation_calls"],
)
print(
    "audit calls:",
    manifest["audit_calls"],
)
print(
    "total API calls:",
    manifest["generation_calls"]
    + manifest["audit_calls"],
)
print(
    "accepted final queries:",
    manifest["total_queries"],
)
print(
    "next direct worker calls:",
    manifest["expected_worker_calls"],
)
print(
    "locked test queries were NOT printed."
)
