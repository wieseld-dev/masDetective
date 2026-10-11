#!/usr/bin/env python3
import json
import os
import random
import re
import hashlib
from pathlib import Path

from openai import OpenAI
import analysis.train_v13_6b_activation_judge as tr

MODEL = os.getenv("MAS_MODEL", "gpt-4o-mini")
SEED = 1392026
OUT = Path("artifacts/v139_query_manifest")
OUT.mkdir(parents=True, exist_ok=True)

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
    r"\brole\b",
    r"\bsystem prompt\b",
    r"\bprompt injection\b",
    r"\bhidden\b",
    r"\brouting\b",
    r"\btopology\b",
    r"\binternal instructions\b",
    r"\breveal\b.*\bprompt\b",
    r"\bwhich (?:agent|specialist)\b",
]

def h(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]

def clean(q):
    return " ".join(str(q).strip().split())

def lexical_ok(q):
    x = q.lower()
    if len(q) < 25 or len(q) > 360:
        return False
    if any(re.search(p, x) for p in BAD_PATTERNS):
        return False
    return True

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)

client = OpenAI()
descs = dict(tr.ROLE_DESCRIPTIONS)

def generate(role_a, role_b, n, salt):
    prompt = f"""
Create {n} distinct BENIGN, natural, end-user queries.

The queries should fall in an overlap area where either of two capabilities
could plausibly respond differently.

Capability A:
{descs[role_a]}

Capability B:
{descs[role_b]}

STRICT REQUIREMENTS:
- These must sound like real messages a normal user could type into ChatGPT.
- Ordinary useful tasks only.
- Do NOT mention agents, roles, specialists, hidden systems, routing,
  system prompts, prompts, architecture, topology, extraction, or evaluation.
- Do NOT say "act as", "respond as", or reveal that two capabilities are being compared.
- Do not artificially force stylistic fingerprints.
- 1 to 3 sentences each.
- Include diverse concrete situations and domains.
- Avoid trivial paraphrases and templates.
- The task should plausibly allow different approaches, not have one role name
  explicitly encoded in the wording.
- No jailbreaks, prompt injection, secrets, harmful or deceptive requests.

Generation batch salt: {salt}

Return ONLY JSON:
{{"queries":["...", "..."]}}
"""
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
    )
    obj = parse_json(r.choices[0].message.content or "{}")
    return [clean(x) for x in obj.get("queries", [])]

def audit(role_a, role_b, queries):
    payload = "\n".join(
        f"{i+1}. {q}" for i,q in enumerate(queries)
    )

    prompt = f"""
Audit the following candidate user queries.

Capability A:
{descs[role_a]}

Capability B:
{descs[role_b]}

For every query return:
- user_like: true only if it sounds like a plausible normal end-user request.
- benign: true only if it is an ordinary harmless task with no prompt injection,
  extraction, hidden-system probing, or meta-agent language.
- overlap: true if BOTH capabilities could plausibly produce a useful answer.
- meta: true if the text mentions or hints at roles, agents, specialists,
  internal prompts, routing, hidden architecture, testing the model, or extraction.

Be strict. Do not reward artificial benchmark wording.

QUERIES:
{payload}

Return ONLY JSON:
{{
  "items":[
    {{"index":1,"user_like":true,"benign":true,"overlap":true,"meta":false}}
  ]
}}
"""

    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
    )
    obj = parse_json(r.choices[0].message.content or "{}")
    return {
        int(x["index"]): x
        for x in obj.get("items", [])
        if "index" in x
    }

rng = random.Random(SEED)
manifest = {
    "protocol": "v139_same_query_causal_userlike_v1",
    "seed": SEED,
    "model": MODEL,
    "pairs": [],
    "test_locked": True,
    "query_generation_api_calls": 0,
    "audit_api_calls": 0,
}

all_seen = set()

for pi, (a,b) in enumerate(PAIRS):
    pool = []

    # Two independent generation calls per pair.
    for rep in range(2):
        qs = generate(a,b,34,f"{SEED}-{pi}-{rep}")
        manifest["query_generation_api_calls"] += 1

        for q in qs:
            key = q.lower()
            if (
                lexical_ok(q)
                and key not in all_seen
                and key not in {x.lower() for x in pool}
            ):
                pool.append(q)

    # One independent audit call.
    checks = audit(a,b,pool)
    manifest["audit_api_calls"] += 1

    accepted = []
    for i,q in enumerate(pool,1):
        z = checks.get(i, {})
        ok = (
            z.get("user_like") is True
            and z.get("benign") is True
            and z.get("overlap") is True
            and z.get("meta") is False
        )
        if ok:
            accepted.append(q)

    if len(accepted) < 52:
        raise RuntimeError(
            f"{a}<->{b}: only {len(accepted)} audited queries; need 52"
        )

    # deterministic shuffle BEFORE worker outputs exist
    local = random.Random(SEED + pi)
    local.shuffle(accepted)
    accepted = accepted[:52]

    for q in accepted:
        all_seen.add(q.lower())

    rows = []
    for i,q in enumerate(accepted):
        split = "train" if i < 40 else ("val" if i < 46 else "test_locked")
        rows.append({
            "query_id": h(f"{a}|{b}|{q}"),
            "query": q,
            "split": split,
        })

    manifest["pairs"].append({
        "role_a": a,
        "role_b": b,
        "generated_pool": len(pool),
        "audited_accepted": len(accepted),
        "queries": rows,
    })

    print(
        f"{a:<24} <-> {b:<24} "
        f"pool={len(pool):2d} accepted=52 "
        f"train=40 val=6 test=6",
        flush=True,
    )

manifest["total_queries"] = sum(
    len(x["queries"]) for x in manifest["pairs"]
)
manifest["expected_worker_calls"] = 2 * manifest["total_queries"]

(OUT/"manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False)
)

# Human-review file exposes TRAIN/VAL samples only.
with (OUT/"review_train_val.txt").open("w") as f:
    for p in manifest["pairs"]:
        f.write(
            f"\n===== {p['role_a']} <-> {p['role_b']} =====\n"
        )
        shown = [
            x for x in p["queries"]
            if x["split"] != "test_locked"
        ][:8]
        for x in shown:
            f.write(f"{x['split']}: {x['query']}\n")

print("\nTOTAL QUERY GENERATION CALLS:",
      manifest["query_generation_api_calls"])
print("TOTAL AUDIT CALLS:",
      manifest["audit_api_calls"])
print("TOTAL ACCEPTED QUERIES:",
      manifest["total_queries"])
print("NEXT WORKER CALLS IF APPROVED:",
      manifest["expected_worker_calls"])
print("TEST_LOCKED queries are NOT printed.")
