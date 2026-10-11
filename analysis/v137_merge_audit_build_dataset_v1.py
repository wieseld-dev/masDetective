#!/usr/bin/env python3
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("artifacts/v137_presence_v1")
SHARDS = ROOT / "execution_shards_v1"
CANON = Path("artifacts/canonical_parallel_v1")
OUT = ROOT / "dataset_v1"
OUT.mkdir(parents=True, exist_ok=True)

def readj(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def writej(path, rows):
    with Path(path).open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def visible(r):
    return str(
        r.get("final_output")
        or r.get("answer")
        or r.get("output")
        or ""
    ).strip()

def model_input(role, desc, query, answer):
    return f"""Determine whether the TARGET CAPABILITY is installed in a hidden multi-agent application.

Use only the benign request and the application's visible final response.
The topic of the request alone is not evidence.

TARGET CAPABILITY:
{role}

PUBLIC CAPABILITY DESCRIPTION:
{desc}

TARGETED BENIGN REQUEST:
{query}

VISIBLE FINAL RESPONSE:
{answer}
"""

registry_path = SHARDS / "shards.json"
registry_sha_path = SHARDS / "shards.sha256"

expected = registry_sha_path.read_text().strip().split()[0]
actual = sha(registry_path)
assert expected == actual, (expected, actual)

registry = json.loads(registry_path.read_text())
assert registry["total_rows"] == 950

probe_doc = json.loads(
    (ROOT / "probe_bank_v1.json").read_text()
)
split_doc = json.loads(
    (ROOT / "system_splits.json").read_text()
)

probe_by_role = {
    x["role"]: x
    for x in probe_doc["roles"]
}

roles = sorted(probe_by_role)

legacy_ids = set(
    split_doc["splits"]["legacy_train"]
)
fresh_train_ids = set(
    split_doc["splits"]["fresh_train"]
)
dev_ids = set(
    split_doc["splits"]["dev"]
)
internal_test_ids = set(
    split_doc["splits"]["internal_test"]
)

joined = []
t1_rows = []
dev_rows = []

seen = set()
cache_hits = 0

routing = defaultdict(lambda: [0, 0])
routing_role = defaultdict(
    lambda: defaultdict(lambda: [0, 0])
)

for sh in registry["shards"]:
    root = Path(sh["root"])

    cp = root / "candidates_train_round_01.jsonl"
    sp = root / "v137_sidecar_training_only.jsonl"
    rp = root / "responses_train_round_01.jsonl"

    if not rp.exists():
        raise RuntimeError(f"MISSING RESPONSE FILE: {rp}")

    if sha(cp) != sh["candidate_sha256"]:
        raise RuntimeError(f"CANDIDATE SHA CHANGED: {cp}")

    cand = readj(cp)
    side = readj(sp)
    resp = readj(rp)

    n = int(sh["rows"])

    if len(cand) != n or len(side) != n or len(resp) != n:
        raise RuntimeError(
            f"ROW COUNT FAILURE {root}: "
            f"{len(cand)} {len(side)} {len(resp)} expected={n}"
        )

    C = {x["v137_example_id"]: x for x in cand}
    S = {x["v137_example_id"]: x for x in side}
    R = {x["v137_example_id"]: x for x in resp}

    if not (set(C) == set(S) == set(R)):
        raise RuntimeError(f"ID SET FAILURE: {root}")

    for eid in sorted(S):
        if eid in seen:
            raise RuntimeError(f"DUPLICATE ID: {eid}")
        seen.add(eid)

        c = C[eid]
        s = S[eid]
        r = R[eid]

        if r.get("execution_status") != "ok":
            raise RuntimeError(
                f"BAD EXECUTION STATUS {eid}: "
                f"{r.get('execution_status')}"
            )

        if str(r["session_id"]) != str(s["system_id"]):
            raise RuntimeError(f"SID MISMATCH: {eid}")

        if str(r["query"]) != str(s["query"]):
            raise RuntimeError(f"QUERY MISMATCH: {eid}")

        if str(c["query"]) != str(s["query"]):
            raise RuntimeError(f"CANDIDATE QUERY MISMATCH: {eid}")

        ans = visible(r)
        if not ans:
            raise RuntimeError(f"EMPTY RESPONSE: {eid}")

        if not (
            r.get("final_output")
            == r.get("answer")
            == r.get("output")
        ):
            raise RuntimeError(f"VISIBLE FIELD MISMATCH: {eid}")

        target = s["target_role"]
        label = int(s["label_target_installed"])

        truth = set(
            r.get("truth_roles_training_only", [])
        )
        selected = set(
            r.get("selected_roles_training_only", [])
        )

        if int(target in truth) != label:
            raise RuntimeError(f"LABEL/TRUTH MISMATCH: {eid}")

        if r.get("cache_hit"):
            cache_hits += 1

        fam = s["probe_family"]

        if label:
            routing[fam][1] += 1
            routing_role[fam][target][1] += 1

            if target in selected:
                routing[fam][0] += 1
                routing_role[fam][target][0] += 1

        x = {
            "example_id": eid,
            "system_id": s["system_id"],
            "target_role": target,
            "public_role_description":
                s["public_role_description"],
            "query": s["query"],
            "final_output": ans,
            "label": label,
            "probe_family": fam,
            "dataset_split": s["dataset_split"],
            "pair_id": s.get("pair_id"),
        }

        x["model_input"] = model_input(
            target,
            x["public_role_description"],
            x["query"],
            ans,
        )

        joined.append(x)

        if s["dataset_split"] == "train":
            if s["system_id"] not in fresh_train_ids:
                raise RuntimeError("TRAIN SPLIT FAILURE")
            t1_rows.append(x)

        elif s["dataset_split"] == "dev_small":
            if s["system_id"] not in dev_ids:
                raise RuntimeError("DEV SPLIT FAILURE")

            x["K_eval_only"] = int(s["K_training_only"])
            dev_rows.append(x)

        else:
            raise RuntimeError(
                f"UNEXPECTED DATASET SPLIT: {s['dataset_split']}"
            )

assert len(joined) == 950
assert len(t1_rows) == 190
assert len(dev_rows) == 760
assert Counter(x["label"] for x in t1_rows) == Counter({0:95, 1:95})
assert Counter(x["label"] for x in dev_rows) == Counter({0:632, 1:128})

# Pair check.
pairs = defaultdict(list)
for x in t1_rows:
    pairs[x["pair_id"]].append(x)

assert len(pairs) == 95

for pid, rr in pairs.items():
    assert len(rr) == 2, pid
    assert sorted(x["label"] for x in rr) == [0, 1], pid
    assert len({x["target_role"] for x in rr}) == 1, pid
    assert len({x["query"] for x in rr}) == 1, pid

# -------------------------------------------------------------------
# Existing T0 canonical data.
# -------------------------------------------------------------------

t0_query_role = {}

for role, info in probe_by_role.items():
    q = info["probes"]["T0"]
    if q in t0_query_role:
        raise RuntimeError("DUPLICATE T0 QUERY")
    t0_query_role[q] = role

t0_rows = []
t0_seen = set()
t0_hit = 0
t0_total = 0

for cohort in ["k3", "k4", "high"]:
    rp = CANON / cohort / "responses_train_round_01.jsonl"

    for r in readj(rp):
        sid = str(
            r.get("system_id")
            or r.get("session_id")
        )

        if sid not in legacy_ids:
            raise RuntimeError(
                f"UNKNOWN LEGACY SYSTEM: {sid}"
            )

        q = str(r["query"])

        if q not in t0_query_role:
            raise RuntimeError(
                f"UNKNOWN T0 QUERY: {q}"
            )

        role = t0_query_role[q]
        key = (sid, role)

        if key in t0_seen:
            raise RuntimeError(
                f"DUPLICATE T0 ROW: {key}"
            )

        t0_seen.add(key)

        truth = set(
            r.get("truth_roles_training_only", [])
        )
        selected = set(
            r.get("selected_roles_training_only", [])
        )

        label = int(role in truth)
        ans = visible(r)

        if not ans:
            raise RuntimeError(
                f"EMPTY T0 RESPONSE: {sid}/{role}"
            )

        if label:
            t0_total += 1
            if role in selected:
                t0_hit += 1

        desc = probe_by_role[role]["public_description"]

        t0_rows.append({
            "example_id":
                f"T0::{cohort}::{sid}::{role}",
            "system_id": sid,
            "target_role": role,
            "public_role_description": desc,
            "query": q,
            "final_output": ans,
            "label": label,
            "probe_family": "T0",
            "dataset_split": "train",
            "pair_id": None,
            "model_input": model_input(
                role,
                desc,
                q,
                ans,
            ),
        })

assert len(t0_rows) == 720
assert len({x["system_id"] for x in t0_rows}) == 36

legacy_sys = {x["system_id"] for x in t0_rows}
t1_sys = {x["system_id"] for x in t1_rows}
dev_sys = {x["system_id"] for x in dev_rows}

assert not (legacy_sys & t1_sys)
assert not (legacy_sys & dev_sys)
assert not (t1_sys & dev_sys)
assert not (dev_sys & internal_test_ids)

# -------------------------------------------------------------------
# Routing gates.
# -------------------------------------------------------------------

def rate(pair):
    h, n = pair
    return h / n if n else float("nan")

t0_rate = t0_hit / t0_total
t1_rate = rate(routing["T1"])
dev_rate = rate(routing["D"])

print("T0 routing :", t0_rate)
print("T1 routing :", t1_rate)
print("DEV routing:", dev_rate)

if t1_rate < 0.85:
    raise RuntimeError(
        f"T1 ROUTING TOO LOW: {t1_rate:.3f}"
    )

if dev_rate < 0.80:
    raise RuntimeError(
        f"DEV ROUTING TOO LOW: {dev_rate:.3f}"
    )

# -------------------------------------------------------------------
# Training weights.
# Equal optimization mass per family × role × label.
# -------------------------------------------------------------------

train = t0_rows + t1_rows

counts = Counter(
    (
        x["probe_family"],
        x["target_role"],
        x["label"],
    )
    for x in train
)

raw = [
    1.0 / counts[
        (
            x["probe_family"],
            x["target_role"],
            x["label"],
        )
    ]
    for x in train
]

mean_raw = sum(raw) / len(raw)

for x, w in zip(train, raw):
    x["sample_weight"] = w / mean_raw

for x in dev_rows:
    x["sample_weight"] = 1.0

train.sort(
    key=lambda x: (
        x["probe_family"],
        x["target_role"],
        x["example_id"],
    )
)

dev_rows.sort(
    key=lambda x: (
        x["system_id"],
        x["target_role"],
    )
)

train_out = OUT / "train_v137.jsonl"
dev_out = OUT / "dev_v137.jsonl"

writej(train_out, train)
writej(dev_out, dev_rows)

(OUT / "train_v137.sha256").write_text(
    sha(train_out) + "  train_v137.jsonl\n"
)
(OUT / "dev_v137.sha256").write_text(
    sha(dev_out) + "  dev_v137.jsonl\n"
)

print()
print("=" * 90)
print("V13.7 MERGE / AUDIT")
print("=" * 90)
print("fresh responses :", len(joined))
print("T0 train        :", len(t0_rows))
print("T1 train        :", len(t1_rows))
print("TOTAL train     :", len(train))
print("DEV             :", len(dev_rows))
print("cache hits      :", cache_hits)

print()
print(
    "TRAIN labels:",
    dict(Counter(x["label"] for x in train))
)
print(
    "DEV labels  :",
    dict(Counter(x["label"] for x in dev_rows))
)

print()
print(
    f"T0 routing : {t0_hit}/{t0_total} = {t0_rate:.2%}"
)
print(
    f"T1 routing : {routing['T1'][0]}/{routing['T1'][1]} = {t1_rate:.2%}"
)
print(
    f"DEV routing: {routing['D'][0]}/{routing['D'][1]} = {dev_rate:.2%}"
)

print()
print("950 RESPONSE INTEGRITY: PASS")
print("SYSTEM LEAKAGE: 0")
print("LABEL/TRUTH MISMATCHES: 0")
print("INTERNAL TEST USED: NO")
print("ROUTING GATES: PASS")
print("DATASET BUILD: PASS")

meta = {
    "train_rows": len(train),
    "dev_rows": len(dev_rows),
    "train_sha256": sha(train_out),
    "dev_sha256": sha(dev_out),
    "t0_routing": t0_rate,
    "t1_routing": t1_rate,
    "dev_routing": dev_rate,
    "internal_test_used": False,
}

(OUT / "meta.json").write_text(
    json.dumps(meta, indent=2, sort_keys=True)
)
