from pathlib import Path
from collections import defaultdict
import hashlib
import json
import statistics

ROOT = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/q38_utility_sft_v1")
OUT.mkdir(parents=True, exist_ok=True)

all_rows = []

for rnd in range(1, 8):
    p = ROOT / f"grpo_train_round_{rnd:02d}.jsonl"
    if not p.exists():
        print("MISSING:", p)
        continue

    rows = [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

    print(f"ROUND {rnd}: {len(rows)} rows")

    for x in rows:
        x["_source_round"] = rnd
        all_rows.append(x)

groups = defaultdict(list)

for x in all_rows:
    prompt = str(x.get("prompt", "")).strip()
    completion = str(x.get("completion", "")).strip()

    if not prompt or not completion:
        continue

    reward = float(x.get("reward", 0.0))

    # Only genuinely useful probes.
    if reward <= 0:
        continue

    gid = str(x.get("group_id", ""))
    if not gid:
        continue

    groups[gid].append(x)

selected = []

for gid, rows in groups.items():
    rows = sorted(
        rows,
        key=lambda x: (
            float(x.get("reward", 0.0)),
            float(x.get("advantage", 0.0)),
        ),
        reverse=True,
    )

    # At most two strong positive examples per attacker state.
    kept = []
    seen_completion = set()

    for x in rows:
        c = str(x["completion"]).strip()

        if c in seen_completion:
            continue

        seen_completion.add(c)
        kept.append(x)

        if len(kept) == 2:
            break

    selected.extend(kept)

# Exact dedupe
dedup = {}
for x in selected:
    key = (
        str(x["prompt"]).strip(),
        str(x["completion"]).strip(),
    )

    old = dedup.get(key)

    if old is None or float(x["reward"]) > float(old["reward"]):
        dedup[key] = x

selected = list(dedup.values())

train = []
val = []

for x in selected:
    sid = str(x.get("session_id", x["group_id"]))

    bucket = int(
        hashlib.sha256(sid.encode()).hexdigest()[:8],
        16,
    ) % 10

    rec = {
        "prompt": str(x["prompt"]),
        "completion": str(x["completion"]),
        "reward": float(x["reward"]),
        "advantage": float(x.get("advantage", 0.0)),
        "group_id": str(x["group_id"]),
        "session_id": sid,
        "source_round": int(x["_source_round"]),
    }

    if bucket == 0:
        val.append(rec)
    else:
        train.append(rec)

def write(path, rows):
    with path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

write(OUT / "train.jsonl", train)
write(OUT / "val.jsonl", val)

rewards = [x["reward"] for x in train]

summary = {
    "source_rounds": list(range(1, 8)),
    "selection": "top-2 positive-reward probes per group",
    "train_rows": len(train),
    "val_rows": len(val),
    "groups": len(groups),
    "mean_train_reward":
        statistics.mean(rewards) if rewards else None,
    "median_train_reward":
        statistics.median(rewards) if rewards else None,
}

(OUT / "dataset_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print(json.dumps(summary, indent=2))
print("DATASET_OK")
