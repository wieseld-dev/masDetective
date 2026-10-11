#!/usr/bin/env python3

import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import blackbox_commutator_poc as poc


# ---------------------------------------------------------
# Load the ACTUAL V11 probes we already generated before.
# ---------------------------------------------------------
SOURCE = Path("artifacts/v11_response_public.jsonl")

if not SOURCE.exists():
    raise SystemExit(f"ERROR: missing {SOURCE}")

probes = []
seen = set()

with SOURCE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        q = row.get("query")

        if not isinstance(q, str):
            continue

        q = q.strip()

        if len(q) < 5 or q in seen:
            continue

        seen.add(q)
        probes.append(q)

if not probes:
    raise SystemExit(
        "ERROR: found no string field named 'query' in V11 response file."
    )

print(f"[V11] Loaded {len(probes)} UNIQUE real V11 probes")
print("[V11] First 5 probes:")
for i, q in enumerate(probes[:5], 1):
    print(f"  {i}. {q[:250]}")

# Give the existing PoC our V11 bank instead of DEFAULT_TASKS.
poc.DEFAULT_TASKS = probes

# IMPORTANT: completely separate cache from the generic-prompt experiment.
sys.argv = [
    sys.argv[0],
    "--n-prompts", str(len(probes)),
    "--concurrency", "8",
    "--cache", "artifacts/commutator_v11_poc/api_runs.jsonl",
    "--seed", "42",
]

args = poc.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

tasks = probes[:args.n_prompts]

rows = asyncio.run(poc.collect_data(args, tasks))

inp, out, usd = poc.token_cost(
    rows,
    args.input_price_per_m,
    args.output_price_per_m,
)

print("\n===== API USAGE =====")
print(f"Input tokens:  {inp:,}")
print(f"Output tokens: {out:,}")
print(f"Estimated API token cost: ${usd:.4f}")

poc.main_analysis(args, rows)
