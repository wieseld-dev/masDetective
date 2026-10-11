#!/bin/bash
#SBATCH --job-name=adapt-role-check
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/adapt-role-check-%j.out
#SBATCH --error=logs/adapt-role-check-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

python - <<'PY'
import json
from pathlib import Path

# Importing the collector must register/validate the V11 role universe.
import adaptive_collect_cpu

from targets.hierarchical_mas import ROLE_PROMPTS
from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
from v11.role_universe import TRAIN_ROLES

missing = sorted(set(TRAIN_ROLES) - set(ROLE_PROMPTS))

print("TRAIN_ROLES =", len(TRAIN_ROLES))
print("ROLE_PROMPTS =", len(ROLE_PROMPTS))
print("MISSING =", missing)

if missing:
    raise SystemExit(f"FAIL: runtime still missing roles: {missing}")

manifests = sorted(
    Path("artifacts/adaptive_v1").glob("*manifest*.json")
)

if not manifests:
    raise SystemExit("FAIL: adaptive manifest not found")

manifest_path = manifests[-1]
obj = json.loads(manifest_path.read_text(encoding="utf-8"))
systems = obj["systems"]

print("MANIFEST =", manifest_path)
print("SYSTEMS =", len(systems))

for row in systems:
    spec = SingleRouterSystemSpec(
        system_id=row["system_id"],
        split=row["split"],
        execution_mode=row["mode"],
        style=row["style"],
        worker_roles=tuple(row["roles"]),
    )

    # Dummy client: constructor validation only. ZERO API calls.
    SingleRouterFanoutMAS(spec, client=object())

print(f"PASS: all {len(systems)} adaptive systems instantiate correctly")
PY
