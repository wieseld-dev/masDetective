#!/bin/bash
#SBATCH --job-name=v14a-pre
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v14a-pre-%j.out
#SBATCH --error=logs/v14a-pre-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"

"$PY" - <<'PY'
from pathlib import Path
import json

from v14.adaptive_belief_policy import V14AdaptiveBeliefPolicy

root = Path("artifacts/v14_adaptive_role_v1")

required = [
    root / "manifest_privileged.json",
    root / "frozen_belief.joblib",
    root / "states_train_round_00.jsonl",
    root / "states_dev_round_00.jsonl",
    root / "frozen_v14.sha256",
    Path("artifacts/v14_benign_policy/adapter_config.json"),
]

for p in required:
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"missing/empty: {p}")

g = Path("scripts/v14_adaptive_generate_round_v1.py").read_text()
u = Path("scripts/v14_adaptive_grpo_update_v1.py").read_text()

if "V14AdaptiveBeliefPolicy" not in g:
    raise RuntimeError("generator is not using V14AdaptiveBeliefPolicy")

if "V14AdaptiveBeliefPolicy" not in u:
    raise RuntimeError("updater is not using V14AdaptiveBeliefPolicy")

if "artifacts/v11_policy" in g:
    raise RuntimeError("generator still references V11 adapter")

if "artifacts/v11_policy" in u:
    raise RuntimeError("updater still references V11 adapter")

M = json.loads(
    (root / "manifest_privileged.json").read_text()
)

print(
    "manifest train =",
    sum(x["split"] == "train" for x in M["systems"]),
)

print(
    "manifest dev   =",
    sum(x["split"] == "dev" for x in M["systems"]),
)

print("V14-ADAPTIVE PREFLIGHT: PASS")
PY
