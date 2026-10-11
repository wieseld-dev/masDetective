#!/bin/bash
#SBATCH --job-name=oldgen-bg
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=cs-4090-09
#SBATCH --output=logs/oldgen-bg-%j.out
#SBATCH --error=logs/oldgen-bg-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

PY=/home/wieseld/.conda/envs/masdetective/bin/python

export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

# Load API credentials used by the black-box MAS.
if [ -f .env ]; then
    set +u
    set -a
    source .env
    set +a
    set -u
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY still missing after loading .env" >&2
    exit 2
fi

echo "OPENAI_API_KEY: loaded"

OUT="artifacts/ten_system_belief_trace_oldgen_benign_guard_v1"

rm -rf "$OUT"
mkdir -p "$OUT" logs

echo "============================================================"
echo "OLD GENERATOR + LEARNED BENIGN HARD GUARD"
echo "============================================================"
echo "Historical reference mean AUC: 0.7922"
echo "Historical reference median AUC: 0.8922"
echo
echo "Changed component:"
echo "  + standalone_benign_critic_binary_v4_hardening"
echo
echo "Unchanged:"
echo "  + CurriculumAdaptiveProbePolicyV11"
echo "  + robust_visible_generator_v1/policy"
echo "  + ObservationNet"
echo "  + MI selector"
echo "  + visible fingerprint"
echo "  + particle Bayesian update"
echo "  + 10 MAS"
echo "  + 8 rounds"
echo "  + exactly 8 accepted candidates per round"
echo "============================================================"

"$PY" \
  analysis/ten_system_belief_trace_oldgen_benign_guard_v1.py

echo
echo "===== FINAL SUMMARY ====="
cat "$OUT/summary.json"

echo
echo "===== GUARD AUDIT COUNTS ====="

"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

p = Path(
    "artifacts/"
    "ten_system_belief_trace_oldgen_benign_guard_v1/"
    "benign_guard_all_generated.jsonl"
)

rows = [
    json.loads(x)
    for x in p.read_text().splitlines()
    if x.strip()
]

accepted = [
    x for x in rows
    if x["accepted"]
]

rejected = [
    x for x in rows
    if not x["accepted"]
]

print("total generated =", len(rows))
print("accepted        =", len(accepted))
print("rejected        =", len(rejected))

print(
    "acceptance rate =",
    len(accepted) / max(1, len(rows)),
)

if accepted:
    a = np.array(
        [x["p_benign"] for x in accepted],
        dtype=float,
    )
    print("accepted Pben mean =", float(a.mean()))
    print("accepted Pben min  =", float(a.min()))
    print("accepted Pben p25  =", float(np.quantile(a, .25)))

if rejected:
    r = np.array(
        [x["p_benign"] for x in rejected],
        dtype=float,
    )
    print("rejected Pben mean =", float(r.mean()))
    print("rejected Pben max  =", float(r.max()))

# There must be 10 systems × 8 rounds × 8 candidate slots.
expected_accepted = 10 * 8 * 8

print()
print("EXPECTED ACCEPTED =", expected_accepted)
print("ACTUAL ACCEPTED   =", len(accepted))

if len(accepted) != expected_accepted:
    raise SystemExit(
        "FAIL: accepted candidate count is not exactly 640"
    )

print("EXACT-8 GUARANTEE AUDIT: PASS")
PY
