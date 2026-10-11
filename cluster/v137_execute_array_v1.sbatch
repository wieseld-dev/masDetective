#!/bin/bash
#SBATCH --job-name=v137-exec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/v137-exec-%A_%a.out
#SBATCH --error=logs/v137-exec-%A_%a.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

PY="$HOME/.conda/envs/masdetective/bin/python"

REGISTRY="artifacts/v137_presence_v1/execution_shards_v1/shards.json"
IDX="${SLURM_ARRAY_TASK_ID}"

mapfile -t META < <(
"$PY" - "$REGISTRY" "$IDX" <<'PY'
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
idx = int(sys.argv[2])

obj = json.loads(registry.read_text())

shards = obj["shards"]

if idx < 0 or idx >= len(shards):
    raise SystemExit(
        f"invalid shard index {idx}; n={len(shards)}"
    )

s = shards[idx]

print(s["root"])
print(s["manifest"])
print(s["rows"])
print(s["candidate_sha256"])
PY
)

CURR_ROOT="${META[0]}"
CURR_MANIFEST="${META[1]}"
EXPECTED_ROWS="${META[2]}"
EXPECTED_CAND_SHA="${META[3]}"

CAND="$CURR_ROOT/candidates_train_round_01.jsonl"
RESP="$CURR_ROOT/responses_train_round_01.jsonl"

echo "============================================================"
echo "V13.7 EXECUTION SHARD"
echo "============================================================"
echo "array job      : ${SLURM_ARRAY_JOB_ID}"
echo "array task     : ${SLURM_ARRAY_TASK_ID}"
echo "root           : ${CURR_ROOT}"
echo "manifest       : ${CURR_MANIFEST}"
echo "expected rows  : ${EXPECTED_ROWS}"
echo "model          : gpt-4o-mini"
echo "workers        : 4"
echo

[[ -f "$CAND" ]] || {
    echo "missing candidate file: $CAND" >&2
    exit 1
}

[[ -f "$CURR_MANIFEST" ]] || {
    echo "missing manifest: $CURR_MANIFEST" >&2
    exit 1
}

ACTUAL_ROWS=$(wc -l < "$CAND")

if [[ "$ACTUAL_ROWS" -ne "$EXPECTED_ROWS" ]]; then
    echo \
      "candidate row mismatch: actual=$ACTUAL_ROWS expected=$EXPECTED_ROWS" \
      >&2
    exit 1
fi

ACTUAL_SHA=$(
"$PY" - "$CAND" <<'PY'
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])

print(
    hashlib.sha256(
        p.read_bytes()
    ).hexdigest()
)
PY
)

if [[ "$ACTUAL_SHA" != "$EXPECTED_CAND_SHA" ]]; then
    echo "candidate SHA mismatch" >&2
    echo "expected=$EXPECTED_CAND_SHA" >&2
    echo "actual  =$ACTUAL_SHA" >&2
    exit 1
fi

echo "candidate integrity: PASS"
echo

"$PY" -u scripts/adaptive_v11_execute_round.py \
    --round 1 \
    --split train \
    --root "$CURR_ROOT" \
    --manifest "$CURR_MANIFEST" \
    --model gpt-4o-mini \
    --workers 4 \
    --retries 3

[[ -f "$RESP" ]] || {
    echo "response file missing: $RESP" >&2
    exit 1
}

NRESP=$(wc -l < "$RESP")

if [[ "$NRESP" -ne "$EXPECTED_ROWS" ]]; then
    echo \
      "response row mismatch: actual=$NRESP expected=$EXPECTED_ROWS" \
      >&2
    exit 1
fi

echo
echo "response rows: $NRESP"
echo "SHARD EXECUTION INTEGRITY: PASS"
