#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

OUT="artifacts/night_20260926/interface_dump.txt"
mkdir -p "$(dirname "$OUT")" logs

exec > >(tee "$OUT") 2>&1

echo "============================================================"
echo " NIGHT INTERFACE DUMP"
echo "============================================================"
date
hostname
pwd

echo
echo "===== PYTHON ====="
"$PY" --version
"$PY" - <<'PY'
import sys
print("executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch inspect failed:", repr(e))
PY

echo
echo "===== ENV PRESENCE (NO SECRETS PRINTED) ====="
if [[ -f .env ]]; then
    echo ".env: PRESENT"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo ".env: MISSING"
fi

[[ -n "${OPENAI_API_KEY:-}" ]] && echo "OPENAI_API_KEY: PRESENT" || echo "OPENAI_API_KEY: MISSING"
[[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN: PRESENT" || echo "HF_TOKEN: MISSING/OPTIONAL"

echo
echo "===== GIT ====="
git status --short || true
git rev-parse --abbrev-ref HEAD || true
git rev-parse HEAD || true

FILES=(
  analysis/one_system_k3_demo.py
  analysis/adaptive_fanout_count_v13.py
  targets/single_router_fanout_mas.py
  v11/role_universe.py
  analysis/train_v7_benign_output_sensor_pilot_v1.py
  analysis/v7_structured_reranker_v1.py
  analysis/v7_rolewise_calibration_v1.py
  analysis/v7_targeted_toprank_v1.py
)

echo
echo "===== REQUIRED FILES ====="
for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "OK   $f"
    else
        echo "MISS $f"
    fi
done

echo
echo "===== REQUIRED ARTIFACTS ====="
PATHS=(
  artifacts/v7_benign_output_sensor_pilot_v1/best_adapter
  artifacts/v7_benign_final_pool_v2/train.jsonl
  artifacts/v7_benign_final_pool_v2/dev.jsonl
  artifacts/v7_benign_final_pool_v2/test.jsonl
  artifacts/v7_rolewise_calibration_v1
  artifacts/v7_structured_reranker_v1
  artifacts/v7_targeted_toprank_v1/targeted_ranker.pt
)

for p in "${PATHS[@]}"; do
    if [[ -e "$p" ]]; then
        echo "OK   $p"
    else
        echo "MISS $p"
    fi
done

echo
echo "===== PY_COMPILE ====="
"$PY" -m py_compile \
  analysis/one_system_k3_demo.py \
  analysis/adaptive_fanout_count_v13.py \
  targets/single_router_fanout_mas.py \
  v11/role_universe.py \
  analysis/train_v7_benign_output_sensor_pilot_v1.py \
  analysis/v7_structured_reranker_v1.py \
  analysis/v7_rolewise_calibration_v1.py \
  analysis/v7_targeted_toprank_v1.py
echo "PY_COMPILE: PASS"

echo
echo "===== TOP LEVEL SYMBOLS / SIGNATURES ====="
"$PY" - <<'PY'
from __future__ import annotations
import ast
from pathlib import Path

files = [
    "analysis/one_system_k3_demo.py",
    "analysis/adaptive_fanout_count_v13.py",
    "targets/single_router_fanout_mas.py",
    "v11/role_universe.py",
    "analysis/train_v7_benign_output_sensor_pilot_v1.py",
    "analysis/v7_structured_reranker_v1.py",
    "analysis/v7_rolewise_calibration_v1.py",
    "analysis/v7_targeted_toprank_v1.py",
]

for fp in files:
    p = Path(fp)
    print("\n" + "=" * 80)
    print(fp)
    print("=" * 80)
    src = p.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            header = lines[node.lineno - 1].strip()
            print(f"{node.lineno:5d}-{end:5d}  {header}")
PY

echo
echo "===== KEY CODE LOCATIONS ====="

for f in \
  analysis/one_system_k3_demo.py \
  analysis/adaptive_fanout_count_v13.py \
  targets/single_router_fanout_mas.py \
  analysis/train_v7_benign_output_sensor_pilot_v1.py \
  analysis/v7_structured_reranker_v1.py \
  analysis/v7_rolewise_calibration_v1.py \
  analysis/v7_targeted_toprank_v1.py
do
    echo
    echo "### $f"
    grep -nE \
      'SingleRouter|SystemSpec|worker_roles|TRAIN_ROLES|ROLE_DESCRIPTIONS|load_sensor|load_model|sensor_score|sensor_log|calibrated|TargetedRanker|build_features|predict|torch\.save|torch\.load|PeftModel|AutoModelForSequenceClassification|Dataset|DataLoader|loss|optimizer|scheduler|selected_roles|_last_trace|belief|log_odds|query\(' \
      "$f" | head -220 || true
done

echo
echo "===== ROLE UNIVERSE ====="
"$PY" - <<'PY'
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS
print("N TRAIN_ROLES =", len(TRAIN_ROLES))
for i, r in enumerate(TRAIN_ROLES):
    print(i, repr(r), "=>", repr(ROLE_DESCRIPTIONS.get(r)))
PY

echo
echo "===== DATASET ROW SCHEMA ====="
"$PY" - <<'PY'
import json
from pathlib import Path

for split in ("train", "dev", "test"):
    p = Path(f"artifacts/v7_benign_final_pool_v2/{split}.jsonl")
    print("\n---", split, "---")
    with p.open() as f:
        row = json.loads(next(f))

    print("keys:", sorted(row.keys()))
    for k, v in row.items():
        if isinstance(v, str):
            vv = v.replace("\n", "\\n")
            if len(vv) > 500:
                vv = vv[:500] + "...<CUT>"
            print(f"{k}: {vv!r}")
        else:
            print(f"{k}: {v!r}")
PY

echo
echo "===== ARTIFACT FILE LISTS ====="
for d in \
  artifacts/v7_benign_output_sensor_pilot_v1 \
  artifacts/v7_structured_reranker_v1 \
  artifacts/v7_rolewise_calibration_v1 \
  artifacts/v7_targeted_toprank_v1
do
    echo
    echo "--- $d ---"
    find "$d" -maxdepth 2 -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort
done

echo
echo "===== JSON ARTIFACT CONTENT / KEYS ====="
"$PY" - <<'PY'
from pathlib import Path
import json

roots = [
    Path("artifacts/v7_structured_reranker_v1"),
    Path("artifacts/v7_rolewise_calibration_v1"),
    Path("artifacts/v7_targeted_toprank_v1"),
]

for root in roots:
    print("\n###", root)
    if not root.exists():
        continue

    for p in sorted(root.rglob("*.json")):
        print("\nFILE:", p)
        try:
            x = json.loads(p.read_text())
            if isinstance(x, dict):
                print("TOP KEYS:", sorted(x.keys()))
                text = json.dumps(x, indent=2)
                print(text[:8000])
            else:
                print(type(x).__name__, repr(x)[:8000])
        except Exception as e:
            print("ERROR:", repr(e))
PY

echo
echo "===== TARGETED RANKER PT KEYS ====="
"$PY" - <<'PY'
from pathlib import Path
import torch

p = Path("artifacts/v7_targeted_toprank_v1/targeted_ranker.pt")
if not p.exists():
    print("MISSING")
else:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    print("type:", type(obj))
    if isinstance(obj, dict):
        print("keys:", sorted(obj.keys()))
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print(k, "shape=", tuple(v.shape), "dtype=", getattr(v, "dtype", None))
            elif isinstance(v, dict):
                print(k, "dict keys=", list(v.keys())[:100])
            else:
                s = repr(v)
                print(k, s[:1500])
PY

echo
echo "===== ADAPTER CONFIG ====="
"$PY" - <<'PY'
from pathlib import Path
import json

p = Path("artifacts/v7_benign_output_sensor_pilot_v1/best_adapter/adapter_config.json")
if p.exists():
    x = json.loads(p.read_text())
    print(json.dumps(x, indent=2))
else:
    print("adapter_config.json MISSING")
PY

echo
echo "===== ONE SYSTEM DEMO FULL SOURCE ====="
sed -n '1,420p' analysis/one_system_k3_demo.py

echo
echo "===== TARGET CLASS RELEVANT SOURCE ====="
grep -nE \
'class SingleRouter|def __init__|def query|def .*trace|selected_roles|worker_roles|execution_mode' \
targets/single_router_fanout_mas.py || true

echo
echo "===== V7 SENSOR IMPORTANT SOURCE WINDOWS ====="
"$PY" - <<'PY'
from pathlib import Path
import re

targets = {
    "analysis/train_v7_benign_output_sensor_pilot_v1.py": [
        r"load_model", r"format", r"score", r"Dataset", r"collat",
        r"train", r"loss", r"evaluate"
    ],
    "analysis/v7_structured_reranker_v1.py": [
        r"load_sensor", r"sensor_score", r"structured", r"predict",
        r"load.*rerank", r"save"
    ],
    "analysis/v7_rolewise_calibration_v1.py": [
        r"load_adapter", r"calibr", r"predict", r"score"
    ],
    "analysis/v7_targeted_toprank_v1.py": [
        r"TargetedRanker", r"build_features", r"predict"
    ],
}

for fp, patterns in targets.items():
    print("\n" + "="*100)
    print(fp)
    print("="*100)
    src = Path(fp).read_text()
    tree = __import__("ast").parse(src)
    lines = src.splitlines()

    shown = set()
    for node in tree.body:
        if not isinstance(node, (__import__("ast").FunctionDef,
                                 __import__("ast").AsyncFunctionDef,
                                 __import__("ast").ClassDef)):
            continue
        name = getattr(node, "name", "")
        header = lines[node.lineno - 1]
        if not any(re.search(p, name, re.I) or re.search(p, header, re.I)
                   for p in patterns):
            continue
        key = (node.lineno, getattr(node, "end_lineno", node.lineno))
        if key in shown:
            continue
        shown.add(key)

        a = max(1, node.lineno - 3)
        b = min(len(lines), getattr(node, "end_lineno", node.lineno) + 3)
        print(f"\n--- lines {a}-{b} ---")
        print("\n".join(f"{i:5d}: {lines[i-1]}" for i in range(a, b+1)))
PY

echo
echo "============================================================"
echo " INTERFACE DUMP COMPLETE"
echo " SAVED TO: $OUT"
echo "============================================================"
