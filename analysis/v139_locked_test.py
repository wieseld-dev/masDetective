#!/usr/bin/env python3
import json
from pathlib import Path

import analysis.train_v13_6b_activation_judge as tr
import analysis.v139_overnight as v139
import analysis.build_visible_realized_gain_v3 as vg

BASE = Path("artifacts/v13_8b_balanced_repair_2048_run1/best_adapter")
NEW  = Path("artifacts/v139_pilots/mix50_retry/best_adapter")
OUT  = Path("artifacts/v139_locked_test_mix50.json")

rows = v139.load_worker_records()
rd = dict(tr.ROLE_DESCRIPTIONS)

test_items = v139.causal_items(
    rows,
    "test_locked",
    rd,
    tr,
)

assert len(test_items) == 216, len(test_items)

def run(adapter):
    tok, model, dtype = vg.load_sensor(adapter)
    return v139.eval_causal(
        model,
        tok,
        dtype,
        test_items,
    )

print("Evaluating frozen V13.8b baseline...")
base = run(BASE)

print("Evaluating selected V13.9 mix50 epoch1...")
new = run(NEW)

result = {
    "protocol": "v139_locked_test_once_v1",
    "selection_was_based_on": "VAL only",
    "selected_checkpoint": str(NEW),
    "baseline_checkpoint": str(BASE),
    "n_test_causal_pairs": len(test_items),
    "baseline": base,
    "selected_v139": new,
    "delta_accuracy":
        new["accuracy"] - base["accuracy"],
    "delta_mean_margin":
        new["mean_margin"] - base["mean_margin"],
}

OUT.write_text(json.dumps(result, indent=2))

print("\n" + "="*100)
print(json.dumps(result, indent=2))
