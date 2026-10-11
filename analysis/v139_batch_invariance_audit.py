#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_v13_6b_activation_judge as tr

TEST7 = [
    "code",
    "planning",
    "research",
    "database_design",
    "writing",
    "fact_checking",
    "experiment_design",
]

SRC = Path("artifacts/k3_same_query_confounds/summary.json")
OUT = Path("artifacts/v139_batch_audit")
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(SRC.read_text())
desc = dict(tr.ROLE_DESCRIPTIONS)
roles20 = list(desc)

tok, model, dtype = vg.load_sensor(
    Path("artifacts/v13_8b_balanced_repair_2048_run1/best_adapter")
)

print("tokenizer.padding_side =", tok.padding_side)
print("tokenizer.pad_token    =", tok.pad_token)
print("tokenizer.pad_token_id =", tok.pad_token_id)
print("model.pad_token_id     =", getattr(model.config, "pad_token_id", None))
print()

def score(roles, q, answer):
    z = vg.sensor_score_row(
        model, tok, dtype,
        list(roles), desc,
        q, answer, 4096
    )
    return dict(zip(roles, map(float, z)))

rows = []

for i, rec in enumerate(data["rows"], 1):
    q = rec["query"]
    ans = rec["answer"]
    src = rec["forced_source_eval_only"]
    comp = rec["competitor"]

    s1_src = score([src], q, ans)[src]
    s1_cmp = score([comp], q, ans)[comp]

    s7 = score(TEST7, q, ans)
    s20 = score(roles20, q, ans)
    s20r = score(list(reversed(roles20)), q, ans)

    row = {
        "i": i,
        "family": rec["family"],
        "source": src,
        "competitor": comp,
        "source_single": s1_src,
        "source_7": s7[src],
        "source_20": s20[src],
        "source_20_reversed": s20r[src],
        "competitor_single": s1_cmp,
        "competitor_7": s7[comp],
        "competitor_20": s20[comp],
        "competitor_20_reversed": s20r[comp],
    }

    rows.append(row)

    print("=" * 100)
    print(rec["family"], "source=", src, "competitor=", comp)
    print(
        "SOURCE:",
        f"single={s1_src:+.6f}",
        f"b7={s7[src]:+.6f}",
        f"b20={s20[src]:+.6f}",
        f"b20rev={s20r[src]:+.6f}",
    )
    print(
        "COMP  :",
        f"single={s1_cmp:+.6f}",
        f"b7={s7[comp]:+.6f}",
        f"b20={s20[comp]:+.6f}",
        f"b20rev={s20r[comp]:+.6f}",
    )

deltas = []
for r in rows:
    for side in ("source", "competitor"):
        base = r[f"{side}_single"]
        for proto in ("7", "20", "20_reversed"):
            deltas.append(abs(r[f"{side}_{proto}"] - base))

summary = {
    "n_cases": len(rows),
    "max_abs_logit_delta_vs_single": float(max(deltas)),
    "mean_abs_logit_delta_vs_single": float(np.mean(deltas)),
    "rows": rows,
}

print("\n" + "#" * 100)
print(json.dumps({
    k:v for k,v in summary.items() if k != "rows"
}, indent=2))

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)
