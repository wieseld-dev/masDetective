from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("analysis").resolve()))
import q38_vs_r7_utility_summary_v1 as old


OLD_PARTS = Path("artifacts/q38_vs_r7_utility_v1/parts")
NEW_PARTS = Path("artifacts/q38_utility_lora_eval_v1/parts")
OUT = Path("artifacts/q38_utility_lora_eval_v1")

R7_ARM = "R7_QWEN25_1P5B"
LORA_ARM = "QWEN38_27B_UTILITY_LORA"


def read_jsonl_dir(root):
    rows = []
    for p in sorted(root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    old_rows_all = read_jsonl_dir(OLD_PARTS)
    r7_rows = [x for x in old_rows_all if x["arm"] == R7_ARM]

    lora_rows = read_jsonl_dir(NEW_PARTS)
    lora_rows = [x for x in lora_rows if x["arm"] == LORA_ARM]

    print("OLD R7 ROWS =", len(r7_rows))
    print("NEW LoRA ROWS =", len(lora_rows))

    assert len(r7_rows) == 100, len(r7_rows)
    assert len(lora_rows) == 100, len(lora_rows)

    arms = {
        R7_ARM: old.summarize(r7_rows),
        LORA_ARM: old.summarize(lora_rows),
    }

    per_role = {}

    for role in sorted({x["role"] for x in r7_rows + lora_rows}):
        rr = [x for x in r7_rows if x["role"] == role]
        ll = [x for x in lora_rows if x["role"] == role]

        assert len(rr) == 5, (role, len(rr))
        assert len(ll) == 5, (role, len(ll))

        per_role[role] = {
            R7_ARM: old.summarize(rr),
            LORA_ARM: old.summarize(ll),
        }

    r7 = arms[R7_ARM]
    lora = arms[LORA_ARM]

    comparison = {
        "lora_minus_r7_presence_auc":
            lora["target_presence_auc"] - r7["target_presence_auc"],

        "lora_minus_r7_pair_accuracy":
            lora["pair_accuracy"] - r7["pair_accuracy"],

        "lora_minus_r7_mean_margin":
            lora["mean_pair_margin"] - r7["mean_pair_margin"],

        "lora_minus_r7_activation_rate":
            lora["target_activation_rate_plus_eval_only"]
            - r7["target_activation_rate_plus_eval_only"],

        "roles_lora_higher_mean_margin":
            sum(
                per_role[r][LORA_ARM]["mean_pair_margin"]
                > per_role[r][R7_ARM]["mean_pair_margin"]
                for r in per_role
            ),

        "roles_total": len(per_role),
    }

    old_summary = json.loads(
        Path("artifacts/q38_vs_r7_utility_v1/summary.json").read_text()
    )

    base_ref = old_summary["arms"]["QWEN38_27B_BASE"]

    summary = {
        "experiment": "q38_utility_lora_vs_frozen_r7_v1",
        "definition":
            "New Qwen3.8-27B Utility-LoRA probes evaluated on the same "
            "matched PLUS/MINUS protocol; R7 rows reused from the original "
            "frozen benchmark without rerunning R7.",
        "truth_used_for": "evaluation only",
        "arms": arms,
        "comparison_lora_vs_r7": comparison,
        "historical_q38_base_reference": base_ref,
        "per_role": per_role,
    }

    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print()
    print("=" * 100)
    print("Q38 UTILITY-LoRA vs FROZEN R7")
    print("=" * 100)

    print("\nHISTORICAL Q38 BASE")
    print("  presence AUC     =", f"{base_ref['target_presence_auc']:.4f}")
    print("  pair accuracy    =", f"{base_ref['pair_accuracy']:.4f}")
    print("  mean pair margin =", f"{base_ref['mean_pair_margin']:+.5f}")

    for arm in (R7_ARM, LORA_ARM):
        s = arms[arm]
        print("\n" + arm)
        print("  presence AUC     =", f"{s['target_presence_auc']:.4f}")
        print("  pair accuracy    =", f"{s['pair_accuracy']:.4f}")
        print("  mean pair margin =", f"{s['mean_pair_margin']:+.5f}")
        print(
            "  PLUS activation  =",
            f"{s['target_activation_rate_plus_eval_only']:.4f}",
        )

    print("\nCOMPARISON")
    print(json.dumps(comparison, indent=2))
    print("\nQ38_UTILITY_LORA_EVAL_OK")


if __name__ == "__main__":
    main()
