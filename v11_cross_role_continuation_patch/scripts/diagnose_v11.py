from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="artifacts/v11_summary.json")
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print("=== V11 CROSS-ROLE DIAGNOSTIC ===")
    print("starting exact V10.1 adapter:", s.get("starting_adapter_verified_exact_v10"))
    print("train role types:", len(s.get("train_roles", [])))
    print("unseen dev role types:", len(s.get("dev_unseen_roles", [])))
    print("train candidates:", s.get("train_candidates"))
    print("train evaluated:", s.get("train_evaluated"))
    print("confirmed elites:", s.get("train_confirmed_elites"))
    print("unseen-role any activation >=.75:", f"{s.get('holdout_role_any_activation_ge_075')}/{len(s.get('dev_unseen_roles', []))}")
    print("unseen-role any strict SPEC:", f"{s.get('holdout_role_any_strict_specific')}/{len(s.get('dev_unseen_roles', []))}")
    print("unseen-role any strict DELTA:", f"{s.get('holdout_role_any_strict_delta')}/{len(s.get('dev_unseen_roles', []))}")
    print()
    for role, r in s.get("holdout_role_stats", {}).items():
        print(
            f"{role:<24} valid={r.get('valid')}/{r.get('total')} "
            f"ACT={r.get('best_activation')} ROB={r.get('best_robust_specific')} "
            f"strict={r.get('strict_specific_count')} any={r.get('any_strict_specific')}"
        )
    print()
    print("IMPORTANT: this is unseen-role DEVELOPMENT validation, not a final untouched paper test.")


if __name__ == "__main__":
    main()
