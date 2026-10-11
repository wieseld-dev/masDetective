from pathlib import Path
import json
import gc
import torch

import analysis.train_general_role_sensor_v1 as gs


def print_metrics(name, m):
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(f"global_auc          = {m['global_auc']:.6f}")
    print(f"macro_role_auc      = {m['macro_role_auc']:.6f}")
    print(f"best_balanced_acc   = {m['best_balanced_accuracy']:.6f}")
    print(f"best_threshold      = {m['best_threshold']:.6f}")
    print(f"pair_accuracy       = {m['pair_accuracy']:.6f}")
    print(f"mean_pair_margin    = {m['mean_pair_margin']:.6f}")
    print(f"median_pair_margin  = {m['median_pair_margin']:.6f}")
    print(f"positive_margin_rate= {m['positive_margin_rate']:.6f}")

    print()
    print("PROBE SOURCE AUC:")
    for k, v in sorted(m["probe_source_auc"].items()):
        print(f"  {k:<24} {v:.6f}")

    print()
    print("PER-ROLE AUC:")
    for role, auc in sorted(
        m["per_role_auc"].items(),
        key=lambda x: -x[1],
    ):
        print(f"  {role:<32} {auc:.6f}")


def main():
    train_rows = gs.read_jsonl(gs.TRAIN_PATH)
    val_rows = gs.read_jsonl(gs.VAL_PATH)

    train_pairs = gs.build_pairs(train_rows)
    val_pairs = gs.build_pairs(val_rows)

    best_dir = Path(
        "artifacts/general_role_sensor_model_v1/best_adapter"
    )

    print("TRAIN rows :", len(train_rows))
    print("TRAIN pairs:", len(train_pairs))
    print("VAL rows   :", len(val_rows))
    print("VAL pairs  :", len(val_pairs))
    print("adapter    :", best_dir)

    tok, model = gs.make_model(
        gs.MODEL_NAME,
        trainable=False,
        adapter=best_dir,
    )

    print()
    print("EVALUATING TRAIN...", flush=True)

    train_metrics, _ = gs.eval_full(
        model,
        tok,
        train_rows,
        train_pairs,
    )

    print_metrics(
        "SEEN-ROLE TRAIN PERFORMANCE",
        train_metrics,
    )

    print()
    print("EVALUATING UNSEEN-ROLE VAL...", flush=True)

    val_metrics, _ = gs.eval_full(
        model,
        tok,
        val_rows,
        val_pairs,
    )

    print_metrics(
        "UNSEEN-ROLE VAL PERFORMANCE",
        val_metrics,
    )

    result = {
        "train_seen_roles": train_metrics,
        "val_unseen_roles": val_metrics,
    }

    out = Path(
        "artifacts/general_role_sensor_model_v1/"
        "seen_vs_unseen_diagnostic.json"
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("SEEN vs UNSEEN")
    print("=" * 100)

    print(
        "global_auc:",
        f"{train_metrics['global_auc']:.6f}",
        "->",
        f"{val_metrics['global_auc']:.6f}",
    )

    print(
        "macro_role_auc:",
        f"{train_metrics['macro_role_auc']:.6f}",
        "->",
        f"{val_metrics['macro_role_auc']:.6f}",
    )

    print(
        "pair_accuracy:",
        f"{train_metrics['pair_accuracy']:.6f}",
        "->",
        f"{val_metrics['pair_accuracy']:.6f}",
    )

    print()
    print("saved:", out)


if __name__ == "__main__":
    main()
