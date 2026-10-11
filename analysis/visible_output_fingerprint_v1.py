#!/usr/bin/env python3
"""
visible_output_fingerprint_v1.py

Direct black-box fingerprint test.

Question:
Does the VISIBLE final MAS output contain information about hidden agent
activation beyond what was already predictable from the probe text itself?

We compare, on the SAME held-out systems:

  Q       : query only
  O       : final output only
  Q+O     : query + final output
  Q+O-red : query + output after literal role-name redaction

Targets:
  1) selected_roles_training_only  -> which roles actually activated
  2) truth_roles_training_only     -> which roles exist in the hidden MAS

The decisive comparison for "observable fingerprint" is:

    Q+O  versus  Q

because Q controls for what the probe itself already implies.

Q+O-red is a stronger sanity check: if it still improves over Q-only after
literal role names are removed, the signal is not merely the answer spelling
out "risk assessment", "requirements analysis", etc.

Splits:
Uses the exact train/val/test system split from observation_model_v1.
No held-out system appears in training.

Outputs:
  summary.json
  per_role_metrics.json
  output_embeddings.npy
  output_redacted_embeddings.npy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_obs_module():
    sys.path.insert(0, str(Path("analysis").resolve()))
    import observation_model_v1 as om
    return om


def read_jsonl(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_round(path: Path):
    m = re.search(r"round_(\d+)", path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def build_output_map(root: Path):
    """
    key = (parent, system_id/session_id, round, candidate_index)
    """
    mp = {}
    duplicates = 0

    for p in sorted(root.rglob("responses_train_round_*.jsonl")):
        parent = str(p.parent)
        fr = parse_round(p)
        for x in read_jsonl(p):
            sid = x.get("system_id") or x.get("session_id")
            rnd = int(x.get("round", fr))
            ci = int(x.get("candidate_index") or 0)

            output = (
                x.get("final_output")
                or x.get("output")
                or x.get("answer")
                or ""
            )
            key = (parent, sid, rnd, ci)
            if key in mp:
                duplicates += 1
            mp[key] = str(output)

    return mp, duplicates


def redact_role_names(text: str, roles):
    s = text
    # longest first to avoid partial replacement problems
    variants = []
    for r in roles:
        base = r.replace("_", " ")
        variants.extend([
            r,
            base,
            base.replace(" ", "-"),
        ])

    variants = sorted(set(variants), key=len, reverse=True)
    for v in variants:
        if not v:
            continue
        s = re.sub(re.escape(v), "[ROLE]", s, flags=re.IGNORECASE)
    return s


def encode_texts(texts, model_name, device, batch_size):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    enc = SentenceTransformer(model_name, device=device)
    z = enc.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return z.astype(np.float32)


def labels_from_examples(examples, roles, target):
    ri = {r:i for i,r in enumerate(roles)}
    Y = np.zeros((len(examples), len(roles)), dtype=np.int32)

    field = "selected_roles" if target == "selected" else "truth_roles"
    for i, x in enumerate(examples):
        for r in x[field]:
            if r in ri:
                Y[i, ri[r]] = 1
    return Y


def fit_ovr(X, Y):
    models = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if len(np.unique(y)) < 2:
            models.append(None)
            continue
        clf = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        )
        clf.fit(X, y)
        models.append(clf)
    return models


def predict_ovr(models, X):
    P = np.zeros((len(X), len(models)), dtype=np.float64)
    for j, clf in enumerate(models):
        if clf is None:
            P[:, j] = 0.0
        else:
            P[:, j] = clf.predict_proba(X)[:, 1]
    return P


def tune_thresholds(Y, P):
    ts = np.full(Y.shape[1], 0.5, dtype=np.float64)
    grid = np.arange(0.05, 0.951, 0.05)

    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in grid:
            yp = (P[:, j] >= t).astype(int)
            f = f1_score(y, yp, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        ts[j] = best_t

    return ts


def metrics(Y, P, thresholds, roles):
    pred = (P >= thresholds[None, :]).astype(int)

    out = {
        "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(Y, pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(Y, pred)),
        "exact_set_accuracy": float(np.mean(np.all(pred == Y, axis=1))),
        "true_cardinality_mean": float(Y.sum(axis=1).mean()),
        "pred_cardinality_mean": float(pred.sum(axis=1).mean()),
        "prob_cardinality_mean": float(P.sum(axis=1).mean()),
    }

    try:
        out["micro_auroc"] = float(roc_auc_score(Y, P, average="micro"))
    except Exception:
        out["micro_auroc"] = None

    aucs, aps = [], []
    per_role = {}

    for j, r in enumerate(roles):
        y = Y[:, j]
        item = {
            "support_positive": int(y.sum()),
            "threshold": float(thresholds[j]),
        }
        if len(np.unique(y)) >= 2:
            try:
                item["auroc"] = float(roc_auc_score(y, P[:, j]))
                aucs.append(item["auroc"])
            except Exception:
                item["auroc"] = None
            try:
                item["ap"] = float(average_precision_score(y, P[:, j]))
                aps.append(item["ap"])
            except Exception:
                item["ap"] = None
        else:
            item["auroc"] = None
            item["ap"] = None

        item["f1"] = float(
            f1_score(y, pred[:, j], zero_division=0)
        )
        per_role[r] = item

    out["macro_auroc_valid_roles"] = float(np.mean(aucs)) if aucs else None
    out["macro_ap_valid_roles"] = float(np.mean(aps)) if aps else None
    return out, per_role


def delta(a, b):
    # a - b for common numeric keys
    out = {}
    for k in a:
        if k not in b:
            continue
        if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
            if a[k] is not None and b[k] is not None:
                out[k] = float(a[k] - b[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir", default="artifacts/observation_model_v1")
    ap.add_argument("--out", default="artifacts/visible_output_fingerprint_v1")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--embed-batch-size", type=int, default=128)
    args = ap.parse_args()

    om = load_obs_module()
    root = Path(args.root)
    obs_dir = Path(args.obs_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples, roles, parents, skipped = om.collect_examples(root)
    output_map, duplicate_outputs = build_output_map(root)

    outputs = []
    missing = 0
    for x in examples:
        key = (
            x["parent"],
            x["system_id"],
            x["round"],
            x["candidate_index"],
        )
        s = output_map.get(key, "")
        if not s:
            missing += 1
        outputs.append(s)

    if missing:
        raise RuntimeError(f"Missing visible output for {missing}/{len(examples)} examples")

    outputs_red = [redact_role_names(s, roles) for s in outputs]

    split = json.loads((obs_dir / "split_systems.json").read_text())
    trsys, vasys, tesys = set(split["train"]), set(split["val"]), set(split["test"])

    train_idx = np.array([i for i,x in enumerate(examples) if x["system_id"] in trsys], dtype=int)
    val_idx = np.array([i for i,x in enumerate(examples) if x["system_id"] in vasys], dtype=int)
    test_idx = np.array([i for i,x in enumerate(examples) if x["system_id"] in tesys], dtype=int)
    test_r1_idx = np.array([i for i in test_idx if examples[i]["round"] == 1], dtype=int)

    qemb_path = obs_dir / "query_embeddings.npy"
    if not qemb_path.exists():
        raise RuntimeError(f"Missing existing query embeddings: {qemb_path}")
    Q = np.load(qemb_path).astype(np.float32)
    if len(Q) != len(examples):
        raise RuntimeError("Query embedding/example count mismatch")

    O_path = out / "output_embeddings.npy"
    OR_path = out / "output_redacted_embeddings.npy"

    if O_path.exists():
        O = np.load(O_path)
    else:
        print("Encoding visible final outputs...")
        O = encode_texts(outputs, args.text_model, args.device, args.embed_batch_size)
        np.save(O_path, O)

    if OR_path.exists():
        OR = np.load(OR_path)
    else:
        print("Encoding role-name-redacted outputs...")
        OR = encode_texts(outputs_red, args.text_model, args.device, args.embed_batch_size)
        np.save(OR_path, OR)

    feature_sets = {
        "query_only": Q,
        "output_only": O,
        "query_plus_output": np.concatenate([Q, O], axis=1),
        "query_plus_redacted_output": np.concatenate([Q, OR], axis=1),
    }

    result = {
        "setting": {
            "examples": len(examples),
            "roles": len(roles),
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_idx),
            "test_round1_rows": len(test_r1_idx),
            "train_systems": len(trsys),
            "val_systems": len(vasys),
            "test_systems": len(tesys),
            "split_by_system": True,
            "visible_features_only": [
                "query",
                "final_output/output/answer"
            ],
            "hidden_labels_used_only_for_training_and_evaluation": [
                "selected_roles_training_only",
                "truth_roles_training_only"
            ],
            "duplicate_output_keys": duplicate_outputs,
        },
        "targets": {}
    }

    per_role_dump = {}

    for target in ["selected", "truth"]:
        print("\nTARGET:", target)
        Y = labels_from_examples(examples, roles, target)
        target_res = {"all_test": {}, "round1_test": {}}
        per_role_dump[target] = {}

        for name, X in feature_sets.items():
            print("  fitting", name)
            models = fit_ovr(X[train_idx], Y[train_idx])

            Pval = predict_ovr(models, X[val_idx])
            thresholds = tune_thresholds(Y[val_idx], Pval)

            Ptest = predict_ovr(models, X[test_idx])
            Mall, PRall = metrics(Y[test_idx], Ptest, thresholds, roles)
            target_res["all_test"][name] = Mall

            if len(test_r1_idx):
                Pr1 = predict_ovr(models, X[test_r1_idx])
                Mr1, PRr1 = metrics(Y[test_r1_idx], Pr1, thresholds, roles)
                target_res["round1_test"][name] = Mr1
            else:
                PRr1 = {}

            per_role_dump[target][name] = {
                "all_test": PRall,
                "round1_test": PRr1,
            }

        # Critical output-added deltas.
        for scope in ["all_test", "round1_test"]:
            q = target_res[scope]["query_only"]
            qo = target_res[scope]["query_plus_output"]
            qor = target_res[scope]["query_plus_redacted_output"]
            o = target_res[scope]["output_only"]

            target_res[scope]["fingerprint_deltas"] = {
                "query_plus_output_minus_query": delta(qo, q),
                "query_plus_redacted_output_minus_query": delta(qor, q),
                "output_only_minus_query": delta(o, q),
            }

        result["targets"][target] = target_res

    (out / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "per_role_metrics.json").write_text(
        json.dumps(per_role_dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "="*100)
    print("VISIBLE OUTPUT FINGERPRINT V1")
    print("="*100)

    for target in ["selected", "truth"]:
        print("\nTARGET:", target)
        for scope in ["round1_test", "all_test"]:
            print("\n ", scope)
            block = result["targets"][target][scope]
            for name in [
                "query_only",
                "output_only",
                "query_plus_output",
                "query_plus_redacted_output",
            ]:
                m = block[name]
                print(
                    f"{name:28s} "
                    f"microF1={m['micro_f1']:.4f} "
                    f"macroF1={m['macro_f1']:.4f} "
                    f"microAUC={m['micro_auroc']:.4f} "
                    f"macroAUC={m['macro_auroc_valid_roles']:.4f}"
                )

            print("  DELTA Q+O - Q:",
                  block["fingerprint_deltas"]["query_plus_output_minus_query"])
            print("  DELTA Q+O-red - Q:",
                  block["fingerprint_deltas"]["query_plus_redacted_output_minus_query"])

    print("\n[saved]", out / "summary.json")
    print("[saved]", out / "per_role_metrics.json")


if __name__ == "__main__":
    main()
