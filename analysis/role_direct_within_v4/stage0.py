from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())

def auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))

def validate(rows, name):
    required = {
        "system_id", "target_role",
        "query", "final_output", "label"
    }
    if not rows:
        raise RuntimeError(f"{name}: empty dataset")

    for i, row in enumerate(rows):
        if required - set(row):
            raise RuntimeError(
                f"{name}: missing fields in row {i}"
            )
        if int(row["label"]) not in (0, 1):
            raise RuntimeError("Invalid label")
        for key in (
            "system_id", "target_role",
            "query", "final_output"
        ):
            if not str(row[key]).strip():
                raise RuntimeError(f"Blank {key}")

    return {
        "rows": len(rows),
        "systems": len({r["system_id"] for r in rows}),
        "roles": len({r["target_role"] for r in rows}),
        "positive": sum(int(r["label"]) for r in rows)
    }

def render(row, mode):
    role = str(row["target_role"]).replace("_", " ")
    query = str(row["query"])
    answer = str(row["final_output"])

    if mode == "query_only":
        return f"ROLE: {role}\nQUERY: {query}"
    if mode == "answer_only":
        return f"ROLE: {role}\nANSWER: {answer}"
    return (
        f"ROLE: {role}\nQUERY: {query}\n"
        f"ANSWER: {answer}"
    )

def within_metrics(rows, scores):
    groups = collections.defaultdict(list)

    for row, score in zip(rows, scores):
        groups[str(row["system_id"])].append((
            str(row["target_role"]),
            int(row["label"]),
            float(score)
        ))

    systems = []

    for sid, items in sorted(groups.items()):
        labels = [x[1] for x in items]
        values = [x[2] for x in items]
        score = auc(labels, values)

        ranked = sorted(
            items, key=lambda x: (-x[2], x[0])
        )

        k = sum(labels)
        truth = {r for r, y, _ in items if y}
        predicted = {r for r, _, _ in ranked[:k]}

        systems.append({
            "system_id": sid,
            "within_auc": score,
            "k_eval_only": k,
            "exact": predicted == truth,
            "topk": [r for r, _, _ in ranked[:k]],
            "truth_eval_only": sorted(truth)
        })

    good = [
        x for x in systems
        if x["within_auc"] is not None
    ]

    return {
        "mean_within_mas_auc": (
            sum(x["within_auc"] for x in good) / len(good)
            if good else None
        ),
        "exact_topk": sum(x["exact"] for x in good),
        "evaluable_systems": len(good),
        "systems": systems
    }

def main():
    project = Path.cwd().resolve()

    if project.name != "mas_rl_detective":
        raise RuntimeError("Wrong project directory")

    out = (
        project / "artifacts/role_direct_within_v4"
        / "experiments/stage0_offline_control"
    )
    out.mkdir(parents=True, exist_ok=True)

    data = (
        project / "artifacts/v137_presence_v1"
        / "dataset_v1"
    )

    train_path = data / "train_v137.jsonl"
    dev_path = data / "dev_v137.jsonl"

    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)

    stats = {
        "train": validate(train, "train"),
        "dev": validate(dev, "dev"),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (train_path, dev_path)
        }
    }

    train_ids = {str(r["system_id"]) for r in train}
    dev_ids = {str(r["system_id"]) for r in dev}

    train_pairs = {
        str(r["pair_id"]) for r in train
        if r.get("pair_id") is not None
    }
    dev_pairs = {
        str(r["pair_id"]) for r in dev
        if r.get("pair_id") is not None
    }

    def signature(row, include_answer):
        parts = [
            str(row["target_role"]),
            str(row["query"])
        ]
        if include_answer:
            parts.append(str(row["final_output"]))
        return digest(normalize("|".join(parts)))

    exact_overlap = len(
        {signature(r, True) for r in train}
        & {signature(r, True) for r in dev}
    )

    query_overlap = len(
        {signature(r, False) for r in train}
        & {signature(r, False) for r in dev}
    )

    stats["split_checks"] = {
        "shared_system_ids": sorted(train_ids & dev_ids),
        "shared_pair_ids": sorted(train_pairs & dev_pairs),
        "shared_queries": query_overlap,
        "shared_exact_examples": exact_overlap
    }

    (out / "preflight.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    print("PREFLIGHT:", json.dumps(stats), flush=True)

    if (
        train_ids & dev_ids
        or train_pairs & dev_pairs
        or exact_overlap
    ):
        raise RuntimeError(
            "SPLIT_LEAKAGE_DETECTED — training blocked"
        )

    if len(train_ids) < 5 or len(dev_ids) < 3:
        raise RuntimeError("Insufficient systems")

    vocabulary = {
        str(r["target_role"]) for r in train + dev
    }

    dev_groups = collections.defaultdict(list)
    for row in dev:
        dev_groups[str(row["system_id"])].append(
            str(row["target_role"])
        )

    if len(vocabulary) != 20 or any(
        len(group) != 20 or len(set(group)) != 20
        for group in dev_groups.values()
    ):
        raise RuntimeError(
            "Invalid 20-role evaluation groups"
        )

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    y_train = np.array(
        [int(x["label"]) for x in train]
    )
    y_dev = np.array(
        [int(x["label"]) for x in dev]
    )

    results = {
        "protocol": (
            "Development diagnostic; exact overlaps checked. "
            "Role-composition-level independence not yet verified."
        ),
        "models": {}
    }

    for mode in (
        "query_only",
        "answer_only",
        "query_answer"
    ):
        print("TRAINING:", mode, flush=True)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=45000,
            sublinear_tf=True
        )

        x_train = vectorizer.fit_transform(
            [render(x, mode) for x in train]
        )
        x_dev = vectorizer.transform(
            [render(x, mode) for x in dev]
        )

        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1200,
            random_state=20261009
        )

        classifier.fit(x_train, y_train)
        scores = classifier.decision_function(x_dev)

        metric = within_metrics(dev, scores)

        results["models"][mode] = {
            "global_auc": auc(
                y_dev.tolist(), scores.tolist()
            ),
            "average_precision": float(
                average_precision_score(y_dev, scores)
            ),
            **metric
        }

        print(
            "RESULT", mode,
            "Within-AUC:", metric["mean_within_mas_auc"],
            "Exact:", metric["exact_topk"],
            flush=True
        )

    results["response_added_within_auc"] = (
        results["models"]["query_answer"]["mean_within_mas_auc"]
        - results["models"]["query_only"]["mean_within_mas_auc"]
    )

    target = out / "offline_control_report.json"

    target.write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    print("V4_STAGE0_COMPLETE", target, flush=True)
    print(
        "RESPONSE_SIGNAL_DELTA",
        results["response_added_within_auc"],
        flush=True
    )

if __name__ == "__main__":
    main()
