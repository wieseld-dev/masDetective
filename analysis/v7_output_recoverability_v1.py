from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


BAD_V3 = Path(
    "artifacts/v7_retention_sensor_v1/"
    "scored_executions.jsonl"
)

V4 = Path(
    "artifacts/v7_substantive_repair_v4/"
    "sensor/scored_v4.jsonl"
)

OUT = Path(
    "artifacts/v7_output_recoverability_v1"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def mean(xs):
    xs = [
        float(x)
        for x in xs
        if x is not None
        and np.isfinite(x)
    ]

    return (
        float(np.mean(xs))
        if xs
        else None
    )


def build_versions():
    rows = read_jsonl(BAD_V3)

    bad = [
        x for x in rows
        if x["kind"] == "bad"
    ]

    v3 = [
        x for x in rows
        if x["kind"] == "repair"
    ]

    v4 = read_jsonl(V4)

    assert len(bad) == 36, len(bad)
    assert len(v3) == 36, len(v3)
    assert len(v4) == 36, len(v4)

    return {
        "BAD": bad,
        "V3": v3,
        "V4": v4,
    }


def discover_roles(versions):
    roles = set()

    for rows in versions.values():
        for x in rows:
            roles.update(
                map(
                    str,
                    x["truth_roles"],
                )
            )

            roles.update(
                map(
                    str,
                    x["selected_roles_eval_only"],
                )
            )

    # Use the canonical order from one historical row if available
    # only indirectly; alphabetical order is sufficient for this
    # diagnostic because role identity is preserved explicitly.
    return sorted(roles)


def label_matrix(rows, roles):
    Y = np.zeros(
        (len(rows), len(roles)),
        dtype=np.int64,
    )

    for i, x in enumerate(rows):
        selected = set(
            map(
                str,
                x["selected_roles_eval_only"],
            )
        )

        for j, role in enumerate(roles):
            Y[i, j] = int(
                role in selected
            )

    return Y


def fit_role_model(Xtr, ytr):
    uniq = np.unique(ytr)

    if len(uniq) == 1:
        return (
            "constant",
            float(uniq[0]),
        )

    clf = LogisticRegression(
        C=1.0,
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=1389,
    )

    clf.fit(
        Xtr,
        ytr,
    )

    return (
        "logreg",
        clf,
    )


def predict_role(model, X):
    kind, obj = model

    if kind == "constant":
        return np.full(
            len(X),
            float(obj),
            dtype=float,
        )

    return obj.predict_proba(X)[:, 1]


def evaluate_version(
    name,
    rows,
    roles,
    embedder,
):
    texts = [
        str(x["final_output"])
        for x in rows
    ]

    groups = np.asarray([
        str(x["session_id"])
        for x in rows
    ])

    unique_groups = sorted(
        set(groups)
    )

    print(
        f"\n{name}: "
        f"rows={len(rows)} "
        f"systems={len(unique_groups)}",
        flush=True,
    )

    X = np.asarray(
        embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float64,
    )

    Y = label_matrix(
        rows,
        roles,
    )

    splitter = GroupKFold(
        n_splits=4,
    )

    P = np.full(
        Y.shape,
        np.nan,
        dtype=float,
    )

    fold_rows = []

    dummy = np.zeros(
        len(rows),
        dtype=int,
    )

    for fold, (tr, te) in enumerate(
        splitter.split(
            X,
            dummy,
            groups,
        ),
        1,
    ):
        train_groups = set(
            groups[tr]
        )
        test_groups = set(
            groups[te]
        )

        assert not (
            train_groups
            & test_groups
        )

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X[tr]
        )

        Xte = scaler.transform(
            X[te]
        )

        trainable_roles = 0

        for j, role in enumerate(roles):
            model = fit_role_model(
                Xtr,
                Y[tr, j],
            )

            if model[0] == "logreg":
                trainable_roles += 1

            P[te, j] = predict_role(
                model,
                Xte,
            )

        fold_rows.append({
            "fold": fold,
            "train_systems":
                sorted(train_groups),
            "test_systems":
                sorted(test_groups),
            "train_rows":
                int(len(tr)),
            "test_rows":
                int(len(te)),
            "trainable_roles":
                int(trainable_roles),
        })

        print(
            f"  fold {fold}: "
            f"train={len(tr)} "
            f"test={len(te)} "
            f"trainable_roles="
            f"{trainable_roles}/{len(roles)}",
            flush=True,
        )

    assert np.isfinite(P).all()

    flat_y = Y.reshape(-1)
    flat_p = P.reshape(-1)

    micro_auc = (
        float(
            roc_auc_score(
                flat_y,
                flat_p,
            )
        )
        if len(np.unique(flat_y)) == 2
        else None
    )

    row_aucs = []
    topk = []
    exact_topk = []

    for i in range(len(rows)):
        y = Y[i]
        p = P[i]

        if len(np.unique(y)) == 2:
            row_aucs.append(
                float(
                    roc_auc_score(
                        y,
                        p,
                    )
                )
            )

        k = int(y.sum())

        if k > 0:
            order = np.argsort(-p)[:k]

            recall = float(
                y[order].sum() / k
            )

            topk.append(recall)

            exact_topk.append(
                float(
                    recall == 1.0
                    and set(order)
                    == set(
                        np.where(y == 1)[0]
                    )
                )
            )

    # Macro per-role AUC only when role has both classes
    # across the complete held-out predictions.
    role_aucs = {}

    for j, role in enumerate(roles):
        yy = Y[:, j]

        if len(np.unique(yy)) < 2:
            continue

        role_aucs[role] = float(
            roc_auc_score(
                yy,
                P[:, j],
            )
        )

    return {
        "version": name,
        "rows": len(rows),
        "systems": len(unique_groups),
        "roles": len(roles),

        "micro_auc":
            micro_auc,

        "mean_within_row_auc":
            mean(row_aucs),

        "mean_topk_selected_recall":
            mean(topk),

        "exact_selected_set_at_true_k":
            mean(exact_topk),

        "macro_role_auc":
            mean(
                list(
                    role_aucs.values()
                )
            ),

        "roles_with_auc":
            len(role_aucs),

        "per_role_auc":
            role_aucs,

        "folds":
            fold_rows,
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    versions = build_versions()
    roles = discover_roles(
        versions
    )

    print(
        "roles:",
        len(roles),
        roles,
        flush=True,
    )

    print(
        "loading MiniLM...",
        flush=True,
    )

    embedder = SentenceTransformer(
        EMBED_MODEL
    )

    summary = {}

    for name in [
        "BAD",
        "V3",
        "V4",
    ]:
        result = evaluate_version(
            name,
            versions[name],
            roles,
            embedder,
        )

        summary[name] = result

    compact = {
        name: {
            "rows":
                x["rows"],

            "systems":
                x["systems"],

            "micro_auc":
                x["micro_auc"],

            "mean_within_row_auc":
                x["mean_within_row_auc"],

            "mean_topk_selected_recall":
                x["mean_topk_selected_recall"],

            "exact_selected_set_at_true_k":
                x["exact_selected_set_at_true_k"],

            "macro_role_auc":
                x["macro_role_auc"],

            "roles_with_auc":
                x["roles_with_auc"],
        }
        for name, x
        in summary.items()
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUT / "compact_summary.json"
    ).write_text(
        json.dumps(
            compact,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== OUTPUT-ONLY "
        "HELD-OUT-SYSTEM RECOVERABILITY ====="
    )

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
