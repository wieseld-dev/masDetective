#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------
# Reuse the already-tested Forward-Evidence V1 implementation.
# ------------------------------------------------------------

SPEC = importlib.util.spec_from_file_location(
    "fwd",
    "analysis/forward_evidence_eig_v1.py",
)

fwd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fwd)

SEED = 20261008

GROUP_ROOT = Path(
    "artifacts/q38_selector_v41_data"
)

RAW_ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2"
)

OUT = Path(
    "artifacts/oracle_activation_mediator_audit_v1"
)

ROLES = list(fwd.ROLES)
R = len(ROLES)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# ACTIVATION-AWARE RAW INDEX
# ============================================================

def activation_vector(rr):
    if "selected_roles_training_only" not in rr:
        return None

    active = set(
        map(
            str,
            rr.get(
                "selected_roles_training_only"
            )
            or [],
        )
    )

    return np.asarray(
        [
            float(r in active)
            for r in ROLES
        ],
        dtype=np.float32,
    )


def build_index():

    index = defaultdict(list)

    n_resp = 0
    n_join = 0
    n_missing_activation = 0

    dirs = sorted({
        p.parent
        for p in RAW_ROOT.rglob(
            "responses_train_round_*.jsonl"
        )
    })

    print(
        "[raw] dirs =",
        len(dirs),
        flush=True,
    )

    for d in dirs:

        for rp in sorted(
            d.glob(
                "responses_train_round_*.jsonl"
            )
        ):

            rnd = fwd.round_from_name(
                rp
            )

            if rnd is None:
                continue

            sp = (
                d
                / f"scored_train_round_{rnd:02d}.jsonl"
            )

            if not sp.exists():
                continue

            scored = fwd.read_jsonl(
                sp
            )

            score_idx = defaultdict(
                list
            )

            for sc in scored:

                q = fwd.normq(
                    sc.get("query")
                )

                ids = {
                    str(
                        sc.get(
                            "system_id",
                            ""
                        )
                    ),
                    str(
                        sc.get(
                            "session_id",
                            ""
                        )
                    ),
                }

                ids.discard("")

                for sid in ids:
                    score_idx[
                        (sid, q)
                    ].append(
                        sc
                    )

            for rr in fwd.read_jsonl(
                rp
            ):

                n_resp += 1

                if (
                    rr.get(
                        "execution_status"
                    )
                    != "ok"
                ):
                    continue

                if not rr.get(
                    "valid",
                    True,
                ):
                    continue

                avec = activation_vector(
                    rr
                )

                if avec is None:
                    n_missing_activation += 1
                    continue

                q = fwd.normq(
                    rr.get("query")
                )

                ids = []

                for k in (
                    "session_id",
                    "system_id",
                ):
                    sid = str(
                        rr.get(
                            k,
                            ""
                        )
                    )

                    if (
                        sid
                        and sid not in ids
                    ):
                        ids.append(
                            sid
                        )

                candidates = []

                for sid in ids:
                    candidates.extend(
                        score_idx.get(
                            (sid, q),
                            [],
                        )
                    )

                # Remove exact duplicate score rows.
                uniq = {}

                for sc in candidates:

                    sig = json.dumps(
                        {
                            "e135":
                                sc.get(
                                    "evidence",
                                    {},
                                ),
                            "e134":
                                sc.get(
                                    "metadata",
                                    {},
                                ).get(
                                    "evidence_v134",
                                    {},
                                ),
                        },
                        sort_keys=True,
                    )

                    uniq[sig] = sc

                for sc in uniq.values():

                    z = fwd.truth_vector(
                        rr,
                        sc,
                    )

                    y = fwd.evidence_vector(
                        sc
                    )

                    for sid in ids:
                        index[
                            (
                                sid,
                                int(rnd),
                                q,
                            )
                        ].append(
                            {
                                "z": z,
                                "a": avec,
                                "y": y,
                            }
                        )

                    n_join += 1

    print(
        "[raw] responses =",
        n_resp,
    )

    print(
        "[raw] joined =",
        n_join,
    )

    print(
        "[raw] missing activation =",
        n_missing_activation,
    )

    print(
        "[raw] keys =",
        len(index),
    )

    return index


def lookup(
    index,
    sid,
    rnd,
    query,
):

    xs = index.get(
        (
            str(sid),
            int(rnd),
            fwd.normq(query),
        ),
        [],
    )

    if not xs:
        return None

    z0 = xs[0]["z"]
    a0 = xs[0]["a"]

    good = [
        x
        for x in xs
        if (
            np.array_equal(
                x["z"],
                z0,
            )
            and
            np.array_equal(
                x["a"],
                a0,
            )
        )
    ]

    if not good:
        return None

    # If conflicting hidden labels exist for same key,
    # reject instead of silently averaging.
    if len(good) != len(xs):
        return None

    y = np.mean(
        np.stack(
            [
                x["y"]
                for x in good
            ]
        ),
        axis=0,
    ).astype(
        np.float32
    )

    return {
        "z":
            z0.astype(
                np.float32
            ),

        "a":
            a0.astype(
                np.float32
            ),

        "y":
            y,

        "repeats":
            len(good),
    }


# ============================================================
# GROUP ALIGNMENT
# ============================================================

def attach(
    groups,
    index,
):

    rows = []
    missing = 0

    for g in groups:

        sid = str(
            g["system_id"]
        )

        rnd = int(
            g["round"]
        )

        for query in g["queries"]:

            obs = lookup(
                index,
                sid,
                rnd,
                query,
            )

            if obs is None:
                missing += 1
                continue

            rows.append({
                "system_id":
                    sid,

                "state_id":
                    str(
                        g["state_id"]
                    ),

                "round":
                    rnd,

                "query":
                    str(query),

                "history_text":
                    str(
                        g.get(
                            "history_text",
                            "- no previous public interaction",
                        )
                    ),

                "belief":
                    np.asarray(
                        [
                            float(
                                g[
                                    "role_probs_before"
                                ][r]
                            )
                            for r
                            in ROLES
                        ],
                        dtype=np.float32,
                    ),

                "z":
                    obs["z"],

                "a":
                    obs["a"],

                "y":
                    obs["y"],
            })

    total = (
        len(rows)
        + missing
    )

    rate = (
        len(rows)
        / max(
            total,
            1,
        )
    )

    print(
        "[align]",
        len(rows),
        "/",
        total,
        f"= {rate:.3%}",
    )

    if rate < .95:
        raise RuntimeError(
            "Alignment below 95%; abort."
        )

    return rows


# ============================================================
# FEATURES
# ============================================================

def feature(
    row,
    qmap,
    hmap,
    *,
    use_z=False,
    use_a=False,
):

    x = [
        qmap[
            row["query"]
        ],

        hmap[
            row["history_text"]
        ],

        row["belief"],

        np.asarray(
            [
                row["round"]
                / 8.0
            ],
            dtype=np.float32,
        ),
    ]

    if use_z:
        x.append(
            row["z"]
        )

    if use_a:
        x.append(
            row["a"]
        )

    return np.concatenate(
        x
    ).astype(
        np.float32
    )


def xy(
    rows,
    qmap,
    hmap,
    *,
    use_z=False,
    use_a=False,
):

    X = np.stack([
        feature(
            r,
            qmap,
            hmap,
            use_z=use_z,
            use_a=use_a,
        )
        for r in rows
    ])

    Y = np.stack([
        r["y"]
        for r in rows
    ])

    return (
        X.astype(
            np.float32
        ),
        Y.astype(
            np.float32
        ),
    )


# ============================================================
# EXTRA METRICS
# ============================================================

def split_metrics(
    y,
    p,
):

    out = fwd.prediction_metrics(
        y,
        p,
    )

    for name, sl in (
        (
            "v135",
            slice(
                0,
                R,
            ),
        ),
        (
            "v134",
            slice(
                R,
                2*R,
            ),
        ),
    ):

        yy = y[
            :,
            sl
        ]

        pp = p[
            :,
            sl
        ]

        out[
            f"{name}_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        yy
                        - pp
                    )**2
                )
            )
        )

        out[
            f"{name}_mae"
        ] = float(
            np.mean(
                np.abs(
                    yy
                    - pp
                )
            )
        )

    return out


def raw_activation_sensor_auc(
    rows,
):

    A = np.stack([
        r["a"]
        for r in rows
    ])

    Y = np.stack([
        r["y"]
        for r in rows
    ])

    result = {}

    for channel, off in (
        (
            "v135",
            0,
        ),
        (
            "v134",
            R,
        ),
    ):

        aucs = []
        details = {}

        for j, role in enumerate(
            ROLES
        ):

            labels = A[
                :,
                j
            ]

            scores = Y[
                :,
                off+j
            ]

            if len(
                np.unique(
                    labels
                )
            ) < 2:
                continue

            auc = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

            aucs.append(
                auc
            )

            details[
                role
            ] = {
                "auc":
                    auc,

                "mean_active":
                    float(
                        np.mean(
                            scores[
                                labels
                                == 1
                            ]
                        )
                    ),

                "mean_inactive":
                    float(
                        np.mean(
                            scores[
                                labels
                                == 0
                            ]
                        )
                    ),
            }

        result[
            channel
        ] = {
            "macro_auc":
                float(
                    np.mean(
                        aucs
                    )
                )
                if aucs
                else None,

            "per_role":
                details,
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "DEVICE =",
        DEVICE,
    )

    print(
        "NO API CALLS"
    )

    train_groups = fwd.read_jsonl(
        GROUP_ROOT
        / "train_groups.jsonl"
    )

    val_groups = fwd.read_jsonl(
        GROUP_ROOT
        / "val_groups.jsonl"
    )

    index = build_index()

    train_rows = attach(
        train_groups,
        index,
    )

    val_rows = attach(
        val_groups,
        index,
    )

    # --------------------------------------------------------
    # SAME train-system split as Forward-Evidence V1.
    # --------------------------------------------------------

    train_sids = sorted({
        r["system_id"]
        for r in train_rows
    })

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        train_sids
    )

    n_dev = max(
        1,
        int(
            .15
            * len(
                train_sids
            )
        ),
    )

    dev_sids = set(
        train_sids[
            :n_dev
        ]
    )

    fit_sids = set(
        train_sids[
            n_dev:
        ]
    )

    fit_rows = [
        r
        for r in train_rows
        if r[
            "system_id"
        ]
        in fit_sids
    ]

    dev_rows = [
        r
        for r in train_rows
        if r[
            "system_id"
        ]
        in dev_sids
    ]

    print(
        "FIT systems/rows =",
        len(fit_sids),
        len(fit_rows),
    )

    print(
        "DEV systems/rows =",
        len(dev_sids),
        len(dev_rows),
    )

    print(
        "VAL systems/rows =",
        len({
            r["system_id"]
            for r in val_rows
        }),
        len(val_rows),
    )

    # --------------------------------------------------------
    # Frozen text embeddings.
    # --------------------------------------------------------

    encoder = SentenceTransformer(
        fwd.ENCODER_NAME,
        device=DEVICE,
    )

    qmap, hmap = fwd.encode_maps(
        encoder,
        (
            fit_rows
            + dev_rows
            + val_rows
        ),
    )

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    configs = {
        "M0_PUBLIC":
            (
                False,
                False,
            ),

        "M1_PLUS_INSTALLED_Z":
            (
                True,
                False,
            ),

        "M2_PLUS_TRUE_ACTIVATION_A":
            (
                False,
                True,
            ),

        "M3_PLUS_Z_AND_A":
            (
                True,
                True,
            ),
    }

    results = {}

    for name, (
        use_z,
        use_a,
    ) in configs.items():

        print()
        print(
            "="*90
        )
        print(
            name
        )
        print(
            "="*90
        )

        Xf, Yf = xy(
            fit_rows,
            qmap,
            hmap,
            use_z=use_z,
            use_a=use_a,
        )

        Xd, Yd = xy(
            dev_rows,
            qmap,
            hmap,
            use_z=use_z,
            use_a=use_a,
        )

        Xv, Yv = xy(
            val_rows,
            qmap,
            hmap,
            use_z=use_z,
            use_a=use_a,
        )

        model, mu, sd, best_loss = (
            fwd.train_model(
                name,
                Xf,
                Yf,
                Xd,
                Yd,
            )
        )

        pred = fwd.predict(
            model,
            Xv,
            mu,
            sd,
        )

        metrics = split_metrics(
            Yv,
            pred,
        )

        metrics[
            "internal_dev_best_loss"
        ] = float(
            best_loss
        )

        results[
            name
        ] = metrics

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Raw sensor ↔ activation diagnostic.
    # --------------------------------------------------------

    raw_auc = (
        raw_activation_sensor_auc(
            val_rows
        )
    )

    m0 = results[
        "M0_PUBLIC"
    ]

    m1 = results[
        "M1_PLUS_INSTALLED_Z"
    ]

    m2 = results[
        "M2_PLUS_TRUE_ACTIVATION_A"
    ]

    m3 = results[
        "M3_PLUS_Z_AND_A"
    ]

    comparison = {
        "Z_effect_M1_minus_M0": {
            "delta_r2":
                m1[
                    "mean_dim_r2"
                ]
                - m0[
                    "mean_dim_r2"
                ],

            "delta_corr":
                m1[
                    "mean_dim_corr"
                ]
                - m0[
                    "mean_dim_corr"
                ],

            "delta_rmse":
                m1[
                    "rmse"
                ]
                - m0[
                    "rmse"
                ],
        },

        "ACTIVATION_effect_M2_minus_M0": {
            "delta_r2":
                m2[
                    "mean_dim_r2"
                ]
                - m0[
                    "mean_dim_r2"
                ],

            "delta_corr":
                m2[
                    "mean_dim_corr"
                ]
                - m0[
                    "mean_dim_corr"
                ],

            "delta_rmse":
                m2[
                    "rmse"
                ]
                - m0[
                    "rmse"
                ],
        },

        "ACTIVATION_given_Z_M3_minus_M1": {
            "delta_r2":
                m3[
                    "mean_dim_r2"
                ]
                - m1[
                    "mean_dim_r2"
                ],

            "delta_corr":
                m3[
                    "mean_dim_corr"
                ]
                - m1[
                    "mean_dim_corr"
                ],

            "delta_rmse":
                m3[
                    "rmse"
                ]
                - m1[
                    "rmse"
                ],
        },

        "Z_given_activation_M3_minus_M2": {
            "delta_r2":
                m3[
                    "mean_dim_r2"
                ]
                - m2[
                    "mean_dim_r2"
                ],

            "delta_corr":
                m3[
                    "mean_dim_corr"
                ]
                - m2[
                    "mean_dim_corr"
                ],

            "delta_rmse":
                m3[
                    "rmse"
                ]
                - m2[
                    "rmse"
                ],
        },
    }

    # Simple decision aid.
    act_dr2 = comparison[
        "ACTIVATION_effect_M2_minus_M0"
    ][
        "delta_r2"
    ]

    act_drmse = comparison[
        "ACTIVATION_effect_M2_minus_M0"
    ][
        "delta_rmse"
    ]

    if (
        act_dr2 >= .05
        and
        act_drmse <= -.05
    ):
        verdict = (
            "STRONG_MEDIATOR_SIGNAL"
        )

    elif (
        act_dr2 >= .015
        and
        act_drmse < 0
    ):
        verdict = (
            "MODERATE_MEDIATOR_SIGNAL"
        )

    else:
        verdict = (
            "NO_CLEAR_MEDIATOR_SIGNAL"
        )

    summary = {
        "experiment":
            "oracle_activation_mediator_audit_v1",

        "final_val_is_heldout_by_system":
            True,

        "truth_activation_used_at_test":
            True,

        "truth_activation_role":
            (
                "ORACLE DIAGNOSTIC ONLY; "
                "not deployable selector input"
            ),

        "fit_systems":
            len(fit_sids),

        "internal_dev_systems":
            len(dev_sids),

        "final_val_systems":
            len({
                r[
                    "system_id"
                ]
                for r
                in val_rows
            }),

        "train_rows":
            len(train_rows),

        "val_rows":
            len(val_rows),

        "models":
            results,

        "comparisons":
            comparison,

        "raw_sensor_activation_relation":
            raw_auc,

        "decision": {
            "verdict":
                verdict,

            "criterion":
                (
                    "M2 must materially improve held-out "
                    "evidence prediction over M0."
                ),
        },
    }

    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "="*100
    )
    print(
        "FINAL"
    )
    print(
        "="*100
    )

    print(
        "VERDICT =",
        verdict,
    )

    print()
    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print()
    print(
        "RAW SENSOR ACTIVATION MACRO AUC:"
    )

    print(
        "V13.5 =",
        raw_auc[
            "v135"
        ][
            "macro_auc"
        ]
    )

    print(
        "V13.4 =",
        raw_auc[
            "v134"
        ][
            "macro_auc"
        ]
    )

    print()
    print(
        "SAVED:",
        OUT
        / "summary.json",
    )


if __name__ == "__main__":
    main()
