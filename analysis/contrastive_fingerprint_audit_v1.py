#!/usr/bin/env python3
"""Held-out conditional fingerprint information audit for installed MAS roles.

Requires:
 analysis/forward_evidence_eig_v1.py
 analysis/oracle_activation_mediator_audit_v1.py
 artifacts/q38_selector_v41_data/{train,val,challenge}_groups.jsonl

No API calls.
No MAS executions.
Hidden installed-role vector Z is used only as supervised ground-truth label.
"""

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


SEED = 20261008

OUT = Path(
    "artifacts/contrastive_fingerprint_audit_v1"
)

GROUP = Path(
    "artifacts/q38_selector_v41_data"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD EXISTING PROJECT MODULES
# ============================================================

def load(name, file):
    spec = importlib.util.spec_from_file_location(
        name,
        file,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


fwd = load(
    "fwd",
    "analysis/forward_evidence_eig_v1.py",
)

med = load(
    "med",
    "analysis/oracle_activation_mediator_audit_v1.py",
)

ROLES = list(
    fwd.ROLES
)

N = len(
    ROLES
)


# ============================================================
# BASIC METRICS
# ============================================================

def clip(p):
    return np.clip(
        p,
        1e-5,
        1 - 1e-5,
    )


def row_loss(y, p):
    p = clip(
        p
    )

    return np.mean(
        -y * np.log(p)
        - (1 - y) * np.log1p(-p),
        axis=1,
    )


def macro_rank_metrics(y, p):

    auc = []
    ap = []
    per = {}

    for j, role in enumerate(
        ROLES
    ):

        if np.unique(
            y[:, j]
        ).size < 2:
            continue

        a = float(
            roc_auc_score(
                y[:, j],
                p[:, j],
            )
        )

        b = float(
            average_precision_score(
                y[:, j],
                p[:, j],
            )
        )

        per[role] = {
            "auc":
                a,

            "ap":
                b,

            "prevalence":
                float(
                    y[:, j].mean()
                ),
        }

        auc.append(
            a
        )

        ap.append(
            b
        )

    return {
        "macro_auc":
            float(
                np.mean(auc)
            )
            if auc
            else None,

        "macro_ap":
            float(
                np.mean(ap)
            )
            if ap
            else None,

        "per_role":
            per,
    }


def metrics(y, p):

    r = macro_rank_metrics(
        y,
        p,
    )

    r.update({
        "log_loss":
            float(
                row_loss(
                    y,
                    p,
                ).mean()
            ),

        "brier":
            float(
                (
                    (y - p) ** 2
                ).mean()
            ),
    })

    return r


# ============================================================
# SYSTEM-LEVEL BOOTSTRAP
# ============================================================

def system_ci(
    rows,
    delta,
    nboot=10000,
):

    d = defaultdict(
        list
    )

    for row, x in zip(
        rows,
        delta,
    ):
        d[
            row["system_id"]
        ].append(
            float(x)
        )

    s = np.array(
        [
            np.mean(v)
            for v in d.values()
        ],
        dtype=float,
    )

    rng = np.random.default_rng(
        SEED + 49
    )

    ix = rng.integers(
        0,
        len(s),
        size=(
            nboot,
            len(s),
        ),
    )

    boot = s[ix].mean(
        axis=1
    )

    return {
        "n_systems":
            len(s),

        "mean_system_gain":
            float(
                s.mean()
            ),

        "ci95":
            [
                float(v)
                for v in np.quantile(
                    boot,
                    [
                        .025,
                        .975,
                    ],
                )
            ],

        "p_boot_gain_gt_zero":
            float(
                np.mean(
                    boot > 0
                )
            ),
    }


# ============================================================
# TRAIN / DEV SYSTEM SPLIT
# ============================================================

def group_split(rows):

    systems = sorted({
        r["system_id"]
        for r in rows
    })

    random.Random(
        SEED
    ).shuffle(
        systems
    )

    n = max(
        1,
        int(
            .15
            * len(
                systems
            )
        ),
    )

    dev = set(
        systems[
            :n
        ]
    )

    fit = set(
        systems[
            n:
        ]
    )

    return (
        [
            r
            for r in rows
            if r["system_id"]
            in fit
        ],

        [
            r
            for r in rows
            if r["system_id"]
            in dev
        ],
    )


# ============================================================
# FEATURES
#
# PRE:
#   query + history + current belief + round
#
# V135:
#   PRE + causal evidence
#
# V134:
#   PRE + absolute evidence
#
# BOTH:
#   PRE + V13.5 + V13.4
# ============================================================

def features(
    rows,
    qmap,
    hmap,
    kind,
):

    X = []

    for r in rows:

        parts = [
            qmap[
                r["query"]
            ],

            hmap[
                r["history_text"]
            ],

            r["belief"],

            np.array(
                [
                    r["round"]
                    / 8.0
                ],
                dtype=np.float32,
            ),
        ]

        if kind in (
            "V135",
            "BOTH",
        ):
            parts.append(
                r["y"][
                    :N
                ]
            )

        if kind in (
            "V134",
            "BOTH",
        ):
            parts.append(
                r["y"][
                    N:
                ]
            )

        X.append(
            np.concatenate(
                parts
            )
        )

    return np.asarray(
        X,
        dtype=np.float32,
    )


def targets(rows):

    return np.stack([
        r["z"]
        for r in rows
    ]).astype(
        np.float32
    )


# ============================================================
# ROLE CLASSIFIER
# ============================================================

class RoleNet(nn.Module):

    def __init__(
        self,
        d,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d,
                256,
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            ),

            nn.Dropout(
                .13
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                .08
            ),

            nn.Linear(
                128,
                N,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


@torch.no_grad()
def predict(
    model,
    X,
):

    model.eval()

    p = []

    for i in range(
        0,
        len(X),
        1024,
    ):

        xb = torch.from_numpy(
            X[
                i:i+1024
            ]
        ).to(
            DEVICE
        )

        p.append(
            torch.sigmoid(
                model(
                    xb
                )
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        p,
        axis=0,
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    kind,
    fit,
    dev,
    qmap,
    hmap,
):

    X = features(
        fit,
        qmap,
        hmap,
        kind,
    )

    Y = targets(
        fit
    )

    Xd = features(
        dev,
        qmap,
        hmap,
        kind,
    )

    Yd = targets(
        dev
    )

    scaler = StandardScaler().fit(
        X
    )

    X = scaler.transform(
        X
    ).astype(
        np.float32
    )

    Xd = scaler.transform(
        Xd
    ).astype(
        np.float32
    )

    model = RoleNet(
        X.shape[1]
    ).to(
        DEVICE
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3,
    )

    tensorX = torch.from_numpy(
        X
    )

    tensorY = torch.from_numpy(
        Y
    )

    best = None
    best_loss = float(
        "inf"
    )

    bad = 0

    for epoch in range(
        1,
        31,
    ):

        model.train()

        perm = torch.randperm(
            len(X)
        )

        acc = []

        for ids in perm.split(
            256
        ):

            logits = model(
                tensorX[
                    ids
                ].to(
                    DEVICE
                )
            )

            loss = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    tensorY[
                        ids
                    ].to(
                        DEVICE
                    ),
                )
            )

            optim.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optim.step()

            acc.append(
                float(
                    loss
                    .detach()
                    .cpu()
                )
            )

        pd = predict(
            model,
            Xd,
        )

        dl = float(
            row_loss(
                Yd,
                pd,
            ).mean()
        )

        print(
            f"[{kind}] "
            f"epoch={epoch:02d} "
            f"train={np.mean(acc):.5f} "
            f"devNLL={dl:.5f}",
            flush=True,
        )

        if dl < (
            best_loss
            - 0.0001
        ):

            best_loss = dl
            bad = 0

            best = {
                k:
                    v.cpu()
                    .clone()

                for k, v
                in model.state_dict().items()
            }

        else:
            bad += 1

        if bad >= 5:
            break

    model.load_state_dict(
        best
    )

    model.eval()

    return (
        model,
        scaler,
        best_loss,
    )


def evaluate(
    model,
    scaler,
    kind,
    rows,
    qmap,
    hmap,
):

    X = scaler.transform(
        features(
            rows,
            qmap,
            hmap,
            kind,
        )
    ).astype(
        np.float32
    )

    return predict(
        model,
        X,
    )


# ============================================================
# SHUFFLE CONTROL
#
# We keep:
#   - same state
#   - same set of candidate queries
#
# But randomly assign another candidate's fingerprint to each
# query inside the state.
#
# Therefore:
# If REAL BOTH > SHUFFLED BOTH,
# then query-fingerprint correspondence matters.
# ============================================================

def state_shuffle(
    rows,
    seed=987,
):

    rng = np.random.default_rng(
        seed
    )

    by = defaultdict(
        list
    )

    for i, r in enumerate(
        rows
    ):
        by[
            r["state_id"]
        ].append(
            i
        )

    index = np.arange(
        len(rows)
    )

    for ids in by.values():

        if len(ids) > 1:

            index[
                ids
            ] = rng.permutation(
                ids
            )

    out = []

    for r, j in zip(
        rows,
        index,
    ):

        out.append({
            **r,

            "y":
                rows[
                    int(j)
                ][
                    "y"
                ],
        })

    return out


# ============================================================
# MATCHED OBSERVATIONAL AUDIT
#
# For each target role:
#   role PRESENT vs role ABSENT
#
# while requiring:
#   same round
#   EXACT same other installed roles
#   highly semantically similar query
#
# This is NOT a causal intervention.
# It is an observational matched analysis.
# ============================================================

def matched_audit(
    rows,
    qmap,
    cos_min=.88,
):

    buckets = defaultdict(
        lambda: [
            [],
            [],
        ]
    )

    for i, r in enumerate(
        rows
    ):

        mask = np.flatnonzero(
            r["z"] > .5
        )

        for j in range(
            N
        ):

            other = tuple(
                int(k)
                for k in mask
                if k != j
            )

            side = int(
                r["z"][j]
                > .5
            )

            buckets[
                (
                    j,
                    r["round"],
                    other,
                )
            ][
                side
            ].append(
                i
            )

    pairs = defaultdict(
        list
    )

    for (
        role,
        rnd,
        others,
    ), (
        neg,
        pos,
    ) in buckets.items():

        if (
            not neg
            or not pos
        ):
            continue

        negQ = np.stack([
            qmap[
                rows[k][
                    "query"
                ]
            ]
            for k in neg
        ])

        posQ = np.stack([
            qmap[
                rows[k][
                    "query"
                ]
            ]
            for k in pos
        ])

        sims = (
            posQ
            @ negQ.T
        )

        used = set()

        for i, pi in enumerate(
            pos
        ):

            order = np.argsort(
                -sims[
                    i
                ]
            )

            for o in order:

                ni = neg[
                    int(o)
                ]

                if (
                    sims[
                        i,
                        o
                    ]
                    < cos_min
                ):
                    break

                if (
                    ni in used
                    or
                    rows[
                        pi
                    ][
                        "system_id"
                    ]
                    ==
                    rows[
                        ni
                    ][
                        "system_id"
                    ]
                ):
                    continue

                used.add(
                    ni
                )

                pairs[
                    role
                ].append(
                    (
                        pi,
                        ni,
                        float(
                            sims[
                                i,
                                o
                            ]
                        ),
                    )
                )

                break

    result = {}

    pool135 = []
    pool134 = []

    for role, p in pairs.items():

        v135 = []
        v134 = []

        for i, j, s in p:

            d135 = (
                rows[
                    i
                ][
                    "y"
                ][
                    role
                ]
                -
                rows[
                    j
                ][
                    "y"
                ][
                    role
                ]
            )

            d134 = (
                rows[
                    i
                ][
                    "y"
                ][
                    N + role
                ]
                -
                rows[
                    j
                ][
                    "y"
                ][
                    N + role
                ]
            )

            v135.append(
                float(
                    d135 > 0
                )
                +
                .5
                * float(
                    d135 == 0
                )
            )

            v134.append(
                float(
                    d134 > 0
                )
                +
                .5
                * float(
                    d134 == 0
                )
            )

        pool135 += v135
        pool134 += v134

        result[
            ROLES[
                role
            ]
        ] = {
            "pairs":
                len(p),

            "v135_pairwise_concordance":
                float(
                    np.mean(
                        v135
                    )
                ),

            "v134_pairwise_concordance":
                float(
                    np.mean(
                        v134
                    )
                ),
        }

    return {
        "matched_pairs_total":
            len(
                pool135
            ),

        "roles_with_pairs":
            len(
                result
            ),

        "query_cosine_min":
            cos_min,

        "same_other_installed_roles":
            True,

        "same_round":
            True,

        "not_causal_intervention":
            True,

        "pooled_v135_concordance":
            float(
                np.mean(
                    pool135
                )
            )
            if pool135
            else None,

        "pooled_v134_concordance":
            float(
                np.mean(
                    pool134
                )
            )
            if pool134
            else None,

        "adequate_sample_for_interpretation":
            (
                len(
                    pool135
                )
                >= 50
            ),

        "per_role":
            result,
    }


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
        "DEVICE",
        DEVICE,
        "NO API CALLS",
        flush=True,
    )

    # --------------------------------------------------------
    # LOAD GROUP DATA
    # --------------------------------------------------------

    groups = {}

    for split in [
        "train",
        "val",
        "challenge",
    ]:

        p = (
            GROUP
            / f"{split}_groups.jsonl"
        )

        if p.exists():

            groups[
                split
            ] = fwd.read_jsonl(
                p
            )

    if (
        "train"
        not in groups
        or
        "val"
        not in groups
    ):

        raise FileNotFoundError(
            "train/val group files missing"
        )

    # --------------------------------------------------------
    # ATTACH RAW RESPONSE/SENSOR/TRUTH DATA
    # --------------------------------------------------------

    idx = med.build_index()

    rows = {
        k:
            med.attach(
                v,
                idx,
            )

        for k, v
        in groups.items()
    }

    # --------------------------------------------------------
    # TRAIN / INTERNAL DEV SPLIT
    # --------------------------------------------------------

    fit, dev = group_split(
        rows[
            "train"
        ]
    )

    fs = {
        r[
            "system_id"
        ]
        for r in fit
    }

    ds = {
        r[
            "system_id"
        ]
        for r in dev
    }

    assert not (
        fs
        & ds
    )

    for k in (
        "val",
        "challenge",
    ):

        if k in rows:

            assert not (
                {
                    r[
                        "system_id"
                    ]
                    for r in rows[
                        k
                    ]
                }
                &
                (
                    fs
                    | ds
                )
            ), (
                f"system leakage: {k}"
            )

    print(
        "SPLIT fit/dev/val/challenge systems=",
        *[
            len({
                x[
                    "system_id"
                ]
                for x in q
            })
            for q in (
                fit,
                dev,
                rows[
                    "val"
                ],
                rows.get(
                    "challenge",
                    [],
                ),
            )
        ],
        flush=True,
    )

    # --------------------------------------------------------
    # TEXT EMBEDDINGS
    # --------------------------------------------------------

    encoder = SentenceTransformer(
        fwd.ENCODER_NAME,
        device=DEVICE,
    )

    allrows = (
        fit
        + dev
        + rows[
            "val"
        ]
        + rows.get(
            "challenge",
            [],
        )
    )

    qmap, hmap = fwd.encode_maps(
        encoder,
        allrows,
    )

    del encoder

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # OUTPUT STRUCTURE
    # --------------------------------------------------------

    pred = {
        k: {}
        for k in rows
    }

    results = {
        "protocol": {
            "trained_on":
                (
                    "TRAIN only, internal DEV "
                    "selected by system"
                ),

            "targets":
                "installed Z only",

            "post_response_input":
                (
                    "frozen V13.5/V13.4 evidence"
                ),

            "no_true_activation_at_test":
                True,

            "no_true_Z_at_test":
                True,

            "note":
                (
                    "Repeatedly inspected VAL/CHALLENGE "
                    "are exploratory, not fresh final tests"
                ),
        },

        "split_sizes": {
            k: {
                "systems":
                    len({
                        r[
                            "system_id"
                        ]
                        for r in rr
                    }),

                "rows":
                    len(
                        rr
                    ),
            }

            for k, rr
            in rows.items()
        },

        "splits": {},
    }

    trained = {}

    # --------------------------------------------------------
    # TRAIN 4 MODELS
    # --------------------------------------------------------

    for kind in [
        "PRE",
        "V135",
        "V134",
        "BOTH",
    ]:

        model, scaler, dv = train(
            kind,
            fit,
            dev,
            qmap,
            hmap,
        )

        trained[
            kind
        ] = (
            model,
            scaler,
        )

        for name, rr in rows.items():

            pred[
                name
            ][
                kind
            ] = evaluate(
                model,
                scaler,
                kind,
                rr,
                qmap,
                hmap,
            )

        torch.save(
            {
                "model":
                    model.state_dict(),

                "scaler_mean":
                    scaler.mean_,

                "scaler_scale":
                    scaler.scale_,

                "input_dim":
                    len(
                        scaler.mean_
                    ),

                "roles":
                    ROLES,

                "kind":
                    kind,

                "encoder":
                    fwd.ENCODER_NAME,
            },

            OUT
            / f"{kind}.pt",
        )

        print(
            f"[{kind}] "
            f"best dev NLL="
            f"{dv:.5f}",
            flush=True,
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for split, rr in rows.items():

        truth = targets(
            rr
        )

        ybase = pred[
            split
        ][
            "PRE"
        ]

        base_loss = row_loss(
            truth,
            ybase,
        )

        splitres = {
            "models": {
                k:
                    metrics(
                        truth,
                        p,
                    )

                for k, p
                in pred[
                    split
                ].items()
            },

            "paired_vs_pre": {},

            "controls": {},
        }

        # Current belief itself as a reference predictor.
        splitres[
            "models"
        ][
            "PRIOR_BELIEF"
        ] = metrics(
            truth,
            np.stack([
                r[
                    "belief"
                ]
                for r in rr
            ]),
        )

        # ----------------------------------------------------
        # CONDITIONAL INFORMATION:
        #
        # Does the response fingerprint reduce uncertainty
        # beyond PRE-query information?
        # ----------------------------------------------------

        for kind in [
            "V135",
            "V134",
            "BOTH",
        ]:

            pp = pred[
                split
            ][
                kind
            ]

            gain = (
                base_loss
                -
                row_loss(
                    truth,
                    pp,
                )
            )

            splitres[
                "paired_vs_pre"
            ][
                kind
            ] = system_ci(
                rr,
                gain,
            )

            splitres[
                "paired_vs_pre"
            ][
                kind
            ][
                "row_mean_gain"
            ] = float(
                np.mean(
                    gain
                )
            )

        # ----------------------------------------------------
        # SHUFFLE CONTROL:
        #
        # Destroy candidate-query ↔ evidence correspondence
        # while keeping state constant.
        # ----------------------------------------------------

        shuffled = state_shuffle(
            rr,
            seed=SEED + 17,
        )

        sboth = evaluate(
            *trained[
                "BOTH"
            ],
            "BOTH",
            shuffled,
            qmap,
            hmap,
        )

        splitres[
            "controls"
        ][
            "WITHIN_STATE_SHUFFLED_BOTH"
        ] = metrics(
            truth,
            sboth,
        )

        real_both = pred[
            split
        ][
            "BOTH"
        ]

        splitres[
            "controls"
        ][
            "BOTH_REAL_MINUS_WITHIN_STATE_SHUFFLED"
        ] = system_ci(
            rr,

            row_loss(
                truth,
                sboth,
            )
            -
            row_loss(
                truth,
                real_both,
            ),
        )

        # ----------------------------------------------------
        # MATCHED ROLE-PRESENT / ROLE-ABSENT AUDIT
        # ----------------------------------------------------

        splitres[
            "matched_config_query_pairs"
        ] = matched_audit(
            rr,
            qmap,
        )

        results[
            "splits"
        ][
            split
        ] = splitres

        print()
        print(
            "===",
            split.upper(),
            "===",
            flush=True,
        )

        for key, m in splitres[
            "models"
        ].items():

            print(
                f"{key:12s} "
                f"NLL={m['log_loss']:.5f} "
                f"Brier={m['brier']:.5f} "
                f"macroAUC={m['macro_auc']:.4f} "
                f"macroAP={m['macro_ap']:.4f}",
                flush=True,
            )

        for key, x in splitres[
            "paired_vs_pre"
        ].items():

            print(
                key,
                "gain_vs_PRE=",
                x[
                    "mean_system_gain"
                ],
                "CI95=",
                x[
                    "ci95"
                ],
                flush=True,
            )

        print(
            "WITHIN-STATE SHUFFLE control=",
            splitres[
                "controls"
            ][
                "BOTH_REAL_MINUS_WITHIN_STATE_SHUFFLED"
            ],
            flush=True,
        )

        print(
            "STRICT MATCHED PAIRS=",
            splitres[
                "matched_config_query_pairs"
            ][
                "matched_pairs_total"
            ],
            flush=True,
        )

    # --------------------------------------------------------
    # FINAL DIAGNOSTIC VERDICT
    # --------------------------------------------------------

    vals = [
        results[
            "splits"
        ][
            s
        ][
            "paired_vs_pre"
        ][
            "BOTH"
        ]

        for s in (
            "val",
            "challenge",
        )

        if s in results[
            "splits"
        ]
    ]

    if (
        vals
        and
        all(
            x[
                "ci95"
            ][
                0
            ] > 0

            for x in vals
        )
        and
        all(
            x[
                "mean_system_gain"
            ] > 0

            for x in vals
        )
    ):

        verdict = (
            "EVIDENCE_CONDITIONAL_SIGNAL_ON_EVALUATED_SPLITS"
        )

    elif (
        vals
        and
        any(
            x[
                "ci95"
            ][
                0
            ] > 0

            for x in vals
        )
    ):

        verdict = (
            "MIXED_EVIDENCE_CONDITIONAL_SIGNAL"
        )

    else:

        verdict = (
            "NO_RELIABLE_EVIDENCE_GAIN"
        )

    results[
        "decision"
    ] = {
        "verdict":
            verdict,

        "scientific_scope":
            (
                "Post-response discrimination only; "
                "does NOT prove a prospective selector "
                "can anticipate which query is informative."
            ),
    }

    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "VERDICT =",
        verdict,
        "\nSAVED =",
        OUT
        / "summary.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
