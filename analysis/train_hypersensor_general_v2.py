from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analysis.train_hypersensor_general_v1 as V1


TRAIN_PATH = V1.TRAIN_PATH
VAL_PATH = V1.VAL_PATH

OUT = Path("artifacts/hypersensor_general_v2")

DEVICE = V1.DEVICE

ROLE_FOLDS = 5
SYSTEM_FOLDS = 3
SEEDS = [1397, 4099, 7919]

EPOCHS = 260
LR = 2e-3
WEIGHT_DECAY = 1e-2
HYPER_RANK = 12

BCE_WEIGHT = 0.05
CF_WEIGHT = 0.50

SHUFFLE_REPEATS = 10


def dump(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# RESPONSE-ANCHORED GEOMETRY
# ============================================================

def delta_geom(data):
    """
    Response-specific displacement.

    Exactly zero when output == query.

    This removes all static role/query geometry from the
    evidence itself.
    """
    return (
        data.G
        - data.G_query_only
    ).astype(np.float32)


def fit_scale(arrays):
    """
    TRAIN-only feature scaling.

    IMPORTANT:
    no centering.

    Delta=0 must remain the exact null anchor.
    """
    X = np.concatenate(
        arrays,
        axis=0,
    ).astype(np.float32)

    sd = X.std(
        axis=0
    ).astype(np.float32)

    positive = sd[sd > 1e-6]

    if len(positive):
        floor = max(
            1e-3,
            0.05 * float(
                np.median(positive)
            ),
        )
    else:
        floor = 1e-3

    return np.maximum(
        sd,
        floor,
    ).astype(np.float32)


# ============================================================
# MODELS
# ============================================================

class AnchoredScorer(nn.Module):
    """
    General response-evidence scorer.

    Key constraints:

    1. score is based only on:
         G(real response) - G(query-only null)

    2. NO context-dependent bias.

    3. role/query may rotate the evidence direction,
       but not create arbitrary score offsets.

    4. decoding vector is unit-normalized so scores
       from different roles remain comparable.
    """

    def __init__(
        self,
        context_dim=None,
        alpha=0.0,
        use_magnitude=False,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.use_context = (
            context_dim is not None
            and self.alpha > 0
        )

        self.use_magnitude = bool(
            use_magnitude
        )

        self.base_w = nn.Parameter(
            torch.empty(20)
        )

        nn.init.normal_(
            self.base_w,
            mean=0.0,
            std=0.10,
        )

        if self.use_context:
            self.context = nn.Linear(
                context_dim,
                HYPER_RANK,
            )

            self.delta_w = nn.Linear(
                HYPER_RANK,
                20,
                bias=False,
            )

            nn.init.normal_(
                self.context.weight,
                mean=0.0,
                std=0.02,
            )

            nn.init.zeros_(
                self.context.bias
            )

            nn.init.normal_(
                self.delta_w.weight,
                mean=0.0,
                std=0.02,
            )

        if self.use_magnitude:
            # GLOBAL coefficient only.
            # Never role-specific.
            self.mag_weight = nn.Parameter(
                torch.zeros(())
            )

    def forward(self, d, c=None):
        # Magnitude before sample normalization.
        mag = torch.linalg.vector_norm(
            d,
            dim=1,
        )

        # Same evidence scale for every role/query.
        u = d / torch.clamp(
            mag[:, None],
            min=1e-6,
        )

        if self.use_context:
            h = torch.tanh(
                self.context(c)
            )

            dw = torch.tanh(
                self.delta_w(h)
            )

            wraw = (
                self.base_w[None, :]
                + self.alpha * dw
            )

        else:
            wraw = self.base_w[None, :]

        # CRITICAL:
        # role-conditioned direction, shared scale.
        w = F.normalize(
            wraw,
            p=2,
            dim=1,
            eps=1e-6,
        )

        s = (
            u * w
        ).sum(dim=1)

        if self.use_magnitude:
            s = (
                s
                + self.mag_weight
                * torch.log1p(mag)
            )

        return s


# ============================================================
# TRAINING
# ============================================================

def make_rank_groups(rows):
    return V1.make_rank_groups(rows)


def make_cf_pairs(rows):
    return V1.make_cf_pairs(rows)


def rank_loss(scores, groups):
    return V1.rank_loss(
        scores,
        groups,
    )


def balanced_bce(scores, y):
    return V1.balanced_bce(
        scores,
        y,
    )


def context_array(data, context_kind):
    if context_kind == "none":
        return None

    if context_kind == "role":
        return data.C_role

    if context_kind == "rq":
        return data.C_role_query

    raise ValueError(context_kind)


def train_model(
    data0,
    idx0,
    seed,
    *,
    context_kind,
    alpha,
    use_magnitude=False,
    data1=None,
    idx1=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx0 = np.asarray(
        idx0,
        dtype=int,
    )

    rows0 = [
        data0.rows[i]
        for i in idx0
    ]

    D0_raw = delta_geom(
        data0
    )[idx0]

    scale_inputs = [
        D0_raw
    ]

    rows1 = None
    D1_raw = None

    if (
        data1 is not None
        and idx1 is not None
        and len(idx1) > 0
    ):
        idx1 = np.asarray(
            idx1,
            dtype=int,
        )

        rows1 = [
            data1.rows[i]
            for i in idx1
        ]

        D1_raw = delta_geom(
            data1
        )[idx1]

        scale_inputs.append(
            D1_raw
        )

    sd = fit_scale(
        scale_inputs
    )

    D0 = torch.tensor(
        D0_raw / sd,
        dtype=torch.float32,
        device=DEVICE,
    )

    C0_np = context_array(
        data0,
        context_kind,
    )

    C0 = (
        None
        if C0_np is None
        else torch.tensor(
            C0_np[idx0],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    y0 = torch.tensor(
        [
            int(x["label"])
            for x in rows0
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    groups = make_rank_groups(
        rows0
    )

    context_dim = (
        None
        if C0 is None
        else int(C0.shape[1])
    )

    model = AnchoredScorer(
        context_dim=context_dim,
        alpha=alpha,
        use_magnitude=use_magnitude,
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    if rows1 is not None:
        D1 = torch.tensor(
            D1_raw / sd,
            dtype=torch.float32,
            device=DEVICE,
        )

        C1_np = context_array(
            data1,
            context_kind,
        )

        C1 = (
            None
            if C1_np is None
            else torch.tensor(
                C1_np[idx1],
                dtype=torch.float32,
                device=DEVICE,
            )
        )

        y1 = torch.tensor(
            [
                int(x["label"])
                for x in rows1
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        cf_pairs = make_cf_pairs(
            rows1
        )

    else:
        D1 = None
        C1 = None
        y1 = None
        cf_pairs = []

    model.train()

    for epoch in range(EPOCHS):
        opt.zero_grad(
            set_to_none=True
        )

        s0 = model(
            D0,
            C0,
        )

        loss = rank_loss(
            s0,
            groups,
        )

        loss = (
            loss
            + BCE_WEIGHT
            * balanced_bce(
                s0,
                y0,
            )
        )

        if (
            D1 is not None
            and cf_pairs
        ):
            s1 = model(
                D1,
                C1,
            )

            cf = torch.stack([
                F.softplus(
                    s1[ni]
                    - s1[pi]
                )
                for pi, ni in cf_pairs
            ]).mean()

            loss = (
                loss
                + CF_WEIGHT * cf
                + 0.02
                * balanced_bce(
                    s1,
                    y1,
                )
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"nonfinite loss "
                f"epoch={epoch}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

    return model, sd


@torch.no_grad()
def predict(
    model,
    data,
    idx,
    sd,
    context_kind,
    override_output=None,
):
    idx = np.asarray(
        idx,
        dtype=int,
    )

    if override_output is None:
        Draw = delta_geom(
            data
        )[idx]

    else:
        # override_output = output embedding matrix
        Draw = np.stack([
            V1.geom_from_vectors(
                data.R[i],
                data.Q[i],
                override_output[i],
            )
            - data.G_query_only[i]
            for i in idx
        ]).astype(np.float32)

    D = torch.tensor(
        Draw / sd,
        dtype=torch.float32,
        device=DEVICE,
    )

    Cnp = context_array(
        data,
        context_kind,
    )

    C = (
        None
        if Cnp is None
        else torch.tensor(
            Cnp[idx],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    model.eval()

    return (
        model(D, C)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def shuffled_outputs(data, seed):
    rng = np.random.default_rng(seed)

    out = data.O.copy()

    by_role = defaultdict(list)

    for i, x in enumerate(data.rows):
        by_role[
            str(x["target_role"])
        ].append(i)

    for _, idxs in by_role.items():
        idxs = np.asarray(
            idxs,
            dtype=int,
        )

        perm = idxs.copy()

        rng.shuffle(perm)

        out[idxs] = data.O[perm]

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise SystemExit(
            f"{OUT} non-empty; refusing overwrite"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = V1.read_jsonl(
        TRAIN_PATH
    )

    val = V1.read_jsonl(
        VAL_PATH
    )

    t0 = [
        x for x in train
        if str(x["probe_family"]) == "T0"
    ]

    t1 = [
        x for x in train
        if str(x["probe_family"]) == "T1"
    ]

    assert len(t0) == 720
    assert len(t1) == 190
    assert len(val) == 760

    roles = sorted({
        str(x["target_role"])
        for x in t0
    })

    systems = sorted({
        str(x["system_id"])
        for x in t0
    })

    print("=" * 110)
    print("HYPERSENSOR GENERAL V2 — ANCHORED EVIDENCE")
    print("=" * 110)

    print("device:", DEVICE)
    print("T0:", len(t0))
    print("T1:", len(t1))
    print("VAL:", len(val))
    print("roles:", len(roles))
    print("systems:", len(systems))
    print("INTERNAL TEST NOT LOADED")
    print()

    emb = V1.BatchedEmbeddings(
        [t0, t1, val]
    )

    D0 = V1.build_data(
        t0,
        emb,
    )

    D1 = V1.build_data(
        t1,
        emb,
    )

    DD = V1.build_data(
        val,
        emb,
    )

    role_folds = V1.make_folds(
        roles,
        ROLE_FOLDS,
        seed=20260928,
    )

    system_folds = V1.make_folds(
        systems,
        SYSTEM_FOLDS,
        seed=20260929,
    )

    variants = {
        "anchor_shared": {
            "context": "none",
            "alpha": 0.0,
            "mag": False,
        },

        "anchor_hyper_role_a25": {
            "context": "role",
            "alpha": 0.25,
            "mag": False,
        },

        "anchor_hyper_rq_a10": {
            "context": "rq",
            "alpha": 0.10,
            "mag": False,
        },

        "anchor_hyper_rq_a25": {
            "context": "rq",
            "alpha": 0.25,
            "mag": False,
        },

        "anchor_hyper_rq_a50": {
            "context": "rq",
            "alpha": 0.50,
            "mag": False,
        },

        "anchor_hyper_rq_a25_mag": {
            "context": "rq",
            "alpha": 0.25,
            "mag": True,
        },
    }

    # --------------------------------------------------------
    # A. STRICT T0:
    # unseen role AND unseen system
    # --------------------------------------------------------

    sums = {
        k: np.zeros(
            len(t0),
            dtype=np.float64,
        )
        for k in variants
    }

    ns = {
        k: np.zeros(
            len(t0),
            dtype=np.int64,
        )
        for k in variants
    }

    print(
        "\n=== A. STRICT T0 DOUBLE HOLDOUT ===",
        flush=True,
    )

    for rf, held_roles0 in enumerate(
        role_folds,
        1,
    ):
        held_roles = set(
            held_roles0
        )

        seen_roles = (
            set(roles)
            - held_roles
        )

        for sf, held_systems0 in enumerate(
            system_folds,
            1,
        ):
            held_systems = set(
                held_systems0
            )

            seen_systems = (
                set(systems)
                - held_systems
            )

            tr = V1.indices_where(
                t0,
                roles=seen_roles,
                systems=seen_systems,
            )

            ev = V1.indices_where(
                t0,
                roles=held_roles,
                systems=held_systems,
            )

            print(
                f"rf={rf}/5 sf={sf}/3 "
                f"train={len(tr)} eval={len(ev)}",
                flush=True,
            )

            for name, cfg in variants.items():
                seed_scores = []

                for seed in SEEDS:
                    model, sd = train_model(
                        D0,
                        tr,
                        seed,
                        context_kind=cfg["context"],
                        alpha=cfg["alpha"],
                        use_magnitude=cfg["mag"],
                    )

                    seed_scores.append(
                        predict(
                            model,
                            D0,
                            ev,
                            sd,
                            cfg["context"],
                        )
                    )

                    del model

                pred = np.mean(
                    np.stack(
                        seed_scores
                    ),
                    axis=0,
                )

                sums[name][ev] += pred
                ns[name][ev] += 1

    strict_scores = {}
    strict_metrics = {}

    for name in variants:
        assert np.all(
            ns[name] == 1
        ), (name, np.unique(ns[name], return_counts=True))

        strict_scores[name] = (
            sums[name]
            / ns[name]
        )

        strict_metrics[name] = V1.evaluate(
            t0,
            strict_scores[name],
        )

    # --------------------------------------------------------
    # B. ROLE-HELDOUT T0 -> T1 / D
    # --------------------------------------------------------

    d_sum = {
        k: np.zeros(
            len(val),
            dtype=np.float64,
        )
        for k in variants
    }

    d_n = {
        k: np.zeros(
            len(val),
            dtype=np.int64,
        )
        for k in variants
    }

    t1_sum = {
        k: np.zeros(
            len(t1),
            dtype=np.float64,
        )
        for k in variants
    }

    t1_n = {
        k: np.zeros(
            len(t1),
            dtype=np.int64,
        )
        for k in variants
    }

    # Main multifamily candidate.
    d_multi_sum = np.zeros(
        len(val),
        dtype=np.float64,
    )

    d_multi_n = np.zeros(
        len(val),
        dtype=np.int64,
    )

    # Shuffled output on the primary a25 model.
    shuf_outputs = [
        shuffled_outputs(
            DD,
            93000 + i,
        )
        for i in range(
            SHUFFLE_REPEATS
        )
    ]

    shuf_sum = [
        np.zeros(
            len(val),
            dtype=np.float64,
        )
        for _ in range(
            SHUFFLE_REPEATS
        )
    ]

    shuf_n = [
        np.zeros(
            len(val),
            dtype=np.int64,
        )
        for _ in range(
            SHUFFLE_REPEATS
        )
    ]

    print(
        "\n=== B. UNSEEN-ROLE TRANSFER ===",
        flush=True,
    )

    for rf, held_roles0 in enumerate(
        role_folds,
        1,
    ):
        held_roles = set(
            held_roles0
        )

        seen_roles = (
            set(roles)
            - held_roles
        )

        tr0 = V1.indices_where(
            t0,
            roles=seen_roles,
        )

        ev1 = V1.indices_where(
            t1,
            roles=held_roles,
        )

        evd = V1.indices_where(
            val,
            roles=held_roles,
        )

        tr1 = V1.indices_where(
            t1,
            roles=seen_roles,
        )

        print(
            f"rf={rf}/5 "
            f"T0train={len(tr0)} "
            f"T1eval={len(ev1)} "
            f"Deval={len(evd)} "
            f"T1aux={len(tr1)}",
            flush=True,
        )

        for name, cfg in variants.items():
            ds = []
            ts = []

            shuffled_seed_scores = [
                []
                for _ in range(
                    SHUFFLE_REPEATS
                )
            ]

            for seed in SEEDS:
                model, sd = train_model(
                    D0,
                    tr0,
                    seed,
                    context_kind=cfg["context"],
                    alpha=cfg["alpha"],
                    use_magnitude=cfg["mag"],
                )

                ds.append(
                    predict(
                        model,
                        DD,
                        evd,
                        sd,
                        cfg["context"],
                    )
                )

                ts.append(
                    predict(
                        model,
                        D1,
                        ev1,
                        sd,
                        cfg["context"],
                    )
                )

                if name == "anchor_hyper_rq_a25":
                    for si, osh in enumerate(
                        shuf_outputs
                    ):
                        shuffled_seed_scores[
                            si
                        ].append(
                            predict(
                                model,
                                DD,
                                evd,
                                sd,
                                "rq",
                                override_output=osh,
                            )
                        )

                del model

            d_pred = np.mean(
                np.stack(ds),
                axis=0,
            )

            t_pred = np.mean(
                np.stack(ts),
                axis=0,
            )

            d_sum[name][evd] += d_pred
            d_n[name][evd] += 1

            t1_sum[name][ev1] += t_pred
            t1_n[name][ev1] += 1

            if name == "anchor_hyper_rq_a25":
                for si in range(
                    SHUFFLE_REPEATS
                ):
                    sp = np.mean(
                        np.stack(
                            shuffled_seed_scores[
                                si
                            ]
                        ),
                        axis=0,
                    )

                    shuf_sum[si][evd] += sp
                    shuf_n[si][evd] += 1

        # T0 + T1 seen roles -> D unseen role.
        seed_scores = []

        for seed in SEEDS:
            model, sd = train_model(
                D0,
                tr0,
                seed,
                context_kind="rq",
                alpha=0.25,
                use_magnitude=False,
                data1=D1,
                idx1=tr1,
            )

            seed_scores.append(
                predict(
                    model,
                    DD,
                    evd,
                    sd,
                    "rq",
                )
            )

            del model

        pp = np.mean(
            np.stack(
                seed_scores
            ),
            axis=0,
        )

        d_multi_sum[evd] += pp
        d_multi_n[evd] += 1

    d_metrics = {}
    t1_metrics = {}

    for name in variants:
        assert np.all(
            d_n[name] == 1
        )

        assert np.all(
            t1_n[name] == 1
        )

        ds = (
            d_sum[name]
            / d_n[name]
        )

        ts = (
            t1_sum[name]
            / t1_n[name]
        )

        d_metrics[name] = V1.evaluate(
            val,
            ds,
        )

        t1_metrics[name] = V1.matched_pair_metrics(
            t1,
            ts,
        )

    assert np.all(
        d_multi_n == 1
    )

    d_multi = (
        d_multi_sum
        / d_multi_n
    )

    d_metrics[
        "anchor_hyper_rq_a25_T0plusT1"
    ] = V1.evaluate(
        val,
        d_multi,
    )

    shuf_metrics = []

    for si in range(
        SHUFFLE_REPEATS
    ):
        assert np.all(
            shuf_n[si] == 1
        )

        ss = (
            shuf_sum[si]
            / shuf_n[si]
        )

        shuf_metrics.append(
            V1.evaluate(
                val,
                ss,
            )
        )

    shuf_summary = {
        "repeats":
            SHUFFLE_REPEATS,

        "mean_global_auc":
            float(np.mean([
                x["global_auc"]
                for x in shuf_metrics
            ])),

        "mean_role_auc":
            float(np.mean([
                x["mean_role_auc"]
                for x in shuf_metrics
            ])),

        "mean_within_system_auc":
            float(np.mean([
                x["mean_within_system_auc"]
                for x in shuf_metrics
            ])),
    }

    summary = {
        "version":
            "hypersensor_general_v2_anchored",

        "device":
            DEVICE,

        "internal_test_used":
            False,

        "core_change":
            "score response-specific delta geometry; "
            "no context-dependent bias; "
            "unit-normalized decoding direction",

        "variants":
            variants,

        "strict_T0_double_holdout":
            strict_metrics,

        "T0_to_T1_unseen_role":
            t1_metrics,

        "T0_to_D_unseen_role":
            d_metrics,

        "D_shuffled_primary":
            shuf_summary,
    }

    dump(
        OUT / "summary.json",
        summary,
    )

    print()
    print("=" * 110)
    print("FINAL ANCHORED GENERALIZATION SUMMARY")
    print("=" * 110)

    print(
        "\n--- STRICT T0: UNSEEN ROLE + UNSEEN SYSTEM ---"
    )

    for name, m in strict_metrics.items():
        print(
            f"{name:<34} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"topK*={m['exact_topK_oracleK_rate']:.4f}"
        )

    print(
        "\n--- T1 MATCHED: UNSEEN ROLE ---"
    )

    for name, m in t1_metrics.items():
        print(
            f"{name:<34} "
            f"pairAcc={m['pair_accuracy']:.4f} "
            f"margin={m['mean_margin']:+.4f}"
        )

    print(
        "\n--- D: UNSEEN ROLE + NEW SYSTEM + NEW WORDING ---"
    )

    for name, m in d_metrics.items():
        print(
            f"{name:<34} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"topK*={m['exact_topK_oracleK_rate']:.4f}"
        )

    real = d_metrics[
        "anchor_hyper_rq_a25"
    ]

    print(
        "\n--- SHUFFLED OUTPUT CONTROL, a25 ---"
    )

    print(
        f"real roleAUC={real['mean_role_auc']:.4f} "
        f"real sysAUC={real['mean_within_system_auc']:.4f}"
    )

    print(
        f"shuf roleAUC={shuf_summary['mean_role_auc']:.4f} "
        f"shuf sysAUC={shuf_summary['mean_within_system_auc']:.4f}"
    )

    # Diagnostic only; uses eval labels to flip.
    print(
        "\n--- ORIENTATION-FREE DIAGNOSTIC ONLY ---"
    )

    for section, metrics in [
        ("T0", strict_metrics),
        ("D", d_metrics),
    ]:
        for name, m in metrics.items():
            aucs = list(
                m.get(
                    "per_role_auc",
                    {}
                ).values()
            )

            if aucs:
                of = float(np.mean([
                    max(x, 1.0 - x)
                    for x in aucs
                ]))

                print(
                    f"{section:<3} {name:<34} "
                    f"orientation_free={of:.4f}"
                )

    print()
    print("INTERNAL TEST UNTOUCHED.")
    print(
        "Summary:",
        OUT / "summary.json"
    )


if __name__ == "__main__":
    main()
