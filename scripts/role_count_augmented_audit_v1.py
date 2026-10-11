from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


ROOT = Path("artifacts/role_count_spectral_v1")
OUT = Path("artifacts/role_count_augmented_audit_v1")
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "manifest.json"
CACHE = ROOT / "api_runs.jsonl"

SEED = 73
DIM = 16
RANK = 4
ENSEMBLE = 5
STEPS = 3000

HORDER = ["00", "10", "01", "11"]

TYPE_CONTRASTS = [
    ("10", "01"),
]

COUNT_CONTRASTS = [
    ("00", "10"),
    ("00", "01"),
    ("10", "11"),
    ("01", "11"),
]

BUDGETS = [1, 2, 3, 5, 10]


def safe_rho(a, b):
    r = float(spearmanr(a, b).statistic)
    return r if np.isfinite(r) else 0.0


def metric(score, actual):
    score = np.asarray(score, float)
    actual = np.asarray(actual, float)

    rho = safe_rho(score, actual)

    n = len(score)
    k = max(1, math.ceil(0.20 * n))

    top = np.argsort(score)[::-1][:k]
    oracle = np.argsort(actual)[::-1][:k]

    base = float(actual.mean())
    selected = float(actual[top].mean())
    best = float(actual[oracle].mean())

    return {
        "rho": rho,
        "lift20": selected / max(base, 1e-12),
        "oracle_lift20": best / max(base, 1e-12),
    }


def load_all():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))

    rows = [
        json.loads(x)
        for x in CACHE.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    pairs = [
        (
            str(x["pair_id"]),
            str(x["A"]),
            str(x["B"]),
        )
        for x in m["pairs"]
    ]

    probes = [
        (
            str(x["probe_id"]),
            str(x["query"]),
        )
        for x in m["probes"]
    ]

    fixed_R = str(m["fixed_R"])
    fixed_M = str(m["fixed_M"])

    return m, rows, pairs, probes, fixed_R, fixed_M


M, ROWS, PAIRS, PROBES, FIXED_R, FIXED_M = load_all()


def seq_for(A, B, h):
    if h == "00":
        return [FIXED_R, FIXED_M]
    if h == "10":
        return [FIXED_R, A, FIXED_M]
    if h == "01":
        return [FIXED_R, B, FIXED_M]
    if h == "11":
        return [FIXED_R, A, B, FIXED_M]
    raise ValueError(h)


def prepare_text():
    pids = [p for p, _ in PROBES]
    queries = [q for _, q in PROBES]

    pos = {
        pid: i
        for i, pid in enumerate(pids)
    }

    outputs = {
        (pair_id, h): [""] * len(PROBES)
        for pair_id, _, _ in PAIRS
        for h in HORDER
    }

    for r in ROWS:
        pair_id = str(r["pair_id"])
        h = str(r["hypothesis_eval_only"])
        pid = str(r["probe_id"])

        key = (pair_id, h)

        if key in outputs and pid in pos:
            outputs[key][pos[pid]] = str(r["final"])

    for key, vals in outputs.items():
        if any(not x for x in vals):
            raise RuntimeError(f"Incomplete outputs for {key}")

    return pids, queries, outputs


class RoleDynamics(nn.Module):
    """
    Two variants:

    context_free:
        h_0 = q
        h'  = A_r h + b_r

    augmented:
        h_0 = 0
        h'  = A_r h + B_r q + b_r

    A and B are low-rank residual maps.
    """

    def __init__(self, roles, d, rank, augmented):
        super().__init__()

        self.roles = list(roles)
        self.ri = {
            r: i
            for i, r in enumerate(self.roles)
        }

        self.d = d
        self.rank = rank
        self.augmented = augmented

        nr = len(self.roles)

        self.UA = nn.Parameter(
            torch.randn(nr, d, rank) * 0.03
        )
        self.VA = nn.Parameter(
            torch.randn(nr, d, rank) * 0.03
        )

        if augmented:
            self.UB = nn.Parameter(
                torch.randn(nr, d, rank) * 0.03
            )
            self.VB = nn.Parameter(
                torch.randn(nr, d, rank) * 0.03
            )
        else:
            self.UB = None
            self.VB = None

        self.bias = nn.Parameter(
            torch.zeros(nr, d)
        )

    def ABb(self, role):
        i = self.ri[role]

        I = torch.eye(
            self.d,
            device=self.UA.device,
        )

        A = (
            I
            + (
                self.UA[i]
                @ self.VA[i].T
            )
            / math.sqrt(self.rank)
        )

        if self.augmented:
            B = (
                self.UB[i]
                @ self.VB[i].T
            ) / math.sqrt(self.rank)
        else:
            B = torch.zeros_like(A)

        b = self.bias[i]

        return A, B, b

    def affine_map(self, seq):
        """
        Returns F where:

            h_final = F @ [q; 1]

        F has shape d x (d+1).
        """

        if self.augmented:
            P = torch.zeros(
                self.d,
                self.d,
                device=self.UA.device,
            )
        else:
            # context-free starts h0=q
            P = torch.eye(
                self.d,
                device=self.UA.device,
            )

        c = torch.zeros(
            self.d,
            device=self.UA.device,
        )

        for role in seq:
            A, B, b = self.ABb(role)

            P = A @ P + B
            c = A @ c + b

        return torch.cat(
            [
                P,
                c[:, None],
            ],
            dim=1,
        )

    def predict(self, q, seq):
        F = self.affine_map(seq)

        qbar = torch.cat(
            [
                q,
                torch.ones(
                    len(q),
                    1,
                    device=q.device,
                ),
            ],
            dim=1,
        )

        return qbar @ F.T


def fit_shared_model(
    qtr,
    ztr,
    *,
    augmented,
    seed,
):
    torch.manual_seed(seed)

    roles = {
        FIXED_R,
        FIXED_M,
    }

    for _, A, B in PAIRS:
        roles.add(A)
        roles.add(B)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = RoleDynamics(
        sorted(roles),
        qtr.shape[1],
        RANK,
        augmented,
    ).to(device)

    tq = torch.tensor(
        qtr,
        dtype=torch.float32,
        device=device,
    )

    tz = {
        key: torch.tensor(
            val,
            dtype=torch.float32,
            device=device,
        )
        for key, val in ztr.items()
    }

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.001,
    )

    best = float("inf")
    best_state = None

    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)

        losses = []

        for pair_id, A, B in PAIRS:
            for h in HORDER:
                pred = model.predict(
                    tq,
                    seq_for(A, B, h),
                )

                losses.append(
                    (
                        (
                            pred
                            - tz[(pair_id, h)]
                        )
                        ** 2
                    ).mean()
                )

        data_loss = torch.stack(losses).mean()

        # Keep transformations modest.
        reg = torch.tensor(
            0.0,
            device=device,
        )

        for p in model.parameters():
            reg = reg + (p ** 2).mean()

        loss = data_loss + 1e-4 * reg

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

        val = float(loss.detach().cpu())

        if val < best:
            best = val
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if step % 1000 == 0:
            print(
                f"[fit {'AUG' if augmented else 'CTX'} "
                f"seed={seed}] "
                f"step={step} "
                f"loss={val:.6f}",
                flush=True,
            )

    model.load_state_dict(best_state)

    return model, best


def get_shared_ensemble_maps(
    qtr,
    ztr,
    *,
    augmented,
):
    all_maps = []
    losses = []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for e in range(ENSEMBLE):
        model, loss = fit_shared_model(
            qtr,
            ztr,
            augmented=augmented,
            seed=SEED + 100 * e + (
                1000 if augmented else 0
            ),
        )

        losses.append(loss)

        maps = {}

        with torch.no_grad():
            for pair_id, A, B in PAIRS:
                for h in HORDER:
                    F = model.affine_map(
                        seq_for(A, B, h)
                    )

                    maps[(pair_id, h)] = (
                        F.detach()
                        .cpu()
                        .numpy()
                    )

        all_maps.append(maps)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_maps = {}

    for key in all_maps[0]:
        mean_maps[key] = np.mean(
            [
                x[key]
                for x in all_maps
            ],
            axis=0,
        )

    return mean_maps, losses


def fit_direct_affine(qtr, ztr):
    maps = {}

    for key, target in ztr.items():
        rr = Ridge(
            alpha=3.0,
            fit_intercept=True,
        ).fit(
            qtr,
            target,
        )

        maps[key] = np.concatenate(
            [
                rr.coef_,
                rr.intercept_[:, None],
            ],
            axis=1,
        )

    return maps


def predict_from_maps(q, maps):
    qbar = np.concatenate(
        [
            q,
            np.ones(
                (len(q), 1),
                dtype=float,
            ),
        ],
        axis=1,
    )

    return {
        key: qbar @ F.T
        for key, F in maps.items()
    }


def estimate_precision(pred_tr, ztr):
    residuals = []

    for key in sorted(ztr):
        residuals.append(
            pred_tr[key]
            - ztr[key]
        )

    residuals = np.vstack(residuals)

    cov = LedoitWolf().fit(
        residuals
    ).covariance_

    cov = (
        cov
        + np.eye(cov.shape[0])
        * 1e-6
    )

    return np.linalg.pinv(cov)


def d2(x, Sinv=None):
    if Sinv is None:
        return np.sum(
            x * x,
            axis=1,
        )

    return np.einsum(
        "ni,ij,nj->n",
        x,
        Sinv,
        x,
    )


def contrast_eval(
    qte,
    zte,
    maps,
    pred,
    contrasts,
    Sinv,
):
    qbar = np.concatenate(
        [
            qte,
            np.ones(
                (len(qte), 1)
            ),
        ],
        axis=1,
    )

    out = []

    exact_errors = []

    for pair_id, A, B in PAIRS:
        for h0, h1 in contrasts:
            F0 = maps[(pair_id, h0)]
            F1 = maps[(pair_id, h1)]

            K = F1 - F0

            # MODEL-PREDICTED difference from two routes.
            pdiff = (
                pred[(pair_id, h1)]
                -
                pred[(pair_id, h0)]
            )

            # ANALYTIC spectral difference.
            kdiff = qbar @ K.T

            exact_errors.append(
                float(
                    np.max(
                        np.abs(
                            pdiff
                            - kdiff
                        )
                    )
                )
            )

            true_diff = (
                zte[(pair_id, h1)]
                -
                zte[(pair_id, h0)]
            )

            score_e = d2(kdiff)
            actual_e = d2(true_diff)

            score_m = d2(
                kdiff,
                Sinv,
            )

            actual_m = d2(
                true_diff,
                Sinv,
            )

            out.append({
                "pair_id": pair_id,
                "contrast": f"{h0}_vs_{h1}",
                "euclid": metric(
                    score_e,
                    actual_e,
                ),
                "maha": metric(
                    score_m,
                    actual_m,
                ),
                "exact_error":
                    exact_errors[-1],
            })

    return out


def global_eval(
    qte,
    zte,
    maps,
    pred,
    Sinv,
):
    rows_e = []
    rows_m = []

    for pair_id, A, B in PAIRS:
        score_e = np.zeros(
            len(qte),
            float,
        )
        actual_e = np.zeros(
            len(qte),
            float,
        )

        score_m = np.zeros(
            len(qte),
            float,
        )
        actual_m = np.zeros(
            len(qte),
            float,
        )

        # Uniform prior over four hypotheses.
        p = 0.25

        for i, hi in enumerate(HORDER):
            for j in range(i + 1, len(HORDER)):
                hj = HORDER[j]

                pd = (
                    pred[(pair_id, hi)]
                    -
                    pred[(pair_id, hj)]
                )

                ad = (
                    zte[(pair_id, hi)]
                    -
                    zte[(pair_id, hj)]
                )

                w = p * p

                score_e += w * d2(pd)
                actual_e += w * d2(ad)

                score_m += w * d2(
                    pd,
                    Sinv,
                )
                actual_m += w * d2(
                    ad,
                    Sinv,
                )

        rows_e.append(
            metric(
                score_e,
                actual_e,
            )
        )

        rows_m.append(
            metric(
                score_m,
                actual_m,
            )
        )

    return rows_e, rows_m


def macro(entries, field):
    vals = [
        x[field]
        for x in entries
    ]

    return {
        "rho": float(
            np.mean([
                x["rho"]
                for x in vals
            ])
        ),
        "lift20": float(
            np.mean([
                x["lift20"]
                for x in vals
            ])
        ),
        "oracle_lift20": float(
            np.mean([
                x["oracle_lift20"]
                for x in vals
            ])
        ),
    }


def macro_plain(vals):
    return {
        "rho": float(
            np.mean([
                x["rho"]
                for x in vals
            ])
        ),
        "lift20": float(
            np.mean([
                x["lift20"]
                for x in vals
            ])
        ),
        "oracle_lift20": float(
            np.mean([
                x["oracle_lift20"]
                for x in vals
            ])
        ),
    }


def posterior_identification(
    qte,
    zte,
    pred,
    maps,
    Sinv,
    *,
    policy,
    repeats,
    seed,
):
    rng = np.random.default_rng(seed)

    accum = {
        b: defaultdict(list)
        for b in BUDGETS
        if b <= len(qte)
    }

    budgets = sorted(accum)

    for rep in range(repeats):
        for pair_id, A, B in PAIRS:
            for true_h in HORDER:
                post = np.ones(4) / 4.0
                unused = list(
                    range(len(qte))
                )

                for step in range(
                    1,
                    max(budgets) + 1,
                ):
                    if policy == "spectral":
                        scores = np.zeros(
                            len(unused),
                            float,
                        )

                        for i, hi in enumerate(HORDER):
                            for j in range(
                                i + 1,
                                len(HORDER),
                            ):
                                hj = HORDER[j]

                                diff = (
                                    pred[(pair_id, hi)][unused]
                                    -
                                    pred[(pair_id, hj)][unused]
                                )

                                scores += (
                                    post[i]
                                    * post[j]
                                    * d2(
                                        diff,
                                        Sinv,
                                    )
                                )

                        chosen = unused[
                            int(
                                np.argmax(
                                    scores
                                )
                            )
                        ]

                    else:
                        chosen = int(
                            rng.choice(
                                unused
                            )
                        )

                    obs = zte[
                        (pair_id, true_h)
                    ][chosen]

                    ll = []

                    for h in HORDER:
                        err = (
                            obs
                            -
                            pred[(pair_id, h)][chosen]
                        )

                        ll.append(
                            -0.5
                            * float(
                                err
                                @ Sinv
                                @ err
                            )
                        )

                    logp = (
                        np.log(
                            np.clip(
                                post,
                                1e-12,
                                1.0,
                            )
                        )
                        + np.asarray(ll)
                    )

                    logp -= logp.max()

                    post = np.exp(logp)
                    post /= post.sum()

                    unused.remove(chosen)

                    if step not in accum:
                        continue

                    pred_h = HORDER[
                        int(
                            np.argmax(post)
                        )
                    ]

                    # K = number of agents in this controlled chain.
                    k_by_h = {
                        h: len(
                            seq_for(
                                A,
                                B,
                                h,
                            )
                        )
                        for h in HORDER
                    }

                    pk = defaultdict(float)

                    for i, h in enumerate(HORDER):
                        pk[k_by_h[h]] += post[i]

                    pred_k = max(
                        pk,
                        key=pk.get,
                    )

                    true_k = k_by_h[
                        true_h
                    ]

                    pA = (
                        post[HORDER.index("10")]
                        +
                        post[HORDER.index("11")]
                    )

                    pB = (
                        post[HORDER.index("01")]
                        +
                        post[HORDER.index("11")]
                    )

                    true_A = int(
                        true_h in ("10", "11")
                    )

                    true_B = int(
                        true_h in ("01", "11")
                    )

                    accum[step]["H_acc"].append(
                        float(
                            pred_h
                            == true_h
                        )
                    )

                    accum[step]["K_acc"].append(
                        float(
                            pred_k
                            == true_k
                        )
                    )

                    accum[step]["A_acc"].append(
                        float(
                            int(pA >= 0.5)
                            == true_A
                        )
                    )

                    accum[step]["B_acc"].append(
                        float(
                            int(pB >= 0.5)
                            == true_B
                        )
                    )

    return {
        b: {
            k: float(
                np.mean(v)
            )
            for k, v in accum[b].items()
        }
        for b in budgets
    }


def heldout_mse(pred, zte):
    return float(
        np.mean([
            np.mean(
                (
                    pred[key]
                    -
                    zte[key]
                )
                ** 2
            )
            for key in zte
        ])
    )


def run():
    pids, queries, outputs = prepare_text()

    idx = np.arange(
        len(PROBES)
    )

    tr_idx, te_idx = train_test_split(
        idx,
        train_size=0.70,
        random_state=SEED,
        shuffle=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 110)
    print("AUGMENTED-STATE SPECTRAL AUDIT")
    print("=" * 110)
    print("device:", device)
    print(
        f"probes={len(idx)} "
        f"train={len(tr_idx)} "
        f"heldout={len(te_idx)}"
    )

    enc = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )

    print("[embed] queries")

    q_full = enc.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    z_full = {}

    for key in sorted(outputs):
        print("[embed]", key)

        z_full[key] = enc.encode(
            outputs[key],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # Shared PCA fit ONLY on training data.
    pca_train = np.vstack(
        [
            q_full[tr_idx],
            *[
                z_full[key][tr_idx]
                for key in sorted(z_full)
            ],
        ]
    )

    d = min(
        DIM,
        pca_train.shape[0] - 1,
        pca_train.shape[1],
    )

    pca = PCA(
        n_components=d,
        random_state=SEED,
    ).fit(
        pca_train
    )

    # IMPORTANT:
    # No L2 normalization after PCA.
    # Keeps affine/operator geometry internally consistent.
    q = pca.transform(
        q_full
    )

    z = {
        key: pca.transform(val)
        for key, val in z_full.items()
    }

    qtr = q[tr_idx]
    qte = q[te_idx]

    ztr = {
        key: val[tr_idx]
        for key, val in z.items()
    }

    zte = {
        key: val[te_idx]
        for key, val in z.items()
    }

    models = {}

    print("\n[1/3] fitting DIRECT AFFINE")

    models["DIRECT_AFFINE"] = {
        "maps":
            fit_direct_affine(
                qtr,
                ztr,
            ),
        "train_losses": [],
    }

    print("\n[2/3] fitting CONTEXT-FREE shared operators")

    ctx_maps, ctx_losses = (
        get_shared_ensemble_maps(
            qtr,
            ztr,
            augmented=False,
        )
    )

    models["CONTEXT_FREE"] = {
        "maps": ctx_maps,
        "train_losses": ctx_losses,
    }

    print("\n[3/3] fitting AUGMENTED q-reinjection operators")

    aug_maps, aug_losses = (
        get_shared_ensemble_maps(
            qtr,
            ztr,
            augmented=True,
        )
    )

    models["AUGMENTED_Q"] = {
        "maps": aug_maps,
        "train_losses": aug_losses,
    }

    result = {
        "latent_dim": d,
        "rank": RANK,
        "ensemble": ENSEMBLE,
        "heldout_probe_ids": [
            pids[i]
            for i in te_idx
        ],
        "models": {},
    }

    report = []

    report.append(
        "=" * 118
    )
    report.append(
        "AUGMENTED-STATE SPECTRAL AUDIT — ROLE / COUNT"
    )
    report.append(
        "=" * 118
    )

    report.append(
        f"40 cached probes | "
        f"train={len(tr_idx)} | "
        f"heldout={len(te_idx)} | "
        f"d={d} | low-rank={RANK} | ensemble={ENSEMBLE}"
    )

    report.append(
        "PRIMARY QUESTION: does q-reinjection h'=A_r h + B_r q + b_r "
        "restore model-consistent spectral ranking?"
    )

    for name, obj in models.items():
        maps = obj["maps"]

        ptr = predict_from_maps(
            qtr,
            maps,
        )

        pte = predict_from_maps(
            qte,
            maps,
        )

        Sinv = estimate_precision(
            ptr,
            ztr,
        )

        type_rows = contrast_eval(
            qte,
            zte,
            maps,
            pte,
            TYPE_CONTRASTS,
            Sinv,
        )

        count_rows = contrast_eval(
            qte,
            zte,
            maps,
            pte,
            COUNT_CONTRASTS,
            Sinv,
        )

        ge, gm = global_eval(
            qte,
            zte,
            maps,
            pte,
            Sinv,
        )

        type_e = macro(
            type_rows,
            "euclid",
        )

        type_m = macro(
            type_rows,
            "maha",
        )

        count_e = macro(
            count_rows,
            "euclid",
        )

        count_m = macro(
            count_rows,
            "maha",
        )

        global_e = macro_plain(
            ge
        )

        global_m = macro_plain(
            gm
        )

        exact_err = max(
            [
                x["exact_error"]
                for x in (
                    type_rows
                    + count_rows
                )
            ]
        )

        mse = heldout_mse(
            pte,
            zte,
        )

        spectral_id = (
            posterior_identification(
                qte,
                zte,
                pte,
                maps,
                Sinv,
                policy="spectral",
                repeats=1,
                seed=SEED + 10,
            )
        )

        random_id = (
            posterior_identification(
                qte,
                zte,
                pte,
                maps,
                Sinv,
                policy="random",
                repeats=300,
                seed=SEED + 20,
            )
        )

        result["models"][name] = {
            "heldout_mse": mse,
            "spectral_exactness_max_abs_error":
                exact_err,
            "type_euclidean": type_e,
            "type_mahalanobis": type_m,
            "count_euclidean": count_e,
            "count_mahalanobis": count_m,
            "global_euclidean": global_e,
            "global_mahalanobis": global_m,
            "spectral_identification":
                spectral_id,
            "random_identification":
                random_id,
            "train_losses":
                obj["train_losses"],
        }

        report.append("")
        report.append(
            "-" * 118
        )
        report.append(name)
        report.append(
            "-" * 118
        )

        report.append(
            f"Held-out output MSE       : {mse:.6f}"
        )

        report.append(
            f"Spectral exactness error  : {exact_err:.3e}"
        )

        report.append("")
        report.append(
            "EUCLIDEAN actual/predicted separation"
        )

        report.append(
            f"  TYPE   rho={type_e['rho']:+.4f} "
            f"lift={type_e['lift20']:.3f}x "
            f"oracle={type_e['oracle_lift20']:.3f}x"
        )

        report.append(
            f"  COUNT  rho={count_e['rho']:+.4f} "
            f"lift={count_e['lift20']:.3f}x "
            f"oracle={count_e['oracle_lift20']:.3f}x"
        )

        report.append(
            f"  GLOBAL rho={global_e['rho']:+.4f} "
            f"lift={global_e['lift20']:.3f}x "
            f"oracle={global_e['oracle_lift20']:.3f}x"
        )

        report.append("")
        report.append(
            "NOISE-AWARE MAHALANOBIS separation"
        )

        report.append(
            f"  TYPE   rho={type_m['rho']:+.4f} "
            f"lift={type_m['lift20']:.3f}x "
            f"oracle={type_m['oracle_lift20']:.3f}x"
        )

        report.append(
            f"  COUNT  rho={count_m['rho']:+.4f} "
            f"lift={count_m['lift20']:.3f}x "
            f"oracle={count_m['oracle_lift20']:.3f}x"
        )

        report.append(
            f"  GLOBAL rho={global_m['rho']:+.4f} "
            f"lift={global_m['lift20']:.3f}x "
            f"oracle={global_m['oracle_lift20']:.3f}x"
        )

        report.append("")
        report.append(
            "Adaptive identification "
            "(Mahalanobis spectral selection)"
        )

        report.append(
            "budget | H spec/random | K spec/random | "
            "A-pres | B-pres"
        )

        for b in sorted(
            spectral_id
        ):
            s = spectral_id[b]
            r = random_id[b]

            report.append(
                f"{b:6d} | "
                f"{s['H_acc']:.3f}/{r['H_acc']:.3f}     | "
                f"{s['K_acc']:.3f}/{r['K_acc']:.3f}     | "
                f"{s['A_acc']:.3f}  | "
                f"{s['B_acc']:.3f}"
            )

    # Diagnosis.
    ctx = result["models"][
        "CONTEXT_FREE"
    ]

    aug = result["models"][
        "AUGMENTED_Q"
    ]

    direct = result["models"][
        "DIRECT_AFFINE"
    ]

    report.append("")
    report.append(
        "=" * 118
    )
    report.append(
        "DIAGNOSTIC"
    )
    report.append(
        "=" * 118
    )

    report.append(
        "Interpretation rules:"
    )

    report.append(
        "1) exactness error ~1e-6 or smaller => spectral algebra/code is numerically consistent with model predictions."
    )

    report.append(
        "2) AUGMENTED_Q >> CONTEXT_FREE => repeated task injection q was a real missing-state variable."
    )

    report.append(
        "3) DIRECT_AFFINE good but AUGMENTED_Q weak => information is predictable, but shared primitive role factorization is still wrong."
    )

    report.append(
        "4) DIRECT_AFFINE also weak => current embedding/linear family cannot predict which held-out probes reveal role/count."
    )

    ctx_g = ctx[
        "global_mahalanobis"
    ]["rho"]

    aug_g = aug[
        "global_mahalanobis"
    ]["rho"]

    dir_g = direct[
        "global_mahalanobis"
    ]["rho"]

    if (
        aug_g >= 0.20
        and
        aug_g >= ctx_g + 0.10
    ):
        verdict = (
            "AUGMENTED_STATE_SUPPORTED"
        )

    elif (
        dir_g >= 0.20
        and
        aug_g < 0.15
    ):
        verdict = (
            "SIGNAL_EXISTS_SHARED_FACTORIZATION_STILL_WEAK"
        )

    elif (
        max(
            abs(
                aug[
                    "spectral_exactness_max_abs_error"
                ]
            ),
            abs(
                ctx[
                    "spectral_exactness_max_abs_error"
                ]
            ),
        )
        > 1e-5
    ):
        verdict = (
            "IMPLEMENTATION_INCONSISTENCY"
        )

    else:
        verdict = (
            "LINEAR_LATENT_MODEL_STILL_INSUFFICIENT"
        )

    report.append("")
    report.append(
        f"VERDICT: {verdict}"
    )

    report.append(
        f"global Mahalanobis rho: "
        f"context={ctx_g:+.4f}, "
        f"augmented={aug_g:+.4f}, "
        f"direct={dir_g:+.4f}"
    )

    report.append(
        "=" * 118
    )

    txt = "\n".join(
        report
    )

    (
        OUT / "REPORT.txt"
    ).write_text(
        txt + "\n",
        encoding="utf-8",
    )

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(txt)


if __name__ == "__main__":
    run()
