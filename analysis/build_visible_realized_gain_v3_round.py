#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import analysis.observation_model_v1 as om
import analysis.realized_posterior_gain_diag_v1 as rg
import analysis.train_v13_6b_activation_judge as tr


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def stable_seed(*parts):
    s = "|".join(map(str, parts)).encode()
    return int.from_bytes(
        hashlib.sha256(s).digest()[:8], "big"
    ) % (2**32)


def load_raw(root):
    raw = {}

    for p in sorted(root.rglob("responses_*_round_*.jsonl")):
        parent = str(p.parent)

        for x in read_jsonl(p):
            sid = str(
                x.get("system_id")
                or x.get("session_id")
                or ""
            )
            rnd = x.get("round")
            ci = x.get("candidate_index")
            q = str(x.get("query") or "")

            if not sid or rnd is None or ci is None:
                continue

            key = (
                parent,
                sid,
                int(rnd),
                int(ci),
                q,
            )

            raw[key] = str(
                x.get("final_output")
                or x.get("answer")
                or x.get("output")
                or ""
            )

    return raw


def load_sensor(adapter):
    from peft import PeftModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    adapter = Path(adapter)
    cfg = json.loads(
        (adapter / "adapter_config.json").read_text()
    )
    base_name = cfg["base_model_name_or_path"]

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            adapter,
            use_fast=True,
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            base_name,
            use_fast=True,
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=dtype,
    )

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
    )
    model.eval()

    return tok, model, dtype


@torch.no_grad()
def sensor_score_row(
    model,
    tok,
    dtype,
    roles,
    role_desc,
    query,
    answer,
    max_length,
):
    # Fixed deployment protocol:
    # all 20 candidate roles for ONE visible response
    # are scored in the same batch.
    texts = [
        tr.presence_text(
            role,
            role_desc[role],
            role,          # SELF target encoding
            query,
            answer,
        )
        for role in roles
    ]

    x = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    x = {
        k: v.to("cuda:0")
        for k, v in x.items()
    }

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=True,
    ):
        z = model(**x).logits[:, 0].float()

    return z.detach().cpu().numpy().astype(np.float64)


def transfer_metrics(rows, roles):
    ys = []
    ss = []

    for x in rows:
        selected = set(x["selected_roles"])
        for j, r in enumerate(roles):
            ys.append(int(r in selected))
            ss.append(float(x["sensor_scores"][j]))

    y = np.asarray(ys)
    s = np.asarray(ss)

    micro = float(roc_auc_score(y, s))

    role_aucs = []
    for j, role in enumerate(roles):
        yy = np.asarray([
            int(role in set(x["selected_roles"]))
            for x in rows
        ])
        zz = np.asarray([
            x["sensor_scores"][j]
            for x in rows
        ])

        if len(np.unique(yy)) == 2:
            role_aucs.append(
                float(roc_auc_score(yy, zz))
            )

    row_aucs = []
    topk = []
    margins = []

    for x in rows:
        selected = set(x["selected_roles"])
        yy = np.asarray(
            [int(r in selected) for r in roles]
        )
        zz = np.asarray(x["sensor_scores"])

        if len(np.unique(yy)) == 2:
            row_aucs.append(
                float(roc_auc_score(yy, zz))
            )

        k = int(yy.sum())
        if k:
            order = np.argsort(-zz)[:k]
            topk.append(float(yy[order].sum() / k))

            margins.append(float(
                zz[yy == 1].mean()
                - zz[yy == 0].mean()
            ))

    return {
        "rows": len(rows),
        "micro_auc": micro,
        "macro_role_auc":
            float(np.mean(role_aucs)),
        "roles_with_auc": len(role_aucs),
        "mean_within_row_auc":
            float(np.mean(row_aucs)),
        "mean_topk_selected_recall":
            float(np.mean(topk)),
        "mean_selected_minus_absent_margin":
            float(np.mean(margins)),
    }


def fit_calibrator(train_rows, roles):
    R = len(roles)

    X = []
    y = []

    prevalence = {}

    for j, role in enumerate(roles):
        yy = [
            int(role in set(x["selected_roles"]))
            for x in train_rows
        ]
        prevalence[role] = float(np.mean(yy))

    for row in train_rows:
        selected = set(row["selected_roles"])

        for j, role in enumerate(roles):
            feat = np.zeros(1 + R, dtype=np.float64)
            feat[0] = float(row["sensor_scores"][j])
            feat[1 + j] = 1.0

            X.append(feat)
            y.append(int(role in selected))

    X = np.stack(X)
    y = np.asarray(y, dtype=int)

    cal = LogisticRegression(
        C=10.0,
        max_iter=2000,
        solver="lbfgs",
    )
    cal.fit(X, y)

    coef = cal.coef_[0]
    slope = float(coef[0])

    if slope <= 0:
        raise RuntimeError(
            f"bad calibration slope: {slope}"
        )

    return {
        "intercept": float(cal.intercept_[0]),
        "score_slope": slope,
        "role_bias": {
            role: float(coef[1 + j])
            for j, role in enumerate(roles)
        },
        "role_prevalence": prevalence,
    }


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -40, 40))
    )


def calibrated_probs(scores, roles, cal):
    scores = np.asarray(scores, dtype=np.float64)

    b = float(cal["intercept"])
    a = float(cal["score_slope"])

    rb = np.asarray(
        [cal["role_bias"][r] for r in roles],
        dtype=np.float64,
    )

    return sigmoid(
        b + a * scores + rb
    )


def sensor_log_lr(scores, roles, cal):
    """
    Convert calibrated P(Y=1|score,role) into
    log p(score|Y=1) / p(score|Y=0).

    odds posterior / odds prior.
    """
    q = np.clip(
        calibrated_probs(scores, roles, cal),
        1e-4,
        1 - 1e-4,
    )

    pi = np.asarray([
        np.clip(
            cal["role_prevalence"][r],
            1e-4,
            1 - 1e-4,
        )
        for r in roles
    ])

    logit_q = np.log(q) - np.log1p(-q)
    logit_pi = np.log(pi) - np.log1p(-pi)

    return np.clip(
        logit_q - logit_pi,
        -7.0,
        7.0,
    )


def visible_posterior_update(
    p_activation,
    sensor_scores,
    roles,
    calibration,
    prior_weights,
    beta,
):
    """
    p_activation: [M,R] = P(Y_r=1 | Z_m, q)
    sensor score gives continuous evidence about Y.

    Up to a particle-independent constant:

      p(score_r | Z,q)
      ∝ (1-a_r) + a_r * LR(score_r)

    generalized Bayes beta tempers independence error.
    """
    log_lr = sensor_log_lr(
        sensor_scores,
        roles,
        calibration,
    )

    lr = np.exp(log_lr)[None, :]

    a = np.clip(
        np.asarray(p_activation, dtype=np.float64),
        1e-5,
        1 - 1e-5,
    )

    mix = (
        (1.0 - a)
        + a * lr
    )

    loglik = (
        float(beta)
        * np.log(
            np.clip(mix, 1e-12, None)
        ).sum(axis=1)
    )

    w = np.asarray(
        prior_weights,
        dtype=np.float64,
    )

    logw = (
        np.log(np.clip(w, 1e-300, None))
        + loglik
    )

    logw -= np.max(logw)

    wp = np.exp(logw)
    wp /= wp.sum()

    return wp


def choose_train_rows(groups, per_state):
    out = []

    for key in sorted(groups):
        xs = sorted(
            groups[key],
            key=lambda x: x["candidate_index"],
        )

        if len(xs) <= per_state:
            out.extend(xs)
            continue

        idx = np.linspace(
            0,
            len(xs) - 1,
            per_state,
            dtype=int,
        )

        out.extend(xs[i] for i in idx)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )
    ap.add_argument(
        "--obs-dir",
        default="artifacts/observation_model_v1",
    )
    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/"
            "v13_8b_balanced_repair_2048_run1/"
            "best_adapter"
        ),
    )
    ap.add_argument(
        "--out",
        default="artifacts/visible_realized_gain_v3",
    )

    ap.add_argument(
        "--train-per-state",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--particles-per-k",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--obs-batch-size",
        type=int,
        default=4096,
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--round",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1389,
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    root = Path(args.root)
    obsdir = Path(args.obs_dir)
    out = Path(args.out)

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"refusing non-empty output dir: {out}"
        )

    out.mkdir(parents=True)

    # ----------------------------------------------------
    # 1. Source data / exact split
    # ----------------------------------------------------
    print("[1/6] loading examples + exact raw outputs", flush=True)

    examples, roles, _, _ = om.collect_examples(root)
    roles = list(roles)
    role_to_i = {
        r: i for i, r in enumerate(roles)
    }

    split = json.loads(
        (obsdir / "split_systems.json").read_text()
    )

    train_systems = set(map(str, split["train"]))
    val_systems = set(map(str, split["val"]))
    test_systems = set(map(str, split["test"]))

    assert not (
        train_systems & val_systems
        or train_systems & test_systems
        or val_systems & test_systems
    )

    raw = load_raw(root)

    train_groups = defaultdict(list)
    val_groups = defaultdict(list)

    for idx, x in enumerate(examples):
        if int(x["round"]) != args.round:
            continue

        sid = str(x["system_id"])

        if sid in test_systems:
            continue

        if sid not in train_systems and sid not in val_systems:
            continue

        key_raw = (
            str(x["parent"]),
            sid,
            int(x["round"]),
            int(x["candidate_index"]),
            str(x["query"]),
        )

        answer = raw.get(key_raw)
        if not answer:
            continue

        z = {
            **x,
            "_example_index": idx,
            "_answer": answer,
        }

        state_key = (
            str(x["parent"]),
            sid,
            int(x["round"]),
        )

        if sid in train_systems:
            train_groups[state_key].append(z)
        else:
            val_groups[state_key].append(z)

    train_rows = choose_train_rows(
        train_groups,
        args.train_per_state,
    )

    # VAL keeps the full candidate pool.
    val_rows = [
        x
        for key in sorted(val_groups)
        for x in sorted(
            val_groups[key],
            key=lambda z: z["candidate_index"],
        )
    ]

    print(
        f"round{args.round} train states={len(train_groups)} "
        f"train candidates={len(train_rows)}",
        flush=True,
    )
    print(
        f"round{args.round} val states={len(val_groups)} "
        f"val candidates={len(val_rows)}",
        flush=True,
    )
    print("TEST USED: 0", flush=True)

    if len(train_groups) < 100:
        raise RuntimeError("too few train states")
    if len(val_groups) < 20:
        raise RuntimeError("too few val states")

    # ----------------------------------------------------
    # 2. V13.8b visible sensor
    # ----------------------------------------------------
    print("\n[2/6] scoring frozen V13.8b", flush=True)

    role_desc = dict(tr.ROLE_DESCRIPTIONS)
    tok, sensor, dtype = load_sensor(args.adapter)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    for name, rows in (
        ("train", train_rows),
        ("val", val_rows),
    ):
        for i, x in enumerate(rows, 1):
            x["sensor_scores"] = sensor_score_row(
                sensor,
                tok,
                dtype,
                roles,
                role_desc,
                str(x["query"]),
                x["_answer"],
                args.max_length,
            ).tolist()

            if i % 100 == 0 or i == len(rows):
                print(
                    f"  sensor {name}: "
                    f"{i}/{len(rows)}",
                    flush=True,
                )

    transfer = {
        "train": transfer_metrics(
            train_rows, roles
        ),
        "val": transfer_metrics(
            val_rows, roles
        ),
    }

    print("\nV13.8b fixed-batch transfer:")
    print(json.dumps(transfer, indent=2))

    # Hard safety gate.
    if (
        transfer["val"]["micro_auc"] < 0.75
        or
        transfer["val"]["mean_within_row_auc"] < 0.80
    ):
        raise RuntimeError(
            "V13.8b transfer gate failed"
        )

    # ----------------------------------------------------
    # 3. TRAIN-only score calibration
    # ----------------------------------------------------
    print("\n[3/6] fitting TRAIN-only calibration", flush=True)

    calibration = fit_calibrator(
        train_rows,
        roles,
    )

    print(
        "score slope:",
        calibration["score_slope"],
        flush=True,
    )

    # Calibration validation diagnostics.
    val_y = []
    val_p = []

    for x in val_rows:
        p = calibrated_probs(
            x["sensor_scores"],
            roles,
            calibration,
        )
        selected = set(x["selected_roles"])

        for j, r in enumerate(roles):
            val_y.append(int(r in selected))
            val_p.append(float(p[j]))

    val_y = np.asarray(val_y)
    val_p = np.asarray(val_p)

    calibration["val_micro_auc"] = float(
        roc_auc_score(val_y, val_p)
    )
    calibration["val_brier"] = float(
        np.mean((val_p - val_y) ** 2)
    )

    print(
        "VAL calibrated AUC:",
        calibration["val_micro_auc"],
    )
    print(
        "VAL calibrated Brier:",
        calibration["val_brier"],
    )

    (out / "channel_calibration.json").write_text(
        json.dumps(
            calibration,
            indent=2,
            sort_keys=True,
        )
    )

    # Save sensor values before freeing 3B.
    sensor_dump = {}

    for name, rows in (
        ("train", train_rows),
        ("val", val_rows),
    ):
        sensor_dump[name] = {
            (
                str(x["parent"]),
                str(x["system_id"]),
                int(x["round"]),
                int(x["candidate_index"]),
            ): list(x["sensor_scores"])
            for x in rows
        }

    del sensor
    del tok
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 4. Observation model / particles
    # ----------------------------------------------------
    print("\n[4/6] loading activation observation model", flush=True)

    ckpt = torch.load(
        obsdir / "model.pt",
        map_location="cpu",
    )

    if list(ckpt["role_order"]) != roles:
        raise RuntimeError("role order mismatch")

    obs_model = om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],
        n_roles=len(roles),
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to("cuda:0")

    obs_model.load_state_dict(
        ckpt["state_dict"]
    )
    obs_model.eval()

    qemb = np.load(
        obsdir / "query_embeddings.npy"
    ).astype(np.float32)

    if len(qemb) != len(examples):
        raise RuntimeError(
            "query embedding/example mismatch"
        )

    max_round = max(
        int(x["round"])
        for x in examples
    )

    # ----------------------------------------------------
    # 5. Visible posterior gains
    # ----------------------------------------------------
    print("\n[5/6] computing truth-side realized gains", flush=True)

    def process_split(name, groups, allowed_rows):
        allowed = {
            (
                str(x["parent"]),
                str(x["system_id"]),
                int(x["round"]),
                int(x["candidate_index"]),
            )
            for x in allowed_rows
        }

        output_rows = []

        done = 0

        for state_key in sorted(groups):
            rows = [
                x for x in groups[state_key]
                if (
                    str(x["parent"]),
                    str(x["system_id"]),
                    int(x["round"]),
                    int(x["candidate_index"]),
                ) in allowed
            ]

            rows = sorted(
                rows,
                key=lambda z: z["candidate_index"],
            )

            if not rows:
                continue

            x0 = rows[0]

            belief = np.asarray(
                [
                    x0["role_probs"][r]
                    for r in roles
                ],
                dtype=np.float32,
            )

            # Exact input convention of observation_model_v1.
            kvec = np.asarray(
                [
                    x0["k_probs"].get(str(k), 0.0)
                    for k in [3, 4, 5, 6]
                ],
                dtype=np.float32,
            )

            rnd = int(x0["round"])
            rnorm = float(
                rnd / max(max_round, 1)
            )

            rng = np.random.default_rng(
                stable_seed(
                    *state_key,
                    args.seed,
                )
            )

            Z, W, Klabels, Kprior = (
                rg.stratified_particles(
                    x0["k_probs"],
                    x0["role_probs"],
                    roles,
                    args.particles_per_k,
                    rng,
                )
            )

            k_values = sorted(Kprior.keys())

            qg = np.stack([
                qemb[x["_example_index"]]
                for x in rows
            ]).astype(np.float32)

            P = rg.particle_predictions(
                obs_model,
                qg,
                belief,
                kvec,
                rnorm,
                Z,
                "cuda:0",
                args.obs_batch_size,
            )

            mi_k, mi_id, mi_z, exposure = (
                rg.predicted_mi(
                    P,
                    W,
                    Klabels,
                    Kprior,
                )
            )

            true_roles = set(
                map(str, x0["truth_roles"])
            )

            true_z = np.zeros(
                len(roles),
                dtype=np.float64,
            )

            for r in true_roles:
                true_z[role_to_i[r]] = 1.0

            true_k = len(true_roles)

            prior = rg.metrics_from_weights(
                W,
                Z,
                Klabels,
                true_z,
                true_k,
                k_values,
            )

            for j, x in enumerate(rows):
                skey = (
                    str(x["parent"]),
                    str(x["system_id"]),
                    int(x["round"]),
                    int(x["candidate_index"]),
                )

                scores = sensor_dump[name][skey]

                Wp = visible_posterior_update(
                    P[j],
                    scores,
                    roles,
                    calibration,
                    W,
                    args.beta,
                )

                post = rg.metrics_from_weights(
                    Wp,
                    Z,
                    Klabels,
                    true_z,
                    true_k,
                    k_values,
                )

                output_rows.append({
                    "state_id":
                        "::".join(map(str, state_key)),
                    "parent": str(x["parent"]),
                    "system_id": str(x["system_id"]),
                    "round": int(x["round"]),
                    "candidate_index":
                        int(x["candidate_index"]),
                    "example_index":
                        int(x["_example_index"]),
                    "query": str(x["query"]),

                    "role_probs_before": {
                        r: float(
                            x0["role_probs"][r]
                        )
                        for r in roles
                    },
                    "k_probs_before": {
                        str(k): float(v)
                        for k, v
                        in x0["k_probs"].items()
                    },

                    "sensor_scores_visible_only":
                        list(map(float, scores)),

                    "mi_k_baseline":
                        float(mi_k[j]),
                    "mi_identity_baseline":
                        float(mi_id[j]),
                    "mi_z_baseline":
                        float(mi_z[j]),
                    "predicted_exposure_baseline":
                        float(exposure[j]),

                    # TRAINING/EVALUATION LABELS ONLY:
                    "role_brier_gain_training_only":
                        float(
                            prior["role_brier"]
                            - post["role_brier"]
                        ),
                    "k_brier_gain_training_only":
                        float(
                            prior["k_brier"]
                            - post["k_brier"]
                        ),
                    "k_log_gain_training_only":
                        float(
                            post["k_logprob"]
                            - prior["k_logprob"]
                        ),
                    "k_abs_gain_training_only":
                        float(
                            prior["k_expected_abs_error"]
                            - post["k_expected_abs_error"]
                        ),

                    "prior_role_brier_training_only":
                        float(prior["role_brier"]),
                    "post_role_brier_training_only":
                        float(post["role_brier"]),
                    "prior_k_brier_training_only":
                        float(prior["k_brier"]),
                    "post_k_brier_training_only":
                        float(post["k_brier"]),

                    "truth_roles_training_only":
                        sorted(true_roles),
                    "true_k_training_only":
                        int(true_k),

                    "selected_roles_not_used_for_update":
                        list(x["selected_roles"]),

                    "visible_update_beta":
                        float(args.beta),
                    "sensor_encoding":
                        "self",
                    "sensor_batch_protocol":
                        "20_roles_same_response",
                })

            done += 1

            if done % 50 == 0:
                print(
                    f"  gains {name}: "
                    f"{done}/{len(groups)} states",
                    flush=True,
                )

        return output_rows

    train_out = process_split(
        "train",
        train_groups,
        train_rows,
    )

    val_out = process_split(
        "val",
        val_groups,
        val_rows,
    )

    write_jsonl(
        out / "train.jsonl",
        train_out,
    )
    write_jsonl(
        out / "val.jsonl",
        val_out,
    )

    # ----------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------
    def summarize_gain(rows):
        keys = [
            "role_brier_gain_training_only",
            "k_brier_gain_training_only",
            "k_log_gain_training_only",
            "k_abs_gain_training_only",
        ]

        ans = {
            "rows": len(rows),
            "states": len(set(
                x["state_id"]
                for x in rows
            )),
        }

        for k in keys:
            a = np.asarray([
                x[k] for x in rows
            ], dtype=float)

            ans[k] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "p10": float(np.quantile(a, .10)),
                "median": float(np.median(a)),
                "p90": float(np.quantile(a, .90)),
                "positive_fraction":
                    float(np.mean(a > 0)),
            }

        return ans

    summary = {
        "protocol":
            f"visible_realized_gain_v3_round{args.round}",
        "sensor":
            str(args.adapter),
        "sensor_encoding":
            "self",
        "sensor_batch_protocol":
            "20_roles_same_response",
        "posterior_update":
            "continuous_sensor_LR_generalized_bayes",
        "beta": args.beta,
        "particles_per_k":
            args.particles_per_k,
        "test_systems_used": 0,
        "transfer": transfer,
        "calibration": {
            "score_slope":
                calibration["score_slope"],
            "val_micro_auc":
                calibration["val_micro_auc"],
            "val_brier":
                calibration["val_brier"],
        },
        "train": summarize_gain(train_out),
        "val": summarize_gain(val_out),
    }

    (out / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print("\n[6/6] DONE")
    print(json.dumps(summary, indent=2))
    print("\nArtifacts:")
    print(out / "train.jsonl")
    print(out / "val.jsonl")
    print(out / "channel_calibration.json")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
