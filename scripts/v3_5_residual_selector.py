#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


SEEDS = (7, 17, 29)


def load_v31(path: Path):
    spec = importlib.util.spec_from_file_location("v31", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def no_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def residual_scores(model, item):
    """
    Round 1 / no observations:
        ordinary score, because no system-specific evidence exists yet.

    After >=1 observation:
        score(real state, q) - score(neutral state of SAME history length, q)

    This cancels query-only/global and history-length priors.
    """
    if item.turns.shape[0] == 0:
        return model.score(
            item.turns,
            item.belief,
            item.queries,
        )

    neutral_turns = torch.zeros_like(
        item.turns
    )

    neutral_belief = torch.full_like(
        item.belief,
        0.5,
    )

    actual = model.score(
        item.turns,
        item.belief,
        item.queries,
    )

    neutral = model.score(
        neutral_turns,
        neutral_belief,
        item.queries,
    )

    return actual - neutral


def ranking_score(v31, model, items):
    vals = []

    model.eval()

    with torch.no_grad():
        for x in items:
            s = residual_scores(
                model,
                x,
            )

            pick = int(
                s.argmax()
            )

            u = x.utility

            rand = float(
                u.mean()
            )

            oracle = float(
                u.max()
            )

            learned = float(
                u[pick]
            )

            denom = oracle - rand

            if abs(denom) > 1e-8:
                vals.append(
                    (learned - rand)
                    / denom
                )

    return float(
        np.mean(vals)
    ) if vals else 0.0


def fit_once(
    v31,
    items,
    emb_dim,
    seed,
    epochs,
):
    v31.seed_all(seed)

    model = v31.AUCSelector(
        emb_dim,
        "history_belief_query",
    )

    no_dropout(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=4e-4,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        order = list(
            range(len(items))
        )

        random.shuffle(order)

        model.train()

        total = 0.0

        for i in order:
            x = items[i]

            scores = residual_scores(
                model,
                x,
            )

            loss = v31.group_loss(
                scores,
                x.utility,
            )

            opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += float(
                loss.detach()
            )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"[REFIT] epoch={epoch:03d}/{epochs:03d} "
                f"loss={total/max(1,len(items)):.6f}",
                flush=True,
            )

    return model.eval()


def train_seed(
    v31,
    all_items,
    emb_dim,
    seed,
):
    sids = sorted(
        {
            x.group.sid
            for x in all_items
        }
    )

    rng = random.Random(
        seed + 991
    )

    rng.shuffle(sids)

    n_val = max(
        3,
        round(
            0.17 * len(sids)
        ),
    )

    val_sids = set(
        sids[:n_val]
    )

    fit_items = [
        x
        for x in all_items
        if x.group.sid
        not in val_sids
    ]

    val_items = [
        x
        for x in all_items
        if x.group.sid
        in val_sids
    ]

    v31.seed_all(seed)

    model = v31.AUCSelector(
        emb_dim,
        "history_belief_query",
    )

    no_dropout(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=4e-4,
    )

    best_epoch = 1
    best_score = -1e9
    bad = 0

    for epoch in range(1, 81):
        order = list(
            range(len(fit_items))
        )

        random.shuffle(order)

        model.train()

        total = 0.0

        for i in order:
            x = fit_items[i]

            scores = residual_scores(
                model,
                x,
            )

            loss = v31.group_loss(
                scores,
                x.utility,
            )

            opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += float(
                loss.detach()
            )

        sc = ranking_score(
            v31,
            model,
            val_items,
        )

        if sc > best_score + 1e-5:
            best_score = sc
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"[TRAIN] seed={seed} "
                f"epoch={epoch:03d} "
                f"loss={total/max(1,len(fit_items)):.6f} "
                f"val_gap_closed={sc:.4f} "
                f"best={best_score:.4f}@{best_epoch}",
                flush=True,
            )

        if (
            epoch >= 15
            and bad >= 15
        ):
            break

    print(
        f"[SELECT] seed={seed} "
        f"epoch={best_epoch} "
        f"val_gap_closed={best_score:.4f}",
        flush=True,
    )

    # Same philosophy as V3.1:
    # choose epoch internally, then refit on every TRAIN system.
    final = fit_once(
        v31,
        all_items,
        emb_dim,
        seed + 10000,
        best_epoch,
    )

    return (
        final,
        best_epoch,
        best_score,
    )


@torch.no_grad()
def evaluate(model, items):
    rows = []

    for x in items:
        scores = residual_scores(
            model,
            x,
        )

        pick = int(
            scores.argmax()
        )

        u = x.utility

        oracle_idx = int(
            u.argmax()
        )

        rows.append(
            {
                "before":
                float(
                    x.group.before_auc
                ),

                "learned":
                float(
                    u[pick]
                ),

                "random":
                float(
                    u.mean()
                ),

                "oracle":
                float(
                    u[oracle_idx]
                ),

                "positive":
                float(
                    u[pick] > 0
                ),

                "exact":
                float(
                    pick == oracle_idx
                ),
            }
        )

    def M(k):
        return float(
            np.mean(
                [
                    x[k]
                    for x in rows
                ]
            )
        )

    rand = M("random")
    learned = M("learned")
    oracle = M("oracle")
    before = M("before")

    return {
        "groups":
        len(rows),

        "mean_auc_before":
        before,

        "mean_delta_auc_random":
        rand,

        "mean_delta_auc_residual":
        learned,

        "mean_delta_auc_oracle":
        oracle,

        "mean_auc_after_residual":
        before + learned,

        "oracle_gap_closed":
        (
            learned - rand
        ) / (
            oracle - rand
            + 1e-12
        ),

        "positive_delta_rate":
        M("positive"),

        "oracle_exact_hit_rate":
        M("exact"),
    }


def read_jsonl(path):
    out = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(
                    json.loads(line)
                )

    return out


def build_audit_groups(
    states,
    bank,
):
    groups = []

    for i, st in enumerate(states):
        cands = []

        if i == 0:
            cands = [
                SimpleNamespace(
                    query=q
                )
                for q in bank
            ]

        groups.append(
            SimpleNamespace(
                sid=str(
                    st["session_id"]
                ),
                history=list(
                    st["history"]
                ),
                role_probs=dict(
                    st["role_probs"]
                ),
                candidates=cands,
            )
        )

    return groups


def turns_of(
    v31,
    state,
    cache,
    emb_dim,
):
    hist = state["history"]

    if not hist:
        return torch.zeros(
            0,
            emb_dim,
            dtype=torch.float32,
        )

    return torch.stack(
        [
            cache[
                v31.turn_text(h)
            ].float()
            for h in hist
        ]
    )


def belief_of(
    v31,
    state,
):
    return torch.tensor(
        [
            float(
                state[
                    "role_probs"
                ][r]
            )
            for r
            in v31.TRAIN_ROLES
        ],
        dtype=torch.float32,
    )


@torch.no_grad()
def ensemble_residual(
    models,
    turns,
    belief,
    qvec,
):
    vals = []

    for model in models:
        if turns.shape[0] == 0:
            vals.append(
                model.score(
                    turns,
                    belief,
                    qvec,
                )
            )
        else:
            zturns = torch.zeros_like(
                turns
            )

            zbelief = torch.full_like(
                belief,
                0.5,
            )

            vals.append(
                model.score(
                    turns,
                    belief,
                    qvec,
                )
                -
                model.score(
                    zturns,
                    zbelief,
                    qvec,
                )
            )

    return torch.stack(
        vals
    ).mean(0)


def rank_corr(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    ra = np.empty(
        len(a),
        dtype=float,
    )

    rb = np.empty(
        len(b),
        dtype=float,
    )

    ra[
        np.argsort(-a)
    ] = np.arange(len(a))

    rb[
        np.argsort(-b)
    ] = np.arange(len(b))

    if (
        np.std(ra) < 1e-12
        or np.std(rb) < 1e-12
    ):
        return 1.0

    return float(
        np.corrcoef(
            ra,
            rb,
        )[0, 1]
    )


def state_sensitivity_audit(
    v31,
    models,
    emb_dim,
    root,
    outdir,
):
    pilot = Path(
        "artifacts/"
        "eqs_v3_2_closed_loop/"
        "pilot"
    )

    bank = json.loads(
        (
            pilot
            / "query_bank.json"
        ).read_text()
    )["queries"]

    all_states = []
    states_by_round = {}

    for r in range(1, 7):
        p = (
            pilot
            / "history_belief_query"
            / (
                f"states_dev_round_"
                f"{r-1:02d}.jsonl"
            )
        )

        xs = read_jsonl(p)

        xs.sort(
            key=lambda x:
            str(x["session_id"])
        )

        states_by_round[r] = xs

        all_states.extend(xs)

    audit_embed = (
        outdir
        / "audit_embeddings"
    )

    audit_embed.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    groups = build_audit_groups(
        all_states,
        bank,
    )

    cache, d = (
        v31.load_or_build_embeddings(
            root,
            audit_embed,
            groups,
            device,
        )
    )

    if int(d) != emb_dim:
        raise RuntimeError(
            f"audit dim {d} != {emb_dim}"
        )

    qvec = torch.stack(
        [
            cache[
                v31.query_text(q)
            ].float()
            for q in bank
        ]
    )

    round_rows = []

    for r in range(1, 7):
        states = states_by_round[r]

        true_scores = []

        same_zero_hist = []
        same_uniform = []
        same_length_only = []
        same_shuffle_hist = []
        same_shuffle_belief = []

        top_queries = []

        for i, state in enumerate(states):
            turns = turns_of(
                v31,
                state,
                cache,
                emb_dim,
            )

            belief = belief_of(
                v31,
                state,
            )

            score = ensemble_residual(
                models,
                turns,
                belief,
                qvec,
            )

            true_scores.append(
                score.numpy()
            )

            true_pick = int(
                score.argmax()
            )

            top_queries.append(
                bank[true_pick]
            )

            # ZERO history content,
            # same number of turns.
            zturns = torch.zeros_like(
                turns
            )

            s_zero = ensemble_residual(
                models,
                zturns,
                belief,
                qvec,
            )

            same_zero_hist.append(
                int(
                    s_zero.argmax()
                )
                == true_pick
            )

            # Uniform belief.
            ub = torch.full_like(
                belief,
                0.5,
            )

            s_uniform = ensemble_residual(
                models,
                turns,
                ub,
                qvec,
            )

            same_uniform.append(
                int(
                    s_uniform.argmax()
                )
                == true_pick
            )

            # LENGTH ONLY.
            s_length = ensemble_residual(
                models,
                zturns,
                ub,
                qvec,
            )

            same_length_only.append(
                int(
                    s_length.argmax()
                )
                == true_pick
            )

            # Swap state components
            # with another system.
            j = (
                i + 1
            ) % len(states)

            other_turns = turns_of(
                v31,
                states[j],
                cache,
                emb_dim,
            )

            other_belief = belief_of(
                v31,
                states[j],
            )

            s_sh = ensemble_residual(
                models,
                other_turns,
                belief,
                qvec,
            )

            same_shuffle_hist.append(
                int(
                    s_sh.argmax()
                )
                == true_pick
            )

            s_sb = ensemble_residual(
                models,
                turns,
                other_belief,
                qvec,
            )

            same_shuffle_belief.append(
                int(
                    s_sb.argmax()
                )
                == true_pick
            )

        pair_corr = []

        for a, b in itertools.combinations(
            true_scores,
            2,
        ):
            pair_corr.append(
                rank_corr(a, b)
            )

        rr = {
            "round": r,

            "unique_top1":
            len(
                set(top_queries)
            ),

            "same_top1_zero_history":
            float(
                np.mean(
                    same_zero_hist
                )
            ),

            "same_top1_uniform_belief":
            float(
                np.mean(
                    same_uniform
                )
            ),

            "same_top1_length_only":
            float(
                np.mean(
                    same_length_only
                )
            ),

            "same_top1_history_swap":
            float(
                np.mean(
                    same_shuffle_hist
                )
            ),

            "same_top1_belief_swap":
            float(
                np.mean(
                    same_shuffle_belief
                )
            ),

            "cross_system_rank_corr":
            float(
                np.mean(
                    pair_corr
                )
            ) if pair_corr else 1.0,

            "example_top1":
            top_queries[0],
        }

        round_rows.append(
            rr
        )

        print(
            "\n"
            f"[AUDIT r={r}] "
            f"uniqueQ={rr['unique_top1']} "
            f"zeroHist={rr['same_top1_zero_history']:.3f} "
            f"uniformBelief={rr['same_top1_uniform_belief']:.3f} "
            f"lengthOnly={rr['same_top1_length_only']:.3f} "
            f"historySwap={rr['same_top1_history_swap']:.3f} "
            f"beliefSwap={rr['same_top1_belief_swap']:.3f} "
            f"rankCorr={rr['cross_system_rank_corr']:.4f}",
            flush=True,
        )

    # Round 1 cannot be adaptive:
    # there is no previous observation.
    post = [
        x
        for x in round_rows
        if x["round"] >= 2
    ]

    aggregate = {
        "mean_unique_top1_rounds_2_6":
        float(
            np.mean(
                [
                    x["unique_top1"]
                    for x in post
                ]
            )
        ),

        "mean_same_top1_length_only_rounds_2_6":
        float(
            np.mean(
                [
                    x["same_top1_length_only"]
                    for x in post
                ]
            )
        ),

        "mean_same_top1_history_swap_rounds_2_6":
        float(
            np.mean(
                [
                    x["same_top1_history_swap"]
                    for x in post
                ]
            )
        ),

        "mean_same_top1_belief_swap_rounds_2_6":
        float(
            np.mean(
                [
                    x["same_top1_belief_swap"]
                    for x in post
                ]
            )
        ),

        "mean_cross_system_rank_corr_rounds_2_6":
        float(
            np.mean(
                [
                    x["cross_system_rank_corr"]
                    for x in post
                ]
            )
        ),
    }

    result = {
        "rounds":
        round_rows,

        "aggregate_post_observation":
        aggregate,
    }

    (
        outdir
        / "state_sensitivity.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=" * 96)
    print("V3.5 STATE SENSITIVITY — ROUNDS 2–6")
    print("=" * 96)

    for k, v in aggregate.items():
        print(
            f"{k}: {v:.6f}"
        )

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/v14_adaptive_role_v1",
    )

    ap.add_argument(
        "--manifest",
        default=(
            "artifacts/v14_adaptive_role_v1/"
            "manifest_privileged.json"
        ),
    )

    ap.add_argument(
        "--belief",
        default=(
            "artifacts/v14_adaptive_role_v1/"
            "frozen_belief.joblib"
        ),
    )

    ap.add_argument(
        "--v31",
        default="scripts/v3_1_mas_auc_semantic.py",
    )

    ap.add_argument(
        "--v31-out",
        default="artifacts/eqs_v3_1_mas_auc",
    )

    ap.add_argument(
        "--outdir",
        default="artifacts/eqs_v3_5_residual",
    )

    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    v31 = load_v31(
        Path(args.v31)
    )

    rounds = [
        1, 2, 3, 4, 5, 6
    ]

    print("=" * 96)
    print("V3.5 — RESIDUAL STATE-CONDITIONED SELECTOR")
    print("=" * 96)
    print("NO OPENAI")
    print("NO NEW MAS EXECUTIONS")
    print(
        "SCORING AFTER OBSERVATION: "
        "F(real_state,q) - F(neutral_same_length,q)"
    )
    print()

    train_groups = v31.load_groups(
        root,
        Path(args.manifest),
        Path(args.belief),
        "train",
        rounds,
    )

    dev_groups = v31.load_groups(
        root,
        Path(args.manifest),
        Path(args.belief),
        "dev",
        rounds,
    )

    print(
        f"GROUPS train={len(train_groups)} "
        f"dev={len(dev_groups)}"
    )

    # Reuse exact V3.1 semantic embedding cache.
    cache, emb_dim = (
        v31.load_or_build_embeddings(
            root,
            Path(args.v31_out),
            train_groups + dev_groups,
            (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
        )
    )

    print(
        f"EMBED_DIM={emb_dim}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_items = (
        v31.build_selector_items(
            train_groups,
            cache,
            emb_dim,
        )
    )

    dev_items = (
        v31.build_selector_items(
            dev_groups,
            cache,
            emb_dim,
        )
    )

    models = []

    seed_results = []

    for seed in SEEDS:
        print()
        print("=" * 96)
        print(f"SEED {seed}")
        print("=" * 96)

        model, epoch, val = (
            train_seed(
                v31,
                train_items,
                emb_dim,
                seed,
            )
        )

        result = evaluate(
            model,
            dev_items,
        )

        print(
            "[DEV]",
            json.dumps(
                result,
                sort_keys=True,
            ),
            flush=True,
        )

        ck = {
            "state_dict":
            model.state_dict(),

            "emb_dim":
            emb_dim,

            "mode":
            "history_belief_query",

            "scoring":
            (
                "raw_if_no_history_else_"
                "real_minus_neutral_same_length"
            ),

            "neutral_belief":
            0.5,

            "seed":
            seed,

            "selected_epoch":
            epoch,

            "internal_val_gap_closed":
            val,

            "roles":
            list(
                v31.TRAIN_ROLES
            ),
        }

        torch.save(
            ck,
            outdir
            / (
                "selector_residual_"
                f"seed{seed}.pt"
            ),
        )

        models.append(
            model
        )

        seed_results.append(
            {
                "seed": seed,
                "selected_epoch":
                epoch,
                "internal_val_gap_closed":
                val,
                "dev":
                result,
            }
        )

    aggregate = {}

    metric_keys = [
        "mean_delta_auc_residual",
        "mean_auc_after_residual",
        "oracle_gap_closed",
        "positive_delta_rate",
        "oracle_exact_hit_rate",
    ]

    for k in metric_keys:
        vals = [
            x["dev"][k]
            for x in seed_results
        ]

        aggregate[k] = {
            "mean":
            float(
                np.mean(vals)
            ),

            "std":
            float(
                np.std(vals)
            ),

            "values":
            vals,
        }

    print()
    print("=" * 96)
    print("V3.5 ONE-STEP DEV RESULTS")
    print("=" * 96)

    for k, x in aggregate.items():
        print(
            f"{k}: "
            f"{x['mean']:.6f} "
            f"+/- {x['std']:.6f}"
        )

    sensitivity = (
        state_sensitivity_audit(
            v31,
            models,
            emb_dim,
            root,
            outdir,
        )
    )

    final = {
        "experiment":
        "V3.5 residual state-conditioned selector",

        "principle":
        (
            "After the first observation, "
            "subtract the same-query score under "
            "a neutral state with identical history length."
        ),

        "reason":
        (
            "Pure query priors and round/history-length "
            "priors cancel algebraically."
        ),

        "seed_results":
        seed_results,

        "aggregate_dev":
        aggregate,

        "state_sensitivity":
        sensitivity,

        "no_new_mas_calls":
        True,
    }

    (
        outdir
        / "results.json"
    ).write_text(
        json.dumps(
            final,
            indent=2,
        )
    )

    print()
    print(
        "SAVED:",
        outdir / "results.json",
    )


if __name__ == "__main__":
    main()
