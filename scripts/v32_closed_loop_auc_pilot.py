#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.io import read_jsonl, write_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


POLICIES = (
    "random",
    "query_only",
    "history_belief_query",
)

SEEDS = (7, 17, 29)


def log(x=""):
    print(x, flush=True)


def normq(q):
    return " ".join(str(q).strip().lower().split())



def v31_frame(h):
    """
    Convert EvidenceFrame -> the dict interface expected by
    V3.1 turn_text(). This is representation-only:
    no truth/private information is added.
    """
    if isinstance(h, dict):
        return h

    return {
        "query": str(getattr(h, "query", "")),
        "answer": str(getattr(h, "answer", "")),
        "v135": dict(getattr(h, "v135", {}) or {}),
        "v134": dict(getattr(h, "v134", {}) or {}),
    }


def sint(*xs):
    s = "\x1f".join(map(str, xs))
    return int(
        hashlib.sha256(s.encode()).hexdigest()[:16],
        16,
    )


def run(cmd):
    log("\n[CMD] " + " ".join(map(str, cmd)))
    t0 = time.time()

    subprocess.run(
        cmd,
        check=True,
    )

    log(
        f"[DONE] {time.time() - t0:.1f}s"
    )


def load_v31(path):
    spec = importlib.util.spec_from_file_location(
        "v31_semantic",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot import {path}"
        )

    m = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = m
    spec.loader.exec_module(m)

    for k in (
        "AUCSelector",
        "load_or_build_embeddings",
        "turn_text",
        "query_text",
    ):
        if not hasattr(m, k):
            raise RuntimeError(
                f"V3.1 missing {k}"
            )

    return m


def build_bank(root, n=512):
    seen = {}

    for r in range(1, 7):
        p = (
            root
            / f"candidates_train_round_{r:02d}.jsonl"
        )

        if not p.exists():
            continue

        for x in read_jsonl(p):
            q = str(
                x.get("query", "")
            ).strip()

            if (
                x.get("valid")
                and len(q) >= 12
            ):
                seen.setdefault(
                    normq(q),
                    q,
                )

    qs = list(seen.values())

    qs.sort(
        key=lambda q: hashlib.sha256(
            (
                "v32|"
                + normq(q)
            ).encode()
        ).hexdigest()
    )

    if len(qs) < 64:
        raise RuntimeError(
            f"too few TRAIN-only queries: "
            f"{len(qs)}"
        )

    return qs[: min(n, len(qs))]


def load_ensemble(
    v31,
    ckdir,
    mode,
):
    models = []
    emb_dim = None

    for seed in SEEDS:
        p = (
            ckdir
            / f"selector_{mode}_seed{seed}.pt"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        ck = torch.load(
            p,
            map_location="cpu",
        )

        d = int(
            ck["emb_dim"]
        )

        if emb_dim is None:
            emb_dim = d

        if d != emb_dim:
            raise RuntimeError(
                "selector dims differ"
            )

        if ck["mode"] != mode:
            raise RuntimeError(
                f"mode mismatch in {p}"
            )

        model = v31.AUCSelector(
            d,
            mode,
        )

        model.load_state_dict(
            ck["state_dict"],
            strict=True,
        )

        model.eval()

        models.append(model)

    return models, emb_dim


def role_auc(
    state,
    true_roles,
):
    truth = set(
        map(str, true_roles)
    )

    y = np.asarray(
        [
            int(r in truth)
            for r in TRAIN_ROLES
        ],
        dtype=np.int64,
    )

    p = np.asarray(
        [
            float(
                state.role_probs[r]
            )
            for r in TRAIN_ROLES
        ],
        dtype=np.float64,
    )

    return float(
        roc_auc_score(y, p)
    )


def make_embedding_groups(
    states,
    queries,
):
    groups = []

    for i, state in enumerate(states):
        candidates = []

        if i == 0:
            candidates = [
                SimpleNamespace(
                    query=q
                )
                for q in queries
            ]

        groups.append(
            SimpleNamespace(
                sid=state.session_id,
                round_idx=state.round_index,
                role_probs=dict(
                    state.role_probs
                ),
                history=[v31_frame(h) for h in state.history],
                candidates=candidates,
                old_chosen=None,
                before_auc=0.0,
            )
        )

    return groups


def get_embeddings(
    v31,
    source_root,
    embed_dir,
    states,
    queries,
    device,
    round_idx,
):
    groups = make_embedding_groups(
        states,
        queries,
    )

    embed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache, emb_dim = (
        v31.load_or_build_embeddings(
            source_root,
            embed_dir,
            groups,
            device,
        )
    )

    needed = {
        v31.query_text(q)
        for q in queries
    }

    for state in states:
        for h in state.history:
            needed.add(
                v31.turn_text(v31_frame(h))
            )

    missing = [
        k
        for k in needed
        if k not in cache
    ]

    if missing:
        alt = (
            embed_dir.parent
            / (
                f"embed_round_"
                f"{round_idx:02d}_rebuild"
            )
        )

        if alt.exists():
            shutil.rmtree(alt)

        alt.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache, emb_dim = (
            v31.load_or_build_embeddings(
                source_root,
                alt,
                groups,
                device,
            )
        )

        missing = [
            k
            for k in needed
            if k not in cache
        ]

        if missing:
            raise RuntimeError(
                "embedding cache missing "
                f"{len(missing)} keys"
            )

    return cache, int(emb_dim)


@torch.no_grad()
def choose_query(
    v31,
    policy,
    state,
    bank,
    cache,
    models,
    emb_dim,
    round_idx,
):
    used = {
        normq(h.query)
        for h in state.history
    }

    available = [
        q
        for q in bank
        if normq(q) not in used
    ]

    if not available:
        raise RuntimeError(
            f"bank exhausted: "
            f"{state.session_id}"
        )

    if policy == "random":
        rng = random.Random(
            sint(
                "v32",
                policy,
                state.session_id,
                round_idx,
            )
        )

        q = available[
            rng.randrange(
                len(available)
            )
        ]

        return q, {}

    qvec = torch.stack(
        [
            cache[
                v31.query_text(q)
            ].float()
            for q in available
        ]
    )

    belief = torch.tensor(
        [
            float(
                state.role_probs[r]
            )
            for r in TRAIN_ROLES
        ],
        dtype=torch.float32,
    )

    if state.history:
        turns = torch.stack(
            [
                cache[
                    v31.turn_text(v31_frame(h))
                ].float()
                for h in state.history
            ]
        )
    else:
        turns = torch.zeros(
            0,
            emb_dim,
            dtype=torch.float32,
        )

    scores = torch.stack(
        [
            model.score(
                turns,
                belief,
                qvec,
            ).float()
            for model in models
        ]
    ).mean(0)

    pick = int(
        scores.argmax()
    )

    order = torch.argsort(
        scores,
        descending=True,
    )[:5]

    diag = {
        "score": float(
            scores[pick]
        ),
        "top5": [
            {
                "q": available[int(j)],
                "s": float(
                    scores[int(j)]
                ),
            }
            for j in order
        ],
    }

    return (
        available[pick],
        diag,
    )


def prepare_roots(
    outdir,
    source_root,
    reset,
):
    # V32_RESUME_BRANCH
    start_round = int(
        os.environ.get("V32_START_ROUND", "1")
    )

    if start_round > 1:
        if not outdir.exists():
            raise RuntimeError(
                f"cannot resume: {outdir} does not exist"
            )

        shared = outdir / "shared_cache"
        public_cache = (
            shared / "target_public_cache.jsonl"
        )

        truth_cache = (
            shared
            / "target_truth_cache_training_only.jsonl"
        )

        if not public_cache.exists():
            raise RuntimeError(
                f"missing resume cache: {public_cache}"
            )

        if not truth_cache.exists():
            raise RuntimeError(
                f"missing resume cache: {truth_cache}"
            )

        arms = {
            policy: outdir / policy
            for policy in POLICIES
        }

        previous_round = start_round - 1

        for policy, root in arms.items():
            state_file = (
                root
                / (
                    "states_dev_round_"
                    f"{previous_round:02d}.jsonl"
                )
            )

            if not state_file.exists():
                raise RuntimeError(
                    "cannot resume; missing "
                    f"{state_file}"
                )

        print(
            "[RESUME] preserving existing experiment; "
            f"starting at round {start_round}",
            flush=True,
        )

        return arms, public_cache

    if (
        outdir.exists()
        and any(outdir.iterdir())
    ):
        if not reset:
            raise RuntimeError(
                f"{outdir} non-empty"
            )

        shutil.rmtree(
            outdir
        )

    outdir.mkdir(
        parents=True
    )

    shared = (
        outdir
        / "shared_cache"
    )

    shared.mkdir()

    public_cache = (
        shared
        / "target_public_cache.jsonl"
    )

    truth_cache = (
        shared
        / "target_truth_cache_training_only.jsonl"
    )

    public_cache.touch()
    truth_cache.touch()

    arms = {}

    for policy in POLICIES:
        root = (
            outdir
            / policy
        )

        root.mkdir()

        shutil.copy2(
            source_root
            / "states_dev_round_00.jsonl",
            root
            / "states_dev_round_00.jsonl",
        )

        (
            root
            / "target_public_cache.jsonl"
        ).symlink_to(
            public_cache.resolve()
        )

        (
            root
            / "target_truth_cache_training_only.jsonl"
        ).symlink_to(
            truth_cache.resolve()
        )

        arms[policy] = root

    return (
        arms,
        public_cache,
    )


def signflip(
    effects,
):
    x = np.asarray(
        effects,
        dtype=float,
    )

    observed = float(
        x.mean()
    )

    null = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=len(x),
    ):
        null.append(
            float(
                (
                    x
                    * np.asarray(signs)
                ).mean()
            )
        )

    return float(
        sum(
            v >= observed - 1e-15
            for v in null
        )
        / len(null)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source-root",
        default=(
            "artifacts/"
            "v14_adaptive_role_v1"
        ),
    )

    ap.add_argument(
        "--manifest",
        default=(
            "artifacts/"
            "v14_adaptive_role_v1/"
            "manifest_privileged.json"
        ),
    )

    ap.add_argument(
        "--belief",
        default=(
            "artifacts/"
            "v14_adaptive_role_v1/"
            "frozen_belief.joblib"
        ),
    )

    ap.add_argument(
        "--v31-script",
        default=(
            "scripts/"
            "v3_1_mas_auc_semantic.py"
        ),
    )

    ap.add_argument(
        "--v31-ckpts",
        default=(
            "artifacts/"
            "eqs_v3_1_mas_auc"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=(
            "artifacts/"
            "eqs_v3_2_closed_loop/"
            "pilot"
        ),
    )

    ap.add_argument(
        "--rounds",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--bank-size",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--reset",
        action="store_true",
    )

    args = ap.parse_args()

    source_root = Path(
        args.source_root
    )

    manifest_path = Path(
        args.manifest
    )

    belief_path = Path(
        args.belief
    )

    ckpt_root = Path(
        args.v31_ckpts
    )

    outdir = Path(
        args.outdir
    )

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY not set"
        )

    v31 = load_v31(
        Path(args.v31_script)
    )

    query_models, qdim = (
        load_ensemble(
            v31,
            ckpt_root,
            "query_only",
        )
    )

    history_models, hdim = (
        load_ensemble(
            v31,
            ckpt_root,
            "history_belief_query",
        )
    )

    if qdim != hdim:
        raise RuntimeError(
            "selector dimensions differ"
        )

    query_bank = build_bank(
        source_root,
        args.bank_size,
    )

    arms, public_cache = (
        prepare_roots(
            outdir,
            source_root,
            args.reset,
        )
    )

    (
        outdir
        / "query_bank.json"
    ).write_text(
        json.dumps(
            {
                "source": (
                    "V14 TRAIN "
                    "valid candidates only"
                ),
                "queries": query_bank,
            },
            indent=2,
        )
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    truth = {
        str(x["session_id"]):
        list(map(str, x["roles"]))
        for x in manifest["systems"]
        if x.get("split") == "dev"
    }

    states0 = [
        PublicBeliefState.from_public_dict(
            x
        )
        for x in read_jsonl(
            source_root
            / "states_dev_round_00.jsonl"
        )
    ]

    sids = sorted(
        x.session_id
        for x in states0
    )

    if sorted(truth) != sids:
        raise RuntimeError(
            "DEV state0/manifest mismatch"
        )

    initial_auc = {
        state.session_id:
        role_auc(
            state,
            truth[
                state.session_id
            ],
        )
        for state in states0
    }

    log(
        "[INIT] "
        f"systems={len(sids)} "
        f"bank={len(query_bank)} "
        f"AUC0="
        f"{np.mean(list(initial_auc.values())):.6f} "
        f"dim={qdim}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    selection_log = []

    embed_dir = (
        outdir
        / "embedding_runtime"
    )

    for round_idx in range(
        int(os.environ.get("V32_START_ROUND", "1")),
        args.rounds + 1,
    ):
        log(
            "\n"
            + "=" * 80
        )

        log(
            f"ROUND "
            f"{round_idx}/"
            f"{args.rounds}"
        )

        log(
            "=" * 80
        )

        states_by_policy = {}
        all_states = []

        for policy in POLICIES:
            states = [
                PublicBeliefState.from_public_dict(
                    x
                )
                for x in read_jsonl(
                    arms[policy]
                    / (
                        f"states_dev_round_"
                        f"{round_idx-1:02d}.jsonl"
                    )
                )
            ]

            states_by_policy[
                policy
            ] = states

            all_states.extend(
                states
            )

        cache, runtime_dim = (
            get_embeddings(
                v31,
                source_root,
                embed_dir,
                all_states,
                query_bank,
                device,
                round_idx,
            )
        )

        if runtime_dim != qdim:
            raise RuntimeError(
                f"embedding dim "
                f"{runtime_dim} "
                f"!= selector dim {qdim}"
            )

        for policy in POLICIES:
            if policy == "random":
                models = None

            elif policy == "query_only":
                models = query_models

            else:
                models = history_models

            candidate_rows = []

            for state in (
                states_by_policy[
                    policy
                ]
            ):
                q, diag = (
                    choose_query(
                        v31,
                        policy,
                        state,
                        query_bank,
                        cache,
                        models,
                        qdim,
                        round_idx,
                    )
                )

                selection_log.append(
                    {
                        "policy": policy,
                        "round": round_idx,
                        "session_id":
                        state.session_id,
                        "query": q,
                        "diagnostics": diag,
                    }
                )

                candidate_rows.append(
                    {
                        "session_id":
                        state.session_id,
                        "split": "dev",
                        "round": round_idx,
                        "candidate_index": 0,
                        "group_id": (
                            f"v32-{policy}-"
                            f"{state.session_id}-"
                            f"r{round_idx:02d}"
                        ),
                        "phase":
                        "v32_closed_loop_auc",
                        "seed": (
                            sint(
                                policy,
                                state.session_id,
                                round_idx,
                            )
                            % (2**31 - 1)
                        ),
                        "query": q,
                        "valid": True,
                        "invalid_reason": None,
                    }
                )

            candidate_path = (
                arms[policy]
                / (
                    f"candidates_dev_round_"
                    f"{round_idx:02d}.jsonl"
                )
            )

            write_jsonl(
                candidate_path,
                candidate_rows,
            )

            run(
                [
                    sys.executable,
                    "scripts/"
                    "adaptive_v11_execute_round.py",
                    "--round",
                    str(round_idx),
                    "--split",
                    "dev",
                    "--root",
                    str(
                        arms[policy]
                    ),
                    "--manifest",
                    str(
                        manifest_path
                    ),
                    "--workers",
                    str(
                        args.workers
                    ),
                ]
            )

            run(
                [
                    sys.executable,
                    "scripts/"
                    "adaptive_v11_score_round.py",
                    "--round",
                    str(round_idx),
                    "--split",
                    "dev",
                    "--root",
                    str(
                        arms[policy]
                    ),
                ]
            )

            run(
                [
                    sys.executable,
                    "scripts/"
                    "kboot_advance_role_only.py",
                    "--round",
                    str(round_idx),
                    "--split",
                    "dev",
                    "--root",
                    str(
                        arms[policy]
                    ),
                    "--belief",
                    str(
                        belief_path
                    ),
                    "--manifest",
                    str(
                        manifest_path
                    ),
                ]
            )

            metrics = read_jsonl(
                arms[policy]
                / (
                    f"kboot_metrics_dev_"
                    f"round_{round_idx:02d}.jsonl"
                )
            )

            mean_auc = np.mean(
                [
                    float(
                        x["role_auc"]
                    )
                    for x in metrics
                ]
            )

            log(
                f"[AUC] "
                f"r={round_idx} "
                f"policy={policy} "
                f"mean={mean_auc:.6f}"
            )

        (
            outdir
            / "selection_log.jsonl"
        ).write_text(
            "".join(
                json.dumps(x)
                + "\n"
                for x in selection_log
            )
        )

    per_session = {
        sid: {
            "auc0":
            initial_auc[sid]
        }
        for sid in sids
    }

    trajectories = {}

    for policy in POLICIES:
        curve = [
            float(
                np.mean(
                    list(
                        initial_auc.values()
                    )
                )
            )
        ]

        by_round = {
            0:
            dict(initial_auc)
        }

        for round_idx in range(
            1,
            args.rounds + 1,
        ):
            rows = read_jsonl(
                arms[policy]
                / (
                    f"kboot_metrics_dev_"
                    f"round_{round_idx:02d}.jsonl"
                )
            )

            rr = {
                str(x["session_id"]):
                float(x["role_auc"])
                for x in rows
            }

            by_round[
                round_idx
            ] = rr

            curve.append(
                float(
                    np.mean(
                        [
                            rr[sid]
                            for sid in sids
                        ]
                    )
                )
            )

        trajectories[
            policy
        ] = {
            "mean_auc_curve_t0_to_tN":
            curve,
            "mean_final_auc":
            curve[-1],
            "mean_gain_at_final":
            curve[-1] - curve[0],
            "mean_auc_over_probes_1_to_N":
            float(
                np.mean(
                    curve[1:]
                )
            ),
        }

        for sid in sids:
            per_session[
                sid
            ][policy] = [
                by_round[r][sid]
                for r in range(
                    args.rounds + 1
                )
            ]

    comparisons = {}

    for A, B in (
        (
            "history_belief_query",
            "query_only",
        ),
        (
            "history_belief_query",
            "random",
        ),
        (
            "query_only",
            "random",
        ),
    ):
        final_effects = [
            (
                per_session[
                    sid
                ][A][-1]
                -
                per_session[
                    sid
                ][B][-1]
            )
            for sid in sids
        ]

        area_effects = [
            (
                float(
                    np.mean(
                        per_session[
                            sid
                        ][A][1:]
                    )
                )
                -
                float(
                    np.mean(
                        per_session[
                            sid
                        ][B][1:]
                    )
                )
            )
            for sid in sids
        ]

        comparisons[
            f"{A}_minus_{B}"
        ] = {
            "mean_final_auc_difference":
            float(
                np.mean(
                    final_effects
                )
            ),
            "sessions_A_gt_B_final":
            int(
                sum(
                    x > 0
                    for x in final_effects
                )
            ),
            "exact_signflip_p_one_sided_final":
            signflip(
                final_effects
            ),
            "mean_probe_area_auc_difference":
            float(
                np.mean(
                    area_effects
                )
            ),
            "session_final_effects":
            dict(
                zip(
                    sids,
                    map(
                        float,
                        final_effects,
                    ),
                )
            ),
        }

    real_calls = sum(
        bool(x.strip())
        for x in (
            public_cache
            .read_text()
            .splitlines()
        )
    )

    summary = {
        "experiment":
        "V3.2 closed-loop ROLE-AUC pilot",

        "status":
        (
            "pilot_on_existing_dev_systems_"
            "not_final_fresh_system_test"
        ),

        "n_systems":
        len(sids),

        "policies":
        list(POLICIES),

        "probes_per_system":
        args.rounds,

        "nominal_policy_system_probe_executions":
        (
            len(sids)
            * len(POLICIES)
            * args.rounds
        ),

        "unique_real_MAS_query_executions_via_shared_cache":
        real_calls,

        "query_bank":
        {
            "size":
            len(query_bank),

            "source":
            (
                "V14 TRAIN valid "
                "candidates only"
            ),

            "dev_queries_used_to_build_bank":
            False,
        },

        "selector":
        {
            "checkpoint_seeds_ensembled":
            list(SEEDS),

            "semantic_embedding_dim":
            qdim,
        },

        "trajectories":
        trajectories,

        "comparisons":
        comparisons,

        "per_session":
        per_session,

        "limitations":
        [
            (
                "Existing 8 DEV systems "
                "were already used in V3.1 "
                "offline evaluation; this "
                "is a prospective closed-loop "
                "pilot, not the final "
                "fresh-system claim."
            )
        ],
    }

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    log(
        "\n"
        "===== FINAL V3.2 "
        "CLOSED-LOOP PILOT RESULTS ====="
    )

    for policy in POLICIES:
        x = trajectories[
            policy
        ]

        log(
            f"{policy:24s} "
            f"curve="
            f"{[round(v,4) for v in x['mean_auc_curve_t0_to_tN']]} "
            f"final="
            f"{x['mean_final_auc']:.6f} "
            f"gain="
            f"{x['mean_gain_at_final']:+.6f}"
        )

    for name, x in (
        comparisons.items()
    ):
        log(
            f"{name}: "
            f"final_diff="
            f"{x['mean_final_auc_difference']:+.6f} "
            f"wins="
            f"{x['sessions_A_gt_B_final']}/"
            f"{len(sids)} "
            f"p="
            f"{x['exact_signflip_p_one_sided_final']:.6f} "
            f"area_diff="
            f"{x['mean_probe_area_auc_difference']:+.6f}"
        )

    log(
        f"MAS unique real executions="
        f"{real_calls}"
    )

    log(
        "SAVED: "
        f"{outdir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
