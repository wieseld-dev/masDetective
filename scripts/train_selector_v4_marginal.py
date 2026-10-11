#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_v11_active.io import (
    read_jsonl,
)

from adaptive_v11_active.schema import (
    PublicBeliefState,
)

from adaptive_v11_curriculum_v2.selector_v4 import (
    FEATURE_DIM_V4,
    build_model,
    feature_vector_v4,
)


ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2"
)

OUT = Path(
    "artifacts/adaptive_v11_curriculum_v2/"
    "selector_v4_marginal"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 20260922
EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 4096
LR = 3e-4
WEIGHT_DECAY = 1e-4


def stable_bucket(text: str) -> int:
    d = hashlib.blake2b(
        str(text).encode("utf-8"),
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(
            d,
            "big",
        )
        % 100
    )


def split_for_system(
    system_id: str,
) -> str:
    b = stable_bucket(
        system_id
    )

    if b < 70:
        return "train"

    if b < 85:
        return "val"

    return "test"


def entropy01(p: float) -> float:
    p = min(
        1.0 - 1e-8,
        max(
            1e-8,
            float(p),
        ),
    )

    return (
        -p * math.log(p)
        - (1.0 - p)
        * math.log(
            1.0 - p
        )
    ) / math.log(2.0)


def is_usable_response(r):
    return (
        bool(
            r.get(
                "valid",
                True,
            )
        )
        and r.get(
            "execution_status",
            "ok",
        ) == "ok"
    )


def parse_stage(p: Path) -> int:
    return int(
        p.name.split("_")[-1]
    )


def parse_round(p: Path) -> int:
    return int(
        p.stem.split("_")[-1]
    )


def load_groups():
    groups = []

    for sdir in sorted(
        ROOT.glob("stage_*")
    ):
        try:
            stage = parse_stage(
                sdir
            )
        except Exception:
            continue

        for cdir in sorted(
            sdir.glob("cycle_*")
        ):
            cycle = cdir.name

            # This follows the REAL historical continuation path.
            # It is used only to build privileged training labels.
            exposed_before = defaultdict(
                set
            )

            response_files = sorted(
                cdir.glob(
                    "responses_train_round_*.jsonl"
                ),
                key=parse_round,
            )

            for rp in response_files:
                rnd = parse_round(
                    rp
                )

                sp = (
                    cdir
                    / (
                        "states_train_round_"
                        f"{rnd - 1:02d}.jsonl"
                    )
                )

                if not sp.exists():
                    continue

                states = {
                    str(x["session_id"]):
                        x
                    for x
                    in read_jsonl(sp)
                }

                raw_groups = defaultdict(
                    list
                )

                for r in read_jsonl(
                    rp
                ):
                    raw_groups[
                        str(
                            r[
                                "session_id"
                            ]
                        )
                    ].append(r)

                # First build every label using the SAME
                # exposed-before set.
                pending_updates = {}

                for sid, raw_rows in (
                    raw_groups.items()
                ):
                    if sid not in states:
                        continue

                    usable = [
                        r
                        for r
                        in raw_rows
                        if is_usable_response(
                            r
                        )
                    ]

                    if len(usable) < 2:
                        continue

                    truth_variants = {
                        tuple(
                            sorted(
                                r.get(
                                    "truth_roles_training_only",
                                    [],
                                )
                            )
                        )
                        for r
                        in usable
                    }

                    if (
                        len(
                            truth_variants
                        )
                        != 1
                    ):
                        raise RuntimeError(
                            f"inconsistent truth: "
                            f"{sdir.name} "
                            f"{cycle} "
                            f"r{rnd} "
                            f"{sid}"
                        )

                    truth = set(
                        next(
                            iter(
                                truth_variants
                            )
                        )
                    )

                    if not truth:
                        continue

                    state = (
                        PublicBeliefState
                        .from_public_dict(
                            states[sid]
                        )
                    )

                    already = set(
                        exposed_before[
                            sid
                        ]
                    )

                    X = []
                    utility = []
                    new_count = []
                    new_roles_all = []
                    selected_roles_all = []
                    candidate_indices = []
                    was_chosen = []

                    usable = sorted(
                        usable,
                        key=lambda r:
                            int(
                                r[
                                    "candidate_index"
                                ]
                            ),
                    )

                    for r in usable:
                        idx = int(
                            r[
                                "candidate_index"
                            ]
                        )

                        q = str(
                            r.get(
                                "query",
                                "",
                            )
                        )

                        selected = (
                            set(
                                r.get(
                                    "selected_roles_training_only",
                                    [],
                                )
                            )
                            & truth
                        )

                        new_roles = (
                            selected
                            - already
                        )

                        # Primary objective:
                        # number of NEW roles.
                        #
                        # Secondary tie-break:
                        # discovering roles that the current
                        # public belief is still uncertain about.
                        uncertainty_bonus = sum(
                            entropy01(
                                float(
                                    state.role_probs.get(
                                        role,
                                        0.5,
                                    )
                                )
                            )
                            for role
                            in new_roles
                        )

                        y = (
                            float(
                                len(
                                    new_roles
                                )
                            )
                            + 0.15
                            * float(
                                uncertainty_bonus
                            )
                        )

                        X.append(
                            feature_vector_v4(
                                state,
                                q,
                                idx,
                            )
                        )

                        utility.append(
                            y
                        )

                        new_count.append(
                            len(
                                new_roles
                            )
                        )

                        new_roles_all.append(
                            set(
                                new_roles
                            )
                        )

                        selected_roles_all.append(
                            set(
                                selected
                            )
                        )

                        candidate_indices.append(
                            idx
                        )

                        was_chosen.append(
                            bool(
                                r.get(
                                    "chosen_for_continuation",
                                    False,
                                )
                            )
                        )

                    system_id = str(
                        usable[0].get(
                            "system_id",
                            sid,
                        )
                    )

                    groups.append({
                        "stage":
                            stage,

                        "cycle":
                            cycle,

                        "round":
                            rnd,

                        "session_id":
                            sid,

                        "system_id":
                            system_id,

                        "split":
                            split_for_system(
                                system_id
                            ),

                        "state":
                            state,

                        "truth":
                            set(
                                truth
                            ),

                        "already":
                            set(
                                already
                            ),

                        "X":
                            np.stack(
                                X
                            ).astype(
                                np.float32
                            ),

                        "utility":
                            np.asarray(
                                utility,
                                dtype=np.float32,
                            ),

                        "new_count":
                            np.asarray(
                                new_count,
                                dtype=np.float32,
                            ),

                        "new_roles":
                            new_roles_all,

                        "selected_roles":
                            selected_roles_all,

                        "candidate_indices":
                            candidate_indices,

                        "was_chosen":
                            was_chosen,
                    })

                    chosen = [
                        r
                        for r
                        in raw_rows
                        if (
                            bool(
                                r.get(
                                    "chosen_for_continuation",
                                    False,
                                )
                            )
                            and is_usable_response(
                                r
                            )
                        )
                    ]

                    if chosen:
                        pending_updates[
                            sid
                        ] = (
                            set(
                                chosen[0].get(
                                    "selected_roles_training_only",
                                    [],
                                )
                            )
                            & truth
                        )

                # Advance actual historical path only AFTER
                # all candidate labels for this round were built.
                for sid, roles in (
                    pending_updates.items()
                ):
                    exposed_before[
                        sid
                    ].update(
                        roles
                    )

    return groups


def fit_scaler(groups):
    arr = np.concatenate([
        g["X"]
        for g in groups
        if g["split"] == "train"
    ])

    mu = arr.mean(
        axis=0
    ).astype(
        np.float32
    )

    sd = arr.std(
        axis=0
    ).astype(
        np.float32
    )

    sd[
        sd < 1e-6
    ] = 1.0

    return (
        mu,
        sd,
    )


def make_pair_data(
    groups,
    mu,
    sd,
):
    all_x = []

    left = []
    right = []
    sign = []
    weight = []

    offset = 0

    training_groups = 0

    for g in groups:
        if (
            g["split"]
            != "train"
        ):
            continue

        X = (
            g["X"]
            - mu
        ) / sd

        y = g["utility"]
        c = g["new_count"]

        all_x.append(
            X.astype(
                np.float32
            )
        )

        training_groups += 1

        n = len(y)

        for i in range(n):
            for j in range(
                i + 1,
                n,
            ):
                util_gap = float(
                    y[i] - y[j]
                )

                count_gap = float(
                    c[i] - c[j]
                )

                # Completely equivalent candidates
                # provide no ranking information.
                if (
                    abs(
                        util_gap
                    )
                    < 0.02
                ):
                    continue

                left.append(
                    offset + i
                )

                right.append(
                    offset + j
                )

                sign.append(
                    1.0
                    if util_gap > 0
                    else -1.0
                )

                # A difference in the number of NEW roles
                # matters much more than an entropy tie-break.
                if (
                    abs(
                        count_gap
                    )
                    > 0
                ):
                    w = (
                        1.0
                        + 0.75
                        * min(
                            3.0,
                            abs(
                                count_gap
                            ),
                        )
                    )
                else:
                    w = 0.25

                weight.append(
                    w
                )

        offset += n

    if not all_x:
        raise RuntimeError(
            "no training features"
        )

    X = torch.from_numpy(
        np.concatenate(
            all_x,
            axis=0,
        )
    )

    left = torch.tensor(
        left,
        dtype=torch.long,
    )

    right = torch.tensor(
        right,
        dtype=torch.long,
    )

    sign = torch.tensor(
        sign,
        dtype=torch.float32,
    )

    weight = torch.tensor(
        weight,
        dtype=torch.float32,
    )

    if not len(sign):
        raise RuntimeError(
            "no pairwise examples"
        )

    return (
        X,
        left,
        right,
        sign,
        weight,
        training_groups,
    )


@torch.no_grad()
def group_scores(
    model,
    g,
    mu,
    sd,
):
    X = (
        g["X"]
        - mu
    ) / sd

    t = torch.from_numpy(
        X.astype(
            np.float32
        )
    )

    return (
        model(t)
        .squeeze(-1)
        .cpu()
        .numpy()
    )


@torch.no_grad()
def evaluate(
    model,
    groups,
    mu,
    sd,
    split,
    stage=None,
):
    rows = []

    for g in groups:
        if (
            g["split"]
            != split
        ):
            continue

        if (
            stage is not None
            and g["stage"]
            != stage
        ):
            continue

        scores = group_scores(
            model,
            g,
            mu,
            sd,
        )

        chosen = int(
            np.argmax(
                scores
            )
        )

        counts = (
            g[
                "new_count"
            ].astype(
                float
            )
        )

        utility = (
            g[
                "utility"
            ].astype(
                float
            )
        )

        oracle_count = float(
            counts.max()
        )

        random_count = float(
            counts.mean()
        )

        chosen_count = float(
            counts[
                chosen
            ]
        )

        oracle_utility = float(
            utility.max()
        )

        chosen_utility = float(
            utility[
                chosen
            ]
        )

        rows.append({
            "chosen_count":
                chosen_count,

            "random_count":
                random_count,

            "oracle_count":
                oracle_count,

            "count_regret":
                (
                    oracle_count
                    - chosen_count
                ),

            "top1_count":
                float(
                    abs(
                        chosen_count
                        - oracle_count
                    )
                    < 1e-9
                ),

            "chosen_utility":
                chosen_utility,

            "oracle_utility":
                oracle_utility,
        })

    if not rows:
        return {}

    out = {
        "groups":
            len(
                rows
            ),
    }

    for key in rows[0]:
        out[key] = float(
            np.mean([
                x[key]
                for x in rows
            ])
        )

    denom = (
        out[
            "oracle_count"
        ]
        - out[
            "random_count"
        ]
    )

    out[
        "signal_captured"
    ] = (
        (
            out[
                "chosen_count"
            ]
            - out[
                "random_count"
            ]
        )
        / denom
        if denom > 1e-9
        else float(
            "nan"
        )
    )

    return out


@torch.no_grad()
def logged_stage10_replay(
    model,
    groups,
    mu,
    sd,
    split="test",
):
    trajectories = defaultdict(
        list
    )

    for g in groups:
        if (
            g["stage"] == 10
            and g["split"]
            == split
        ):
            key = (
                g["cycle"],
                g["system_id"],
            )

            trajectories[
                key
            ].append(
                g
            )

    records = []

    for (
        cycle,
        system_id,
    ), gg in trajectories.items():
        gg = sorted(
            gg,
            key=lambda x:
                x["round"],
        )

        truth = set(
            gg[0][
                "truth"
            ]
        )

        current_seen = set()
        v4_seen = set()
        oracle_seen = set()

        for g in gg:
            scores = group_scores(
                model,
                g,
                mu,
                sd,
            )

            v4_idx = int(
                np.argmax(
                    scores
                )
            )

            v4_seen.update(
                g[
                    "selected_roles"
                ][
                    v4_idx
                ]
                & truth
            )

            chosen_indices = [
                i
                for i, x
                in enumerate(
                    g[
                        "was_chosen"
                    ]
                )
                if x
            ]

            if chosen_indices:
                current_seen.update(
                    g[
                        "selected_roles"
                    ][
                        chosen_indices[0]
                    ]
                    & truth
                )

            # Privileged same-pool greedy upper bound.
            gains = []

            for i, roles in enumerate(
                g[
                    "selected_roles"
                ]
            ):
                new = (
                    roles
                    & truth
                    - oracle_seen
                )

                gains.append((
                    len(
                        new
                    ),
                    len(
                        roles
                        & truth
                    ),
                    -i,
                    i,
                ))

            oracle_idx = max(
                gains
            )[-1]

            oracle_seen.update(
                g[
                    "selected_roles"
                ][
                    oracle_idx
                ]
                & truth
            )

        K = len(
            truth
        )

        records.append({
            "cycle":
                cycle,

            "system_id":
                system_id,

            "K":
                K,

            "current_coverage":
                len(
                    current_seen
                )
                / K,

            "v4_coverage":
                len(
                    v4_seen
                )
                / K,

            "oracle_coverage":
                len(
                    oracle_seen
                )
                / K,

            "current_exact":
                float(
                    current_seen
                    == truth
                ),

            "v4_exact":
                float(
                    v4_seen
                    == truth
                ),

            "oracle_exact":
                float(
                    oracle_seen
                    == truth
                ),
        })

    def summarize(rows):
        if not rows:
            return {}

        return {
            "systems":
                len(
                    rows
                ),

            "current_coverage":
                float(
                    np.mean([
                        x[
                            "current_coverage"
                        ]
                        for x in rows
                    ])
                ),

            "v4_coverage":
                float(
                    np.mean([
                        x[
                            "v4_coverage"
                        ]
                        for x in rows
                    ])
                ),

            "oracle_coverage":
                float(
                    np.mean([
                        x[
                            "oracle_coverage"
                        ]
                        for x in rows
                    ])
                ),

            "current_exact":
                float(
                    np.mean([
                        x[
                            "current_exact"
                        ]
                        for x in rows
                    ])
                ),

            "v4_exact":
                float(
                    np.mean([
                        x[
                            "v4_exact"
                        ]
                        for x in rows
                    ])
                ),

            "oracle_exact":
                float(
                    np.mean([
                        x[
                            "oracle_exact"
                        ]
                        for x in rows
                    ])
                ),
        }

    return {
        "overall":
            summarize(
                records
            ),

        "cycle_01":
            summarize([
                x
                for x in records
                if x[
                    "cycle"
                ]
                == "cycle_01"
            ]),

        "cycle_02":
            summarize([
                x
                for x in records
                if x[
                    "cycle"
                ]
                == "cycle_02"
            ]),

        "note":
            (
                "Logged-pool replay: V4 choices use candidate pools "
                "generated under the historical trajectory. "
                "No privileged field enters V4 features."
            ),
    }


def pct(x):
    return (
        "nan"
        if not math.isfinite(
            float(
                x
            )
        )
        else f"{100*x:.2f}%"
    )


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

    torch.set_num_threads(
        max(
            1,
            min(
                4,
                torch.get_num_threads(),
            ),
        )
    )

    print(
        "[load] reconstructing "
        "marginal-gain groups...",
        flush=True,
    )

    groups = load_groups()

    split_counts = {
        s: sum(
            g[
                "split"
            ] == s
            for g
            in groups
        )
        for s
        in (
            "train",
            "val",
            "test",
        )
    }

    stage10_counts = {
        s: sum(
            (
                g[
                    "split"
                ] == s
                and g[
                    "stage"
                ] == 10
            )
            for g
            in groups
        )
        for s
        in (
            "train",
            "val",
            "test",
        )
    }

    print(
        "[data] "
        f"groups={len(groups)} "
        f"splits={split_counts} "
        f"stage10={stage10_counts}",
        flush=True,
    )

    mu, sd = fit_scaler(
        groups
    )

    (
        X,
        pair_left,
        pair_right,
        pair_sign,
        pair_weight,
        training_groups,
    ) = make_pair_data(
        groups,
        mu,
        sd,
    )

    n_pairs = len(
        pair_sign
    )

    print(
        "[pairs] "
        f"candidates={len(X)} "
        f"pairs={n_pairs} "
        f"train_groups={training_groups}",
        flush=True,
    )

    model = build_model()

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )

    best_metric = -1e18
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        perm = torch.randperm(
            n_pairs
        )

        losses = []

        for start in range(
            0,
            n_pairs,
            BATCH_SIZE,
        ):
            idx = perm[
                start:
                start
                + BATCH_SIZE
            ]

            li = pair_left[
                idx
            ]

            ri = pair_right[
                idx
            ]

            sg = pair_sign[
                idx
            ]

            wt = pair_weight[
                idx
            ]

            sa = (
                model(
                    X[li]
                )
                .squeeze(-1)
            )

            sb = (
                model(
                    X[ri]
                )
                .squeeze(-1)
            )

            pair_loss = (
                F.softplus(
                    -sg
                    * (
                        sa
                        - sb
                    )
                )
                * wt
            ).mean()

            optimizer.zero_grad(
                set_to_none=True
            )

            pair_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    pair_loss.item()
                )
            )

        model.eval()

        val = evaluate(
            model,
            groups,
            mu,
            sd,
            "val",
        )

        metric = (
            float(
                val.get(
                    "chosen_count",
                    -1e9,
                )
            )
            - 0.05
            * float(
                val.get(
                    "count_regret",
                    1e9,
                )
            )
        )

        improved = (
            metric
            > best_metric
            + 1e-8
        )

        if improved:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict()
                .items()
            }

        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:02d}] "
            f"loss={np.mean(losses):.5f} "
            f"val_chosen={val.get('chosen_count', float('nan')):.4f} "
            f"val_random={val.get('random_count', float('nan')):.4f} "
            f"val_oracle={val.get('oracle_count', float('nan')):.4f} "
            f"signal={val.get('signal_captured', float('nan')):.4f} "
            f"top1={pct(val.get('top1_count', float('nan')))} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            bad_epochs
            >= PATIENCE
        ):
            print(
                "[early-stop]",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "no best model"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    train_metrics = evaluate(
        model,
        groups,
        mu,
        sd,
        "train",
    )

    val_metrics = evaluate(
        model,
        groups,
        mu,
        sd,
        "val",
    )

    test_metrics = evaluate(
        model,
        groups,
        mu,
        sd,
        "test",
    )

    stage10_test = evaluate(
        model,
        groups,
        mu,
        sd,
        "test",
        stage=10,
    )

    replay = (
        logged_stage10_replay(
            model,
            groups,
            mu,
            sd,
            split="test",
        )
    )

    checkpoint = (
        OUT
        / "selector_v4.pt"
    )

    torch.save({
        "state_dict":
            model.state_dict(),

        "mu":
            mu,

        "sd":
            sd,

        "feature_dim":
            FEATURE_DIM_V4,

        "training_groups":
            training_groups,

        "pairwise_examples":
            n_pairs,

        "best_epoch":
            best_epoch,

        "objective":
            (
                "privileged-training-only "
                "marginal unseen-role "
                "pairwise ranking"
            ),

        "inference_inputs":
            (
                "public belief state + "
                "candidate query + "
                "candidate index"
            ),

        "internal_test_used":
            False,
    }, checkpoint)

    report = {
        "checkpoint":
            str(
                checkpoint
            ),

        "feature_dim":
            FEATURE_DIM_V4,

        "best_epoch":
            best_epoch,

        "split_counts":
            split_counts,

        "stage10_counts":
            stage10_counts,

        "pairwise_examples":
            n_pairs,

        "train":
            train_metrics,

        "val":
            val_metrics,

        "test":
            test_metrics,

        "stage10_test_groups":
            stage10_test,

        "stage10_logged_replay":
            replay,

        "internal_test_used":
            False,
    }

    (
        OUT
        / "REPORT.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SELECTOR V4 — FINAL REPORT"
    )

    print(
        "=" * 100
    )

    print(
        "checkpoint:",
        checkpoint,
    )

    print(
        "best epoch:",
        best_epoch,
    )

    print(
        "pairs:",
        n_pairs,
    )

    for name, m in [
        (
            "VAL",
            val_metrics,
        ),
        (
            "TEST",
            test_metrics,
        ),
        (
            "STAGE10 TEST",
            stage10_test,
        ),
    ]:
        print()
        print(
            f"[{name}]"
        )

        print(
            "groups:",
            m.get(
                "groups"
            ),
        )

        print(
            "chosen new roles:",
            m.get(
                "chosen_count"
            ),
        )

        print(
            "random new roles:",
            m.get(
                "random_count"
            ),
        )

        print(
            "oracle new roles:",
            m.get(
                "oracle_count"
            ),
        )

        print(
            "signal captured:",
            pct(
                m.get(
                    "signal_captured",
                    float(
                        "nan"
                    ),
                )
            ),
        )

        print(
            "top1:",
            pct(
                m.get(
                    "top1_count",
                    float(
                        "nan"
                    ),
                )
            ),
        )

    print()
    print(
        "=" * 100
    )

    print(
        "STAGE10 LOGGED 8-ROUND REPLAY"
    )

    print(
        "=" * 100
    )

    for name in [
        "overall",
        "cycle_01",
        "cycle_02",
    ]:
        x = replay[
            name
        ]

        if not x:
            continue

        print()
        print(
            name.upper()
        )

        print(
            "systems:",
            x[
                "systems"
            ],
        )

        print(
            "CURRENT coverage:",
            pct(
                x[
                    "current_coverage"
                ]
            ),
        )

        print(
            "V4 coverage     :",
            pct(
                x[
                    "v4_coverage"
                ]
            ),
        )

        print(
            "ORACLE coverage :",
            pct(
                x[
                    "oracle_coverage"
                ]
            ),
        )

        print(
            "CURRENT exact:",
            pct(
                x[
                    "current_exact"
                ]
            ),
        )

        print(
            "V4 exact     :",
            pct(
                x[
                    "v4_exact"
                ]
            ),
        )

        print(
            "ORACLE exact :",
            pct(
                x[
                    "oracle_exact"
                ]
            ),
        )

    print()
    print(
        "PRIVILEGED DATA USED FOR LABELS ONLY: YES"
    )

    print(
        "PRIVILEGED DATA USED BY SELECTOR FEATURES: NO"
    )

    print(
        "INTERNAL TEST USED: NO"
    )

    print(
        "V4 TRAINING: DONE"
    )


if __name__ == "__main__":
    main()
