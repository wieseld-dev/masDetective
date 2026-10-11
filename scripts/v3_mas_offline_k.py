from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import zlib

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


KS = [3, 4, 5, 6]
K2I = {k: i for i, k in enumerate(KS)}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# =============================================================================
# BASIC UTILS
# =============================================================================

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: Path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# =============================================================================
# TEXT REPRESENTATION
#
# Deliberately self-contained:
# no downloads
# no transformers
# no new packages
#
# We first test whether epistemic query selection is present in the REAL MAS
# data at all.
# =============================================================================

def hist_text(
    history,
    max_turns=8,
    max_chars_per_answer=5000,
):
    parts = []

    for h in history[-max_turns:]:
        q = str(h.get("query", ""))
        a = str(h.get("answer", ""))[:max_chars_per_answer]

        parts.append(
            "Q " + q + "\nA " + a
        )

    return "\n".join(parts)


def append_text(history, q, a):
    base = hist_text(history)

    extra = (
        "Q "
        + str(q)
        + "\nA "
        + str(a)[:5000]
    )

    return (base + "\n" + extra).strip()


def stable_bucket(s: str, dim: int):
    return zlib.crc32(
        s.encode("utf-8")
    ) % dim


def hash_text(
    text: str,
    dim=2048,
    max_tokens=900,
):
    toks = TOKEN_RE.findall(
        text.lower()
    )[:max_tokens]

    x = torch.zeros(
        dim + 8,
        dtype=torch.float32,
    )

    # unigrams + bigrams
    for i, token in enumerate(toks):

        x[
            stable_bucket(
                "u:" + token,
                dim,
            )
        ] += 1.0

        if i:
            x[
                stable_bucket(
                    "b:"
                    + toks[i - 1]
                    + "_"
                    + token,
                    dim,
                )
            ] += 0.7

    # cheap structural signals
    nchar = len(text)
    nword = len(toks)

    x[dim + 0] = math.log1p(nchar) / 10.0
    x[dim + 1] = math.log1p(nword) / 8.0
    x[dim + 2] = text.count("\n") / 50.0
    x[dim + 3] = text.count("?") / 20.0
    x[dim + 4] = text.count(":") / 30.0
    x[dim + 5] = text.count("```") / 10.0

    x[dim + 6] = (
        sum(c.isdigit() for c in text)
        / max(1, nchar)
    )

    x[dim + 7] = (
        sum(c.isupper() for c in text)
        / max(1, nchar)
    )

    z = x[:dim]

    z.copy_(
        torch.log1p(z)
    )

    norm = torch.linalg.vector_norm(z)

    if float(norm) > 0:
        z.div_(norm)

    return x


# =============================================================================
# REAL MAS DATA
# =============================================================================

@dataclass
class Candidate:
    idx: int
    query: str
    answer: str
    valid: bool


@dataclass
class Group:
    sid: str
    split: str
    round: int
    true_k: int
    history: list
    candidates: list
    old_chosen: int | None


def load_groups(
    root: Path,
    manifest_path: Path,
    split: str,
    rounds,
):
    manifest = json.loads(
        manifest_path.read_text()
    )

    truth = {
        str(x["session_id"]): int(x["k"])
        for x in manifest["systems"]
        if x.get("split") == split
    }

    groups = []

    for r in rounds:

        state_path = (
            root
            / f"states_{split}_round_{r-1:02d}.jsonl"
        )

        candidate_path = (
            root
            / f"candidates_{split}_round_{r:02d}.jsonl"
        )

        response_path = (
            root
            / f"responses_{split}_round_{r:02d}.jsonl"
        )

        metrics_path = (
            root
            / f"metrics_{split}_round_{r:02d}.jsonl"
        )

        if not (
            state_path.exists()
            and candidate_path.exists()
            and response_path.exists()
        ):
            continue

        states = {
            str(x["session_id"]): x
            for x in read_jsonl(state_path)
        }

        candidates = read_jsonl(
            candidate_path
        )

        responses = read_jsonl(
            response_path
        )

        response_by = {
            (
                str(x["session_id"]),
                int(x["candidate_index"]),
            ): x
            for x in responses
        }

        old_chosen = {}

        if metrics_path.exists():
            old_chosen = {
                str(x["session_id"]):
                    int(x["chosen_candidate"])
                for x in read_jsonl(
                    metrics_path
                )
                if "chosen_candidate" in x
            }

        by_session = defaultdict(list)

        for c in candidates:

            sid = str(
                c["session_id"]
            )

            idx = int(
                c["candidate_index"]
            )

            response = response_by.get(
                (sid, idx),
                {},
            )

            ok = (
                bool(c.get("valid"))
                and response.get(
                    "execution_status"
                ) == "ok"
                and bool(
                    str(
                        response.get(
                            "final_output",
                            "",
                        )
                    ).strip()
                )
            )

            if not ok:
                continue

            by_session[sid].append(
                Candidate(
                    idx=idx,
                    query=str(
                        c.get(
                            "query",
                            "",
                        )
                    ),
                    answer=str(
                        response.get(
                            "final_output",
                            "",
                        )
                    ),
                    valid=True,
                )
            )

        for sid, cs in by_session.items():

            if sid not in states:
                continue

            if sid not in truth:
                continue

            true_k = truth[sid]

            if true_k not in K2I:
                continue

            if len(cs) < 2:
                continue

            groups.append(
                Group(
                    sid=sid,
                    split=split,
                    round=r,
                    true_k=true_k,
                    history=states[sid].get(
                        "history",
                        [],
                    ),
                    candidates=sorted(
                        cs,
                        key=lambda x: x.idx,
                    ),
                    old_chosen=old_chosen.get(
                        sid
                    ),
                )
            )

    return groups


# =============================================================================
# K SENSOR
#
# It estimates:
#
#     P(K | history)
#
# and after a candidate response:
#
#     P(K | history, q, response)
#
# IMPORTANT:
# Train utilities use out-of-fold sensor predictions by SESSION.
# So the utility teacher never trains on the same MAS session it labels.
# =============================================================================

class KSensor(nn.Module):

    def __init__(self, din):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                din,
                256,
            ),
            nn.GELU(),

            nn.LayerNorm(
                256
            ),

            nn.Dropout(
                0.08
            ),

            nn.Linear(
                256,
                96,
            ),

            nn.GELU(),

            nn.Linear(
                96,
                4,
            ),
        )

    def forward(self, x):
        return self.net(x)


def sensor_examples(
    groups,
    dim,
):
    X = []
    y = []
    sids = []

    for g in groups:

        base = hist_text(
            g.history
        )

        X.append(
            hash_text(
                base,
                dim,
            )
        )

        y.append(
            K2I[g.true_k]
        )

        sids.append(
            g.sid
        )

        for c in g.candidates:

            full = append_text(
                g.history,
                c.query,
                c.answer,
            )

            X.append(
                hash_text(
                    full,
                    dim,
                )
            )

            y.append(
                K2I[g.true_k]
            )

            sids.append(
                g.sid
            )

    return (
        torch.stack(X),
        torch.tensor(
            y,
            dtype=torch.long,
        ),
        sids,
    )


def train_sensor(
    X,
    y,
    epochs=35,
    seed=0,
):
    seed_all(seed)

    model = KSensor(
        X.shape[1]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=2e-4,
    )

    n = len(y)
    batch_size = min(
        192,
        n,
    )

    for _ in range(epochs):

        perm = torch.randperm(
            n
        )

        model.train()

        for start in range(
            0,
            n,
            batch_size,
        ):
            ii = perm[
                start:start
                + batch_size
            ]

            logits = model(
                X[ii]
            )

            loss = F.cross_entropy(
                logits,
                y[ii],
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

    return model.eval()


@torch.no_grad()
def probs(
    model,
    texts,
    dim,
):
    X = torch.stack(
        [
            hash_text(
                text,
                dim,
            )
            for text in texts
        ]
    )

    return F.softmax(
        model(X),
        dim=-1,
    )


# =============================================================================
# REALIZED K UTILITY
#
# Utility =
#
# log P(true K | history + candidate response)
# -
# log P(true K | history)
#
# This is NOT the old V14 role reward.
# =============================================================================

def build_train_utilities(
    train_groups,
    dim=2048,
    folds=4,
):
    train_sids = sorted(
        set(
            g.sid
            for g in train_groups
        )
    )

    fold_of = {
        sid: i % folds
        for i, sid
        in enumerate(train_sids)
    }

    utility_map = {}

    before_acc = []

    for fold in range(folds):

        fit_groups = [
            g
            for g in train_groups
            if fold_of[g.sid]
            != fold
        ]

        held_groups = [
            g
            for g in train_groups
            if fold_of[g.sid]
            == fold
        ]

        X, y, _ = sensor_examples(
            fit_groups,
            dim,
        )

        sensor = train_sensor(
            X,
            y,
            seed=100 + fold,
        )

        for g in held_groups:

            texts = [
                hist_text(
                    g.history
                )
            ]

            texts += [
                append_text(
                    g.history,
                    c.query,
                    c.answer,
                )
                for c
                in g.candidates
            ]

            P = probs(
                sensor,
                texts,
                dim,
            )

            yi = K2I[
                g.true_k
            ]

            before = float(
                P[
                    0,
                    yi
                ].clamp_min(
                    1e-8
                ).log()
            )

            before_acc.append(
                int(
                    int(
                        P[0].argmax()
                    )
                    == yi
                )
            )

            utilities = [
                float(
                    P[
                        j + 1,
                        yi
                    ].clamp_min(
                        1e-8
                    ).log()
                )
                - before
                for j
                in range(
                    len(
                        g.candidates
                    )
                )
            ]

            utility_map[
                (
                    g.sid,
                    g.round,
                )
            ] = (
                P[0].tolist(),
                utilities,
                P[1:].tolist(),
            )

    # Final sensor:
    # train-only data, used only to score DEV.
    X, y, _ = sensor_examples(
        train_groups,
        dim,
    )

    all_train_sensor = train_sensor(
        X,
        y,
        epochs=45,
        seed=777,
    )

    acc = (
        sum(before_acc)
        / max(
            1,
            len(before_acc),
        )
    )

    return (
        utility_map,
        all_train_sensor,
        acc,
    )


def build_dev_utilities(
    groups,
    sensor,
    dim,
):
    out = {}

    before_acc = []
    before_nll = []

    for g in groups:

        texts = [
            hist_text(
                g.history
            )
        ]

        texts += [
            append_text(
                g.history,
                c.query,
                c.answer,
            )
            for c
            in g.candidates
        ]

        P = probs(
            sensor,
            texts,
            dim,
        )

        yi = K2I[
            g.true_k
        ]

        p0 = float(
            P[
                0,
                yi
            ].clamp_min(
                1e-8
            )
        )

        before_acc.append(
            int(
                int(
                    P[0].argmax()
                )
                == yi
            )
        )

        before_nll.append(
            -math.log(
                p0
            )
        )

        before = math.log(
            p0
        )

        utilities = [
            math.log(
                float(
                    P[
                        j + 1,
                        yi
                    ].clamp_min(
                        1e-8
                    )
                )
            )
            - before
            for j
            in range(
                len(
                    g.candidates
                )
            )
        ]

        out[
            (
                g.sid,
                g.round,
            )
        ] = (
            P[0].tolist(),
            utilities,
            P[1:].tolist(),
        )

    metrics = {
        "before_k_accuracy":
            sum(before_acc)
            / max(
                1,
                len(before_acc),
            ),

        "before_k_nll":
            sum(before_nll)
            / max(
                1,
                len(before_nll),
            ),
    }

    return out, metrics


# =============================================================================
# EQS SELECTOR
#
# History ---> h
# Query   ---> q
#
# score(history, q)
#
# Query-only is trained separately as an adaptivity ablation.
# =============================================================================

class Selector(nn.Module):

    def __init__(
        self,
        din,
        query_only=False,
    ):
        super().__init__()

        self.query_only = (
            query_only
        )

        self.henc = nn.Sequential(
            nn.Linear(
                din + 4,
                256,
            ),
            nn.GELU(),

            nn.LayerNorm(
                256
            ),

            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
        )

        self.qenc = nn.Sequential(
            nn.Linear(
                din,
                256,
            ),
            nn.GELU(),

            nn.LayerNorm(
                256
            ),

            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(
                512,
                160,
            ),

            nn.GELU(),

            nn.Dropout(
                0.08
            ),

            nn.Linear(
                160,
                1,
            ),
        )

    def score(
        self,
        history_x,
        belief_probs,
        query_x,
    ):
        q = self.qenc(
            query_x
        )

        if self.query_only:

            h = torch.zeros_like(
                q
            )

        else:

            h = self.henc(
                torch.cat(
                    [
                        history_x,
                        belief_probs,
                    ],
                    dim=-1,
                )
            )

        if h.ndim == 1:
            hh = (
                h.unsqueeze(0)
                .expand(
                    q.shape[0],
                    -1,
                )
            )
        else:
            hh = h

        features = torch.cat(
            [
                hh,
                q,
                hh * q,
                (
                    hh - q
                ).abs(),
            ],
            dim=-1,
        )

        return self.head(
            features
        ).squeeze(-1)


def make_selector_groups(
    groups,
    utility_map,
    dim,
):
    data = []

    for g in groups:

        key = (
            g.sid,
            g.round,
        )

        if key not in utility_map:
            continue

        item = utility_map[
            key
        ]

        before_probs = item[0]
        utilities = item[1]
        after_probs = item[2]

        if len(
            utilities
        ) != len(
            g.candidates
        ):
            continue

        history_x = hash_text(
            hist_text(
                g.history
            ),
            dim,
        )

        query_x = torch.stack(
            [
                hash_text(
                    c.query,
                    dim,
                )
                for c
                in g.candidates
            ]
        )

        data.append(
            (
                g,
                history_x,

                torch.tensor(
                    before_probs,
                    dtype=torch.float32,
                ),

                query_x,

                torch.tensor(
                    utilities,
                    dtype=torch.float32,
                ),

                after_probs,
            )
        )

    return data


def train_selector(
    data,
    din,
    seed=0,
    query_only=False,
    epochs=60,
):
    seed_all(seed)

    model = Selector(
        din,
        query_only=query_only,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=3e-4,
    )

    for epoch in range(
        epochs
    ):

        order = list(
            range(
                len(data)
            )
        )

        random.shuffle(
            order
        )

        model.train()

        epoch_loss = 0.0

        for i in order:

            (
                g,
                history_x,
                belief_probs,
                query_x,
                utility,
                _after,
            ) = data[i]

            scores = model.score(
                history_x,
                belief_probs,
                query_x,
            )

            utility_z = (
                utility
                - utility.mean()
            ) / (
                utility.std()
                + 1e-5
            )

            target = F.softmax(
                utility_z
                / 0.55,
                dim=0,
            )

            ranking_loss = -(
                target
                * F.log_softmax(
                    scores,
                    dim=0,
                )
            ).sum()

            score_z = (
                scores
                - scores.mean()
            ) / (
                scores.std()
                + 1e-5
            )

            regression_loss = (
                F.smooth_l1_loss(
                    score_z,
                    utility_z,
                )
            )

            loss = (
                ranking_loss
                + 0.20
                * regression_loss
            )

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            epoch_loss += (
                float(
                    loss.detach()
                )
            )

        if (
            epoch == 0
            or (
                epoch + 1
            ) % 10 == 0
        ):
            print(
                f"selector query_only={query_only} "
                f"epoch={epoch+1:03d} "
                f"loss={epoch_loss/max(1,len(data)):.5f}",
                flush=True,
            )

    return model.eval()


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def eval_selector(
    model,
    data,
):
    rows = []

    for (
        g,
        history_x,
        belief_probs,
        query_x,
        utility,
        after,
    ) in data:

        scores = model.score(
            history_x,
            belief_probs,
            query_x,
        )

        pick = int(
            scores.argmax()
        )

        oracle = int(
            utility.argmax()
        )

        vals = utility.tolist()

        random_utility = (
            sum(vals)
            / len(vals)
        )

        old_utility = None

        if g.old_chosen is not None:

            for j, c in enumerate(
                g.candidates
            ):
                if (
                    c.idx
                    == g.old_chosen
                ):
                    old_utility = vals[j]
                    break

        order = sorted(
            range(
                len(vals)
            ),
            key=lambda j:
                vals[j],
            reverse=True,
        )

        top20 = set(
            order[
                :max(
                    1,
                    math.ceil(
                        0.2
                        * len(order)
                    ),
                )
            ]
        )

        yi = K2I[
            g.true_k
        ]

        after_predictions = [
            max(
                range(4),
                key=lambda k:
                    after[j][k],
            )
            for j
            in range(
                len(after)
            )
        ]

        learned_acc = int(
            after_predictions[pick]
            == yi
        )

        random_acc = (
            sum(
                int(
                    pred == yi
                )
                for pred
                in after_predictions
            )
            / len(
                after_predictions
            )
        )

        oracle_acc = int(
            after_predictions[
                oracle
            ] == yi
        )

        old_acc = None

        if g.old_chosen is not None:

            for j, c in enumerate(
                g.candidates
            ):
                if (
                    c.idx
                    == g.old_chosen
                ):
                    old_acc = int(
                        after_predictions[j]
                        == yi
                    )
                    break

        rows.append(
            {
                "learned":
                    vals[pick],

                "random":
                    random_utility,

                "oracle":
                    vals[oracle],

                "old":
                    old_utility,

                "oracle_hit":
                    int(
                        pick == oracle
                    ),

                "top20":
                    int(
                        pick in top20
                    ),

                "kacc_learned":
                    learned_acc,

                "kacc_random":
                    random_acc,

                "kacc_oracle":
                    oracle_acc,

                "kacc_old":
                    old_acc,
            }
        )

    def mean(key):
        xs = [
            r[key]
            for r in rows
            if r[key] is not None
        ]

        return (
            sum(xs)
            / max(
                1,
                len(xs),
            )
        )

    r = mean(
        "random"
    )

    l = mean(
        "learned"
    )

    o = mean(
        "oracle"
    )

    return {
        "groups":
            len(rows),

        "mean_utility_random":
            r,

        "mean_utility_learned":
            l,

        "mean_utility_old_v14":
            mean("old"),

        "mean_utility_oracle":
            o,

        "oracle_gap_closed":
            (
                l - r
            ) / (
                o - r
                + 1e-9
            ),

        "oracle_exact_hit_rate":
            mean(
                "oracle_hit"
            ),

        "top20_hit_rate":
            mean(
                "top20"
            ),

        "k_accuracy_after_learned":
            mean(
                "kacc_learned"
            ),

        "k_accuracy_after_random_candidate":
            mean(
                "kacc_random"
            ),

        "k_accuracy_after_old_v14":
            mean(
                "kacc_old"
            ),

        "k_accuracy_after_oracle_utility":
            mean(
                "kacc_oracle"
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
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
        "--outdir",
        default=(
            "artifacts/"
            "eqs_v3_mas"
        ),
    )

    ap.add_argument(
        "--dim",
        type=int,
        default=2048,
    )

    ap.add_argument(
        "--rounds",
        default=(
            "1,2,3,4,5,6"
        ),
    )

    args = ap.parse_args()

    torch.set_num_threads(
        int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                "4",
            )
        )
    )

    root = Path(
        args.root
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounds = [
        int(x)
        for x
        in args.rounds.split(",")
    ]

    print(
        "=" * 88
    )

    print(
        "V3-MAS OFFLINE K QUERY SELECTION"
    )

    print(
        "=" * 88
    )

    print(
        "NO SYNTHETIC WORLDS"
    )

    print(
        "NO MAS DOWNLOAD"
    )

    print(
        "USING EXISTING REAL MAS RESPONSES"
    )

    train_groups = load_groups(
        root,
        Path(
            args.manifest
        ),
        "train",
        rounds,
    )

    dev_groups = load_groups(
        root,
        Path(
            args.manifest
        ),
        "dev",
        rounds,
    )

    print(
        f"GROUPS "
        f"train={len(train_groups)} "
        f"dev={len(dev_groups)}"
    )

    print(
        f"SESSIONS "
        f"train={len(set(g.sid for g in train_groups))} "
        f"dev={len(set(g.sid for g in dev_groups))}"
    )

    assert train_groups
    assert dev_groups

    print()
    print(
        "===== BUILD CROSS-FITTED K UTILITY LABELS ====="
    )

    (
        train_utilities,
        train_sensor,
        oof_sensor_accuracy,
    ) = build_train_utilities(
        train_groups,
        dim=args.dim,
        folds=4,
    )

    (
        dev_utilities,
        dev_sensor_metrics,
    ) = build_dev_utilities(
        dev_groups,
        train_sensor,
        args.dim,
    )

    print(
        "OOF_K_SENSOR_BEFORE_ACC:",
        round(
            oof_sensor_accuracy,
            4,
        ),
    )

    print(
        "DEV_K_SENSOR:",
        json.dumps(
            dev_sensor_metrics,
            sort_keys=True,
        ),
    )

    train_data = (
        make_selector_groups(
            train_groups,
            train_utilities,
            args.dim,
        )
    )

    dev_data = (
        make_selector_groups(
            dev_groups,
            dev_utilities,
            args.dim,
        )
    )

    print(
        f"SELECTOR GROUPS "
        f"train={len(train_data)} "
        f"dev={len(dev_data)}"
    )

    results = []

    for seed in [
        7,
        17,
        29,
    ]:

        print()
        print(
            "=" * 88
        )

        print(
            f"SEED {seed}"
        )

        print(
            "=" * 88
        )

        full_model = train_selector(
            train_data,
            args.dim + 8,
            seed=seed,
            query_only=False,
        )

        query_only_model = (
            train_selector(
                train_data,
                args.dim + 8,
                seed=seed,
                query_only=True,
            )
        )

        full_result = (
            eval_selector(
                full_model,
                dev_data,
            )
        )

        query_only_result = (
            eval_selector(
                query_only_model,
                dev_data,
            )
        )

        print(
            "HISTORY_QUERY",
            json.dumps(
                full_result,
                sort_keys=True,
            ),
        )

        print(
            "QUERY_ONLY",
            json.dumps(
                query_only_result,
                sort_keys=True,
            ),
        )

        torch.save(
            {
                "model":
                    full_model.state_dict(),

                "dim":
                    args.dim,

                "ks":
                    KS,
            },
            outdir
            / f"selector_seed{seed}.pt",
        )

        results.append(
            {
                "seed":
                    seed,

                "history_query":
                    full_result,

                "query_only":
                    query_only_result,
            }
        )

    def avg(path):
        vals = []

        for row in results:
            x = row

            for key in path:
                x = x[key]

            vals.append(x)

        return (
            sum(vals)
            / len(vals)
        )

    history_keys = [
        k
        for k
        in results[0][
            "history_query"
        ]
        if k != "groups"
    ]

    query_keys = [
        k
        for k
        in results[0][
            "query_only"
        ]
        if k != "groups"
    ]

    aggregate = {
        "train_groups":
            len(train_data),

        "dev_groups":
            len(dev_data),

        "train_sessions":
            len(
                set(
                    g.sid
                    for g
                    in train_groups
                )
            ),

        "dev_sessions":
            len(
                set(
                    g.sid
                    for g
                    in dev_groups
                )
            ),

        "oof_k_sensor_before_acc":
            oof_sensor_accuracy,

        "dev_k_sensor":
            dev_sensor_metrics,

        "history_query_mean":
            {
                key:
                    avg(
                        [
                            "history_query",
                            key,
                        ]
                    )
                for key
                in history_keys
            },

        "query_only_mean":
            {
                key:
                    avg(
                        [
                            "query_only",
                            key,
                        ]
                    )
                for key
                in query_keys
            },
    }

    final = {
        "aggregate":
            aggregate,

        "seeds":
            results,
    }

    results_path = (
        outdir
        / "results.json"
    )

    results_path.write_text(
        json.dumps(
            final,
            indent=2,
        )
    )

    print()
    print(
        "=" * 88
    )

    print(
        "FINAL V3-MAS OFFLINE RESULTS"
    )

    print(
        "=" * 88
    )

    print(
        json.dumps(
            aggregate,
            indent=2,
        )
    )

    print()
    print(
        "SAVED:",
        results_path,
    )


if __name__ == "__main__":
    main()
