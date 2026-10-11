#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2"
)

OUT = Path(
    "artifacts/allk_k_aggregator_v1"
)

CACHE = OUT / "cache"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE.mkdir(
    parents=True,
    exist_ok=True,
)

ROLES = list(TRAIN_ROLES)
ROLE_TO_I = {
    r: i
    for i, r in enumerate(ROLES)
}

N_ROLES = len(ROLES)

K_VALUES = np.asarray(
    [3, 4, 5, 6, 7, 8],
    dtype=np.float32,
)

K_TO_CLASS = {
    k: k - 3
    for k in range(3, 9)
}

SEED = 20260923

EMBED_MODEL = "all-MiniLM-L6-v2"

# Fixed inference budget.
BAG_SIZE = 8

# ALL-K participation sensor.
SENSOR_EPOCHS = 120
SENSOR_PATIENCE = 15
SENSOR_BATCH = 512
SENSOR_LR = 4e-4

# K ensemble.
K_MODEL_SEEDS = [
    101,
    211,
    307,
    419,
    523,
    631,
    743,
    857,
]

K_EPOCHS = 300
K_PATIENCE = 30
K_STEPS_PER_EPOCH = 120
K_BATCH = 96
K_LR = 7e-4

# Deterministic bags per system during validation/test.
EVAL_BAGS = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# IO
# =====================================================================

def read_jsonl(path: Path):
    out = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                out.append(
                    json.loads(line)
                )

    return out


def stable_int(text: str):
    h = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        h,
        "big",
    )


def safe_stage(path: Path):
    m = re.search(
        r"stage_(\d+)",
        str(path),
    )

    return (
        int(m.group(1))
        if m
        else -1
    )


# =====================================================================
# TEXT FEATURES
# =====================================================================

def text_stats(text: str):
    text = str(text)
    words = text.split()

    nchar = len(text)
    nword = len(words)

    uniq = (
        len(
            set(
                w.lower()
                for w in words
            )
        )
        / max(
            1,
            nword,
        )
    )

    digits = sum(
        c.isdigit()
        for c in text
    )

    uppers = sum(
        c.isupper()
        for c in text
    )

    punct = sum(
        c in ".,;:!?-()[]{}"
        for c in text
    )

    bullets = sum(
        line.lstrip().startswith(
            (
                "-",
                "*",
                "•",
            )
        )
        for line
        in text.splitlines()
    )

    avg_word = (
        sum(
            len(w)
            for w in words
        )
        / max(
            1,
            nword,
        )
    )

    return np.asarray([
        math.log1p(nchar),
        math.log1p(nword),
        math.log1p(
            len(
                text.splitlines()
            )
        ),
        uniq,
        digits / max(1, nchar),
        uppers / max(1, nchar),
        punct / max(1, nchar),
        math.log1p(
            text.count(":")
        ),
        math.log1p(
            bullets
        ),
        avg_word,
    ], dtype=np.float32)


# =====================================================================
# LOAD ALL CURRICULUM DATA
# =====================================================================

def collect_source_files():
    files = sorted(
        ROOT.glob(
            "stage_*/cycle_*/"
            "responses_train_round_*.jsonl"
        )
    )

    if not files:
        raise RuntimeError(
            "No curriculum response files found."
        )

    return files


def visible_output(x):
    return str(
        x.get(
            "final_output",
            "",
        )
        or x.get(
            "output",
            "",
        )
        or x.get(
            "answer",
            "",
        )
        or ""
    ).strip()


def valid_row(x):
    if (
        x.get(
            "execution_status"
        )
        != "ok"
    ):
        return False

    if not bool(
        x.get(
            "valid",
            True,
        )
    ):
        return False

    if not visible_output(x):
        return False

    truth = x.get(
        "truth_roles_training_only",
        [],
    )

    selected = x.get(
        "selected_roles_training_only",
        [],
    )

    if not truth:
        return False

    if selected is None:
        return False

    return True


def source_cache_paths(path):
    rel = str(
        path.relative_to(ROOT)
    )

    key = hashlib.sha1(
        rel.encode("utf-8")
    ).hexdigest()[:16]

    return (
        CACHE / f"{key}.npy",
        CACHE / f"{key}.json",
    )


def response_digest(texts):
    h = hashlib.sha256()

    for t in texts:
        h.update(
            t.encode(
                "utf-8"
            )
        )

        h.update(
            b"\0"
        )

    return h.hexdigest()


def prepare_dataset():
    print(
        f"[device] {DEVICE}",
        flush=True,
    )

    files = collect_source_files()

    print(
        f"[sources] {len(files)} response files",
        flush=True,
    )

    print(
        "[encoder] loading MiniLM...",
        flush=True,
    )

    encoder = SentenceTransformer(
        EMBED_MODEL,
        local_files_only=True,
        device=str(DEVICE),
    )

    role_texts = [
        (
            f"{role}. "
            f"{ROLE_DESCRIPTIONS.get(role, role)} "
            f"{anchor_probe(role)}"
        )
        for role in ROLES
    ]

    role_emb = encoder.encode(
        role_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    rows = []
    feat_chunks = []

    for nfile, path in enumerate(
        files,
        start=1,
    ):
        raw = read_jsonl(
            path
        )

        usable = [
            x
            for x in raw
            if valid_row(x)
        ]

        if not usable:
            continue

        texts = [
            visible_output(x)
            for x in usable
        ]

        digest = response_digest(
            texts
        )

        npy_path, meta_path = (
            source_cache_paths(
                path
            )
        )

        cached = False

        if (
            npy_path.exists()
            and meta_path.exists()
        ):
            meta = json.loads(
                meta_path.read_text()
            )

            if (
                meta.get("digest")
                == digest
                and int(
                    meta.get(
                        "rows",
                        -1,
                    )
                )
                == len(
                    texts
                )
            ):
                X = np.load(
                    npy_path
                )

                cached = True

        if not cached:
            print(
                f"[embed {nfile}/{len(files)}] "
                f"{path} rows={len(texts)}",
                flush=True,
            )

            remb = encoder.encode(
                texts,
                batch_size=128,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(
                np.float32
            )

            r_role = (
                remb @ role_emb.T
            ).astype(
                np.float32
            )

            stats = np.stack([
                text_stats(t)
                for t in texts
            ])

            # Response-only public evidence.
            X = np.concatenate([
                remb,
                r_role,
                stats,
            ], axis=1).astype(
                np.float32
            )

            np.save(
                npy_path,
                X,
            )

            meta_path.write_text(
                json.dumps({
                    "digest":
                        digest,

                    "rows":
                        len(texts),

                    "dim":
                        int(
                            X.shape[1]
                        ),

                    "source":
                        str(path),
                }, indent=2)
            )

        else:
            print(
                f"[cache {nfile}/{len(files)}] "
                f"HIT {path}",
                flush=True,
            )

        if len(X) != len(usable):
            raise RuntimeError(
                "Cache/data row mismatch"
            )

        stage = safe_stage(
            path
        )

        for local_i, x in enumerate(
            usable
        ):
            truth = tuple(
                sorted(
                    map(
                        str,
                        x[
                            "truth_roles_training_only"
                        ],
                    )
                )
            )

            selected = set(
                map(
                    str,
                    x.get(
                        "selected_roles_training_only",
                        [],
                    ),
                )
            )

            k = len(truth)

            if k not in K_TO_CLASS:
                continue

            y_route = np.asarray([
                1.0
                if role in selected
                else 0.0
                for role in ROLES
            ], dtype=np.float32)

            y_roles = np.asarray([
                1.0
                if role in truth
                else 0.0
                for role in ROLES
            ], dtype=np.float32)

            rows.append({
                "session_id":
                    str(
                        x.get(
                            "system_id",
                            x[
                                "session_id"
                            ],
                        )
                    ),

                "truth_roles":
                    truth,

                "signature":
                    "|".join(
                        truth
                    ),

                "K":
                    k,

                "mode":
                    str(
                        x.get(
                            "mode",
                            "unknown",
                        )
                    ),

                "stage":
                    stage,

                "round":
                    int(
                        x.get(
                            "round",
                            0,
                        )
                    ),

                "chosen":
                    bool(
                        x.get(
                            "chosen_for_continuation",
                            False,
                        )
                    ),

                "route_y":
                    y_route,

                "installed_y":
                    y_roles,
            })

        feat_chunks.append(
            X
        )

    if not rows:
        raise RuntimeError(
            "No usable curriculum rows."
        )

    X = np.concatenate(
        feat_chunks,
        axis=0,
    )

    if len(X) != len(rows):
        raise RuntimeError(
            f"Feature mismatch: X={len(X)} rows={len(rows)}"
        )

    print(
        f"[dataset] rows={len(rows)} dim={X.shape[1]}",
        flush=True,
    )

    return rows, X


# =====================================================================
# SYSTEM-LEVEL SPLIT
# =====================================================================

def build_systems(rows):
    systems = {}

    for i, x in enumerate(
        rows
    ):
        sid = x[
            "session_id"
        ]

        if sid not in systems:
            systems[sid] = {
                "sid":
                    sid,

                "K":
                    x["K"],

                "truth_roles":
                    x[
                        "truth_roles"
                    ],

                "signature":
                    x[
                        "signature"
                    ],

                "mode":
                    x[
                        "mode"
                    ],

                "stage":
                    x[
                        "stage"
                    ],

                "row_indices":
                    [],

                "chosen_indices":
                    [],
            }

        s = systems[sid]

        if (
            s["truth_roles"]
            != x[
                "truth_roles"
            ]
        ):
            raise RuntimeError(
                f"Inconsistent truth for {sid}"
            )

        s[
            "row_indices"
        ].append(
            i
        )

        if x["chosen"]:
            s[
                "chosen_indices"
            ].append(
                i
            )

    return systems


def make_split(systems):
    # Group identical installed-role compositions together.
    by_k_signature = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    for sid, s in systems.items():
        by_k_signature[
            s["K"]
        ][
            s["signature"]
        ].append(
            sid
        )

    sig_split = {}

    for k in range(
        3,
        9,
    ):
        sigs = list(
            by_k_signature[
                k
            ]
        )

        sigs.sort(
            key=lambda s:
                stable_int(
                    f"{SEED}|K{k}|{s}"
                )
        )

        n = len(
            sigs
        )

        if n < 6:
            raise RuntimeError(
                f"Too few compositions for K={k}: {n}"
            )

        n_test = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        n_val = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        n_train = (
            n
            - n_val
            - n_test
        )

        for sig in sigs[
            :n_train
        ]:
            sig_split[sig] = (
                "train"
            )

        for sig in sigs[
            n_train:
            n_train+n_val
        ]:
            sig_split[sig] = (
                "val"
            )

        for sig in sigs[
            n_train+n_val:
        ]:
            sig_split[sig] = (
                "test"
            )

        print(
            f"[split] K={k} "
            f"compositions={n} "
            f"train={n_train} "
            f"val={n_val} "
            f"test={n_test}",
            flush=True,
        )

    for s in systems.values():
        s["split"] = (
            sig_split[
                s["signature"]
            ]
        )

    # Composition leakage check.
    split_sigs = {
        split: {
            s["signature"]
            for s
            in systems.values()
            if s["split"]
            == split
        }
        for split in (
            "train",
            "val",
            "test",
        )
    }

    overlap = (
        len(
            split_sigs["train"]
            & split_sigs["val"]
        )
        + len(
            split_sigs["train"]
            & split_sigs["test"]
        )
        + len(
            split_sigs["val"]
            & split_sigs["test"]
        )
    )

    if overlap:
        raise RuntimeError(
            f"Composition leakage={overlap}"
        )

    print(
        "[split] composition leakage=0",
        flush=True,
    )

    counts = Counter(
        (
            s["split"],
            s["K"],
        )
        for s in systems.values()
    )

    print(
        "[system split counts]",
        dict(
            sorted(
                counts.items()
            )
        ),
        flush=True,
    )


# =====================================================================
# ALL-K PARTICIPATION SENSOR
# =====================================================================

class ParticipationSensor(
    nn.Module
):
    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                384,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),

            nn.Linear(
                384,
                192,
            ),
            nn.GELU(),
            nn.Dropout(
                0.08
            ),

            nn.Linear(
                192,
                N_ROLES,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


def row_indices_for_split(
    rows,
    systems,
    split,
):
    return np.asarray([
        i
        for i, x
        in enumerate(rows)
        if systems[
            x["session_id"]
        ][
            "split"
        ]
        == split
    ], dtype=int)


def safe_macro_auc(
    y,
    p,
):
    vals = []

    for j in range(
        y.shape[1]
    ):
        if (
            len(
                np.unique(
                    y[:, j]
                )
            )
            < 2
        ):
            continue

        vals.append(
            roc_auc_score(
                y[:, j],
                p[:, j],
            )
        )

    if not vals:
        return float(
            "nan"
        )

    return float(
        np.mean(
            vals
        )
    )


@torch.no_grad()
def sensor_predict(
    model,
    X,
    mu,
    sd,
):
    model.eval()

    out = []

    for start in range(
        0,
        len(X),
        2048,
    ):
        xb = (
            X[
                start:
                start+2048
            ]
            - mu
        ) / sd

        xb = torch.from_numpy(
            xb.astype(
                np.float32
            )
        ).to(
            DEVICE
        )

        p = torch.sigmoid(
            model(
                xb
            )
        )

        out.append(
            p.detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    )


def train_participation_sensor(
    rows,
    X,
    systems,
):
    tr = row_indices_for_split(
        rows,
        systems,
        "train",
    )

    va = row_indices_for_split(
        rows,
        systems,
        "val",
    )

    te = row_indices_for_split(
        rows,
        systems,
        "test",
    )

    Y = np.stack([
        x["route_y"]
        for x in rows
    ]).astype(
        np.float32
    )

    mu = X[tr].mean(
        axis=0
    ).astype(
        np.float32
    )

    sd = X[tr].std(
        axis=0
    ).astype(
        np.float32
    )

    sd[
        sd < 1e-6
    ] = 1.0

    pos = Y[
        tr
    ].sum(
        axis=0
    )

    neg = (
        len(tr)
        - pos
    )

    pos_weight = np.clip(
        neg
        / np.maximum(
            pos,
            1.0,
        ),
        0.5,
        20.0,
    ).astype(
        np.float32
    )

    model = ParticipationSensor(
        X.shape[1]
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SENSOR_LR,
        weight_decay=1e-4,
    )

    loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(
                pos_weight
            ).to(
                DEVICE
            )
        )
    )

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    rng = np.random.default_rng(
        SEED
    )

    print()
    print(
        "=" * 100
    )
    print(
        "TRAINING ALL-K PARTICIPATION SENSOR"
    )
    print(
        "=" * 100
    )

    for epoch in range(
        1,
        SENSOR_EPOCHS + 1,
    ):
        model.train()

        perm = rng.permutation(
            tr
        )

        losses = []

        for start in range(
            0,
            len(perm),
            SENSOR_BATCH,
        ):
            ids = perm[
                start:
                start+SENSOR_BATCH
            ]

            xb = (
                X[ids]
                - mu
            ) / sd

            xb = torch.from_numpy(
                xb.astype(
                    np.float32
                )
            ).to(
                DEVICE
            )

            yb = torch.from_numpy(
                Y[
                    ids
                ]
            ).to(
                DEVICE
            )

            logits = model(
                xb
            )

            loss = loss_fn(
                logits,
                yb,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        pval = sensor_predict(
            model,
            X[va],
            mu,
            sd,
        )

        val_auc = safe_macro_auc(
            Y[va],
            pval,
        )

        improved = (
            val_auc
            > best_auc
            + 1e-5
        )

        if improved:
            best_auc = (
                val_auc
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                {
                    k:
                        v.detach()
                        .cpu()
                    for k, v
                    in model
                    .state_dict()
                    .items()
                }
            )

            bad = 0

        else:
            bad += 1

        print(
            f"[sensor] "
            f"epoch={epoch:03d} "
            f"loss={np.mean(losses):.4f} "
            f"val_macro_auc={val_auc:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            bad
            >= SENSOR_PATIENCE
        ):
            print(
                "[sensor] early stop",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No sensor checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )

    ptest = sensor_predict(
        model,
        X[te],
        mu,
        sd,
    )

    test_auc = safe_macro_auc(
        Y[te],
        ptest,
    )

    test_micro = float(
        roc_auc_score(
            Y[te].ravel(),
            ptest.ravel(),
        )
    )

    test_ap = float(
        average_precision_score(
            Y[te].ravel(),
            ptest.ravel(),
        )
    )

    print()
    print(
        "[sensor final]"
    )

    print(
        f"best epoch     = {best_epoch}"
    )

    print(
        f"test macro AUC = {test_auc:.4f}"
    )

    print(
        f"test micro AUC = {test_micro:.4f}"
    )

    print(
        f"test micro AP  = {test_ap:.4f}"
    )

    ckpt = {
        "state_dict":
            best_state,

        "input_dim":
            int(
                X.shape[1]
            ),

        "mu":
            mu,

        "sd":
            sd,

        "roles":
            ROLES,

        "embedding_model":
            EMBED_MODEL,

        "best_epoch":
            best_epoch,

        "test_macro_auc":
            test_auc,

        "test_micro_auc":
            test_micro,

        "internal_test_used":
            False,
    }

    torch.save(
        ckpt,
        OUT /
        "allk_participation_sensor.pt",
    )

    # Generate public evidence for every row.
    p_all = sensor_predict(
        model,
        X,
        mu,
        sd,
    )

    return (
        p_all.astype(
            np.float32
        ),
        {
            "best_epoch":
                best_epoch,

            "test_macro_auc":
                test_auc,

            "test_micro_auc":
                test_micro,

            "test_micro_ap":
                test_ap,
        },
    )


# =====================================================================
# K INPUT: PUBLIC PARTICIPATION EVIDENCE ONLY
# =====================================================================

def probability_features(
    p,
):
    p = np.clip(
        p,
        1e-4,
        1.0 - 1e-4,
    )

    logits = np.log(
        p / (
            1.0 - p
        )
    )

    uncertainty = (
        p * (
            1.0 - p
        )
    )

    return np.concatenate([
        p,
        logits,
        uncertainty,
    ], axis=-1).astype(
        np.float32
    )


def attach_system_evidence(
    systems,
    p_all,
):
    for s in systems.values():
        ids = np.asarray(
            s[
                "row_indices"
            ],
            dtype=int,
        )

        p = p_all[
            ids
        ]

        s[
            "participation_probs"
        ] = p

        s[
            "evidence"
        ] = probability_features(
            p
        )

    min_pool = min(
        len(
            s["row_indices"]
        )
        for s
        in systems.values()
    )

    print(
        f"[K data] systems={len(systems)} "
        f"minimum response pool={min_pool}",
        flush=True,
    )

    if min_pool < BAG_SIZE:
        print(
            "[warning] some systems have "
            "fewer than 8 valid responses; "
            "sampling with replacement.",
            flush=True,
        )


# =====================================================================
# DEEP SETS K ESTIMATOR
# =====================================================================

class KDeepSet(
    nn.Module
):
    def __init__(
        self,
        item_dim,
    ):
        super().__init__()

        self.item = nn.Sequential(
            nn.Linear(
                item_dim,
                160,
            ),
            nn.GELU(),
            nn.LayerNorm(
                160
            ),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                160,
                160,
            ),
            nn.GELU(),
            nn.LayerNorm(
                160
            ),
        )

        # mean + max + std of hidden representation:
        # 160 * 3
        #
        # Explicit participation summaries:
        # mean/max/std/noisy-or = 20*4
        trunk_dim = (
            160 * 3
            + N_ROLES * 4
        )

        self.trunk = nn.Sequential(
            nn.Linear(
                trunk_dim,
                320,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),

            nn.Linear(
                320,
                160,
            ),
            nn.GELU(),
            nn.Dropout(
                0.08
            ),
        )

        self.k_head = nn.Linear(
            160,
            len(
                K_VALUES
            ),
        )

        self.role_head = nn.Linear(
            160,
            N_ROLES,
        )

    def forward(
        self,
        x,
    ):
        # x: B x 8 x 60

        h = self.item(
            x
        )

        h_mean = h.mean(
            dim=1
        )

        h_max = h.max(
            dim=1
        ).values

        h_std = h.std(
            dim=1,
            unbiased=False,
        )

        # First 20 dims are participation probabilities.
        p = torch.clamp(
            x[
                :,
                :,
                :N_ROLES
            ],
            1e-4,
            1.0 - 1e-4,
        )

        p_mean = p.mean(
            dim=1
        )

        p_max = p.max(
            dim=1
        ).values

        p_std = p.std(
            dim=1,
            unbiased=False,
        )

        # Probability that each role appeared at least once
        # under conditional-independence approximation.
        p_noisy_or = (
            1.0
            - torch.prod(
                1.0 - p,
                dim=1,
            )
        )

        z = torch.cat([
            h_mean,
            h_max,
            h_std,
            p_mean,
            p_max,
            p_std,
            p_noisy_or,
        ], dim=1)

        z = self.trunk(
            z
        )

        return (
            self.k_head(
                z
            ),
            self.role_head(
                z
            ),
        )


# =====================================================================
# BAG SAMPLING
# =====================================================================

def sample_bag(
    system,
    rng,
):
    pool = system[
        "evidence"
    ]

    n = len(
        pool
    )

    replace = (
        n < BAG_SIZE
    )

    ids = rng.choice(
        n,
        size=BAG_SIZE,
        replace=replace,
    )

    return pool[
        ids
    ].astype(
        np.float32
    )


def installed_target(
    system,
):
    truth = set(
        system[
            "truth_roles"
        ]
    )

    return np.asarray([
        1.0
        if r in truth
        else 0.0
        for r in ROLES
    ], dtype=np.float32)


def class_balanced_batch(
    systems,
    by_k,
    rng,
    batch_size,
):
    xs = []
    ks = []
    roles = []

    available_k = sorted(
        by_k
    )

    for _ in range(
        batch_size
    ):
        k = int(
            rng.choice(
                available_k
            )
        )

        sid = str(
            rng.choice(
                by_k[k]
            )
        )

        s = systems[
            sid
        ]

        xs.append(
            sample_bag(
                s,
                rng,
            )
        )

        ks.append(
            K_TO_CLASS[
                s["K"]
            ]
        )

        roles.append(
            installed_target(
                s
            )
        )

    return (
        torch.from_numpy(
            np.stack(
                xs
            )
        ).to(
            DEVICE
        ),

        torch.tensor(
            ks,
            dtype=torch.long,
            device=DEVICE,
        ),

        torch.from_numpy(
            np.stack(
                roles
            )
        ).to(
            DEVICE
        ),
    )


# =====================================================================
# K EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate_k_model(
    model,
    systems,
    sids,
    seed,
):
    model.eval()

    predictions = {}

    for sid in sids:
        s = systems[
            sid
        ]

        rng = np.random.default_rng(
            stable_int(
                f"EVAL|{seed}|{sid}"
            )
            % (
                2**32
            )
        )

        bags = np.stack([
            sample_bag(
                s,
                rng,
            )
            for _ in range(
                EVAL_BAGS
            )
        ])

        xb = torch.from_numpy(
            bags
        ).to(
            DEVICE
        )

        k_logits, role_logits = (
            model(
                xb
            )
        )

        kp = torch.softmax(
            k_logits,
            dim=1,
        ).mean(
            dim=0
        )

        rp = torch.sigmoid(
            role_logits
        ).mean(
            dim=0
        )

        predictions[
            sid
        ] = {
            "true_k":
                int(
                    s["K"]
                ),

            "k_probs":
                kp.detach()
                .cpu()
                .numpy(),

            "role_probs":
                rp.detach()
                .cpu()
                .numpy(),

            "truth_roles":
                s[
                    "truth_roles"
                ],
        }

    metrics = metrics_from_system_predictions(
        predictions
    )

    return (
        metrics,
        predictions,
    )


def metrics_from_system_predictions(
    predictions,
):
    y = []
    pred = []

    abs_err = []

    role_true = []
    role_pred = []

    per_k = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
        }
    )

    for sid, x in predictions.items():
        true_k = int(
            x["true_k"]
        )

        pred_k = (
            int(
                np.argmax(
                    x[
                        "k_probs"
                    ]
                )
            )
            + 3
        )

        y.append(
            true_k
        )

        pred.append(
            pred_k
        )

        abs_err.append(
            abs(
                true_k
                - pred_k
            )
        )

        per_k[
            true_k
        ][
            "n"
        ] += 1

        if pred_k == true_k:
            per_k[
                true_k
            ][
                "correct"
            ] += 1

        truth = set(
            x[
                "truth_roles"
            ]
        )

        yt = np.asarray([
            1
            if r in truth
            else 0
            for r in ROLES
        ])

        yp = (
            np.asarray(
                x[
                    "role_probs"
                ]
            )
            >= 0.5
        ).astype(
            int
        )

        role_true.append(
            yt
        )

        role_pred.append(
            yp
        )

    y = np.asarray(
        y
    )

    pred = np.asarray(
        pred
    )

    role_true = np.stack(
        role_true
    )

    role_pred = np.stack(
        role_pred
    )

    exact = float(
        np.mean(
            y == pred
        )
    )

    mae = float(
        np.mean(
            np.abs(
                y - pred
            )
        )
    )

    within1 = float(
        np.mean(
            np.abs(
                y - pred
            )
            <= 1
        )
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=[
            3,
            4,
            5,
            6,
            7,
            8,
        ],
    )

    per_k_acc = {
        str(k):
            (
                v["correct"]
                / max(
                    1,
                    v["n"],
                )
            )
        for k, v
        in sorted(
            per_k.items()
        )
    }

    return {
        "n_systems":
            int(
                len(y)
            ),

        "exact_k":
            exact,

        "mae":
            mae,

        "within_1":
            within1,

        "installed_role_macro_f1":
            float(
                f1_score(
                    role_true,
                    role_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "installed_role_micro_f1":
            float(
                f1_score(
                    role_true,
                    role_pred,
                    average="micro",
                    zero_division=0,
                )
            ),

        "per_k_accuracy":
            per_k_acc,

        "confusion_matrix":
            cm.tolist(),
    }


# =====================================================================
# TRAIN ONE K MODEL
# =====================================================================

def train_one_k_model(
    systems,
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    train_sids = [
        sid
        for sid, s
        in systems.items()
        if s["split"]
        == "train"
    ]

    val_sids = [
        sid
        for sid, s
        in systems.items()
        if s["split"]
        == "val"
    ]

    test_sids = [
        sid
        for sid, s
        in systems.items()
        if s["split"]
        == "test"
    ]

    by_k = defaultdict(
        list
    )

    for sid in train_sids:
        by_k[
            systems[
                sid
            ][
                "K"
            ]
        ].append(
            sid
        )

    item_dim = (
        systems[
            train_sids[0]
        ][
            "evidence"
        ].shape[1]
    )

    model = KDeepSet(
        item_dim
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=K_LR,
        weight_decay=2e-4,
    )

    rng = np.random.default_rng(
        seed
    )

    best_state = None
    best_epoch = 0

    best_val_exact = -1.0
    best_val_mae = 999.0

    bad = 0

    print()
    print(
        "=" * 100
    )

    print(
        f"K MODEL seed={seed}"
    )

    print(
        "=" * 100
    )

    for epoch in range(
        1,
        K_EPOCHS + 1,
    ):
        model.train()

        losses = []

        for _ in range(
            K_STEPS_PER_EPOCH
        ):
            xb, kb, rb = (
                class_balanced_batch(
                    systems,
                    by_k,
                    rng,
                    K_BATCH,
                )
            )

            k_logits, role_logits = (
                model(
                    xb
                )
            )

            ce = F.cross_entropy(
                k_logits,
                kb,
            )

            role_loss = (
                F.binary_cross_entropy_with_logits(
                    role_logits,
                    rb,
                )
            )

            k_probs = torch.softmax(
                k_logits,
                dim=1,
            )

            kvals = torch.tensor(
                K_VALUES,
                dtype=torch.float32,
                device=DEVICE,
            )

            expected_k = (
                k_probs
                * kvals[
                    None,
                    :
                ]
            ).sum(
                dim=1
            )

            true_k_float = (
                kb.float()
                + 3.0
            )

            count_loss = (
                F.smooth_l1_loss(
                    expected_k,
                    true_k_float,
                )
            )

            installed_count = (
                torch.sigmoid(
                    role_logits
                )
                .sum(
                    dim=1
                )
            )

            role_count_loss = (
                F.smooth_l1_loss(
                    installed_count,
                    true_k_float,
                )
            )

            loss = (
                ce
                + 0.75
                * role_loss
                + 0.20
                * count_loss
                + 0.10
                * role_count_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        val_metrics, _ = (
            evaluate_k_model(
                model,
                systems,
                val_sids,
                seed=424242,
            )
        )

        exact = val_metrics[
            "exact_k"
        ]

        mae = val_metrics[
            "mae"
        ]

        improved = (
            exact
            > best_val_exact
            + 1e-9
            or (
                abs(
                    exact
                    - best_val_exact
                )
                < 1e-9
                and mae
                < best_val_mae
                - 1e-9
            )
        )

        if improved:
            best_val_exact = (
                exact
            )

            best_val_mae = (
                mae
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                {
                    k:
                        v.detach()
                        .cpu()
                    for k, v
                    in model
                    .state_dict()
                    .items()
                }
            )

            bad = 0

        else:
            bad += 1

        if (
            epoch == 1
            or epoch % 5 == 0
            or improved
        ):
            print(
                f"[K seed={seed}] "
                f"epoch={epoch:03d} "
                f"loss={np.mean(losses):.4f} "
                f"val_exact={100*exact:.2f}% "
                f"val_MAE={mae:.4f} "
                f"{'*' if improved else ''}",
                flush=True,
            )

        if (
            bad
            >= K_PATIENCE
        ):
            print(
                f"[K seed={seed}] "
                f"early stop epoch={epoch}",
                flush=True,
            )

            break

    if best_state is None:
        raise RuntimeError(
            "K model did not train"
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )

    val_metrics, val_pred = (
        evaluate_k_model(
            model,
            systems,
            val_sids,
            seed=424242,
        )
    )

    test_metrics, test_pred = (
        evaluate_k_model(
            model,
            systems,
            test_sids,
            seed=515151,
        )
    )

    ckpt_path = (
        OUT
        / f"k_model_seed_{seed}.pt"
    )

    torch.save({
        "state_dict":
            best_state,

        "item_dim":
            int(
                item_dim
            ),

        "roles":
            ROLES,

        "K_values":
            K_VALUES.tolist(),

        "bag_size":
            BAG_SIZE,

        "best_epoch":
            best_epoch,

        "val_metrics":
            val_metrics,

        "test_metrics":
            test_metrics,

        "public_inference_features_only":
            True,

        "internal_test_used":
            False,
    }, ckpt_path)

    print(
        f"[K seed={seed}] "
        f"BEST epoch={best_epoch} "
        f"VAL exact={100*val_metrics['exact_k']:.2f}% "
        f"TEST exact={100*test_metrics['exact_k']:.2f}% "
        f"TEST MAE={test_metrics['mae']:.4f}",
        flush=True,
    )

    return {
        "seed":
            seed,

        "checkpoint":
            str(
                ckpt_path
            ),

        "best_epoch":
            best_epoch,

        "val":
            val_metrics,

        "test":
            test_metrics,

        "val_predictions":
            val_pred,

        "test_predictions":
            test_pred,
    }


# =====================================================================
# ENSEMBLE
# =====================================================================

def ensemble_predictions(
    model_results,
    field,
):
    sids = sorted(
        model_results[0][
            field
        ]
    )

    out = {}

    for sid in sids:
        base = (
            model_results[0][
                field
            ][sid]
        )

        kp = np.mean(
            [
                x[field][
                    sid
                ][
                    "k_probs"
                ]
                for x in model_results
            ],
            axis=0,
        )

        rp = np.mean(
            [
                x[field][
                    sid
                ][
                    "role_probs"
                ]
                for x in model_results
            ],
            axis=0,
        )

        out[sid] = {
            "true_k":
                base[
                    "true_k"
                ],

            "truth_roles":
                base[
                    "truth_roles"
                ],

            "k_probs":
                kp,

            "role_probs":
                rp,
        }

    return out


# =====================================================================
# MAIN
# =====================================================================

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

    if DEVICE.type == "cpu":
        torch.set_num_threads(
            8
        )

    print(
        "=" * 110
    )

    print(
        "ALL-K BENIGN MAS K ESTIMATOR V1"
    )

    print(
        "=" * 110
    )

    print(
        "Target: K in {3,4,5,6,7,8}"
    )

    print(
        "Inference inputs: visible responses only"
    )

    print(
        "Privileged labels: TRAINING ONLY"
    )

    print(
        "Final untouched internal test: NOT USED"
    )

    print(
        f"Device: {DEVICE}"
    )

    rows, X = prepare_dataset()

    systems = build_systems(
        rows
    )

    print()
    print(
        "[systems]",
        len(
            systems
        ),
    )

    k_counts = Counter(
        s["K"]
        for s
        in systems.values()
    )

    print(
        "[K distribution]",
        dict(
            sorted(
                k_counts.items()
            )
        ),
    )

    stage_counts = Counter(
        (
            s["stage"],
            s["K"],
        )
        for s
        in systems.values()
    )

    print(
        "[stage/K count entries]",
        len(
            stage_counts
        ),
    )

    make_split(
        systems
    )

    split_manifest = {
        sid: {
            "split":
                s["split"],

            "K":
                s["K"],

            "signature":
                s["signature"],

            "stage":
                s["stage"],

            "mode":
                s["mode"],
        }
        for sid, s
        in systems.items()
    }

    (
        OUT
        / "SYSTEM_SPLIT.json"
    ).write_text(
        json.dumps(
            split_manifest,
            indent=2,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------------
    # Stage A:
    # all-K visible-response -> participation sensor
    # --------------------------------------------------------------

    p_all, sensor_report = (
        train_participation_sensor(
            rows,
            X,
            systems,
        )
    )

    np.save(
        OUT
        / "participation_probabilities.npy",
        p_all,
    )

    # --------------------------------------------------------------
    # Stage B:
    # 8 observations -> installed roles + direct K
    # --------------------------------------------------------------

    attach_system_evidence(
        systems,
        p_all,
    )

    results = []

    for seed in K_MODEL_SEEDS:
        result = (
            train_one_k_model(
                systems,
                seed,
            )
        )

        results.append(
            result
        )

    # Choose top 5 only using validation.
    ranked = sorted(
        results,
        key=lambda x: (
            -x["val"][
                "exact_k"
            ],
            x["val"][
                "mae"
            ],
        ),
    )

    selected = ranked[
        :min(
            5,
            len(
                ranked
            ),
        )
    ]

    ensemble_val_pred = (
        ensemble_predictions(
            selected,
            "val_predictions",
        )
    )

    ensemble_test_pred = (
        ensemble_predictions(
            selected,
            "test_predictions",
        )
    )

    ensemble_val = (
        metrics_from_system_predictions(
            ensemble_val_pred
        )
    )

    ensemble_test = (
        metrics_from_system_predictions(
            ensemble_test_pred
        )
    )

    selected_seeds = [
        x["seed"]
        for x in selected
    ]

    report = {
        "sensor":
            sensor_report,

        "systems":
            len(
                systems
            ),

        "K_distribution":
            {
                str(k):
                    int(v)
                for k, v
                in sorted(
                    k_counts.items()
                )
            },

        "bag_size":
            BAG_SIZE,

        "models":
            [
                {
                    "seed":
                        x["seed"],

                    "checkpoint":
                        x[
                            "checkpoint"
                        ],

                    "best_epoch":
                        x[
                            "best_epoch"
                        ],

                    "val":
                        x[
                            "val"
                        ],

                    "test":
                        x[
                            "test"
                        ],
                }
                for x in results
            ],

        "selected_ensemble_seeds":
            selected_seeds,

        "ensemble_val":
            ensemble_val,

        "ensemble_test":
            ensemble_test,

        "inference_inputs":
            (
                "eight visible final responses "
                "converted to participation probabilities"
            ),

        "forbidden_inference_fields":
            [
                "K",
                "truth_roles",
                "selected_roles",
                "mode",
                "stage",
                "system_id",
                "execution trace",
            ],

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

    (
        OUT
        / "ENSEMBLE.json"
    ).write_text(
        json.dumps({
            "selected_seeds":
                selected_seeds,

            "checkpoints":
                [
                    x[
                        "checkpoint"
                    ]
                    for x
                    in selected
                ],

            "bag_size":
                BAG_SIZE,

            "roles":
                ROLES,

            "K_values":
                K_VALUES.tolist(),
        }, indent=2)
    )

    print()
    print(
        "=" * 110
    )

    print(
        "FINAL ALL-K K ESTIMATOR"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "PARTICIPATION SENSOR"
    )

    print(
        f"test macro AUC : "
        f"{sensor_report['test_macro_auc']:.4f}"
    )

    print(
        f"test micro AUC : "
        f"{sensor_report['test_micro_auc']:.4f}"
    )

    print()
    print(
        "SELECTED ENSEMBLE SEEDS:",
        selected_seeds,
    )

    print()
    print(
        "ENSEMBLE VALIDATION"
    )

    print(
        f"Exact K : "
        f"{100*ensemble_val['exact_k']:.2f}%"
    )

    print(
        f"MAE     : "
        f"{ensemble_val['mae']:.4f}"
    )

    print(
        f"Within 1: "
        f"{100*ensemble_val['within_1']:.2f}%"
    )

    print()
    print(
        "ENSEMBLE HELD-OUT TEST"
    )

    print(
        f"Exact K : "
        f"{100*ensemble_test['exact_k']:.2f}%"
    )

    print(
        f"MAE     : "
        f"{ensemble_test['mae']:.4f}"
    )

    print(
        f"Within 1: "
        f"{100*ensemble_test['within_1']:.2f}%"
    )

    print(
        f"Installed-role macro F1: "
        f"{ensemble_test['installed_role_macro_f1']:.4f}"
    )

    print(
        f"Installed-role micro F1: "
        f"{ensemble_test['installed_role_micro_f1']:.4f}"
    )

    print()
    print(
        "PER-K EXACT ACCURACY"
    )

    for k, acc in (
        ensemble_test[
            "per_k_accuracy"
        ].items()
    ):
        print(
            f"K={k}: "
            f"{100*acc:.2f}%"
        )

    print()
    print(
        "CONFUSION MATRIX "
        "[rows=true 3..8, cols=pred 3..8]"
    )

    for row in (
        ensemble_test[
            "confusion_matrix"
        ]
    ):
        print(
            row
        )

    print()
    print(
        "PRIVILEGED INPUT AT INFERENCE: NO"
    )

    print(
        "INTERNAL FINAL TEST USED: NO"
    )

    print(
        f"REPORT: {OUT / 'REPORT.json'}"
    )

    print(
        f"ENSEMBLE: {OUT / 'ENSEMBLE.json'}"
    )


if __name__ == "__main__":
    main()
