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
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)

ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2/stage_10"
)

OUT = Path(
    "artifacts/adaptive_v11_curriculum_v2/selector_v5_router"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260922
EMBED_MODEL = "all-MiniLM-L6-v2"

EPOCHS = 60
PATIENCE = 8
BATCH = 512
LR = 5e-4
WD = 1e-4


def bucket(x: str) -> int:
    d = hashlib.blake2b(
        str(x).encode(),
        digest_size=8,
    ).digest()

    return int.from_bytes(d, "big") % 100


def split_for_system(sid: str) -> str:
    b = bucket(sid)

    if b < 70:
        return "train"

    if b < 85:
        return "val"

    return "test"


def entropy01(p):
    p = min(
        1 - 1e-7,
        max(1e-7, float(p)),
    )

    return (
        -p * math.log(p)
        - (1-p) * math.log(1-p)
    ) / math.log(2)


def load_raw():
    rows = []

    for cycle in (
        "cycle_01",
        "cycle_02",
    ):
        cdir = ROOT / cycle

        for rnd in range(1, 9):
            sp = (
                cdir
                / f"states_train_round_{rnd-1:02d}.jsonl"
            )

            rp = (
                cdir
                / f"responses_train_round_{rnd:02d}.jsonl"
            )

            states = {
                str(x["session_id"]): x
                for x in read_jsonl(sp)
            }

            for x in read_jsonl(rp):
                if (
                    x.get("execution_status") != "ok"
                    or not bool(x.get("valid", True))
                ):
                    continue

                sid = str(x["session_id"])

                if sid not in states:
                    continue

                st = PublicBeliefState.from_public_dict(
                    states[sid]
                )

                system_id = str(
                    x.get(
                        "system_id",
                        sid,
                    )
                )

                truth_selected = set(
                    x.get(
                        "selected_roles_training_only",
                        [],
                    )
                )

                y = np.asarray(
                    [
                        1.0 if r in truth_selected else 0.0
                        for r in TRAIN_ROLES
                    ],
                    dtype=np.float32,
                )

                probs = np.asarray(
                    [
                        float(st.role_probs[r])
                        for r in TRAIN_ROLES
                    ],
                    dtype=np.float32,
                )

                ent = np.asarray(
                    [
                        entropy01(st.role_probs[r])
                        for r in TRAIN_ROLES
                    ],
                    dtype=np.float32,
                )

                rows.append({
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

                    "candidate_index":
                        int(x["candidate_index"]),

                    "chosen":
                        bool(
                            x.get(
                                "chosen_for_continuation",
                                False,
                            )
                        ),

                    "query":
                        str(x["query"]),

                    "probs":
                        probs,

                    "entropy":
                        ent,

                    "y":
                        y,
                })

    return rows


class RouterNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(
                128,
                len(TRAIN_ROLES),
            ),
        )

    def forward(self, x):
        return self.net(x)


def make_features(rows, encoder):
    queries = [
        x["query"]
        for x in rows
    ]

    print(
        f"[embed] queries={len(queries)}",
        flush=True,
    )

    qemb = encoder.encode(
        queries,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    role_text = [
        (
            f"{r}. "
            f"{ROLE_DESCRIPTIONS.get(r, r)} "
            f"{anchor_probe(r)}"
        )
        for r in TRAIN_ROLES
    ]

    remb = encoder.encode(
        role_text,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    feats = []

    for i, x in enumerate(rows):
        sim = (
            qemb[i]
            @ remb.T
        ).astype(np.float32)

        probs = x["probs"]
        ent = x["entropy"]

        cand = np.zeros(
            8,
            dtype=np.float32,
        )

        cand[
            x["candidate_index"] % 8
        ] = 1.0

        f = np.concatenate([
            qemb[i],             # 384
            sim,                 # 20
            probs,               # 20
            ent,                 # 20
            sim * ent,           # 20
            sim * (1.0-probs),   # 20
            cand,                # 8
            np.asarray(
                [x["round"]/8.0],
                dtype=np.float32,
            ),
        ])

        feats.append(f)

    return (
        np.stack(feats),
        remb,
    )


def split_arrays(rows, X):
    out = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        idx = np.asarray(
            [
                i
                for i, x
                in enumerate(rows)
                if x["split"] == split
            ],
            dtype=int,
        )

        out[split] = (
            idx,
            X[idx],
            np.stack([
                rows[i]["y"]
                for i in idx
            ]),
        )

    return out


@torch.no_grad()
def predict(model, X):
    model.eval()

    outs = []

    for s in range(
        0,
        len(X),
        1024,
    ):
        xb = torch.from_numpy(
            X[s:s+1024]
        )

        outs.append(
            torch.sigmoid(
                model(xb)
            ).cpu().numpy()
        )

    return np.concatenate(
        outs,
        axis=0,
    )


def auc_metrics(y, p):
    aucs = []

    for j in range(
        y.shape[1]
    ):
        if len(
            np.unique(
                y[:, j]
            )
        ) < 2:
            continue

        aucs.append(
            roc_auc_score(
                y[:, j],
                p[:, j],
            )
        )

    return {
        "macro_auc":
            float(
                np.mean(aucs)
            ),

        "micro_auc":
            float(
                roc_auc_score(
                    y.ravel(),
                    p.ravel(),
                )
            ),
    }


def selector_replay(
    rows,
    probs_pred,
    split,
):
    by_group = defaultdict(list)

    for i, x in enumerate(rows):
        if x["split"] != split:
            continue

        key = (
            x["cycle"],
            x["session_id"],
            x["round"],
        )

        by_group[key].append(
            (i, x)
        )

    chosen_hits = []
    random_hits = []
    oracle_hits = []

    selected_by_traj = defaultdict(set)
    current_by_traj = defaultdict(set)
    truth_union = defaultdict(set)

    for key, group in by_group.items():
        cycle, sid, rnd = key

        scores = []

        for i, x in group:
            # Expected activation of currently uncertain roles.
            score = float(
                np.sum(
                    probs_pred[i]
                    * x["entropy"]
                )
            )

            scores.append(
                (
                    score,
                    i,
                    x,
                )
            )

        _, best_i, best_x = max(
            scores,
            key=lambda z: z[0],
        )

        actual_best = set(
            TRAIN_ROLES[j]
            for j, z
            in enumerate(
                best_x["y"]
            )
            if z > 0.5
        )

        selected_by_traj[
            (cycle, sid)
        ].update(
            actual_best
        )

        for _, _, x in scores:
            actual = set(
                TRAIN_ROLES[j]
                for j, z
                in enumerate(
                    x["y"]
                )
                if z > 0.5
            )

            truth_union[
                (cycle, sid)
            ].update(
                actual
            )

            if x["chosen"]:
                current_by_traj[
                    (cycle, sid)
                ].update(
                    actual
                )

        # Diagnostic: number of actually activated
        # roles for one selected candidate.
        candidate_counts = [
            int(
                x["y"].sum()
            )
            for _, _, x
            in scores
        ]

        chosen_hits.append(
            int(
                best_x["y"].sum()
            )
        )

        random_hits.append(
            float(
                np.mean(
                    candidate_counts
                )
            )
        )

        oracle_hits.append(
            max(
                candidate_counts
            )
        )

    return {
        "groups":
            len(by_group),

        "chosen_mean_activated":
            float(
                np.mean(
                    chosen_hits
                )
            ),

        "random_mean_activated":
            float(
                np.mean(
                    random_hits
                )
            ),

        "oracle_mean_activated":
            float(
                np.mean(
                    oracle_hits
                )
            ),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(
        max(
            1,
            min(
                4,
                torch.get_num_threads(),
            ),
        )
    )

    rows = load_raw()

    print(
        "[data]",
        {
            s: sum(
                x["split"] == s
                for x in rows
            )
            for s in (
                "train",
                "val",
                "test",
            )
        },
        flush=True,
    )

    print(
        "[encoder] loading local MiniLM...",
        flush=True,
    )

    encoder = SentenceTransformer(
        EMBED_MODEL,
        local_files_only=True,
    )

    X, role_emb = make_features(
        rows,
        encoder,
    )

    data = split_arrays(
        rows,
        X,
    )

    train_idx, Xtr, Ytr = data["train"]
    val_idx, Xv, Yv = data["val"]
    test_idx, Xt, Yt = data["test"]

    mu = Xtr.mean(
        axis=0
    ).astype(np.float32)

    sd = Xtr.std(
        axis=0
    ).astype(np.float32)

    sd[
        sd < 1e-6
    ] = 1.0

    Xtr = (
        Xtr - mu
    ) / sd

    Xv = (
        Xv - mu
    ) / sd

    Xt = (
        Xt - mu
    ) / sd

    pos = Ytr.sum(
        axis=0
    )

    neg = (
        len(Ytr)
        - pos
    )

    pos_weight = np.clip(
        neg
        / np.maximum(
            pos,
            1.0,
        ),
        0.5,
        10.0,
    ).astype(np.float32)

    model = RouterNet(
        X.shape[1]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(
                pos_weight
            )
        )
    )

    tx = torch.from_numpy(
        Xtr
    )

    ty = torch.from_numpy(
        Ytr
    )

    best_auc = -1
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        perm = torch.randperm(
            len(tx)
        )

        losses = []

        for start in range(
            0,
            len(tx),
            BATCH,
        ):
            idx = perm[
                start:start+BATCH
            ]

            logits = model(
                tx[idx]
            )

            loss = criterion(
                logits,
                ty[idx],
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

        pv = predict(
            model,
            Xv,
        )

        vm = auc_metrics(
            Yv,
            pv,
        )

        score = vm[
            "macro_auc"
        ]

        improved = (
            score
            > best_auc
            + 1e-5
        )

        if improved:
            best_auc = score
            best_epoch = epoch
            bad = 0

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
            bad += 1

        print(
            f"[epoch {epoch:02d}] "
            f"loss={np.mean(losses):.4f} "
            f"val_macro_auc={vm['macro_auc']:.4f} "
            f"val_micro_auc={vm['micro_auc']:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if bad >= PATIENCE:
            print(
                "[early-stop]",
                flush=True,
            )
            break

    model.load_state_dict(
        best_state
    )

    model.eval()

    ptr = predict(
        model,
        Xtr,
    )

    pv = predict(
        model,
        Xv,
    )

    pt = predict(
        model,
        Xt,
    )

    metrics = {
        "train":
            auc_metrics(
                Ytr,
                ptr,
            ),

        "val":
            auc_metrics(
                Yv,
                pv,
            ),

        "test":
            auc_metrics(
                Yt,
                pt,
            ),
    }

    # Put predictions back in global row order
    pall = np.zeros(
        (
            len(rows),
            len(TRAIN_ROLES),
        ),
        dtype=np.float32,
    )

    pall[
        train_idx
    ] = ptr

    pall[
        val_idx
    ] = pv

    pall[
        test_idx
    ] = pt

    replay = selector_replay(
        rows,
        pall,
        "test",
    )

    ckpt = (
        OUT
        / "selector_v5_router.pt"
    )

    torch.save({
        "state_dict":
            model.state_dict(),

        "input_dim":
            int(
                X.shape[1]
            ),

        "mu":
            mu,

        "sd":
            sd,

        "roles":
            list(
                TRAIN_ROLES
            ),

        "embedding_model":
            EMBED_MODEL,

        "role_embeddings":
            role_emb,

        "best_epoch":
            best_epoch,

        "objective":
            "predict selected_roles_training_only",

        "inference_privileged_inputs":
            False,

        "internal_test_used":
            False,
    }, ckpt)

    report = {
        "checkpoint":
            str(
                ckpt
            ),

        "best_epoch":
            best_epoch,

        "n_rows":
            len(rows),

        "metrics":
            metrics,

        "test_selector_replay":
            replay,

        "internal_test_used":
            False,
    }

    (
        OUT / "REPORT.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("="*90)
    print("SELECTOR V5 ROUTER")
    print("="*90)

    print(
        "best epoch:",
        best_epoch,
    )

    for split, m in (
        metrics.items()
    ):
        print(
            f"{split}: "
            f"macro_auc={m['macro_auc']:.4f} "
            f"micro_auc={m['micro_auc']:.4f}"
        )

    print()
    print("[HELD-OUT SELECTOR]")

    for k, v in (
        replay.items()
    ):
        print(
            f"{k}: {v}"
        )

    print()
    print(
        "PRIVILEGED LABELS AT TRAINING: YES"
    )

    print(
        "PRIVILEGED INPUTS AT INFERENCE: NO"
    )

    print(
        "INTERNAL TEST USED: NO"
    )

    print(
        "V5 DONE"
    )


if __name__ == "__main__":
    main()
