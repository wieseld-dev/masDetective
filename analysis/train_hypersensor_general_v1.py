from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from v8.embedding import FrozenOutputEmbedderV8
from v11.role_universe import ROLE_DESCRIPTIONS


TRAIN_PATH = Path(
    "artifacts/v137_presence_v1/dataset_v1/train_v137.jsonl"
)
VAL_PATH = Path(
    "artifacts/v137_presence_v1/dataset_v1/dev_v137.jsonl"
)

OUT = Path("artifacts/hypersensor_general_v1")

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def dump(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def role_desc(row):
    role = str(row["target_role"])

    return str(
        row.get("public_role_description")
        or ROLE_DESCRIPTIONS.get(role, role)
    )


# ============================================================
# EMBEDDINGS
# ============================================================

class BatchedEmbeddings:
    def __init__(self, datasets):
        print(
            f"Loading FrozenOutputEmbedderV8 on {DEVICE}...",
            flush=True,
        )

        self.embedder = FrozenOutputEmbedderV8(
            device=DEVICE
        )

        texts = set()

        for rows in datasets:
            for x in rows:
                texts.add(role_desc(x))
                texts.add(str(x["query"]))
                texts.add(str(x["final_output"]))

        texts = sorted(texts)

        print(
            "unique texts:",
            len(texts),
            flush=True,
        )

        vecs = self.embedder.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        self.map = {
            t: np.asarray(v, dtype=np.float32)
            for t, v in zip(texts, vecs)
        }

        self.dim = int(vecs.shape[1])

        print(
            "embedding dim:",
            self.dim,
            flush=True,
        )

    def get(self, text):
        return self.map[str(text)]


# ============================================================
# EXACT 20D GEOMETRY
# ============================================================

def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def geom_from_vectors(r, q, o):
    oq = o - q
    or_ = o - r
    rq = r - q
    qr = q - r

    d_or = float(np.linalg.norm(or_))
    d_oq = float(np.linalg.norm(oq))
    d_qr = float(np.linalg.norm(q - r))

    cos_or = cosine(o, r)
    cos_oq = cosine(o, q)
    cos_qr = cosine(q, r)

    return np.asarray(
        [
            cos_or,
            cos_oq,
            cos_qr,

            d_or,
            d_oq,
            d_qr,

            cos_or - cos_qr,
            cos_oq - cos_qr,

            cosine(oq, r),
            cosine(oq, rq),

            cosine(or_, q),
            cosine(or_, qr),

            d_oq,
            d_or,
            d_qr,

            float(np.dot(oq, r)),
            float(np.dot(oq, rq)),

            d_oq + d_qr - d_or,
            d_or + d_qr - d_oq,

            d_oq / max(d_qr, 1e-6),
        ],
        dtype=np.float32,
    )


@dataclass
class Data:
    rows: list
    R: np.ndarray
    Q: np.ndarray
    O: np.ndarray

    G: np.ndarray
    G_query_only: np.ndarray

    C_role: np.ndarray
    C_role_query: np.ndarray


def build_data(rows, emb: BatchedEmbeddings):
    R = np.stack([
        emb.get(role_desc(x))
        for x in rows
    ]).astype(np.float32)

    Q = np.stack([
        emb.get(str(x["query"]))
        for x in rows
    ]).astype(np.float32)

    O = np.stack([
        emb.get(str(x["final_output"]))
        for x in rows
    ]).astype(np.float32)

    G = np.stack([
        geom_from_vectors(r, q, o)
        for r, q, o in zip(R, Q, O)
    ]).astype(np.float32)

    Gq = np.stack([
        geom_from_vectors(r, q, q)
        for r, q in zip(R, Q)
    ]).astype(np.float32)

    # Role embedding itself.
    C_role = R.copy()

    # Semantic context.
    #
    # Each component is public:
    # role description + benign query.
    #
    # Product receives sqrt(d) scaling so its norm
    # is not vanishingly small relative to R/Q.
    d = R.shape[1]

    prod = math.sqrt(d) * (R * Q)
    delta = np.abs(R - Q)

    C_rq = np.concatenate(
        [R, Q, prod, delta],
        axis=1,
    ).astype(np.float32)

    # Fixed per-sample normalization.
    # No labels / held-out data statistics.
    n = np.linalg.norm(
        C_rq,
        axis=1,
        keepdims=True,
    )

    C_rq = C_rq / np.maximum(n, 1e-6)

    return Data(
        rows=rows,
        R=R,
        Q=Q,
        O=O,
        G=G,
        G_query_only=Gq,
        C_role=C_role,
        C_role_query=C_rq,
    )


# ============================================================
# METRICS
# ============================================================

def auc_safe(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    if len(np.unique(y)) < 2:
        return None

    return float(
        roc_auc_score(y, s)
    )


def evaluate(rows, scores):
    y = np.asarray(
        [int(x["label"]) for x in rows],
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    out = {
        "rows": len(rows),
        "systems": len(set(
            str(x["system_id"])
            for x in rows
        )),
        "global_auc": auc_safe(y, scores),
    }

    by_role = defaultdict(list)

    for i, x in enumerate(rows):
        by_role[str(x["target_role"])].append(i)

    role_aucs = {}

    for role, idxs in sorted(by_role.items()):
        a = auc_safe(
            y[idxs],
            scores[idxs],
        )

        if a is not None:
            role_aucs[role] = a

    out["per_role_auc"] = role_aucs

    out["mean_role_auc"] = (
        float(np.mean(list(role_aucs.values())))
        if role_aucs
        else None
    )

    out["median_role_auc"] = (
        float(np.median(list(role_aucs.values())))
        if role_aucs
        else None
    )

    by_system = defaultdict(list)

    for i, x in enumerate(rows):
        by_system[str(x["system_id"])].append(i)

    system_aucs = []
    perfect = []
    exact_topk = []
    margins = []

    for sid, idxs in by_system.items():
        yy = y[idxs]
        ss = scores[idxs]

        p = np.where(yy == 1)[0]
        n = np.where(yy == 0)[0]

        if len(p) == 0 or len(n) == 0:
            continue

        system_aucs.append(
            float(
                roc_auc_score(
                    yy,
                    ss,
                )
            )
        )

        margin = float(
            np.min(ss[p])
            - np.max(ss[n])
        )

        margins.append(margin)

        is_perfect = float(
            margin > 0
        )

        perfect.append(
            is_perfect
        )

        k = int(np.sum(yy))

        top = np.argsort(-ss)[:k]

        exact = float(
            set(top.tolist())
            ==
            set(p.tolist())
        )

        exact_topk.append(
            exact
        )

    out["mean_within_system_auc"] = (
        float(np.mean(system_aucs))
        if system_aucs else None
    )

    out["perfect_order_rate"] = (
        float(np.mean(perfect))
        if perfect else None
    )

    out["exact_topK_oracleK_rate"] = (
        float(np.mean(exact_topk))
        if exact_topk else None
    )

    out["mean_full_separation_margin"] = (
        float(np.mean(margins))
        if margins else None
    )

    return out


def matched_pair_metrics(rows, scores):
    groups = defaultdict(list)

    for i, x in enumerate(rows):
        pid = x.get("pair_id")

        if pid is not None:
            groups[str(pid)].append(i)

    wins = []
    margins = []

    by_role = defaultdict(list)

    for pid, idxs in groups.items():
        if len(idxs) != 2:
            continue

        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]

        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if len(pos) != 1 or len(neg) != 1:
            continue

        pi = pos[0]
        ni = neg[0]

        m = float(
            scores[pi]
            - scores[ni]
        )

        if m > 0:
            w = 1.0
        elif m < 0:
            w = 0.0
        else:
            w = 0.5

        wins.append(w)
        margins.append(m)

        role = str(
            rows[pi]["target_role"]
        )

        by_role[role].append(
            (w, m)
        )

    per_role = {}

    for role, vals in sorted(by_role.items()):
        per_role[role] = {
            "pairs": len(vals),
            "pair_accuracy":
                float(np.mean([x[0] for x in vals])),
            "mean_margin":
                float(np.mean([x[1] for x in vals])),
        }

    return {
        "pairs": len(wins),
        "pair_accuracy":
            float(np.mean(wins)),
        "mean_margin":
            float(np.mean(margins)),
        "median_margin":
            float(np.median(margins)),
        "per_role": per_role,
    }


# ============================================================
# MODELS
# ============================================================

class SharedLinear(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            20,
            1,
        )

    def forward(self, g, c=None):
        return self.linear(g).squeeze(-1)


class HyperLinear(nn.Module):
    """
    Role/query semantics generate the decoding rule
    applied to the 20D behavioral geometry.

    No role IDs.
    No per-role heads.
    No per-role calibration.
    """

    def __init__(
        self,
        context_dim,
        rank=HYPER_RANK,
    ):
        super().__init__()

        self.base_w = nn.Parameter(
            torch.zeros(20)
        )

        self.base_b = nn.Parameter(
            torch.zeros(())
        )

        self.context = nn.Linear(
            context_dim,
            rank,
            bias=True,
        )

        self.delta_w = nn.Linear(
            rank,
            20,
            bias=False,
        )

        self.delta_b = nn.Linear(
            rank,
            1,
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

        nn.init.normal_(
            self.delta_b.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(self, g, c):
        h = torch.tanh(
            self.context(c)
        )

        dw = torch.tanh(
            self.delta_w(h)
        )

        db = 0.25 * self.delta_b(
            h
        ).squeeze(-1)

        w = (
            self.base_w[None, :]
            + dw
        )

        return (
            (g * w).sum(dim=1)
            + self.base_b
            + db
        )


# ============================================================
# TRAINING
# ============================================================

def geom_array(data, mode):
    if mode == "full":
        return data.G

    if mode == "query_only":
        return data.G_query_only

    raise ValueError(mode)


def context_array(data, kind):
    if kind == "shared":
        return None

    if kind == "hyper_role":
        return data.C_role

    if kind == "hyper_rq":
        return data.C_role_query

    raise ValueError(kind)


def fit_scaler(arrays):
    X = np.concatenate(
        arrays,
        axis=0,
    ).astype(np.float32)

    mu = X.mean(
        axis=0,
    ).astype(np.float32)

    sd = X.std(
        axis=0,
    ).astype(np.float32)

    positive = sd[
        sd > 1e-6
    ]

    if len(positive):
        floor = max(
            1e-3,
            0.05 * float(
                np.median(positive)
            ),
        )
    else:
        floor = 1e-3

    sd = np.maximum(
        sd,
        floor,
    ).astype(np.float32)

    return mu, sd


def make_rank_groups(rows):
    by_system = defaultdict(list)

    for i, x in enumerate(rows):
        by_system[
            str(x["system_id"])
        ].append(i)

    groups = []

    for _, idxs in by_system.items():
        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]

        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if not pos or not neg:
            continue

        groups.append(
            (
                torch.tensor(
                    pos,
                    device=DEVICE,
                    dtype=torch.long,
                ),
                torch.tensor(
                    neg,
                    device=DEVICE,
                    dtype=torch.long,
                ),
            )
        )

    return groups


def make_cf_pairs(rows):
    by_pair = defaultdict(list)

    for i, x in enumerate(rows):
        pid = x.get("pair_id")

        if pid is not None:
            by_pair[str(pid)].append(i)

    out = []

    for _, idxs in by_pair.items():
        if len(idxs) != 2:
            continue

        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]

        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if len(pos) == 1 and len(neg) == 1:
            out.append(
                (
                    pos[0],
                    neg[0],
                )
            )

    return out


def rank_loss(scores, groups):
    vals = []

    for p, n in groups:
        sp = scores[p]
        sn = scores[n]

        vals.append(
            F.softplus(
                sn[:, None]
                - sp[None, :]
            ).mean()
        )

    if not vals:
        return scores.sum() * 0.0

    return torch.stack(
        vals
    ).mean()


def balanced_bce(scores, y):
    p = y == 1
    n = y == 0

    terms = []

    if torch.any(p):
        terms.append(
            F.binary_cross_entropy_with_logits(
                scores[p],
                torch.ones_like(
                    scores[p]
                ),
            )
        )

    if torch.any(n):
        terms.append(
            F.binary_cross_entropy_with_logits(
                scores[n],
                torch.zeros_like(
                    scores[n]
                ),
            )
        )

    return torch.stack(
        terms
    ).mean()


def train_model(
    kind,
    data0,
    idx0,
    seed,
    geom_mode="full",
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

    G0_raw = geom_array(
        data0,
        geom_mode,
    )[idx0]

    scaler_inputs = [
        G0_raw
    ]

    rows1 = None
    G1_raw = None

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

        G1_raw = geom_array(
            data1,
            geom_mode,
        )[idx1]

        scaler_inputs.append(
            G1_raw
        )

    mu, sd = fit_scaler(
        scaler_inputs
    )

    G0 = torch.tensor(
        (G0_raw - mu) / sd,
        dtype=torch.float32,
        device=DEVICE,
    )

    C0_np = context_array(
        data0,
        kind,
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

    if kind == "shared":
        model = SharedLinear()

    else:
        model = HyperLinear(
            context_dim=C0.shape[1],
        )

    model = model.to(
        DEVICE
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    if rows1 is not None:
        G1 = torch.tensor(
            (G1_raw - mu) / sd,
            dtype=torch.float32,
            device=DEVICE,
        )

        C1_np = context_array(
            data1,
            kind,
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
        G1 = None
        C1 = None
        y1 = None
        cf_pairs = []

    model.train()

    for epoch in range(EPOCHS):
        opt.zero_grad(
            set_to_none=True
        )

        s0 = model(
            G0,
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
            G1 is not None
            and cf_pairs
        ):
            s1 = model(
                G1,
                C1,
            )

            cf = torch.stack([
                F.softplus(
                    s1[ni]
                    - s1[pi]
                )
                for pi, ni
                in cf_pairs
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
                f"nonfinite loss: "
                f"kind={kind} seed={seed} "
                f"epoch={epoch}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        opt.step()

    return model, mu, sd


@torch.no_grad()
def predict(
    model,
    data,
    idx,
    mu,
    sd,
    kind,
    geom_mode="full",
    override_geom=None,
):
    idx = np.asarray(
        idx,
        dtype=int,
    )

    if override_geom is None:
        Graw = geom_array(
            data,
            geom_mode,
        )[idx]

    else:
        Graw = override_geom[
            idx
        ]

    G = torch.tensor(
        (Graw - mu) / sd,
        dtype=torch.float32,
        device=DEVICE,
    )

    Cnp = context_array(
        data,
        kind,
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
        model(
            G,
            C,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


# ============================================================
# FOLDS
# ============================================================

def make_folds(items, n, seed):
    items = list(
        sorted(items)
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        items
    )

    folds = [
        []
        for _ in range(n)
    ]

    for i, x in enumerate(items):
        folds[i % n].append(x)

    return folds


def indices_where(rows, roles=None, systems=None):
    out = []

    role_set = (
        None
        if roles is None
        else set(roles)
    )

    sys_set = (
        None
        if systems is None
        else set(systems)
    )

    for i, x in enumerate(rows):
        if (
            role_set is not None
            and str(x["target_role"])
            not in role_set
        ):
            continue

        if (
            sys_set is not None
            and str(x["system_id"])
            not in sys_set
        ):
            continue

        out.append(i)

    return np.asarray(
        out,
        dtype=int,
    )


# ============================================================
# SHUFFLED OUTPUT CONTROL
# ============================================================

def shuffled_geom(
    data,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    out = np.empty_like(
        data.G
    )

    by_role = defaultdict(list)

    for i, x in enumerate(data.rows):
        by_role[
            str(x["target_role"])
        ].append(i)

    for role, idxs in by_role.items():
        idxs = np.asarray(
            idxs,
            dtype=int,
        )

        perm = idxs.copy()

        rng.shuffle(
            perm
        )

        for i, j in zip(
            idxs,
            perm,
        ):
            out[i] = geom_from_vectors(
                data.R[i],
                data.Q[i],
                data.O[j],
            )

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise SystemExit(
            f"{OUT} is non-empty; "
            "refusing overwrite."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = read_jsonl(
        TRAIN_PATH
    )

    val = read_jsonl(
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

    if len(t0) != 720:
        raise RuntimeError(
            f"T0 expected 720, got {len(t0)}"
        )

    if len(t1) != 190:
        raise RuntimeError(
            f"T1 expected 190, got {len(t1)}"
        )

    if len(val) != 760:
        raise RuntimeError(
            f"VAL expected 760, got {len(val)}"
        )

    roles = sorted({
        str(x["target_role"])
        for x in t0
    })

    systems = sorted({
        str(x["system_id"])
        for x in t0
    })

    print("=" * 100)
    print("HYPERSENSOR GENERAL V1")
    print("=" * 100)
    print("device:", DEVICE)
    print("T0 rows:", len(t0))
    print("T1 rows:", len(t1))
    print("VAL rows:", len(val))
    print("roles:", len(roles))
    print("T0 systems:", len(systems))
    print("INTERNAL TEST NOT LOADED")
    print()

    emb = BatchedEmbeddings(
        [t0, t1, val]
    )

    print("\nBuilding tensors...", flush=True)

    D0 = build_data(
        t0,
        emb,
    )

    D1 = build_data(
        t1,
        emb,
    )

    DD = build_data(
        val,
        emb,
    )

    role_folds = make_folds(
        roles,
        ROLE_FOLDS,
        seed=20260928,
    )

    system_folds = make_folds(
        systems,
        SYSTEM_FOLDS,
        seed=20260929,
    )

    print("\nROLE FOLDS:")
    for i, f in enumerate(
        role_folds,
        1,
    ):
        print(
            f"  fold {i}: {f}"
        )

    print("\nSYSTEM FOLD SIZES:")
    for i, f in enumerate(
        system_folds,
        1,
    ):
        print(
            f"  fold {i}: {len(f)}"
        )

    # --------------------------------------------------------
    # A. STRICT DOUBLE-CROSSFIT T0
    #
    # Every T0 row is predicted by a model that saw neither:
    # - that row's role
    # - that row's system
    # --------------------------------------------------------

    variants = {
        "shared_geom": (
            "shared",
            "full",
        ),
        "hyper_role_geom": (
            "hyper_role",
            "full",
        ),
        "hyper_rq_geom": (
            "hyper_rq",
            "full",
        ),
        "hyper_rq_query_only_control": (
            "hyper_rq",
            "query_only",
        ),
    }

    strict_sum = {
        k: np.zeros(
            len(t0),
            dtype=np.float64,
        )
        for k in variants
    }

    strict_n = {
        k: np.zeros(
            len(t0),
            dtype=np.int64,
        )
        for k in variants
    }

    strict_fold_reports = []

    print(
        "\n=== A. STRICT ROLE + SYSTEM DOUBLE-CROSSFIT T0 ===",
        flush=True,
    )

    for rf, held_roles in enumerate(
        role_folds,
        1,
    ):
        held_roles = set(
            held_roles
        )

        seen_roles = set(
            roles
        ) - held_roles

        for sf, held_systems in enumerate(
            system_folds,
            1,
        ):
            held_systems = set(
                held_systems
            )

            seen_systems = set(
                systems
            ) - held_systems

            tr_idx = indices_where(
                t0,
                roles=seen_roles,
                systems=seen_systems,
            )

            ev_idx = indices_where(
                t0,
                roles=held_roles,
                systems=held_systems,
            )

            print(
                f"roleFold={rf}/{ROLE_FOLDS} "
                f"sysFold={sf}/{SYSTEM_FOLDS} "
                f"train={len(tr_idx)} eval={len(ev_idx)}",
                flush=True,
            )

            fold_entry = {
                "role_fold": rf,
                "system_fold": sf,
                "held_roles":
                    sorted(held_roles),
                "held_systems":
                    sorted(held_systems),
                "train_rows":
                    int(len(tr_idx)),
                "eval_rows":
                    int(len(ev_idx)),
            }

            for name, (
                kind,
                geom_mode,
            ) in variants.items():

                fold_scores = []

                for seed in SEEDS:
                    model, mu, sd = train_model(
                        kind=kind,
                        data0=D0,
                        idx0=tr_idx,
                        seed=seed,
                        geom_mode=geom_mode,
                    )

                    s = predict(
                        model=model,
                        data=D0,
                        idx=ev_idx,
                        mu=mu,
                        sd=sd,
                        kind=kind,
                        geom_mode=geom_mode,
                    )

                    fold_scores.append(
                        s
                    )

                    del model

                fs = np.mean(
                    np.stack(
                        fold_scores,
                        axis=0,
                    ),
                    axis=0,
                )

                strict_sum[name][ev_idx] += fs
                strict_n[name][ev_idx] += 1

                fold_entry[name] = {
                    "global_auc":
                        auc_safe(
                            [
                                int(t0[i]["label"])
                                for i in ev_idx
                            ],
                            fs,
                        )
                }

            strict_fold_reports.append(
                fold_entry
            )

    strict_scores = {}

    strict_metrics = {}

    for name in variants:
        if np.any(
            strict_n[name] != 1
        ):
            vals, counts = np.unique(
                strict_n[name],
                return_counts=True,
            )

            raise RuntimeError(
                f"{name}: bad strict coverage: "
                f"{dict(zip(vals.tolist(), counts.tolist()))}"
            )

        strict_scores[name] = (
            strict_sum[name]
            / strict_n[name]
        )

        strict_metrics[name] = evaluate(
            t0,
            strict_scores[name],
        )

    # --------------------------------------------------------
    # B. ROLE-HELDOUT TRANSFER:
    #
    # Train on ALL T0 systems but only seen roles.
    # Evaluate:
    # - T1 held-out roles
    # - D held-out roles (D systems are already disjoint)
    #
    # Also train T0+T1(seen roles) and test on D held-out roles.
    # --------------------------------------------------------

    transfer_names = [
        "shared_geom",
        "hyper_role_geom",
        "hyper_rq_geom",
        "hyper_rq_query_only_control",
    ]

    d_sum = {
        k: np.zeros(
            len(val),
            dtype=np.float64,
        )
        for k in transfer_names
    }

    d_n = {
        k: np.zeros(
            len(val),
            dtype=np.int64,
        )
        for k in transfer_names
    }

    t1_sum = {
        k: np.zeros(
            len(t1),
            dtype=np.float64,
        )
        for k in transfer_names
    }

    t1_n = {
        k: np.zeros(
            len(t1),
            dtype=np.int64,
        )
        for k in transfer_names
    }

    # Multi-family model:
    # T0 system-ranking + T1 matched CF,
    # only on roles NOT in held-out role fold.
    d_multi_sum = np.zeros(
        len(val),
        dtype=np.float64,
    )

    d_multi_n = np.zeros(
        len(val),
        dtype=np.int64,
    )

    # Shuffled-response controls for main HyperRQ T0-only model.
    shuffled_D = [
        shuffled_geom(
            DD,
            seed=91000 + i,
        )
        for i in range(
            SHUFFLE_REPEATS
        )
    ]

    d_shuffle_sum = [
        np.zeros(
            len(val),
            dtype=np.float64,
        )
        for _ in range(
            SHUFFLE_REPEATS
        )
    ]

    d_shuffle_n = [
        np.zeros(
            len(val),
            dtype=np.int64,
        )
        for _ in range(
            SHUFFLE_REPEATS
        )
    ]

    print(
        "\n=== B. ROLE-HELDOUT T0 -> T1 / D TRANSFER ===",
        flush=True,
    )

    for rf, held_roles in enumerate(
        role_folds,
        1,
    ):
        held_roles = set(
            held_roles
        )

        seen_roles = set(
            roles
        ) - held_roles

        tr0_idx = indices_where(
            t0,
            roles=seen_roles,
        )

        ev1_idx = indices_where(
            t1,
            roles=held_roles,
        )

        evd_idx = indices_where(
            val,
            roles=held_roles,
        )

        tr1_idx = indices_where(
            t1,
            roles=seen_roles,
        )

        print(
            f"roleFold={rf}/{ROLE_FOLDS} "
            f"T0train={len(tr0_idx)} "
            f"T1eval={len(ev1_idx)} "
            f"Deval={len(evd_idx)} "
            f"T1aux={len(tr1_idx)}",
            flush=True,
        )

        for name in transfer_names:
            kind, geom_mode = variants[
                name
            ]

            s_d_all = []
            s_1_all = []

            shuf_seed_scores = [
                []
                for _ in range(
                    SHUFFLE_REPEATS
                )
            ]

            for seed in SEEDS:
                model, mu, sd = train_model(
                    kind=kind,
                    data0=D0,
                    idx0=tr0_idx,
                    seed=seed,
                    geom_mode=geom_mode,
                )

                sd_pred = predict(
                    model,
                    DD,
                    evd_idx,
                    mu,
                    sd,
                    kind,
                    geom_mode,
                )

                s1_pred = predict(
                    model,
                    D1,
                    ev1_idx,
                    mu,
                    sd,
                    kind,
                    geom_mode,
                )

                s_d_all.append(
                    sd_pred
                )

                s_1_all.append(
                    s1_pred
                )

                if name == "hyper_rq_geom":
                    for si, Gsh in enumerate(
                        shuffled_D
                    ):
                        sh = predict(
                            model,
                            DD,
                            evd_idx,
                            mu,
                            sd,
                            kind="hyper_rq",
                            geom_mode="full",
                            override_geom=Gsh,
                        )

                        shuf_seed_scores[
                            si
                        ].append(
                            sh
                        )

                del model

            d_pred = np.mean(
                np.stack(
                    s_d_all,
                    axis=0,
                ),
                axis=0,
            )

            t1_pred = np.mean(
                np.stack(
                    s_1_all,
                    axis=0,
                ),
                axis=0,
            )

            d_sum[name][evd_idx] += d_pred
            d_n[name][evd_idx] += 1

            t1_sum[name][ev1_idx] += t1_pred
            t1_n[name][ev1_idx] += 1

            if name == "hyper_rq_geom":
                for si in range(
                    SHUFFLE_REPEATS
                ):
                    shp = np.mean(
                        np.stack(
                            shuf_seed_scores[
                                si
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )

                    d_shuffle_sum[
                        si
                    ][evd_idx] += shp

                    d_shuffle_n[
                        si
                    ][evd_idx] += 1

        # ----------------------------------------------
        # Multi-family HyperRQ:
        # T0 + T1 seen roles -> D unseen roles
        # ----------------------------------------------

        multi_seed_scores = []

        for seed in SEEDS:
            model, mu, sd = train_model(
                kind="hyper_rq",
                data0=D0,
                idx0=tr0_idx,
                seed=seed,
                geom_mode="full",
                data1=D1,
                idx1=tr1_idx,
            )

            s = predict(
                model,
                DD,
                evd_idx,
                mu,
                sd,
                kind="hyper_rq",
                geom_mode="full",
            )

            multi_seed_scores.append(
                s
            )

            del model

        multi = np.mean(
            np.stack(
                multi_seed_scores,
                axis=0,
            ),
            axis=0,
        )

        d_multi_sum[
            evd_idx
        ] += multi

        d_multi_n[
            evd_idx
        ] += 1

    # Verify transfer coverage.
    d_scores = {}
    t1_scores = {}

    for name in transfer_names:
        if np.any(
            d_n[name] != 1
        ):
            raise RuntimeError(
                f"D coverage failure: {name}"
            )

        if np.any(
            t1_n[name] != 1
        ):
            raise RuntimeError(
                f"T1 coverage failure: {name}"
            )

        d_scores[name] = (
            d_sum[name]
            / d_n[name]
        )

        t1_scores[name] = (
            t1_sum[name]
            / t1_n[name]
        )

    if np.any(
        d_multi_n != 1
    ):
        raise RuntimeError(
            "D multi-family coverage failure"
        )

    d_multi_scores = (
        d_multi_sum
        / d_multi_n
    )

    d_shuffle_metrics = []

    for si in range(
        SHUFFLE_REPEATS
    ):
        if np.any(
            d_shuffle_n[si] != 1
        ):
            raise RuntimeError(
                f"shuffle coverage failure {si}"
            )

        ss = (
            d_shuffle_sum[si]
            / d_shuffle_n[si]
        )

        d_shuffle_metrics.append(
            evaluate(
                val,
                ss,
            )
        )

    d_metrics = {
        name: evaluate(
            val,
            d_scores[name],
        )
        for name in transfer_names
    }

    d_metrics[
        "hyper_rq_T0plusT1_to_D"
    ] = evaluate(
        val,
        d_multi_scores,
    )

    t1_metrics = {
        name: matched_pair_metrics(
            t1,
            t1_scores[name],
        )
        for name in transfer_names
    }

    shuffle_summary = {
        "repeats":
            SHUFFLE_REPEATS,

        "mean_global_auc":
            float(np.mean([
                x["global_auc"]
                for x in d_shuffle_metrics
            ])),

        "mean_role_auc":
            float(np.mean([
                x["mean_role_auc"]
                for x in d_shuffle_metrics
            ])),

        "mean_within_system_auc":
            float(np.mean([
                x["mean_within_system_auc"]
                for x in d_shuffle_metrics
            ])),

        "all_runs":
            d_shuffle_metrics,
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary = {
        "version":
            "hypersensor_general_v1",

        "device":
            DEVICE,

        "internal_test_used":
            False,

        "protocol": {
            "strict_T0":
                "5 role folds x 3 system folds; "
                "every row predicted by model that saw "
                "neither its role nor its system",

            "T1_transfer":
                "T0-only training; target role entirely unseen",

            "D_transfer":
                "target role entirely unseen; "
                "D systems disjoint from T0",

            "D_multifamily":
                "T0 system-ranking + T1 matched-CF "
                "on seen roles only; D target roles unseen",

            "role_id_used":
                False,

            "role_specific_heads":
                False,

            "role_specific_calibration":
                False,
        },

        "model": {
            "geometry_dim": 20,
            "embedding":
                "sentence-transformers/all-MiniLM-L6-v2",
            "hyper_rank":
                HYPER_RANK,
            "epochs":
                EPOCHS,
            "lr":
                LR,
            "weight_decay":
                WEIGHT_DECAY,
            "seeds":
                SEEDS,
        },

        "role_folds":
            role_folds,

        "system_folds":
            system_folds,

        "strict_T0_double_holdout":
            strict_metrics,

        "T0_to_T1_unseen_role":
            t1_metrics,

        "T0_to_D_unseen_role":
            d_metrics,

        "D_shuffled_output_control":
            shuffle_summary,

        "strict_fold_reports":
            strict_fold_reports,
    }

    dump(
        OUT / "summary.json",
        summary,
    )

    np.savez_compressed(
        OUT / "scores.npz",

        t0_shared=
            strict_scores[
                "shared_geom"
            ],

        t0_hyper_role=
            strict_scores[
                "hyper_role_geom"
            ],

        t0_hyper_rq=
            strict_scores[
                "hyper_rq_geom"
            ],

        t0_query_only=
            strict_scores[
                "hyper_rq_query_only_control"
            ],

        d_shared=
            d_scores[
                "shared_geom"
            ],

        d_hyper_role=
            d_scores[
                "hyper_role_geom"
            ],

        d_hyper_rq=
            d_scores[
                "hyper_rq_geom"
            ],

        d_query_only=
            d_scores[
                "hyper_rq_query_only_control"
            ],

        d_hyper_rq_multifamily=
            d_multi_scores,

        t1_hyper_rq=
            t1_scores[
                "hyper_rq_geom"
            ],
    )

    # --------------------------------------------------------
    # PRINT FINAL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("FINAL GENERALIZATION SUMMARY")
    print("=" * 110)

    def line(name, m):
        print(
            f"{name:<38} "
            f"global={m['global_auc']:.4f} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"topK*={m['exact_topK_oracleK_rate']:.4f}"
        )

    print("\n--- STRICT T0: UNSEEN ROLE + UNSEEN SYSTEM ---")

    for name in [
        "shared_geom",
        "hyper_role_geom",
        "hyper_rq_geom",
        "hyper_rq_query_only_control",
    ]:
        line(
            name,
            strict_metrics[name],
        )

    print("\n--- D: UNSEEN ROLE + NEW SYSTEM + NEW WORDING ---")

    for name in [
        "shared_geom",
        "hyper_role_geom",
        "hyper_rq_geom",
        "hyper_rq_query_only_control",
        "hyper_rq_T0plusT1_to_D",
    ]:
        line(
            name,
            d_metrics[name],
        )

    print("\n--- T1 MATCHED TRANSFER: UNSEEN ROLE ---")

    for name in [
        "shared_geom",
        "hyper_role_geom",
        "hyper_rq_geom",
        "hyper_rq_query_only_control",
    ]:
        m = t1_metrics[name]

        print(
            f"{name:<38} "
            f"pairAcc={m['pair_accuracy']:.4f} "
            f"meanMargin={m['mean_margin']:+.4f}"
        )

    print("\n--- D SHUFFLED OUTPUT CONTROL ---")

    print(
        "hyper_rq real D sysAUC       = "
        f"{d_metrics['hyper_rq_geom']['mean_within_system_auc']:.4f}"
    )

    print(
        "hyper_rq shuffled mean sysAUC= "
        f"{shuffle_summary['mean_within_system_auc']:.4f}"
    )

    print()
    print("--- PER-ROLE STRICT T0 HyperRQ ---")

    for role, auc in sorted(
        strict_metrics[
            "hyper_rq_geom"
        ][
            "per_role_auc"
        ].items(),
        key=lambda kv: kv[1],
    ):
        print(
            f"{role:<30} "
            f"AUC={auc:.4f}"
        )

    print()
    print("--- PER-ROLE D HyperRQ ---")

    for role, auc in sorted(
        d_metrics[
            "hyper_rq_geom"
        ][
            "per_role_auc"
        ].items(),
        key=lambda kv: kv[1],
    ):
        print(
            f"{role:<30} "
            f"AUC={auc:.4f}"
        )

    print()
    print("INTERNAL TEST UNTOUCHED.")
    print("Summary:", OUT / "summary.json")


if __name__ == "__main__":
    main()
