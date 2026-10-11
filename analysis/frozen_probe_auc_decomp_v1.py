#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer

from v11.role_universe import ROLE_DESCRIPTIONS


SRC = Path(
    "artifacts/frozen_probe_auc_v1/responses.jsonl"
)

OUT = Path(
    "artifacts/frozen_probe_auc_decomp_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
]


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


rows = [
    x
    for x in read_jsonl(SRC)
    if x.get(
        "execution_status"
    ) == "ok"
]


assert len(rows) == 320, len(rows)


print(
    "rows:",
    len(rows),
    flush=True,
)


# ------------------------------------------------------------------
# Exact public feature family from SharedPresenceDetector,
# but batched so this diagnostic is much faster.
# ------------------------------------------------------------------

print(
    "loading MiniLM...",
    flush=True,
)

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)


role_texts = {
    role:
        ROLE_DESCRIPTIONS[
            role
        ]
    for role in ROLES
}


unique_queries = sorted(
    {
        str(x["query"])
        for x in rows
    }
)

unique_outputs = sorted(
    {
        str(
            x["final_output"]
        )
        for x in rows
    }
)


print(
    "unique queries:",
    len(unique_queries),
    flush=True,
)

print(
    "unique outputs:",
    len(unique_outputs),
    flush=True,
)


def enc(texts):
    a = embedder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

    return np.asarray(
        a,
        dtype=np.float32,
    )


role_keys = list(
    role_texts
)

role_embs = enc(
    [
        role_texts[r]
        for r in role_keys
    ]
)

query_embs = enc(
    unique_queries
)

output_embs = enc(
    unique_outputs
)


RE = {
    r:
        role_embs[i]
    for i, r in enumerate(
        role_keys
    )
}

QE = {
    q:
        query_embs[i]
    for i, q in enumerate(
        unique_queries
    )
}

OE = {
    o:
        output_embs[i]
    for i, o in enumerate(
        unique_outputs
    )
}


def cosine(a, b):
    return float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )


def style(text):
    t = str(
        text or ""
    )

    words = re.findall(
        r"\b[\w'-]+\b",
        t,
        flags=re.UNICODE,
    )

    chars = max(
        len(t),
        1,
    )

    return np.asarray(
        [
            math.log1p(
                len(t)
            ),
            math.log1p(
                len(words)
            ),
            math.log1p(
                t.count("\n")
            ),
            math.log1p(
                t.count("- ")
                + t.count("* ")
            ),
            float(
                "```" in t
            ),
            sum(
                ch.isdigit()
                for ch in t
            )
            / chars,
            t.count(":")
            / chars,
            t.count("|")
            / chars,
            (
                t.count("{")
                + t.count("}")
            )
            / chars,
        ],
        dtype=np.float32,
    )


def features(row):
    role = str(
        row["target_role"]
    )

    query = str(
        row["query"]
    )

    output = str(
        row["final_output"]
    )

    r = RE[role]
    q = QE[query]
    o = OE[output]

    scalars = np.asarray(
        [
            cosine(o, r),
            cosine(o, q),
            cosine(q, r),
            float(
                np.linalg.norm(
                    o - r
                )
            ),
            float(
                np.linalg.norm(
                    o - q
                )
            ),
            cosine(
                o - q,
                r,
            ),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            scalars,
            style(output),
            (
                o * r
            ).astype(
                np.float32
            ),
            (
                o * q
            ).astype(
                np.float32
            ),
            np.abs(
                o - r
            ).astype(
                np.float32
            ),
            np.abs(
                o - q
            ).astype(
                np.float32
            ),
        ]
    )


print(
    "building features...",
    flush=True,
)

X = np.stack(
    [
        features(x)
        for x in rows
    ]
)


y_installed = np.asarray(
    [
        int(
            x[
                "target_present_eval_only"
            ]
        )
        for x in rows
    ],
    dtype=int,
)


y_selected = np.asarray(
    [
        int(
            x[
                "target_selected_eval_only"
            ]
        )
        for x in rows
    ],
    dtype=int,
)


groups = np.asarray(
    [
        str(
            x["comp_id"]
        )
        for x in rows
    ],
    dtype=object,
)


roles = np.asarray(
    [
        str(
            x["target_role"]
        )
        for x in rows
    ],
    dtype=object,
)


styles = np.asarray(
    [
        str(
            x["style"]
        )
        for x in rows
    ],
    dtype=object,
)


families = np.asarray(
    [
        str(
            x["family"]
        )
        for x in rows
    ],
    dtype=object,
)


reps = np.asarray(
    [
        int(
            x["rep"]
        )
        for x in rows
    ],
    dtype=int,
)


# ------------------------------------------------------------------
# Reproduce the exact 5-fold composition-grouped detector.
# ------------------------------------------------------------------

def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )


oof = np.full(
    len(rows),
    np.nan,
    dtype=float,
)


gkf = GroupKFold(
    n_splits=5
)


for fold, (
    tr,
    te,
) in enumerate(
    gkf.split(
        X,
        y_installed,
        groups,
    ),
    1,
):

    model = make_model()

    model.fit(
        X[tr],
        y_installed[tr],
    )

    oof[te] = (
        model.decision_function(
            X[te]
        )
    )

    print(
        f"fold {fold}: "
        f"train={len(tr)} "
        f"test={len(te)}",
        flush=True,
    )


assert np.isfinite(
    oof
).all()


print(
    "\nGLOBAL CHECK:",
    flush=True,
)

print(
    "installed OOF AUC:",
    roc_auc_score(
        y_installed,
        oof,
    ),
    flush=True,
)


def auc_or_none(
    yy,
    ss,
):
    yy = np.asarray(
        yy,
        dtype=int,
    )

    ss = np.asarray(
        ss,
        dtype=float,
    )

    if (
        len(yy) < 2
        or len(
            np.unique(
                yy
            )
        ) < 2
    ):
        return None

    return float(
        roc_auc_score(
            yy,
            ss,
        )
    )


def fmt(v):
    if v is None:
        return "NA"

    return f"{v:.4f}"


report = {}


print(
    "\n"
    + "=" * 122
)

print(
    "ROLE DECOMPOSITION"
)

print(
    "=" * 122
)

print(
    f"{'role':<12} "
    f"{'route':>8} "
    f"{'instAUC':>9} "
    f"{'actAUC':>9} "
    f"{'sel-v-abs':>10} "
    f"{'miss-v-abs':>11} "
    f"{'oracleRoute':>12}"
)


for role in ROLES:

    rm = (
        roles == role
    )

    pos = (
        rm
        & (
            y_installed
            == 1
        )
    )

    neg = (
        rm
        & (
            y_installed
            == 0
        )
    )

    selected_pos = (
        pos
        & (
            y_selected
            == 1
        )
    )

    missed_pos = (
        pos
        & (
            y_selected
            == 0
        )
    )


    route_n = int(
        pos.sum()
    )

    route_hits = int(
        selected_pos.sum()
    )

    route_rate = (
        route_hits
        / route_n
        if route_n
        else None
    )


    installed_auc = auc_or_none(
        y_installed[rm],
        oof[rm],
    )


    # How much the installed-presence detector score
    # tracks ACTUAL activation.
    activation_auc = auc_or_none(
        y_selected[rm],
        oof[rm],
    )


    # Strongest direct sensor diagnostic:
    # only target-selected positives versus target-absent negatives.
    m_sel_abs = (
        selected_pos
        | neg
    )

    sel_vs_abs = auc_or_none(
        y_installed[
            m_sel_abs
        ],
        oof[
            m_sel_abs
        ],
    )


    # If target was installed but router missed it,
    # can output still separate installation from absence?
    m_miss_abs = (
        missed_pos
        | neg
    )

    miss_vs_abs = auc_or_none(
        y_installed[
            m_miss_abs
        ],
        oof[
            m_miss_abs
        ],
    )


    # Oracle that knows only whether the target
    # was actually selected.
    routing_oracle_auc = (
        auc_or_none(
            y_installed[rm],
            y_selected[rm],
        )
    )


    def mean_score(mask):
        if not mask.any():
            return None

        return float(
            oof[mask].mean()
        )


    role_report = {
        "present_n":
            route_n,

        "routing_hits":
            route_hits,

        "routing_hit_rate":
            route_rate,

        "installed_presence_auc":
            installed_auc,

        "activation_auc_using_installed_sensor_score":
            activation_auc,

        "selected_present_vs_absent_auc":
            sel_vs_abs,

        "missed_present_vs_absent_auc":
            miss_vs_abs,

        "routing_only_oracle_auc":
            routing_oracle_auc,

        "mean_score_selected_present":
            mean_score(
                selected_pos
            ),

        "mean_score_missed_present":
            mean_score(
                missed_pos
            ),

        "mean_score_absent":
            mean_score(
                neg
            ),
    }


    # ----------------------------------------------------------
    # Per-probe routing + AUC
    # ----------------------------------------------------------

    probe_rows = []

    pairs = sorted(
        {
            (
                str(
                    rows[i][
                        "family"
                    ]
                ),
                int(
                    rows[i][
                        "rep"
                    ]
                ),
            )
            for i in np.where(
                rm
            )[0]
        }
    )

    for family, rep in pairs:

        pm = (
            rm
            & (
                families
                == family
            )
            & (
                reps
                == rep
            )
        )

        ppos = (
            pm
            & (
                y_installed
                == 1
            )
        )

        phits = int(
            (
                ppos
                & (
                    y_selected
                    == 1
                )
            ).sum()
        )

        pn = int(
            ppos.sum()
        )

        probe_rows.append({
            "family":
                family,

            "rep":
                rep,

            "present_n":
                pn,

            "routing_hits":
                phits,

            "routing_hit_rate":
                (
                    phits / pn
                    if pn
                    else None
                ),

            "installed_auc":
                auc_or_none(
                    y_installed[
                        pm
                    ],
                    oof[
                        pm
                    ],
                ),
        })


    # ----------------------------------------------------------
    # Per-router-style routing + AUC
    # ----------------------------------------------------------

    style_rows = []

    for style_name in sorted(
        set(
            styles[
                rm
            ].tolist()
        )
    ):

        sm = (
            rm
            & (
                styles
                == style_name
            )
        )

        spos = (
            sm
            & (
                y_installed
                == 1
            )
        )

        shits = int(
            (
                spos
                & (
                    y_selected
                    == 1
                )
            ).sum()
        )

        sn = int(
            spos.sum()
        )

        style_rows.append({
            "style":
                style_name,

            "present_n":
                sn,

            "routing_hits":
                shits,

            "routing_hit_rate":
                (
                    shits / sn
                    if sn
                    else None
                ),

            "installed_auc":
                auc_or_none(
                    y_installed[
                        sm
                    ],
                    oof[
                        sm
                    ],
                ),
        })


    role_report[
        "probes"
    ] = probe_rows

    role_report[
        "styles"
    ] = style_rows

    report[
        role
    ] = role_report


    print(
        f"{role:<12} "
        f"{fmt(route_rate):>8} "
        f"{fmt(installed_auc):>9} "
        f"{fmt(activation_auc):>9} "
        f"{fmt(sel_vs_abs):>10} "
        f"{fmt(miss_vs_abs):>11} "
        f"{fmt(routing_oracle_auc):>12}"
    )


print(
    "\n"
    + "=" * 122
)

print(
    "PER-PROBE DIAGNOSTIC"
)

print(
    "=" * 122
)


for role in ROLES:

    print(
        f"\n### {role}"
    )

    for x in report[
        role
    ][
        "probes"
    ]:

        print(
            f"  {x['family']:<10} "
            f"rep={x['rep']} "
            f"route="
            f"{x['routing_hits']}/"
            f"{x['present_n']} "
            f"({fmt(x['routing_hit_rate'])}) "
            f"installedAUC="
            f"{fmt(x['installed_auc'])}"
        )


print(
    "\n"
    + "=" * 122
)

print(
    "PER-STYLE DIAGNOSTIC"
)

print(
    "=" * 122
)


for role in ROLES:

    print(
        f"\n### {role}"
    )

    for x in report[
        role
    ][
        "styles"
    ]:

        print(
            f"  {x['style']} "
            f"route="
            f"{x['routing_hits']}/"
            f"{x['present_n']} "
            f"({fmt(x['routing_hit_rate'])}) "
            f"installedAUC="
            f"{fmt(x['installed_auc'])}"
        )


summary = {
    "n":
        len(rows),

    "global_installed_oof_auc":
        float(
            roc_auc_score(
                y_installed,
                oof,
            )
        ),

    "global_activation_auc_using_installed_sensor_score":
        float(
            roc_auc_score(
                y_selected,
                oof,
            )
        ),

    "roles":
        report,
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# Save row-level OOF scores for later diagnostics.
with (
    OUT
    / "rows_with_oof_scores.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for i, row in enumerate(
        rows
    ):

        z = {
            **row,

            "installed_presence_oof_score":
                float(
                    oof[i]
                ),
        }

        f.write(
            json.dumps(
                z,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "\nwrote:",
    OUT
    / "summary.json"
)

print(
    "wrote:",
    OUT
    / "rows_with_oof_scores.jsonl"
)
