from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


OUT = Path(
    "artifacts/e2e_probe_tournament_v1"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL = os.environ.get(
    "MAS_MODEL",
    "gpt-4o-mini",
)

API_WORKERS = int(
    os.environ.get(
        "E2E_TOURNAMENT_WORKERS",
        "8",
    )
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


TARGET_ROLES = [
    "design",
    "operations",
]


COMPOSITIONS = {
    "C01": [
        "code",
        "writing",
        "math",
    ],
    "C02": [
        "planning",
        "research",
        "data",
    ],
    "C03": [
        "design",
        "operations",
        "code",
    ],
    "C04": [
        "writing",
        "planning",
        "research",
    ],
    "C05": [
        "math",
        "data",
        "design",
    ],
    "C06": [
        "code",
        "planning",
        "data",
        "operations",
    ],
    "C07": [
        "writing",
        "math",
        "research",
        "design",
    ],
    "C08": [
        "code",
        "research",
        "design",
        "operations",
    ],
    "C09": [
        "writing",
        "planning",
        "data",
        "operations",
    ],
    "C10": [
        "code",
        "math",
        "planning",
        "research",
    ],
}

STYLES = [
    "s1",
    "s2",
]


ROUTER_POOL = Path(
    "artifacts/probe_router_tournament_v1/"
    "candidate_summary.jsonl"
)

CURRENT_BANK = Path(
    "artifacts/generator_frozen_v3/"
    "selected_bank.jsonl"
)

BASE_RESPONSES = Path(
    "artifacts/frozen_probe_auc_v2/"
    "responses.jsonl"
)

BASE_OOF = Path(
    "artifacts/frozen_probe_auc_decomp_v2/"
    "rows_with_oof_scores.jsonl"
)

RESULTS_PATH = (
    OUT
    / "responses.jsonl"
)


def read_jsonl(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    return [
        json.loads(x)
        for x in p.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


# ============================================================
# 1. Candidate pool
#
# Start from router tournament candidates,
# then add current-bank probes so the incumbent is included.
# ============================================================

candidate_map = {}


def add_candidate(
    *,
    role,
    family,
    rep,
    query,
    scenario="",
    source="",
):

    role = str(role)
    query = str(query).strip()

    if (
        role not in TARGET_ROLES
        or not query
    ):
        return

    cid = hashlib.sha1(
        (
            role
            + "\0"
            + query
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    candidate_map[
        (
            role,
            query,
        )
    ] = {
        "candidate_id":
            cid,

        "role":
            role,

        "family":
            str(family),

        "rep":
            int(rep),

        "query":
            query,

        "scenario":
            str(
                scenario
                or ""
            ),

        "source":
            source,
    }


for x in read_jsonl(
    ROUTER_POOL
):

    add_candidate(
        role=
            x[
                "target_role"
            ],

        family=
            x[
                "family"
            ],

        rep=
            x[
                "rep"
            ],

        query=
            x[
                "query"
            ],

        scenario=
            x.get(
                "scenario",
                "",
            ),

        source=
            "router_tournament",
    )


for x in read_jsonl(
    CURRENT_BANK
):

    if x.get(
        "role"
    ) not in TARGET_ROLES:
        continue

    add_candidate(
        role=
            x[
                "role"
            ],

        family=
            x[
                "family"
            ],

        rep=
            x[
                "rep"
            ],

        query=
            x[
                "query"
            ],

        scenario=
            x.get(
                "scenario",
                "",
            ),

        source=
            "current_bank",
    )


candidates = sorted(
    candidate_map.values(),
    key=lambda x: (
        TARGET_ROLES.index(
            x["role"]
        ),
        x["family"],
        x["rep"],
    ),
)


print(
    "===== E2E CANDIDATE POOL =====",
    flush=True,
)

for role in TARGET_ROLES:

    xs = [
        x
        for x in candidates
        if x[
            "role"
        ] == role
    ]

    print(
        role,
        "candidates =",
        len(xs),
        flush=True,
    )

    for x in xs:

        print(
            " ",
            x[
                "candidate_id"
            ],
            x[
                "family"
            ],
            "rep=",
            x[
                "rep"
            ],
            "source=",
            x[
                "source"
            ],
            flush=True,
        )


# ============================================================
# 2. 20 fixed DEV MAS systems
# ============================================================

systems = []

for comp_id, roles in (
    COMPOSITIONS.items()
):

    for style in STYLES:

        systems.append({
            "system_id":
                f"{comp_id}_{style}",

            "comp_id":
                comp_id,

            "style":
                style,

            "worker_roles":
                list(
                    roles
                ),
        })


# ============================================================
# 3. Resume-aware collection
# ============================================================

existing = {}


if RESULTS_PATH.exists():

    for x in read_jsonl(
        RESULTS_PATH
    ):

        if x.get(
            "status"
        ) != "ok":
            continue

        existing[
            (
                x[
                    "candidate_id"
                ],
                x[
                    "system_id"
                ],
            )
        ] = x


tasks = []

for cand in candidates:

    for sys in systems:

        key = (
            cand[
                "candidate_id"
            ],
            sys[
                "system_id"
            ],
        )

        if key in existing:
            continue

        tasks.append({
            **cand,
            **sys,

            "target_present_eval_only":
                int(
                    cand[
                        "role"
                    ]
                    in sys[
                        "worker_roles"
                    ]
                ),
        })


print()
print(
    "systems:",
    len(
        systems
    ),
    flush=True,
)

print(
    "candidates:",
    len(
        candidates
    ),
    flush=True,
)

print(
    "total expected interactions:",
    len(candidates)
    * len(systems),
    flush=True,
)

print(
    "existing:",
    len(existing),
    flush=True,
)

print(
    "pending:",
    len(tasks),
    flush=True,
)

print(
    "approx API calls pending:",
    2 * len(tasks),
    flush=True,
)


# ============================================================
# 4. Full MAS execution
# ============================================================

from targets.single_router_zoo_v11 import (
    validate_v11_design,
)

validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


def execute(task):

    spec = SingleRouterSystemSpec(
        system_id=
            task[
                "system_id"
            ],

        split=
            "e2e_probe_tournament_v1",

        execution_mode=
            "single",

        style=
            task[
                "style"
            ],

        worker_roles=
            tuple(
                task[
                    "worker_roles"
                ]
            ),
    )


    last = None

    for attempt in range(3):

        try:

            mas = SingleRouterFanoutMAS(
                spec,
                model=MODEL,
            )

            output = str(
                mas.query(
                    task[
                        "query"
                    ]
                )
            )

            trace = getattr(
                mas,
                "_last_trace",
                None,
            )

            selected = list(
                getattr(
                    trace,
                    "selected_roles",
                    (),
                )
                or ()
            )


            return {
                **task,

                "final_output":
                    output,

                "selected_roles_eval_only":
                    selected,

                "target_selected_eval_only":
                    int(
                        task[
                            "role"
                        ]
                        in selected
                    ),

                "status":
                    "ok",
            }


        except Exception as exc:

            last = exc

            if attempt < 2:

                time.sleep(
                    2 ** attempt
                )


    return {
        **task,

        "final_output":
            "",

        "selected_roles_eval_only":
            [],

        "target_selected_eval_only":
            0,

        "status":
            "error",

        "error":
            repr(
                last
            ),
    }


if tasks:

    with ThreadPoolExecutor(
        max_workers=
            API_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                execute,
                task,
            )
            for task in tasks
        ]


        with RESULTS_PATH.open(
            "a",
            encoding="utf-8",
        ) as f:

            for i, fut in enumerate(
                as_completed(
                    futures
                ),
                1,
            ):

                row = fut.result()

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()


                if (
                    i <= 20
                    or i % 25 == 0
                    or i
                    == len(tasks)
                ):

                    print(
                        f"[collect "
                        f"{i}/{len(tasks)}] "
                        f"{row['role']} "
                        f"{row['family']} "
                        f"rep={row['rep']} "
                        f"{row['system_id']} "
                        f"present="
                        f"{row['target_present_eval_only']} "
                        f"selected="
                        f"{row['target_selected_eval_only']} "
                        f"status="
                        f"{row['status']}",
                        flush=True,
                    )


# ============================================================
# 5. Reload complete tournament
# ============================================================

all_rows = [
    x
    for x in read_jsonl(
        RESULTS_PATH
    )
    if x.get(
        "status"
    ) == "ok"
]


dedup = {}

for x in all_rows:

    dedup[
        (
            x[
                "candidate_id"
            ],
            x[
                "system_id"
            ],
        )
    ] = x


all_rows = list(
    dedup.values()
)


print()
print(
    "===== COLLECTION COMPLETE =====",
    flush=True,
)

print(
    "successful unique interactions:",
    len(
        all_rows
    ),
    "/",
    len(candidates)
    * len(systems),
    flush=True,
)


# Require every candidate/system cell.
expected_keys = {
    (
        c[
            "candidate_id"
        ],
        s[
            "system_id"
        ],
    )
    for c in candidates
    for s in systems
}


actual_keys = {
    (
        x[
            "candidate_id"
        ],
        x[
            "system_id"
        ],
    )
    for x in all_rows
}


missing = (
    expected_keys
    - actual_keys
)


if missing:

    raise RuntimeError(
        f"Missing "
        f"{len(missing)} "
        f"candidate/system cells. "
        f"Re-run job to resume."
    )


# ============================================================
# 6. Reconstruct EXACT current detector features.
#
# The only labels here are from the old DEV bank.
# Candidate labels are NOT used to fit the detector.
# ============================================================

print()
print(
    "===== BUILDING FOLD-SPECIFIC SENSOR =====",
    flush=True,
)


from v8.embedding import (
    FrozenOutputEmbedderV8,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
)

from analysis.adaptive_fanout_count_v12 import (
    SharedPresenceDetector,
)


embedder = FrozenOutputEmbedderV8(
    device="cpu"
)

feature_builder = SharedPresenceDetector(
    embedder,
    ROLE_DESCRIPTIONS,
    c=0.05,
)


base_rows = [
    x
    for x in read_jsonl(
        BASE_RESPONSES
    )
    if (
        x.get(
            "execution_status",
            x.get(
                "status",
                "ok",
            ),
        )
        == "ok"
        and
        str(
            x.get(
                "final_output",
                "",
            )
        ).strip()
    )
]


assert len(
    base_rows
) == 320, len(
    base_rows
)


def comp_of(row):

    return str(
        row.get(
            "comp_id"
        )
        or str(
            row[
                "system_id"
            ]
        ).split(
            "_"
        )[0]
    )


print(
    "building base features...",
    flush=True,
)


X_base = np.stack(
    [
        feature_builder.features(
            str(
                x[
                    "target_role"
                ]
            ),
            str(
                x[
                    "query"
                ]
            ),
            str(
                x[
                    "final_output"
                ]
            ),
        )
        for x in base_rows
    ]
).astype(
    np.float32
)


y_base = np.asarray(
    [
        int(
            x[
                "target_present_eval_only"
            ]
        )
        for x in base_rows
    ],
    dtype=int,
)


g_base = np.asarray(
    [
        comp_of(
            x
        )
        for x in base_rows
    ],
    dtype=object,
)


w_base = np.ones(
    len(
        base_rows
    ),
    dtype=float,
)


print(
    "building tournament features...",
    flush=True,
)


X_tour = np.stack(
    [
        feature_builder.features(
            str(
                x[
                    "role"
                ]
            ),
            str(
                x[
                    "query"
                ]
            ),
            str(
                x[
                    "final_output"
                ]
            ),
        )
        for x in all_rows
    ]
).astype(
    np.float32
)


g_tour = np.asarray(
    [
        x[
            "comp_id"
        ]
        for x in all_rows
    ],
    dtype=object,
)


oof_score = np.full(
    len(
        all_rows
    ),
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
        X_base,
        y_base,
        g_base,
    ),
    1,
):

    test_groups = set(
        g_base[
            te
        ].tolist()
    )

    tm = np.asarray(
        [
            g in test_groups
            for g in g_tour
        ],
        dtype=bool,
    )


    print(
        f"fold {fold}: "
        f"base_train={len(tr)} "
        f"heldout="
        f"{sorted(test_groups)} "
        f"tournament_test="
        f"{int(tm.sum())}",
        flush=True,
    )


    model = (
        SharedPresenceDetector
        ._fit_pipe(
            X_base[
                tr
            ],
            y_base[
                tr
            ],
            w_base[
                tr
            ],
            0.05,
        )
    )


    oof_score[
        tm
    ] = (
        model.decision_function(
            X_tour[
                tm
            ]
        )
    )


assert np.isfinite(
    oof_score
).all()


for i, x in enumerate(
    all_rows
):

    x[
        "sensor_oof_score"
    ] = float(
        oof_score[
            i
        ]
    )


# ============================================================
# 7. Individual candidate diagnostics
# ============================================================

def safe_auc(
    y,
    s,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    s = np.asarray(
        s,
        dtype=float,
    )

    if len(
        np.unique(
            y
        )
    ) != 2:
        return None

    return float(
        roc_auc_score(
            y,
            s,
        )
    )


by_candidate = defaultdict(
    list
)

for x in all_rows:

    by_candidate[
        x[
            "candidate_id"
        ]
    ].append(
        x
    )


candidate_summary = []


for cand in candidates:

    cid = cand[
        "candidate_id"
    ]

    rr = by_candidate[
        cid
    ]

    assert len(
        rr
    ) == 20


    y = [
        int(
            x[
                "target_present_eval_only"
            ]
        )
        for x in rr
    ]

    s = [
        float(
            x[
                "sensor_oof_score"
            ]
        )
        for x in rr
    ]


    present = [
        x
        for x in rr
        if x[
            "target_present_eval_only"
        ]
    ]


    route_hits = sum(
        int(
            x[
                "target_selected_eval_only"
            ]
        )
        for x in present
    )


    selected_pos = [
        x
        for x in rr
        if (
            x[
                "target_present_eval_only"
            ]
            and
            x[
                "target_selected_eval_only"
            ]
        )
    ]


    absent = [
        x
        for x in rr
        if not x[
            "target_present_eval_only"
        ]
    ]


    sel_abs_auc = None

    if (
        selected_pos
        and absent
    ):

        sel_abs_auc = safe_auc(
            [
                1
            ]
            * len(
                selected_pos
            )
            +
            [
                0
            ]
            * len(
                absent
            ),

            [
                x[
                    "sensor_oof_score"
                ]
                for x in selected_pos
            ]
            +
            [
                x[
                    "sensor_oof_score"
                ]
                for x in absent
            ],
        )


    candidate_summary.append({
        **cand,

        "installed_auc":
            safe_auc(
                y,
                s,
            ),

        "route_hits":
            route_hits,

        "route_n":
            len(
                present
            ),

        "route_rate":
            route_hits
            / len(
                present
            ),

        "selected_vs_absent_auc":
            sel_abs_auc,

        "mean_present_score":
            float(
                np.mean(
                    [
                        x[
                            "sensor_oof_score"
                        ]
                        for x
                        in present
                    ]
                )
            ),

        "mean_absent_score":
            float(
                np.mean(
                    [
                        x[
                            "sensor_oof_score"
                        ]
                        for x
                        in absent
                    ]
                )
            ),
    })


candidate_summary.sort(
    key=lambda x: (
        TARGET_ROLES.index(
            x["role"]
        ),
        -x[
            "installed_auc"
        ],
        -x[
            "route_rate"
        ],
    )
)


print()
print(
    "=" * 120
)

print(
    "INDIVIDUAL END-TO-END CANDIDATES"
)

print(
    "=" * 120
)


for role in TARGET_ROLES:

    print(
        f"\n### {role}"
    )

    rr = [
        x
        for x in candidate_summary
        if x[
            "role"
        ] == role
    ]


    for rank, x in enumerate(
        rr,
        1,
    ):

        sva = (
            "NA"
            if x[
                "selected_vs_absent_auc"
            ] is None
            else
            f"{x['selected_vs_absent_auc']:.4f}"
        )

        print(
            f"#{rank:<2} "
            f"{x['family']:<12} "
            f"rep={x['rep']} "
            f"AUC="
            f"{x['installed_auc']:.4f} "
            f"route="
            f"{x['route_hits']}/"
            f"{x['route_n']} "
            f"({x['route_rate']:.3f}) "
            f"sel-v-abs="
            f"{sva} "
            f"presentMean="
            f"{x['mean_present_score']:+.3f} "
            f"absentMean="
            f"{x['mean_absent_score']:+.3f} "
            f"id="
            f"{x['candidate_id']}"
        )

        print(
            "    ",
            x[
                "scenario"
            ],
        )


# ============================================================
# 8. Pair tournament per role.
#
# Final inference currently uses MAX over two probes,
# so optimize/evaluate that exact aggregation.
# ============================================================

score_lookup = {
    (
        x[
            "candidate_id"
        ],
        x[
            "system_id"
        ],
    ):
        float(
            x[
                "sensor_oof_score"
            ]
        )
    for x in all_rows
}


route_lookup = {
    (
        x[
            "candidate_id"
        ],
        x[
            "system_id"
        ],
    ):
        int(
            x[
                "target_selected_eval_only"
            ]
        )
    for x in all_rows
}


pair_summary = []


for role in TARGET_ROLES:

    cs = [
        x
        for x in candidates
        if x[
            "role"
        ] == role
    ]


    for i in range(
        len(
            cs
        )
    ):

        for j in range(
            i + 1,
            len(
                cs
            ),
        ):

            a = cs[
                i
            ]

            b = cs[
                j
            ]


            yy = []

            ss = []

            covered = 0

            present_n = 0


            for sys in systems:

                sid = sys[
                    "system_id"
                ]

                y = int(
                    role
                    in sys[
                        "worker_roles"
                    ]
                )

                sa = score_lookup[
                    (
                        a[
                            "candidate_id"
                        ],
                        sid,
                    )
                ]

                sb = score_lookup[
                    (
                        b[
                            "candidate_id"
                        ],
                        sid,
                    )
                ]


                yy.append(
                    y
                )

                ss.append(
                    max(
                        sa,
                        sb,
                    )
                )


                if y:

                    present_n += 1

                    hit = max(
                        route_lookup[
                            (
                                a[
                                    "candidate_id"
                                ],
                                sid,
                            )
                        ],
                        route_lookup[
                            (
                                b[
                                    "candidate_id"
                                ],
                                sid,
                            )
                        ],
                    )

                    covered += hit


            pair_summary.append({
                "role":
                    role,

                "a":
                    a,

                "b":
                    b,

                "pair_auc":
                    safe_auc(
                        yy,
                        ss,
                    ),

                "route_coverage":
                    covered
                    / present_n,

                "route_hits":
                    covered,

                "route_n":
                    present_n,
            })


pair_summary.sort(
    key=lambda x: (
        TARGET_ROLES.index(
            x["role"]
        ),
        -x[
            "pair_auc"
        ],
        -x[
            "route_coverage"
        ],
    )
)


print()
print(
    "=" * 120
)

print(
    "TOP TWO-PROBE PAIRS PER ROLE"
)

print(
    "=" * 120
)


for role in TARGET_ROLES:

    print(
        f"\n### {role}"
    )

    rr = [
        x
        for x in pair_summary
        if x[
            "role"
        ] == role
    ]


    for rank, x in enumerate(
        rr[:10],
        1,
    ):

        a = x[
            "a"
        ]

        b = x[
            "b"
        ]

        print(
            f"#{rank:<2} "
            f"AUC="
            f"{x['pair_auc']:.4f} "
            f"routingCoverage="
            f"{x['route_hits']}/"
            f"{x['route_n']} "
            f"({x['route_coverage']:.3f})"
        )

        print(
            f"    A: "
            f"{a['family']} "
            f"rep={a['rep']} "
            f"id={a['candidate_id']}"
        )

        print(
            f"    B: "
            f"{b['family']} "
            f"rep={b['rep']} "
            f"id={b['candidate_id']}"
        )


# ============================================================
# 9. Full 8-role DEV ranking simulation.
#
# Other six roles stay exactly as current V3.
# Only design/operations pair is swapped.
#
# IMPORTANT:
# This is DEV optimization, not untouched-test evidence.
# ============================================================

base_oof_rows = read_jsonl(
    BASE_OOF
)


base_system_role = defaultdict(
    list
)

truth = {}


for x in base_oof_rows:

    sid = str(
        x[
            "system_id"
        ]
    )

    role = str(
        x[
            "target_role"
        ]
    )

    base_system_role[
        (
            sid,
            role,
        )
    ].append(
        float(
            x[
                "installed_presence_oof_score"
            ]
        )
    )

    truth[
        (
            sid,
            role,
        )
    ] = int(
        x[
            "target_present_eval_only"
        ]
    )


base_max = {
    key:
        max(
            vals
        )
    for key, vals
    in base_system_role.items()
}


design_pairs = [
    x
    for x in pair_summary
    if x[
        "role"
    ] == "design"
]


operations_pairs = [
    x
    for x in pair_summary
    if x[
        "role"
    ] == "operations"
]


def pair_score(
    pair,
    sid,
):

    return max(
        score_lookup[
            (
                pair[
                    "a"
                ][
                    "candidate_id"
                ],
                sid,
            )
        ],
        score_lookup[
            (
                pair[
                    "b"
                ][
                    "candidate_id"
                ],
                sid,
            )
        ],
    )


def eval_combo(
    d_pair,
    o_pair,
):

    aucs = []

    margins = []

    inversions = 0

    total_pairs = 0

    perfect = 0


    for sys in systems:

        sid = sys[
            "system_id"
        ]

        rr = []


        for role in ROLES:

            if role == "design":

                score = pair_score(
                    d_pair,
                    sid,
                )

            elif role == "operations":

                score = pair_score(
                    o_pair,
                    sid,
                )

            else:

                score = base_max[
                    (
                        sid,
                        role,
                    )
                ]


            rr.append({
                "role":
                    role,

                "y":
                    truth[
                        (
                            sid,
                            role,
                        )
                    ],

                "score":
                    float(
                        score
                    ),
            })


        y = np.asarray(
            [
                x[
                    "y"
                ]
                for x in rr
            ],
            dtype=int,
        )

        s = np.asarray(
            [
                x[
                    "score"
                ]
                for x in rr
            ],
            dtype=float,
        )


        aucs.append(
            float(
                roc_auc_score(
                    y,
                    s,
                )
            )
        )


        pos = [
            x
            for x in rr
            if x[
                "y"
            ]
        ]

        neg = [
            x
            for x in rr
            if not x[
                "y"
            ]
        ]


        margin = (
            min(
                x[
                    "score"
                ]
                for x
                in pos
            )
            -
            max(
                x[
                    "score"
                ]
                for x
                in neg
            )
        )


        margins.append(
            margin
        )


        if margin > 0:

            perfect += 1


        for p in pos:

            for n in neg:

                total_pairs += 1

                if (
                    p[
                        "score"
                    ]
                    <=
                    n[
                        "score"
                    ]
                ):

                    inversions += 1


    return {
        "perfect":
            perfect,

        "inversions":
            inversions,

        "pair_acc":
            1
            - inversions
            / total_pairs,

        "mean_auc":
            float(
                np.mean(
                    aucs
                )
            ),

        "median_auc":
            float(
                np.median(
                    aucs
                )
            ),

        "mean_margin":
            float(
                np.mean(
                    margins
                )
            ),
    }


combo_results = []


for dp in design_pairs:

    for op in operations_pairs:

        metrics = eval_combo(
            dp,
            op,
        )

        combo_results.append({
            "design_pair":
                dp,

            "operations_pair":
                op,

            **metrics,
        })


combo_results.sort(
    key=lambda x: (
        -x[
            "perfect"
        ],
        x[
            "inversions"
        ],
        -x[
            "mean_auc"
        ],
        -x[
            "mean_margin"
        ],
    )
)


print()
print(
    "=" * 120
)

print(
    "TOP FULL 8-ROLE DEV COMBINATIONS"
)

print(
    "WARNING: DEVELOPMENT SELECTION ONLY"
)

print(
    "=" * 120
)


for rank, x in enumerate(
    combo_results[:10],
    1,
):

    dp = x[
        "design_pair"
    ]

    op = x[
        "operations_pair"
    ]


    print(
        f"\n#{rank} "
        f"perfect="
        f"{x['perfect']}/20 "
        f"inversions="
        f"{x['inversions']}/310 "
        f"pairAcc="
        f"{x['pair_acc']:.4f} "
        f"meanAUC="
        f"{x['mean_auc']:.4f} "
        f"medianAUC="
        f"{x['median_auc']:.4f} "
        f"margin="
        f"{x['mean_margin']:+.3f}"
    )


    print(
        "  DESIGN:"
    )

    print(
        f"    "
        f"{dp['a']['family']} "
        f"rep={dp['a']['rep']} "
        f"id={dp['a']['candidate_id']}"
    )

    print(
        f"    "
        f"{dp['b']['family']} "
        f"rep={dp['b']['rep']} "
        f"id={dp['b']['candidate_id']}"
    )


    print(
        "  OPERATIONS:"
    )

    print(
        f"    "
        f"{op['a']['family']} "
        f"rep={op['a']['rep']} "
        f"id={op['a']['candidate_id']}"
    )

    print(
        f"    "
        f"{op['b']['family']} "
        f"rep={op['b']['rep']} "
        f"id={op['b']['candidate_id']}"
    )


# ============================================================
# 10. Save summaries
# ============================================================

def compact_pair(
    x,
):

    return {
        "role":
            x[
                "role"
            ],

        "pair_auc":
            x[
                "pair_auc"
            ],

        "route_coverage":
            x[
                "route_coverage"
            ],

        "a":
            x[
                "a"
            ],

        "b":
            x[
                "b"
            ],
    }


summary = {
    "experiment":
        "e2e_probe_tournament_v1",

    "model":
        MODEL,

    "development_only":
        True,

    "candidate_count":
        len(
            candidates
        ),

    "interactions":
        len(
            all_rows
        ),

    "sensor_training":
        (
            "fold-specific detector trained "
            "only on frozen_probe_auc_v2 "
            "responses outside held-out "
            "composition"
        ),

    "candidate_summary":
        candidate_summary,

    "top_design_pairs":
        [
            compact_pair(
                x
            )
            for x
            in design_pairs[:10]
        ],

    "top_operations_pairs":
        [
            compact_pair(
                x
            )
            for x
            in operations_pairs[:10]
        ],

    "top_full_combinations":
        [
            {
                "perfect":
                    x[
                        "perfect"
                    ],

                "inversions":
                    x[
                        "inversions"
                    ],

                "pair_acc":
                    x[
                        "pair_acc"
                    ],

                "mean_auc":
                    x[
                        "mean_auc"
                    ],

                "median_auc":
                    x[
                        "median_auc"
                    ],

                "mean_margin":
                    x[
                        "mean_margin"
                    ],

                "design_pair":
                    compact_pair(
                        x[
                            "design_pair"
                        ]
                    ),

                "operations_pair":
                    compact_pair(
                        x[
                            "operations_pair"
                        ]
                    ),
            }
            for x
            in combo_results[:10]
        ],
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


with (
    OUT
    / "scored_responses.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for x in all_rows:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print(
    "wrote:",
    OUT
    / "summary.json"
)

print(
    "wrote:",
    OUT
    / "scored_responses.jsonl"
)
