#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OUT = Path(
    "artifacts/probe_router_tournament_v1"
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
        "ROUTER_TOURNAMENT_WORKERS",
        "10",
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


CURRENT_BANK = Path(
    "artifacts/generator_frozen_v2/"
    "selected_bank.jsonl"
)

POOL_10 = Path(
    "artifacts/generator_v12_10/"
    "all_candidates.jsonl"
)

POOL_11 = Path(
    "artifacts/generator_v12_11/"
    "all_candidates.jsonl"
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


bank = read_jsonl(
    CURRENT_BANK
)

p10 = read_jsonl(
    POOL_10
)

p11 = read_jsonl(
    POOL_11
)


current_queries = {
    str(x["query"])
    for x in bank
}


BAD = re.compile(
    r"\b("
    r"drug|dosage|patient|treatment|"
    r"stock price|trading|investment advice|"
    r"your help|your guidance|"
    r"what initial step|what steps should"
    r")\b",
    re.I,
)


def eligible(row):
    if not row.get(
        "surface_valid",
        False
    ):
        return False

    q = str(
        row.get(
            "query",
            ""
        )
    ).strip()

    scenario = str(
        row.get(
            "scenario",
            ""
        )
    ).strip()

    if not q or not scenario:
        return False

    if q in current_queries:
        return False

    if BAD.search(
        scenario
    ):
        return False

    sim = row.get(
        "target_similarity"
    )

    margin = row.get(
        "target_margin"
    )

    if sim is None or margin is None:
        return False

    if float(sim) < 0.30:
        return False

    if float(margin) < 0.05:
        return False

    return True


# ------------------------------------------------------------
# Repair specification.
#
# code:
#   both probes weak -> search broadly.
#
# math:
#   ambiguity already excellent -> search non-ambiguity only.
#
# design:
#   both weak -> search broadly.
#
# operations:
#   intake already excellent -> search non-intake only.
# ------------------------------------------------------------

candidate_sets = {}


code = [
    x
    for x in p11
    if (
        x.get("role") == "code"
        and eligible(x)
    )
]


math = [
    x
    for x in p10
    if (
        x.get("role") == "math"
        and eligible(x)
        and x.get("family")
        != "ambiguity"
    )
]


design = [
    x
    for x in p10
    if (
        x.get("role") == "design"
        and eligible(x)
    )
]


operations = [
    x
    for x in p11
    if (
        x.get("role")
        == "operations"
        and eligible(x)
        and x.get("family")
        != "intake"
    )
]


for role, xs in [
    ("code", code),
    ("math", math),
    ("design", design),
    ("operations", operations),
]:

    xs.sort(
        key=lambda x: (
            float(
                x.get(
                    "score",
                    -999,
                )
            ),
            float(
                x.get(
                    "target_margin",
                    -999,
                )
            ),
        ),
        reverse=True,
    )

    # Semantic prefilter only.
    # MAS router chooses among at most top 8.
    candidate_sets[
        role
    ] = xs[:8]


print(
    "===== CANDIDATE COUNTS =====",
    flush=True,
)

for role in [
    "code",
    "math",
    "design",
    "operations",
]:

    print(
        role,
        len(
            candidate_sets[
                role
            ]
        ),
        flush=True,
    )

    if not candidate_sets[
        role
    ]:
        raise RuntimeError(
            f"No candidates for {role}"
        )


# ------------------------------------------------------------
# Actual MAS router.
# We call ONLY _router_choice().
# No worker answer, no aggregator.
# ------------------------------------------------------------

from targets.single_router_zoo_v11 import (
    validate_v11_design,
)

validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


# Fixed DEV tournament:
# first three composition families containing target,
# crossed with both router styles.
eval_cells = {}


for role in [
    "code",
    "math",
    "design",
    "operations",
]:

    comp_ids = [
        cid
        for cid, rs
        in COMPOSITIONS.items()
        if role in rs
    ][:3]

    assert len(
        comp_ids
    ) == 3

    cells = []

    for cid in comp_ids:

        for style in STYLES:

            cells.append({
                "comp_id":
                    cid,

                "style":
                    style,

                "roles":
                    COMPOSITIONS[
                        cid
                    ],
            })

    eval_cells[
        role
    ] = cells


tasks = []


for role, candidates in candidate_sets.items():

    for ci, cand in enumerate(
        candidates
    ):

        for cell in eval_cells[
            role
        ]:

            tasks.append({
                "target_role":
                    role,

                "candidate_index":
                    ci,

                "family":
                    cand["family"],

                "rep":
                    int(
                        cand["rep"]
                    ),

                "scenario":
                    cand["scenario"],

                "query":
                    cand["query"],

                "target_similarity":
                    float(
                        cand[
                            "target_similarity"
                        ]
                    ),

                "target_margin":
                    float(
                        cand[
                            "target_margin"
                        ]
                    ),

                "semantic_score":
                    float(
                        cand.get(
                            "score",
                            0.0,
                        )
                    ),

                "comp_id":
                    cell["comp_id"],

                "style":
                    cell["style"],

                "worker_roles":
                    list(
                        cell[
                            "roles"
                        ]
                    ),
            })


print(
    "\nrouter calls planned:",
    len(tasks),
    flush=True,
)


def execute(task):

    sid = (
        "router_tournament_"
        + task["target_role"]
        + "_"
        + task["comp_id"]
        + "_"
        + task["style"]
    )

    spec = SingleRouterSystemSpec(
        system_id=sid,
        split=
            "probe_router_tournament_v1",
        execution_mode="single",
        style=task["style"],
        worker_roles=tuple(
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

            worker_ids = (
                mas._router_choice(
                    task["query"]
                )
            )

            selected_roles = [
                mas.role_by_worker[
                    wid
                ]
                for wid in worker_ids
            ]

            return {
                **task,

                "selected_roles":
                    selected_roles,

                "target_hit":
                    int(
                        task[
                            "target_role"
                        ]
                        in selected_roles
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

        "selected_roles":
            [],

        "target_hit":
            0,

        "status":
            "error",

        "error":
            repr(last),
    }


results = []


with ThreadPoolExecutor(
    max_workers=API_WORKERS
) as pool:

    futures = [
        pool.submit(
            execute,
            task,
        )
        for task in tasks
    ]

    for i, fut in enumerate(
        as_completed(
            futures
        ),
        1,
    ):

        row = fut.result()

        results.append(
            row
        )

        if (
            i <= 20
            or i % 25 == 0
        ):
            print(
                f"[{i}/{len(tasks)}] "
                f"{row['target_role']} "
                f"{row['family']} "
                f"rep={row['rep']} "
                f"{row['comp_id']}/"
                f"{row['style']} "
                f"hit={row['target_hit']}",
                flush=True,
            )


with (
    OUT
    / "routing_results.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in results:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ------------------------------------------------------------
# Aggregate candidate score.
# ------------------------------------------------------------

agg = {}


for row in results:

    key = (
        row["target_role"],
        row["candidate_index"],
    )

    if key not in agg:

        agg[key] = {
            "target_role":
                row[
                    "target_role"
                ],

            "candidate_index":
                row[
                    "candidate_index"
                ],

            "family":
                row["family"],

            "rep":
                row["rep"],

            "scenario":
                row["scenario"],

            "query":
                row["query"],

            "target_similarity":
                row[
                    "target_similarity"
                ],

            "target_margin":
                row[
                    "target_margin"
                ],

            "semantic_score":
                row[
                    "semantic_score"
                ],

            "hits":
                0,

            "n":
                0,

            "style_hits":
                defaultdict(int),

            "style_n":
                defaultdict(int),

            "selected_counter":
                Counter(),
        }

    z = agg[key]

    if row["status"] == "ok":

        z["hits"] += int(
            row[
                "target_hit"
            ]
        )

        z["n"] += 1

        z["style_hits"][
            row["style"]
        ] += int(
            row[
                "target_hit"
            ]
        )

        z["style_n"][
            row["style"]
        ] += 1

        z["selected_counter"].update(
            row[
                "selected_roles"
            ]
        )


summary_rows = []


for z in agg.values():

    n = z["n"]

    hits = z["hits"]

    rate = (
        hits / n
        if n
        else 0.0
    )

    style_rates = {}

    for style in STYLES:

        sn = z[
            "style_n"
        ].get(
            style,
            0,
        )

        sh = z[
            "style_hits"
        ].get(
            style,
            0,
        )

        style_rates[
            style
        ] = (
            sh / sn
            if sn
            else 0.0
        )

    worst_style = min(
        style_rates.values()
    )

    z2 = {
        k: v
        for k, v in z.items()
        if k not in {
            "style_hits",
            "style_n",
            "selected_counter",
        }
    }

    z2[
        "routing_hit_rate"
    ] = rate

    z2[
        "worst_style_hit_rate"
    ] = worst_style

    z2[
        "style_hit_rates"
    ] = style_rates

    z2[
        "selected_counts"
    ] = dict(
        z[
            "selected_counter"
        ]
    )

    summary_rows.append(
        z2
    )


summary_rows.sort(
    key=lambda x: (
        ROLES.index(
            x["target_role"]
        ),
        -x[
            "routing_hit_rate"
        ],
        -x[
            "worst_style_hit_rate"
        ],
        -x[
            "target_margin"
        ],
    )
)


with (
    OUT
    / "candidate_summary.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in summary_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "\n"
    + "=" * 120
)

print(
    "ROUTER TOURNAMENT RESULTS"
)

print(
    "=" * 120
)


for role in [
    "code",
    "math",
    "design",
    "operations",
]:

    print(
        f"\n### {role}"
    )

    xs = [
        x
        for x in summary_rows
        if x[
            "target_role"
        ] == role
    ]

    for rank, x in enumerate(
        xs,
        1,
    ):

        print(
            f"\n#{rank} "
            f"{x['family']} "
            f"rep={x['rep']} "
            f"route="
            f"{x['hits']}/{x['n']} "
            f"({x['routing_hit_rate']:.3f}) "
            f"worstStyle="
            f"{x['worst_style_hit_rate']:.3f} "
            f"margin="
            f"{x['target_margin']:+.3f}"
        )

        print(
            "scenario:",
            x["scenario"],
        )

        print(
            "selected:",
            x[
                "selected_counts"
            ],
        )


print(
    "\nwrote:",
    OUT
    / "candidate_summary.jsonl"
)
