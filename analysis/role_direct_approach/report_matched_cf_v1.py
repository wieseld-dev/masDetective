#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from adaptive_v11_active.io import write_jsonl
from statistics import mean


ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


def sid(x):
    return str(
        x.get(
            "system_id",
            x.get(
                "session_id",
                "",
            ),
        )
    )


def win(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def bootstrap_win_ci(
    rows,
    n=10000,
    seed=20261008,
):

    rng = random.Random(
        seed
    )

    vals = [
        float(x)
        for x in rows
    ]

    boots = []

    for _ in range(n):

        sample = [
            vals[
                rng.randrange(
                    len(vals)
                )
            ]
            for _ in vals
        ]

        boots.append(
            mean(sample)
        )

    boots.sort()

    lo = boots[
        int(
            0.025
            * len(boots)
        )
    ]

    hi = boots[
        int(
            0.975
            * len(boots)
        )
    ]

    return lo, hi


manifest = json.loads(
    (
        ROOT
        / "manifest_privileged.json"
    ).read_text(
        encoding="utf-8"
    )
)

candidates = read_jsonl(
    ROOT
    / "candidates_dev_round_01.jsonl"
)

responses = read_jsonl(
    ROOT
    / "responses_dev_round_01.jsonl"
)

scored = read_jsonl(
    ROOT
    / "scored_dev_round_01.jsonl"
)

assert len(candidates) == 600
assert len(responses) == 600
assert len(scored) == 600


manifest_by = {
    x["session_id"]: x
    for x in manifest["systems"]
}


resp_by = {
    (
        sid(x),
        str(x["query"]),
    ): x
    for x in responses
}


score_by = {
    (
        sid(x),
        str(x["query"]),
    ): x
    for x in scored
}


obs = []

for c in candidates:

    system = str(
        c["session_id"]
    )

    q = str(
        c["query"]
    )

    key = (
        system,
        q,
    )

    assert key in resp_by, key
    assert key in score_by, key

    r = resp_by[key]
    s = score_by[key]

    target = str(
        c["target_role"]
    )

    substitute = str(
        c[
            "substitute_role"
        ]
    )

    selected = set(
        map(
            str,
            r.get(
                "selected_roles_training_only",
                [],
            ),
        )
    )

    obs.append({
        "system_id":
            system,

        "pair_id":
            str(
                c["pair_id"]
            ),

        "target_role":
            target,

        "condition":
            str(
                c["condition"]
            ),

        "probe_rank":
            int(
                c[
                    "direct_probe_rank"
                ]
            ),

        "query":
            q,

        "substitute_role":
            substitute,

        "target_activated":
            target in selected,

        "substitute_activated":
            substitute in selected,

        "exclusive_target":
            selected == {
                target
            },

        "v135":
            float(
                s[
                    "evidence"
                ][
                    target
                ]
            ),

        "v134":
            float(
                s[
                    "metadata"
                ][
                    "evidence_v134"
                ][
                    target
                ]
            ),
    })


# ============================================================
# Exact POS-vs-NEG matched comparisons.
# ============================================================

matched = defaultdict(
    dict
)

for x in obs:

    key = (
        x[
            "target_role"
        ],
        x[
            "pair_id"
        ],
        x[
            "probe_rank"
        ],
    )

    matched[key][
        x["condition"]
    ] = x


comparisons = []

for key, d in matched.items():

    assert set(d) == {
        "TARGET_PRESENT",
        "TARGET_ABSENT",
    }, (
        key,
        d.keys(),
    )

    p = d[
        "TARGET_PRESENT"
    ]

    n = d[
        "TARGET_ABSENT"
    ]

    assert (
        p["query"]
        == n["query"]
    )

    comparisons.append({
        "target_role":
            p[
                "target_role"
            ],

        "pair_id":
            p[
                "pair_id"
            ],

        "probe_rank":
            p[
                "probe_rank"
            ],

        "query":
            p[
                "query"
            ],

        "v135_pos":
            p[
                "v135"
            ],

        "v135_neg":
            n[
                "v135"
            ],

        "v135_delta":
            p[
                "v135"
            ]
            - n[
                "v135"
            ],

        "v135_win":
            win(
                p[
                    "v135"
                ],
                n[
                    "v135"
                ],
            ),

        "v134_pos":
            p[
                "v134"
            ],

        "v134_neg":
            n[
                "v134"
            ],

        "v134_delta":
            p[
                "v134"
            ]
            - n[
                "v134"
            ],

        "v134_win":
            win(
                p[
                    "v134"
                ],
                n[
                    "v134"
                ],
            ),

        "target_routed_pos":
            p[
                "target_activated"
            ],

        "exclusive_target_pos":
            p[
                "exclusive_target"
            ],

        "substitute_routed_neg":
            n[
                "substitute_activated"
            ],
    })


assert len(comparisons) == 300


# ============================================================
# Per pair: accumulate the 3 direct probes.
# ============================================================

pair_groups = defaultdict(
    list
)

for x in comparisons:

    pair_groups[
        (
            x[
                "target_role"
            ],
            x[
                "pair_id"
            ],
        )
    ].append(x)


pair_results = []

for (
    target,
    pair_id,
), rr in sorted(
    pair_groups.items()
):

    rr = sorted(
        rr,
        key=lambda x:
            x[
                "probe_rank"
            ]
    )

    assert len(rr) == 3

    result = {
        "target_role":
            target,

        "pair_id":
            pair_id,
    }

    for k in [
        1,
        2,
        3,
    ]:

        cur = rr[:k]

        p135 = mean(
            x[
                "v135_pos"
            ]
            for x in cur
        )

        n135 = mean(
            x[
                "v135_neg"
            ]
            for x in cur
        )

        p134 = mean(
            x[
                "v134_pos"
            ]
            for x in cur
        )

        n134 = mean(
            x[
                "v134_neg"
            ]
            for x in cur
        )

        result[
            f"v135_win_k{k}"
        ] = win(
            p135,
            n135,
        )

        result[
            f"v135_delta_k{k}"
        ] = (
            p135
            - n135
        )

        result[
            f"v134_win_k{k}"
        ] = win(
            p134,
            n134,
        )

        result[
            f"v134_delta_k{k}"
        ] = (
            p134
            - n134
        )

    pair_results.append(
        result
    )


assert len(pair_results) == 100


# ============================================================
# Per-role metrics.
# ============================================================

by_role_cmp = defaultdict(
    list
)

by_role_pair = defaultdict(
    list
)

for x in comparisons:
    by_role_cmp[
        x[
            "target_role"
        ]
    ].append(x)

for x in pair_results:
    by_role_pair[
        x[
            "target_role"
        ]
    ].append(x)


per_role = {}

for role in sorted(
    by_role_cmp
):

    cc = by_role_cmp[
        role
    ]

    pp = by_role_pair[
        role
    ]

    assert len(cc) == 15
    assert len(pp) == 5

    per_role[role] = {
        "matched_probe_win_v135":
            mean(
                x[
                    "v135_win"
                ]
                for x in cc
            ),

        "matched_probe_delta_v135":
            mean(
                x[
                    "v135_delta"
                ]
                for x in cc
            ),

        "matched_probe_win_v134":
            mean(
                x[
                    "v134_win"
                ]
                for x in cc
            ),

        "pair_mean_win_v135":
            mean(
                x[
                    "v135_win_k3"
                ]
                for x in pp
            ),

        "pair_mean_delta_v135":
            mean(
                x[
                    "v135_delta_k3"
                ]
                for x in pp
            ),

        "pair_mean_win_v134":
            mean(
                x[
                    "v134_win_k3"
                ]
                for x in pp
            ),

        "routing_target_present":
            mean(
                float(
                    x[
                        "target_routed_pos"
                    ]
                )
                for x in cc
            ),

        "routing_exclusive_present":
            mean(
                float(
                    x[
                        "exclusive_target_pos"
                    ]
                )
                for x in cc
            ),

        "routing_substitute_absent":
            mean(
                float(
                    x[
                        "substitute_routed_neg"
                    ]
                )
                for x in cc
            ),
    }


# ============================================================
# Overall accumulation curve.
# Each unit here is a matched system pair.
# n = 100.
# ============================================================

accumulation = {}

for k in [
    1,
    2,
    3,
]:

    vals135 = [
        x[
            f"v135_win_k{k}"
        ]
        for x in pair_results
    ]

    vals134 = [
        x[
            f"v134_win_k{k}"
        ]
        for x in pair_results
    ]

    lo135, hi135 = (
        bootstrap_win_ci(
            vals135
        )
    )

    lo134, hi134 = (
        bootstrap_win_ci(
            vals134
        )
    )

    accumulation[
        str(k)
    ] = {
        "v135_paired_win":
            mean(
                vals135
            ),

        "v135_ci95":
            [
                lo135,
                hi135,
            ],

        "v135_mean_delta":
            mean(
                x[
                    f"v135_delta_k{k}"
                ]
                for x in pair_results
            ),

        "v134_paired_win":
            mean(
                vals134
            ),

        "v134_ci95":
            [
                lo134,
                hi134,
            ],

        "v134_mean_delta":
            mean(
                x[
                    f"v134_delta_k{k}"
                ]
                for x in pair_results
            ),
    }


macro_role_v135 = mean(
    x[
        "pair_mean_win_v135"
    ]
    for x in per_role.values()
)

macro_role_v134 = mean(
    x[
        "pair_mean_win_v134"
    ]
    for x in per_role.values()
)


summary = {
    "experiment":
        "ROLE DIRECT MATCHED COUNTERFACTUAL V1",

    "roles":
        20,

    "matched_pairs":
        100,

    "mas_responses":
        len(obs),

    "matched_probe_comparisons":
        len(comparisons),

    "routing_target_present":
        mean(
            float(
                x[
                    "target_routed_pos"
                ]
            )
            for x in comparisons
        ),

    "routing_exclusive_present":
        mean(
            float(
                x[
                    "exclusive_target_pos"
                ]
            )
            for x in comparisons
        ),

    "routing_substitute_absent":
        mean(
            float(
                x[
                    "substitute_routed_neg"
                ]
            )
            for x in comparisons
        ),

    "v135_probe_matched_win":
        mean(
            x[
                "v135_win"
            ]
            for x in comparisons
        ),

    "v134_probe_matched_win":
        mean(
            x[
                "v134_win"
            ]
            for x in comparisons
        ),

    "v135_macro_role_pair_mean_win":
        macro_role_v135,

    "v134_macro_role_pair_mean_win":
        macro_role_v134,

    "accumulation":
        accumulation,

    "per_role":
        per_role,
}


(
    ROOT
    / "matched_cf_report.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


write_jsonl(
    ROOT
    / "matched_probe_comparisons.jsonl",
    comparisons,
)

write_jsonl(
    ROOT
    / "matched_pair_results.jsonl",
    pair_results,
)


print("=" * 105)
print(
    "ROLE DIRECT — MATCHED COUNTERFACTUAL"
)
print("=" * 105)

print(
    "responses                    =",
    summary[
        "mas_responses"
    ],
)

print(
    "matched pairs                =",
    summary[
        "matched_pairs"
    ],
)

print(
    "routing target | present     =",
    round(
        summary[
            "routing_target_present"
        ],
        4,
    ),
)

print(
    "routing exclusive | present  =",
    round(
        summary[
            "routing_exclusive_present"
        ],
        4,
    ),
)

print(
    "routing substitute | absent  =",
    round(
        summary[
            "routing_substitute_absent"
        ],
        4,
    ),
)

print()
print(
    "V13.5 probe matched win       =",
    round(
        summary[
            "v135_probe_matched_win"
        ],
        4,
    ),
)

print(
    "V13.5 macro role pair win     =",
    round(
        summary[
            "v135_macro_role_pair_mean_win"
        ],
        4,
    ),
)

print()
print(
    "V13.4 probe matched win       =",
    round(
        summary[
            "v134_probe_matched_win"
        ],
        4,
    ),
)

print(
    "V13.4 macro role pair win     =",
    round(
        summary[
            "v134_macro_role_pair_mean_win"
        ],
        4,
    ),
)

print()
print("=" * 105)
print("ACCUMULATION")
print("=" * 105)

for k, x in accumulation.items():

    print(
        f"k={k}  "
        f"V13.5={x['v135_paired_win']:.4f} "
        f"CI=[{x['v135_ci95'][0]:.4f}, "
        f"{x['v135_ci95'][1]:.4f}]  "
        f"delta={x['v135_mean_delta']:+.4f}   "
        f"V13.4={x['v134_paired_win']:.4f}"
    )


print()
print("=" * 105)
print("PER ROLE")
print("=" * 105)

for role, x in per_role.items():

    print(
        f"{role:26s} "
        f"route={x['routing_target_present']:.3f}  "
        f"probe={x['matched_probe_win_v135']:.3f}  "
        f"pair3={x['pair_mean_win_v135']:.3f}  "
        f"delta={x['pair_mean_delta_v135']:+.3f}"
    )

print()
print(
    "REPORT =",
    ROOT
    / "matched_cf_report.json"
)
