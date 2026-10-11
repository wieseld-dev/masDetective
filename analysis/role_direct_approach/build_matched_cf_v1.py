#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_v11_active.io import (
    sha256_json,
    stable_int,
    write_json,
    write_jsonl,
)

from analysis.role_direct_approach.generate_direct_panel_v1 import (
    ROLES,
    CONFUSABLES,
)


ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)

PANEL_SRC = Path(
    "artifacts/"
    "role_direct_approach_directprompt_v1_fix1/"
    "role_direct_probe_panel.jsonl"
)

TEMPLATE_MANIFEST = Path(
    "artifacts/"
    "role_direct_routing_pilot_v1/"
    "manifest_privileged.json"
)

KEEP_STRATEGIES = [
    "CONCRETE ARTIFACT",
    "CONSTRAINED TASK",
    "SPECIALIST DELIVERABLE",
]

PAIRS_PER_ROLE = 5


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


panel_all = read_jsonl(
    PANEL_SRC
)

panel_by_role = defaultdict(list)

for x in panel_all:

    if x["strategy"] in KEEP_STRATEGIES:
        panel_by_role[
            str(x["target_role"])
        ].append(x)


# ============================================================
# Exact 3 already-generated direct probes per role.
# ============================================================

for role in ROLES:

    rows = panel_by_role[role]

    assert len(rows) == 3, (
        role,
        len(rows),
    )

    strategy_order = {
        x: i
        for i, x in enumerate(
            KEEP_STRATEGIES
        )
    }

    rows.sort(
        key=lambda x:
            strategy_order[
                x["strategy"]
            ]
    )


template_manifest = json.loads(
    TEMPLATE_MANIFEST.read_text(
        encoding="utf-8"
    )
)

templates = sorted(
    [
        x
        for x in template_manifest["systems"]
        if x["split"] == "dev"
    ],
    key=lambda x: x["session_id"],
)

assert len(templates) == 5, len(templates)


# ============================================================
# Choose backgrounds deliberately far from TARGET and
# substitute when possible.
# ============================================================

def compatible_backgrounds(
    target: str,
    substitute: str,
    pair_index: int,
):

    strict = []

    for r in ROLES:

        if r in {
            target,
            substitute,
        }:
            continue

        # Avoid direct target confusables.
        if r in CONFUSABLES[target]:
            continue

        # Avoid direct substitute confusables.
        if r in CONFUSABLES.get(
            substitute,
            [],
        ):
            continue

        # Avoid strong reverse-confusable relation.
        if target in CONFUSABLES.get(
            r,
            [],
        ):
            continue

        if substitute in CONFUSABLES.get(
            r,
            [],
        ):
            continue

        strict.append(r)

    relaxed = [
        r
        for r in ROLES
        if r not in {
            target,
            substitute,
        }
        and r not in CONFUSABLES[target]
    ]

    pool = (
        strict
        if len(strict) >= 2
        else relaxed
    )

    assert len(pool) >= 2

    pool = sorted(
        pool,
        key=lambda r: stable_int(
            "ROLE_DIRECT_CF_BG",
            target,
            substitute,
            pair_index,
            r,
        ),
    )

    return pool[:2]


systems = []
pair_rows = []


for role_index, target in enumerate(
    ROLES
):

    conf = list(
        CONFUSABLES[target]
    )

    assert conf

    for pair_index in range(
        PAIRS_PER_ROLE
    ):

        substitute = conf[
            pair_index % len(conf)
        ]

        bg = compatible_backgrounds(
            target,
            substitute,
            pair_index,
        )

        assert len(bg) == 2
        assert len(set(bg)) == 2

        template = templates[
            pair_index
            % len(templates)
        ]

        mode = str(
            template["mode"]
        )

        style = str(
            template["style"]
        )

        pair_id = (
            f"{target}__p"
            f"{pair_index + 1:02d}"
        )

        pos_sid = (
            f"rdcf_{role_index:02d}_"
            f"p{pair_index + 1:02d}_POS"
        )

        neg_sid = (
            f"rdcf_{role_index:02d}_"
            f"p{pair_index + 1:02d}_NEG"
        )

        pos_roles = [
            bg[0],
            bg[1],
            target,
        ]

        neg_roles = [
            bg[0],
            bg[1],
            substitute,
        ]

        assert len(set(pos_roles)) == 3
        assert len(set(neg_roles)) == 3

        assert target in pos_roles
        assert target not in neg_roles

        # Exact matched background.
        assert (
            pos_roles[:2]
            == neg_roles[:2]
        )

        systems.append({
            "session_id":
                pos_sid,

            "split":
                "dev",

            "mode":
                mode,

            "style":
                style,

            "roles":
                pos_roles,

            "k":
                3,

            "pair_id":
                pair_id,

            "pair_index":
                pair_index + 1,

            "condition":
                "TARGET_PRESENT",

            "target_role":
                target,

            "substitute_role":
                substitute,

            "background_roles":
                bg,
        })

        systems.append({
            "session_id":
                neg_sid,

            "split":
                "dev",

            "mode":
                mode,

            "style":
                style,

            "roles":
                neg_roles,

            "k":
                3,

            "pair_id":
                pair_id,

            "pair_index":
                pair_index + 1,

            "condition":
                "TARGET_ABSENT",

            "target_role":
                target,

            "substitute_role":
                substitute,

            "background_roles":
                bg,
        })

        pair_rows.append({
            "pair_id":
                pair_id,

            "target_role":
                target,

            "substitute_role":
                substitute,

            "background_roles":
                bg,

            "mode":
                mode,

            "style":
                style,

            "positive_system":
                pos_sid,

            "negative_system":
                neg_sid,
        })


assert len(systems) == 200
assert len(pair_rows) == 100
assert len({
    x["session_id"]
    for x in systems
}) == 200


manifest = {
    "version":
        "role_direct_matched_cf_v1",

    "seed":
        20261008,

    "systems":
        systems,

    "sha256":
        sha256_json(
            systems
        ),

    "notes": {
        "privileged":
            True,

        "design":
            (
                "Matched counterfactual "
                "ROLE DIRECT evaluation"
            ),

        "k":
            3,

        "pairs_per_role":
            PAIRS_PER_ROLE,

        "target_roles":
            len(ROLES),

        "pair_difference":
            (
                "TARGET_PRESENT replaces "
                "matched confusable substitute; "
                "background/mode/style fixed"
            ),
    },
}

write_json(
    ROOT / "manifest_privileged.json",
    manifest,
)

write_jsonl(
    ROOT / "matched_pairs.jsonl",
    pair_rows,
)


# ============================================================
# Exact same 3 queries go to POS and NEG in each pair.
# ============================================================

candidates = []

system_by_id = {
    x["session_id"]: x
    for x in systems
}

for pair in pair_rows:

    target = pair[
        "target_role"
    ]

    probes = panel_by_role[
        target
    ]

    for condition, sid in [
        (
            "TARGET_PRESENT",
            pair[
                "positive_system"
            ],
        ),
        (
            "TARGET_ABSENT",
            pair[
                "negative_system"
            ],
        ),
    ]:

        m = system_by_id[sid]

        for probe_rank, p in enumerate(
            probes,
            1,
        ):

            candidates.append({
                "session_id":
                    sid,

                "split":
                    "dev",

                "round":
                    1,

                "candidate_index":
                    probe_rank - 1,

                "group_id":
                    (
                        f"{pair['pair_id']}"
                        f"::{condition}"
                    ),

                "query":
                    str(
                        p["query"]
                    ),

                "valid":
                    True,

                "invalid_reason":
                    "",

                "phase":
                    "ROLE_DIRECT_MATCHED_CF",

                "target_role":
                    target,

                "pair_id":
                    pair[
                        "pair_id"
                    ],

                "pair_index":
                    m[
                        "pair_index"
                    ],

                "condition":
                    condition,

                "substitute_role":
                    pair[
                        "substitute_role"
                    ],

                "background_roles":
                    pair[
                        "background_roles"
                    ],

                "direct_probe_rank":
                    probe_rank,

                "source_probe_index":
                    int(
                        p[
                            "probe_index"
                        ]
                    ),

                "probe_strategy":
                    str(
                        p[
                            "strategy"
                        ]
                    ),

                "generator_mode":
                    "DIRECT_PROMPT_NO_BELIEF",

                "chosen_for_continuation":
                    True,
            })


assert len(candidates) == 600


# ============================================================
# HARD MATCHED-DESIGN AUDIT
# ============================================================

by_pair = defaultdict(list)

for m in systems:
    by_pair[
        m["pair_id"]
    ].append(m)


for pair_id, mm in by_pair.items():

    assert len(mm) == 2

    pos = next(
        x
        for x in mm
        if x[
            "condition"
        ]
        == "TARGET_PRESENT"
    )

    neg = next(
        x
        for x in mm
        if x[
            "condition"
        ]
        == "TARGET_ABSENT"
    )

    assert pos["k"] == neg["k"] == 3
    assert pos["mode"] == neg["mode"]
    assert pos["style"] == neg["style"]

    assert (
        pos["background_roles"]
        == neg["background_roles"]
    )

    assert (
        pos["roles"][:2]
        == neg["roles"][:2]
    )

    target = pos[
        "target_role"
    ]

    substitute = pos[
        "substitute_role"
    ]

    assert pos["roles"][2] == target
    assert neg["roles"][2] == substitute

    assert target not in neg["roles"]


# Query equality across POS/NEG.
cq = defaultdict(
    lambda: defaultdict(list)
)

for c in candidates:

    cq[
        c["pair_id"]
    ][
        c["condition"]
    ].append(
        (
            c[
                "direct_probe_rank"
            ],
            c[
                "query"
            ],
        )
    )


for pair_id, d in cq.items():

    pos = sorted(
        d[
            "TARGET_PRESENT"
        ]
    )

    neg = sorted(
        d[
            "TARGET_ABSENT"
        ]
    )

    assert pos == neg
    assert len(pos) == 3


write_jsonl(
    ROOT
    / "candidates_dev_round_01.jsonl",
    candidates,
)


# ============================================================
# Summary
# ============================================================

role_counts = Counter(
    x["target_role"]
    for x in pair_rows
)

assert set(
    role_counts.values()
) == {
    5
}


print("=" * 90)
print(
    "ROLE DIRECT MATCHED COUNTERFACTUAL BUILD"
)
print("=" * 90)

print(
    "roles             =",
    len(ROLES),
)

print(
    "pairs/role        =",
    PAIRS_PER_ROLE,
)

print(
    "matched pairs     =",
    len(pair_rows),
)

print(
    "systems           =",
    len(systems),
)

print(
    "probes/system     =",
    3,
)

print(
    "MAS executions    =",
    len(candidates),
)

print(
    "manifest sha256   =",
    manifest["sha256"],
)

print()
print(
    "MATCHED_CF_BUILD_PASS"
)
