#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

from targets.single_router_zoo_v11 import (
    validate_v11_design,
)

validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


ROOT = Path(
    "artifacts/elicitation_mas_pilot_v1"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

BANK = Path(
    "artifacts/generator_frozen_v2/"
    "selected_bank.jsonl"
)

MODEL = os.environ.get(
    "MAS_MODEL",
    "gpt-4o-mini",
)

WORKERS = int(
    os.environ.get(
        "PILOT_API_WORKERS",
        "6",
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


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


bank = read_jsonl(
    BANK
)

assert len(bank) == 16

counts = Counter(
    x["role"]
    for x in bank
)

for role in ROLES:
    assert counts[role] == 2, (
        role,
        counts[role],
    )


# ------------------------------------------------------------
# Balanced matched K=3 pairs.
#
# For role r_i:
#
# PRESENT:
#   r_i, r_(i+2), r_(i+3)
#
# ABSENT:
#   r_(i+1), r_(i+2), r_(i+3)
#
# Therefore:
# - K stays exactly 3
# - two background roles are identical
# - only one role changes
# - every role participates symmetrically across the cycle
# ------------------------------------------------------------

systems = {}

for i, target_role in enumerate(
    ROLES
):

    substitute = ROLES[
        (i + 1) % len(ROLES)
    ]

    bg1 = ROLES[
        (i + 2) % len(ROLES)
    ]

    bg2 = ROLES[
        (i + 3) % len(ROLES)
    ]

    present_roles = (
        target_role,
        bg1,
        bg2,
    )

    absent_roles = (
        substitute,
        bg1,
        bg2,
    )

    assert (
        target_role
        in present_roles
    )

    assert (
        target_role
        not in absent_roles
    )

    assert len(
        set(present_roles)
    ) == 3

    assert len(
        set(absent_roles)
    ) == 3

    systems[
        target_role
    ] = {
        "present":
            present_roles,

        "absent":
            absent_roles,

        "substitute":
            substitute,

        "background":
            (bg1, bg2),
    }


# Save exact privileged experiment design.
(
    ROOT
    / "system_pairs_privileged.json"
).write_text(
    json.dumps(
        systems,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


tasks = []

for probe_index, probe in enumerate(
    bank
):

    target_role = str(
        probe["role"]
    )

    for condition in [
        "present",
        "absent",
    ]:

        roles = systems[
            target_role
        ][condition]

        tasks.append({
            "probe_index":
                probe_index,

            "target_role":
                target_role,

            "family":
                probe["family"],

            "rep":
                int(probe["rep"]),

            "scenario":
                probe["scenario"],

            "query":
                probe["query"],

            "condition":
                condition,

            "worker_roles":
                list(roles),

            "target_installed_eval_only":
                condition == "present",
        })


assert len(tasks) == 32


def execute_one(task):

    target_role = task[
        "target_role"
    ]

    condition = task[
        "condition"
    ]

    roles = tuple(
        task[
            "worker_roles"
        ]
    )

    sid = (
        f"elicitation_v1_"
        f"{target_role}_"
        f"{task['family']}_"
        f"r{task['rep']}_"
        f"{condition}"
    )

    spec = SingleRouterSystemSpec(
        system_id=sid,
        split="elicitation_presence_pilot_v1",
        execution_mode="single",
        style="s1",
        worker_roles=roles,
    )

    last = None

    for attempt in range(3):

        try:

            t0 = time.time()

            mas = SingleRouterFanoutMAS(
                spec,
                model=MODEL,
            )

            output = str(
                mas.query(
                    task["query"]
                )
            )

            elapsed = (
                time.time()
                - t0
            )

            trace = getattr(
                mas,
                "_last_trace",
                None,
            )

            if trace is None:
                raise RuntimeError(
                    "MAS did not expose "
                    "_last_trace"
                )

            selected = list(
                map(
                    str,
                    getattr(
                        trace,
                        "selected_roles",
                        (),
                    ),
                )
            )

            return {
                **task,

                "system_id":
                    sid,

                "execution_mode":
                    "single",

                "style":
                    "s1",

                "model":
                    MODEL,

                "final_output":
                    output,

                "selected_roles_eval_only":
                    selected,

                "target_selected_eval_only":
                    target_role
                    in selected,

                "elapsed_seconds":
                    elapsed,

                "attempt":
                    attempt,

                "execution_status":
                    "ok",
            }

        except Exception as exc:

            last = exc

            if attempt >= 2:
                break

            time.sleep(
                2 ** attempt
            )

    return {
        **task,

        "system_id":
            sid,

        "execution_mode":
            "single",

        "style":
            "s1",

        "model":
            MODEL,

        "final_output":
            "",

        "selected_roles_eval_only":
            [],

        "target_selected_eval_only":
            False,

        "elapsed_seconds":
            None,

        "attempt":
            2,

        "execution_status":
            "error",

        "error":
            repr(last),
    }


print(
    "===== ELICITATION MAS PRESENCE PILOT V1 =====",
    flush=True,
)

print(
    "model:",
    MODEL,
    flush=True,
)

print(
    "tasks:",
    len(tasks),
    flush=True,
)

print(
    "concurrency:",
    WORKERS,
    flush=True,
)

print(
    "\n===== MATCHED SYSTEM PAIRS =====",
    flush=True,
)

for role in ROLES:

    x = systems[role]

    print(
        role,
        "PRESENT=",
        list(
            x["present"]
        ),
        "ABSENT=",
        list(
            x["absent"]
        ),
        flush=True,
    )


rows = []

with ThreadPoolExecutor(
    max_workers=WORKERS
) as pool:

    futures = {
        pool.submit(
            execute_one,
            task,
        ): task

        for task in tasks
    }

    for n, fut in enumerate(
        as_completed(
            futures
        ),
        1,
    ):

        row = fut.result()

        rows.append(
            row
        )

        print(
            f"[{n:02d}/32] "
            f"{row['target_role']:<10} "
            f"{row['condition']:<7} "
            f"{row['family']:<10} "
            f"selected="
            f"{row['selected_roles_eval_only']} "
            f"hit="
            f"{int(row['target_selected_eval_only'])} "
            f"status="
            f"{row['execution_status']}",
            flush=True,
        )


rows.sort(
    key=lambda x: (
        ROLES.index(
            x["target_role"]
        ),
        x["family"],
        x["rep"],
        x["condition"],
    )
)


with (
    ROOT
    / "responses.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


ok = [
    x
    for x in rows
    if x[
        "execution_status"
    ] == "ok"
]


present = [
    x
    for x in ok
    if x["condition"]
    == "present"
]

absent = [
    x
    for x in ok
    if x["condition"]
    == "absent"
]


summary_roles = {}


for role in ROLES:

    rp = [
        x
        for x in present
        if x[
            "target_role"
        ] == role
    ]

    ra = [
        x
        for x in absent
        if x[
            "target_role"
        ] == role
    ]

    present_hits = sum(
        bool(
            x[
                "target_selected_eval_only"
            ]
        )
        for x in rp
    )

    fallbacks = Counter()

    for x in ra:

        for selected_role in x[
            "selected_roles_eval_only"
        ]:

            fallbacks[
                selected_role
            ] += 1

    summary_roles[
        role
    ] = {
        "present_n":
            len(rp),

        "present_target_hits":
            present_hits,

        "present_target_hit_rate":
            (
                present_hits
                / len(rp)
                if rp
                else None
            ),

        "absent_n":
            len(ra),

        "absent_fallback_counts":
            dict(
                fallbacks
            ),

        "present_selected_roles":
            [
                x[
                    "selected_roles_eval_only"
                ]
                for x in rp
            ],

        "absent_selected_roles":
            [
                x[
                    "selected_roles_eval_only"
                ]
                for x in ra
            ],
    }


overall_hits = sum(
    bool(
        x[
            "target_selected_eval_only"
        ]
    )
    for x in present
)


summary = {
    "experiment":
        "elicitation_mas_presence_pilot_v1",

    "model":
        MODEL,

    "execution_mode":
        "single",

    "style":
        "s1",

    "K":
        3,

    "queries_planned":
        32,

    "queries_ok":
        len(ok),

    "present_queries":
        len(present),

    "absent_queries":
        len(absent),

    "overall_present_target_hits":
        overall_hits,

    "overall_present_target_hit_rate":
        (
            overall_hits
            / len(present)
            if present
            else None
        ),

    "roles":
        summary_roles,

    "protocol": {
        "generator_bank":
            str(BANK),

        "selected_roles_usage":
            "evaluation_only",

        "final_outputs_saved":
            True,

        "truth_used_for_attack":
            False,

        "purpose":
            "routing/exposure sanity check before black-box presence recovery",
    },
}


(
    ROOT
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


print(
    "\n"
    + "=" * 100
)

print(
    "RESULTS"
)

print(
    "=" * 100
)

print(
    f"OK: {len(ok)}/32"
)

print(
    "OVERALL PRESENT TARGET HIT:",
    f"{overall_hits}/{len(present)}",
    (
        f"= {overall_hits / len(present):.3f}"
        if present
        else ""
    ),
)


for role in ROLES:

    x = summary_roles[
        role
    ]

    print(
        f"\n{role}"
    )

    print(
        "  present hit:",
        f"{x['present_target_hits']}/"
        f"{x['present_n']}",
        "=",
        x[
            "present_target_hit_rate"
        ],
    )

    print(
        "  absent fallbacks:",
        x[
            "absent_fallback_counts"
        ],
    )


print(
    "\nwrote:",
    ROOT
    / "responses.jsonl"
)

print(
    "wrote:",
    ROOT
    / "summary.json"
)
