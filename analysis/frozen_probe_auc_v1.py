#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/frozen_probe_auc_v1")
ROOT.mkdir(parents=True, exist_ok=True)

BANK_V1 = Path(
    "artifacts/generator_frozen_v1/selected_bank.jsonl"
)

BANK_V2 = Path(
    "artifacts/generator_frozen_v2/selected_bank.jsonl"
)

BANK_V2_MANIFEST = Path(
    "artifacts/generator_frozen_v2/manifest.json"
)

V1210 = Path(
    "artifacts/generator_v12_10/all_candidates.jsonl"
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


# 10 distinct composition families.
# Counts are nearly balanced:
# code/planning/research=5, others=4.
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


MODEL = os.environ.get(
    "MAS_MODEL",
    "gpt-4o-mini",
)

WORKERS = int(
    os.environ.get(
        "AUC_API_WORKERS",
        "6",
    )
)


def read_jsonl(path):
    p = Path(path)

    if not p.exists():
        return []

    return [
        json.loads(line)
        for line in p.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def append_jsonl(path, row):
    with Path(path).open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


def ensure_frozen_v2():
    if BANK_V2.exists():
        rows = read_jsonl(
            BANK_V2
        )

        if len(rows) != 16:
            raise RuntimeError(
                f"Existing frozen_v2 has "
                f"{len(rows)} rows, expected 16"
            )

        return rows

    print(
        "[bank] generator_frozen_v2 missing; "
        "building it now from frozen_v1.",
        flush=True,
    )

    if not BANK_V1.exists():
        raise FileNotFoundError(
            BANK_V1
        )

    if not V1210.exists():
        raise FileNotFoundError(
            V1210
        )

    base = read_jsonl(
        BANK_V1
    )

    pool = read_jsonl(
        V1210
    )

    assert len(base) == 16

    kept = []

    removed = []

    for row in base:

        if (
            row["role"] == "research"
            and
            row["family"] == "intake"
        ):
            removed.append(
                row
            )
        else:
            kept.append(
                row
            )

    assert len(removed) == 1

    replacement = [
        row
        for row in pool
        if (
            row["role"] == "research"
            and
            row["family"] == "intake"
            and
            int(row["rep"]) == 3
        )
    ]

    assert len(replacement) == 1

    repl = {
        **replacement[0],

        "final_source":
            "V12.10 candidate pool",

        "final_reason":
            "non-medical research replacement",
    }

    final = kept + [
        repl
    ]

    final.sort(
        key=lambda x: (
            ROLES.index(
                x["role"]
            ),
            x["family"],
            int(x["rep"]),
        )
    )

    counts = Counter(
        x["role"]
        for x in final
    )

    assert len(final) == 16

    for role in ROLES:
        assert counts[role] == 2

    BANK_V2.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BANK_V2.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in final:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    sha = hashlib.sha256(
        BANK_V2.read_bytes()
    ).hexdigest()

    manifest = {
        "name":
            "generator_frozen_v2",

        "status":
            "FROZEN",

        "n_roles":
            8,

        "probes_per_role":
            2,

        "n_probes":
            16,

        "development_roles":
            ROLES,

        "research_replacement":
            "V12.10 research intake rep=3",

        "sha256":
            sha,
    }

    BANK_V2_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[bank] wrote:",
        BANK_V2,
        flush=True,
    )

    print(
        "[bank] sha256:",
        sha,
        flush=True,
    )

    return final


bank = ensure_frozen_v2()

counts = Counter(
    x["role"]
    for x in bank
)

assert len(bank) == 16

for role in ROLES:
    assert counts[role] == 2


# ------------------------------------------------------------
# Register role prompts and load actual MAS.
# ------------------------------------------------------------

from targets.single_router_zoo_v11 import (
    validate_v11_design,
)

validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


# ------------------------------------------------------------
# Experimental design
# ------------------------------------------------------------

systems = []

for comp_id, comp_roles in COMPOSITIONS.items():

    for style in STYLES:

        systems.append({
            "system_id":
                f"{comp_id}_{style}",

            "comp_id":
                comp_id,

            "style":
                style,

            "roles":
                list(comp_roles),

            "K":
                len(comp_roles),
        })


tasks = []

for system in systems:

    installed = set(
        system["roles"]
    )

    for probe_index, probe in enumerate(
        bank
    ):

        role = str(
            probe["role"]
        )

        tasks.append({
            "system_id":
                system["system_id"],

            "comp_id":
                system["comp_id"],

            "style":
                system["style"],

            "worker_roles":
                list(
                    system["roles"]
                ),

            "K_eval_only":
                system["K"],

            "probe_index":
                probe_index,

            "target_role":
                role,

            "family":
                probe["family"],

            "rep":
                int(
                    probe["rep"]
                ),

            "query":
                probe["query"],

            "scenario":
                probe["scenario"],

            "target_present_eval_only":
                int(
                    role
                    in installed
                ),
        })


assert len(tasks) == 320


print(
    "===== FROZEN PROBE AUC V1 =====",
    flush=True,
)

print(
    "bank:",
    BANK_V2,
    flush=True,
)

print(
    "systems:",
    len(systems),
    flush=True,
)

print(
    "compositions:",
    len(COMPOSITIONS),
    flush=True,
)

print(
    "styles:",
    STYLES,
    flush=True,
)

print(
    "probes/system:",
    len(bank),
    flush=True,
)

print(
    "MAS interactions:",
    len(tasks),
    flush=True,
)

print(
    "expected API calls:",
    2 * len(tasks),
    flush=True,
)

print(
    "model:",
    MODEL,
    flush=True,
)

print(
    "concurrency:",
    WORKERS,
    flush=True,
)


role_presence = Counter()

for comp in COMPOSITIONS.values():
    role_presence.update(
        comp
    )

print(
    "role presence across composition families:",
    dict(role_presence),
    flush=True,
)


# ------------------------------------------------------------
# Resumable MAS collection
# ------------------------------------------------------------

RESPONSES = (
    ROOT
    / "responses.jsonl"
)


def task_key(row):
    return (
        str(
            row["system_id"]
        ),
        str(
            row["target_role"]
        ),
        str(
            row["family"]
        ),
        int(
            row["rep"]
        ),
    )


existing_rows = read_jsonl(
    RESPONSES
)

existing = {
    task_key(row):
        row
    for row in existing_rows
    if (
        row.get(
            "execution_status"
        )
        == "ok"
    )
}


pending = [
    task
    for task in tasks
    if task_key(task)
    not in existing
]


print(
    "existing:",
    len(existing),
    flush=True,
)

print(
    "pending:",
    len(pending),
    flush=True,
)


def execute_one(task):

    spec = SingleRouterSystemSpec(
        system_id=
            task["system_id"],

        split=
            "frozen_probe_auc_v1",

        execution_mode=
            "single",

        style=
            task["style"],

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
                    "missing _last_trace"
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

                "output":
                    output,

                "final_output":
                    output,

                # Evaluation-only.
                "selected_roles_eval_only":
                    selected,

                "target_selected_eval_only":
                    int(
                        task[
                            "target_role"
                        ]
                        in selected
                    ),

                "elapsed_seconds":
                    elapsed,

                "retries_used":
                    attempt,

                "execution_status":
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

        "output":
            "",

        "final_output":
            "",

        "selected_roles_eval_only":
            [],

        "target_selected_eval_only":
            0,

        "elapsed_seconds":
            None,

        "retries_used":
            2,

        "execution_status":
            "error",

        "error":
            repr(last),
    }


if pending:

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:

        futures = {
            pool.submit(
                execute_one,
                task,
            ): task

            for task in pending
        }

        for n, fut in enumerate(
            as_completed(
                futures
            ),
            1,
        ):

            row = fut.result()

            append_jsonl(
                RESPONSES,
                row,
            )

            if (
                n <= 20
                or n % 20 == 0
                or row[
                    "execution_status"
                ] != "ok"
            ):
                print(
                    f"[collect "
                    f"{n}/{len(pending)}] "
                    f"{row['system_id']} "
                    f"target="
                    f"{row['target_role']} "
                    f"present="
                    f"{row['target_present_eval_only']} "
                    f"selected="
                    f"{row['selected_roles_eval_only']} "
                    f"status="
                    f"{row['execution_status']}",
                    flush=True,
                )


# ------------------------------------------------------------
# Reload completed dataset.
# ------------------------------------------------------------

all_rows = read_jsonl(
    RESPONSES
)

by_key = {}

for row in all_rows:

    if row.get(
        "execution_status"
    ) == "ok":

        by_key[
            task_key(
                row
            )
        ] = row


rows = []

for task in tasks:

    k = task_key(
        task
    )

    if k in by_key:
        rows.append(
            by_key[k]
        )


print(
    "\n===== COLLECTION COMPLETE =====",
    flush=True,
)

print(
    "successful:",
    len(rows),
    "/",
    len(tasks),
    flush=True,
)


if len(rows) != len(tasks):

    raise RuntimeError(
        f"Only {len(rows)}/{len(tasks)} "
        "successful interactions. "
        "Re-run job to resume."
    )


# ------------------------------------------------------------
# Free routing diagnostic.
# This is NEVER supplied to the detector.
# ------------------------------------------------------------

present_rows = [
    row
    for row in rows
    if row[
        "target_present_eval_only"
    ] == 1
]

routing_hits = sum(
    row[
        "target_selected_eval_only"
    ]
    for row in present_rows
)

print(
    "present->target routing:",
    f"{routing_hits}/"
    f"{len(present_rows)}",
    "=",
    routing_hits
    / len(present_rows),
    flush=True,
)


# ------------------------------------------------------------
# BLACK-BOX DIRECT PRESENCE DETECTOR
#
# Features:
#   target role description
#   query
#   final external output
#
# NO:
#   selected role
#   installed roles
#   K
#   router output
#   traces
# ------------------------------------------------------------

print(
    "\n===== LOADING EMBEDDER =====",
    flush=True,
)

from sentence_transformers import (
    SentenceTransformer,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
)

from analysis.adaptive_fanout_count_v12 import (
    SharedPresenceDetector,
)


class NormalizedEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/"
            "all-MiniLM-L6-v2",
            device="cpu",
        )

    def encode(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


embedder = NormalizedEmbedder()

detector = SharedPresenceDetector(
    embedder,
    {
        role:
            ROLE_DESCRIPTIONS[
                role
            ]
        for role in ROLES
    },
    c=0.05,
)


examples = []

for row in rows:

    examples.append({
        "role":
            row["target_role"],

        "query":
            row["query"],

        "output":
            row["final_output"],

        # Privileged label used only during
        # offline training/evaluation.
        "y":
            int(
                row[
                    "target_present_eval_only"
                ]
            ),

        "weight":
            1.0,

        # Both router styles for a composition
        # stay together in the same fold.
        "group":
            row["comp_id"],
    })


print(
    "detector examples:",
    len(examples),
    flush=True,
)

print(
    "OOF groups:",
    len(
        set(
            x["group"]
            for x in examples
        )
    ),
    flush=True,
)


result = detector.fit(
    examples,
    folds=5,
)


per_role = result[
    "per_role_auc"
]

valid_role_aucs = [
    float(v)
    for v in per_role.values()
    if v is not None
]


summary = {
    "experiment":
        "frozen_probe_auc_v1",

    "bank":
        str(BANK_V2),

    "model":
        MODEL,

    "n_interactions":
        len(rows),

    "n_compositions":
        len(COMPOSITIONS),

    "styles":
        STYLES,

    "execution_mode":
        "single",

    "detector_input":
        [
            "candidate_role_description",
            "query",
            "final_output",
        ],

    "forbidden_from_detector":
        [
            "selected_roles",
            "router_output",
            "installed_role_set",
            "K",
            "internal_trace",
        ],

    "cv_protocol":
        "5-fold GroupKFold by composition; "
        "both router styles of a composition "
        "are held out together",

    "oof_auc":
        result["auc"],

    "oof_balanced_accuracy":
        result[
            "balanced_accuracy_at_oof_threshold"
        ],

    "mean_per_role_auc":
        (
            float(
                np.mean(
                    valid_role_aucs
                )
            )
            if valid_role_aucs
            else None
        ),

    "median_per_role_auc":
        (
            float(
                np.median(
                    valid_role_aucs
                )
            )
            if valid_role_aucs
            else None
        ),

    "per_role_auc":
        per_role,

    "present_target_routing_hit_rate_eval_only":
        routing_hits
        / len(
            present_rows
        ),

    "present_target_routing_hits_eval_only":
        routing_hits,

    "present_target_routing_n_eval_only":
        len(
            present_rows
        ),
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
    "FROZEN-PROBE DIRECT PRESENCE RESULT"
)

print(
    "=" * 100
)

print(
    "OOF AUC:",
    f"{summary['oof_auc']:.4f}",
)

print(
    "OOF BAL ACC:",
    f"{summary['oof_balanced_accuracy']:.4f}",
)

print(
    "MEAN PER-ROLE AUC:",
    f"{summary['mean_per_role_auc']:.4f}",
)

print(
    "MEDIAN PER-ROLE AUC:",
    f"{summary['median_per_role_auc']:.4f}",
)


print(
    "\nPER ROLE:"
)

for role in ROLES:

    value = per_role.get(
        role
    )

    if value is None:
        print(
            f"{role:<12} NA"
        )
    else:
        print(
            f"{role:<12} "
            f"{value:.4f}"
        )


print(
    "\nROUTING DIAGNOSTIC "
    "(EVAL ONLY):"
)

print(
    f"{routing_hits}/"
    f"{len(present_rows)} "
    f"= "
    f"{routing_hits / len(present_rows):.4f}"
)

print(
    "\nwrote:",
    ROOT
    / "summary.json"
)
