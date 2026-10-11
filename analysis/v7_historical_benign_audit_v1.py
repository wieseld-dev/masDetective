from __future__ import annotations

import json
import re
import random
from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/v7_historical_benign_audit_v1")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ---------------------------------------------------------------------
# Conservative audit rules.
#
# IMPORTANT:
# TASK_SEMANTIC_DIRECT is NOT a benignness failure.
#
# We only reject here for the three historical failure families we have
# already identified:
#   1. assistant->user / reversed direction
#   2. system/capability introspection
#   3. dangling/deictic prompts that depend on hidden previous context
# ---------------------------------------------------------------------

DIRECTION_PATTERNS = [
    r"\bwhat .* would you like (?:help|assistance) with\b",
    r"\bwhat .* do you need (?:help|assistance) with\b",
    r"\bwhich .* would you like (?:help|assistance) with\b",
    r"\bwhich .* do you need (?:help|assistance) with\b",
    r"\bwould you like (?:help|assistance)\b",
    r"\bdo you need (?:help|assistance)\b",
    r"\bhow can i help you\b",
    r"\bhow can i assist you\b",
    r"\bwhat can i help you with\b",
    r"\bwhat can i assist you with\b",
    r"\bwhat specific .* would you like\b",
    r"\bwhich specific .* would you like\b",
]

SYSTEM_SELF_PATTERNS = [
    r"\bdo you have experience\b",
    r"\bhave you worked with\b",
    r"\bare you experienced\b",
    r"\bare you familiar with\b",
    r"\bare you knowledgeable\b",
    r"\bwhat are your .*capabilit",
    r"\bwhat capabilities do you\b",
    r"\bwhat is your expertise\b",
    r"\bwhat are your strengths\b",
    r"\bwhat can you do\b",
    r"\bcan you handle\b",
    r"\bcan you perform\b",
    r"\bcan you analyze\b",
    r"\bwhich .* agent\b",
    r"\bwhat .* agent\b",
    r"\bdo you have an? .* agent\b",
    r"\bdo you use .* agent\b",
    r"\bwhat tools? do you\b",
    r"\bwhich tools? do you\b",
    r"\bwhat model do you\b",
    r"\bwhich model do you\b",
    r"\byour internal\b",
    r"\byour system\b",
    r"\byour architecture\b",
    r"\byour workflow\b",
    r"\byour hidden\b",
]

DANGLING_PATTERNS = [
    r"\bthese tasks\b",
    r"\bthese projects\b",
    r"\bthese assignments\b",
    r"\bthese activities\b",
    r"\bthese requirements\b",
    r"\bthese problems\b",
    r"\bthese requests\b",
    r"\bthese queries\b",
    r"\bthese analyses\b",
    r"\bthis task\b",
    r"\bthis capability\b",
    r"\bthose tasks\b",
    r"\bthe above\b",
    r"\bthe previous\b",
]

# Not a rejection. Used only to characterize how obvious the semantic
# domain is directly from q.
DIRECT_DOMAIN_TERMS = {
    "accessibility_review": [
        "accessibility",
        "accessible",
        "wcag",
    ],
    "database_design": [
        "database",
        "databases",
        "sql",
        "database query",
        "database queries",
        "schema",
    ],
    "financial_analysis": [
        "financial",
        "finance",
        "revenue",
        "profit",
        "cash flow",
    ],
    "customer_support_triage": [
        "customer support",
        "customer service",
        "customer inquiries",
        "support ticket",
        "support tickets",
        "triage",
    ],
    "experiment_design": [
        "experiment",
        "experimental design",
        "ab test",
        "a/b test",
    ],
    "qa_testing": [
        "quality assurance",
        "qa testing",
        "software testing",
        "test cases",
    ],
    "risk_assessment": [
        "risk assessment",
        "risk analysis",
        "risks",
    ],
    "requirements_analysis": [
        "requirements analysis",
        "requirements",
    ],
    "incident_management": [
        "incident management",
        "incident response",
        "incident",
    ],
    "translation": [
        "translate",
        "translation",
    ],
    "documentation": [
        "documentation",
        "documenting",
    ],
    "interface_design": [
        "user interface",
        "interface",
        "interfaces",
        "usability",
    ],
    "data": [
        "dataset",
        "data analysis",
        "analyze data",
        "large volumes of data",
    ],
    "math": [
        "mathematical",
        "mathematics",
        "statistical",
        "statistics",
    ],
}


def norm(s):
    return " ".join(str(s or "").strip().lower().split())


def any_match(patterns, text):
    return [
        p for p in patterns
        if re.search(p, text, flags=re.I)
    ]


def direct_domains(text):
    hits = []

    for role, terms in DIRECT_DOMAIN_TERMS.items():
        if any(term in text for term in terms):
            hits.append(role)

    return sorted(set(hits))


def classify_query(q):
    t = norm(q)

    direction = any_match(
        DIRECTION_PATTERNS,
        t,
    )

    system_self = any_match(
        SYSTEM_SELF_PATTERNS,
        t,
    )

    dangling = any_match(
        DANGLING_PATTERNS,
        t,
    )

    failures = []

    if direction:
        failures.append("DIRECTION_BAD")

    if system_self:
        failures.append("SYSTEM_SELF_BAD")

    if dangling:
        failures.append("DANGLING_BAD")

    domains = direct_domains(t)

    return {
        "strict_benign":
            len(failures) == 0,

        "failures":
            failures,

        "direction_matches":
            direction,

        "system_self_matches":
            system_self,

        "dangling_matches":
            dangling,

        "task_semantic_direct":
            bool(domains),

        "direct_domains":
            domains,
    }


def stage_cycle_round(path):
    parts = path.parts

    stage = next(
        (
            x for x in parts
            if x.startswith("stage_")
        ),
        None,
    )

    cycle = next(
        (
            x for x in parts
            if x.startswith("cycle_")
        ),
        None,
    )

    m = re.search(
        r"round_(\d+)",
        path.name,
    )

    rnd = (
        int(m.group(1))
        if m
        else None
    )

    return stage, cycle, rnd


files = sorted(
    ROOT.glob(
        "**/responses_train_round_*.jsonl"
    )
)

print(
    "response files:",
    len(files),
    flush=True,
)

all_rows = []

raw_rows = 0
bad_json = 0
missing_query = 0
missing_output = 0
missing_labels = 0
not_ok = 0
not_valid = 0

for path in files:
    stage, cycle, rnd = (
        stage_cycle_round(path)
    )

    with path.open(
        encoding="utf-8"
    ) as f:
        for line_no, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            raw_rows += 1

            try:
                x = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            query = x.get("query")

            output = (
                x.get("final_output")
                or x.get("answer")
                or x.get("output")
            )

            selected = x.get(
                "selected_roles_training_only"
            )

            truth = x.get(
                "truth_roles_training_only"
            )

            if not query:
                missing_query += 1
                continue

            if not output:
                missing_output += 1
                continue

            if not isinstance(
                selected,
                list,
            ):
                missing_labels += 1
                continue

            status = str(
                x.get(
                    "execution_status",
                    "",
                )
            ).lower()

            if status not in {
                "ok",
                "success",
                "completed",
            }:
                not_ok += 1
                continue

            if x.get("valid") is not True:
                not_valid += 1
                continue

            audit = classify_query(
                query
            )

            row = {
                "source_file":
                    str(path),

                "source_line":
                    line_no,

                "stage":
                    stage,

                "cycle":
                    cycle,

                "round":
                    rnd,

                "session_id":
                    str(
                        x.get(
                            "session_id",
                            x.get(
                                "system_id",
                                "",
                            ),
                        )
                    ),

                "system_id":
                    str(
                        x.get(
                            "system_id",
                            "",
                        )
                    ),

                "candidate_index":
                    x.get(
                        "candidate_index"
                    ),

                "chosen_for_continuation":
                    x.get(
                        "chosen_for_continuation"
                    ),

                "query":
                    str(query),

                "final_output":
                    str(output),

                # E: actual selected/activated roles.
                "selected_roles":
                    sorted(
                        map(
                            str,
                            selected,
                        )
                    ),

                # Z: installed roles.
                "truth_roles":
                    sorted(
                        map(
                            str,
                            truth or [],
                        )
                    ),

                "strict_benign":
                    audit[
                        "strict_benign"
                    ],

                "benign_failures":
                    audit[
                        "failures"
                    ],

                "task_semantic_direct":
                    audit[
                        "task_semantic_direct"
                    ],

                "direct_domains":
                    audit[
                        "direct_domains"
                    ],
            }

            all_rows.append(row)


# ---------------------------------------------------------
# Deduplicate exact executions.
#
# Same session + same query + same output + same activation set
# is identical evidence for the purposes of this inventory.
# ---------------------------------------------------------

dedup = {}
duplicate_rows = 0

for x in all_rows:
    key = (
        x["session_id"],
        x["query"],
        x["final_output"],
        tuple(
            x["selected_roles"]
        ),
    )

    if key in dedup:
        duplicate_rows += 1
        continue

    dedup[key] = x

rows = list(
    dedup.values()
)

strict = [
    x for x in rows
    if x["strict_benign"]
]

rejected = [
    x for x in rows
    if not x["strict_benign"]
]


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

failure_counts = Counter()

for x in rows:
    for reason in x[
        "benign_failures"
    ]:
        failure_counts[
            reason
        ] += 1


role_pos = Counter()

for x in strict:
    for r in x[
        "selected_roles"
    ]:
        role_pos[r] += 1


installed_role_counts = Counter()

for x in strict:
    for r in x[
        "truth_roles"
    ]:
        installed_role_counts[
            r
        ] += 1


stage_stats = defaultdict(
    lambda: {
        "all": 0,
        "strict_benign": 0,
        "task_semantic_direct": 0,
        "sessions": set(),
    }
)

for x in rows:
    s = x["stage"]

    stage_stats[s]["all"] += 1

    stage_stats[s][
        "strict_benign"
    ] += int(
        x["strict_benign"]
    )

    stage_stats[s][
        "task_semantic_direct"
    ] += int(
        x[
            "task_semantic_direct"
        ]
    )

    stage_stats[s][
        "sessions"
    ].add(
        x["session_id"]
    )


stage_summary = {}

for s, v in sorted(
    stage_stats.items()
):
    stage_summary[s] = {
        "all":
            v["all"],

        "strict_benign":
            v["strict_benign"],

        "strict_benign_rate":
            (
                v["strict_benign"]
                / v["all"]
                if v["all"]
                else None
            ),

        "task_semantic_direct":
            v[
                "task_semantic_direct"
            ],

        "sessions":
            len(
                v["sessions"]
            ),
    }


sessions_all = {
    x["session_id"]
    for x in rows
    if x["session_id"]
}

sessions_strict = {
    x["session_id"]
    for x in strict
    if x["session_id"]
}


summary = {
    "source": {
        "files":
            len(files),

        "raw_rows":
            raw_rows,

        "bad_json":
            bad_json,

        "missing_query":
            missing_query,

        "missing_output":
            missing_output,

        "missing_selected_roles":
            missing_labels,

        "execution_not_ok":
            not_ok,

        "valid_not_true":
            not_valid,
    },

    "usable_before_benign_filter":
        len(all_rows),

    "exact_duplicates_removed":
        duplicate_rows,

    "unique_usable":
        len(rows),

    "strict_benign":
        len(strict),

    "strict_benign_rate":
        (
            len(strict) / len(rows)
            if rows
            else None
        ),

    "rejected":
        len(rejected),

    "failure_counts":
        dict(
            failure_counts
        ),

    "task_semantic_direct_among_strict":
        sum(
            int(
                x[
                    "task_semantic_direct"
                ]
            )
            for x in strict
        ),

    "task_semantic_direct_rate_among_strict":
        (
            sum(
                int(
                    x[
                        "task_semantic_direct"
                    ]
                )
                for x in strict
            )
            / len(strict)
            if strict
            else None
        ),

    "unique_sessions_all":
        len(sessions_all),

    "unique_sessions_strict":
        len(sessions_strict),

    "selected_role_positive_counts_strict":
        dict(
            sorted(
                role_pos.items()
            )
        ),

    "installed_role_counts_strict":
        dict(
            sorted(
                installed_role_counts.items()
            )
        ),

    "by_stage":
        stage_summary,
}


# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

with (
    OUT / "strict_benign.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in strict:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


with (
    OUT / "rejected.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in rejected:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------
# Human-audit samples.
# ---------------------------------------------------------

def sample_rows(xs, n):
    if len(xs) <= n:
        return list(xs)

    return rng.sample(
        xs,
        n,
    )


audit_sample = {
    "strict_benign_random":
        sample_rows(
            strict,
            30,
        ),

    "task_semantic_direct_strict":
        sample_rows(
            [
                x for x in strict
                if x[
                    "task_semantic_direct"
                ]
            ],
            20,
        ),

    "rejected_direction":
        sample_rows(
            [
                x for x in rejected
                if "DIRECTION_BAD"
                in x[
                    "benign_failures"
                ]
            ],
            15,
        ),

    "rejected_system_self":
        sample_rows(
            [
                x for x in rejected
                if "SYSTEM_SELF_BAD"
                in x[
                    "benign_failures"
                ]
            ],
            15,
        ),

    "rejected_dangling":
        sample_rows(
            [
                x for x in rejected
                if "DANGLING_BAD"
                in x[
                    "benign_failures"
                ]
            ],
            15,
        ),
}


(
    OUT / "human_audit_sample.json"
).write_text(
    json.dumps(
        audit_sample,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------
# Console report.
# ---------------------------------------------------------

print(
    "\n===== HISTORICAL BENIGN SENSOR INVENTORY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print(
    "\n===== 15 RANDOM STRICT-BENIGN EXAMPLES ====="
)

for i, x in enumerate(
    sample_rows(
        strict,
        15,
    ),
    1,
):
    print(
        f"\n[{i:02d}] "
        f"{x['stage']} "
        f"{x['cycle']} "
        f"round={x['round']} "
        f"selected={x['selected_roles']}"
    )

    print(
        "Q:",
        x["query"],
    )

print(
    "\nSaved:"
)

print(
    " ",
    OUT / "summary.json",
)

print(
    " ",
    OUT / "strict_benign.jsonl",
)

print(
    " ",
    OUT / "human_audit_sample.json",
)
