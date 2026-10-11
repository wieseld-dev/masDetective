from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict


SRC = Path(
    "artifacts/v7_historical_benign_audit_v1/strict_benign.jsonl"
)

OUT = Path(
    "artifacts/v7_historical_benign_audit_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ============================================================
# 1. BENIGNNESS FAILURES
# ============================================================

# Assistant -> user phrasing.
REVERSED_DIRECTION = [
    r"\bwould you like\b",
    r"\bdo you want me to\b",
    r"\bwould you prefer\b",
    r"\bhow can i help you\b",
    r"\bhow can i assist you\b",
    r"\bwhat can i help you with\b",
    r"\bwhat can i assist you with\b",
    r"\bwhat .* would you like .* with\b",
    r"\bwhich .* would you like .* with\b",
]

# Asking about the system rather than asking it to solve the user's task.
SYSTEM_SELF = [
    r"\bdo you have experience\b",
    r"\bhave you worked with\b",
    r"\bare you experienced\b",
    r"\bare you familiar with\b",
    r"\bare you knowledgeable\b",
    r"\bwhat are your .*capabilit",
    r"\bwhat capabilities do you\b",
    r"\bwhat are your strengths\b",
    r"\bwhat is your expertise\b",
    r"\bwhat .* are you good at\b",
    r"\bwhat .* do you specialize in\b",
    r"\bwhich .* do you specialize in\b",
    r"\bwhat would you find .*challenging\b",
    r"\bwhich .* would you find .*challenging\b",
    r"\bwhat do you find .*challenging\b",
    r"\byour internal\b",
    r"\byour architecture\b",
    r"\byour hidden\b",
    r"\byour workflow\b",
    r"\byour agents?\b",
    r"\byour tools?\b",
    r"\byour capabilities\b",
]

# Capability / routing probes even if not literally phrased as system-self.
CAPABILITY_ROUTING = [
    r"\bwhich .*skill areas?\b",
    r"\bwhich .*skill set\b",
    r"\bwhat .*skill areas?\b",
    r"\bwhat .*skill set\b",
    r"\bwhich .*capabilit",
    r"\bwhat .*capabilit",
    r"\bcurrent capabilities\b",
    r"\brequire assistance with\b",
    r"\brequires assistance with\b",
    r"\bneed assistance with\b",
    r"\bneed help with\b",
    r"\bwhich .*expertise\b",
    r"\bwhat .*expertise\b",
    r"\bbest suited to\b",
    r"\bbest suited for\b",
]

# Deictic references which are not self-contained in a single black-box turn.
DANGLING = [
    r"\bthese tasks\b",
    r"\bthese questions\b",
    r"\bthese projects\b",
    r"\bthese assignments\b",
    r"\bthese activities\b",
    r"\bthese requirements\b",
    r"\bthese problems\b",
    r"\bthese requests\b",
    r"\bthese queries\b",
    r"\bthese analyses\b",
    r"\bthese pieces\b",
    r"\bthese items\b",
    r"\bthese examples\b",
    r"\bthese documents\b",
    r"\bthese texts\b",
    r"\bthose tasks\b",
    r"\bthose questions\b",
    r"\bthe above\b",
    r"\bthe previous\b",
]


# ============================================================
# 2. USER-LIKE FAILURES
# ============================================================

# Grammatically user->system, but the conversational act is unnatural:
# the user offers to explain/teach/do something for the assistant.
UNNATURAL_USER_ACT = [
    r"^\s*should i explain\b",
    r"^\s*should i start by explaining\b",
    r"^\s*should i begin by explaining\b",
    r"^\s*should i describe\b",
    r"^\s*should i tell you\b",
    r"^\s*should i provide an explanation\b",
    r"^\s*may i explain\b",
    r"^\s*can i explain\b",
]

# Generic capability-selection wording.
META_HELP_SELECTION = [
    r"\bwhat kind of help\b",
    r"\bwhat type of help\b",
    r"\bwhat kind of assistance\b",
    r"\bwhat type of assistance\b",
    r"\bwhich type of assistance\b",
    r"\bwhich kind of assistance\b",
]


# ============================================================
# 3. SUBSTANTIVENESS
# ============================================================

# These are BENIGN, but previous experiments showed that pure
# clarification / information-gathering prompts tend to produce
# generic outputs with weak fingerprints.
GENERIC_CLARIFICATION = [
    r"\bwhat information should i provide\b",
    r"\bwhat details should i provide\b",
    r"\bwhat information do you need\b",
    r"\bwhat details do you need\b",
    r"\bwhat should i tell you\b",
    r"\bwhat context should i provide\b",
    r"\bhow can i best provide .*details\b",
    r"\bhow should i provide .*information\b",
]

# A substantive request normally asks the system to reason, compare,
# recommend, evaluate, explain, identify, plan, analyze, improve, etc.
SUBSTANTIVE_MARKERS = [
    r"\bhow (?:can|should|would) i\b",
    r"\bhow do i\b",
    r"\bhow would you\b",
    r"\bwhat should i\b",
    r"\bwhat would you recommend\b",
    r"\bwhat do you recommend\b",
    r"\bwhich .* should i\b",
    r"\bwhich .* would you recommend\b",
    r"\bwhat .* should i examine\b",
    r"\bwhat .* should i consider\b",
    r"\bwhat .* should i evaluate\b",
    r"\bwhat .* should i prioritize\b",
    r"\bwhat .* should i look for\b",
    r"\bcan you explain\b",
    r"\bcan you compare\b",
    r"\bcan you analyze\b",
    r"\bcan you help me (?:analyze|evaluate|compare|plan|design|understand|improve)\b",
    r"\bhelp me (?:analyze|evaluate|compare|plan|design|understand|improve)\b",
    r"\bexplain\b",
    r"\banalyze\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\brecommend\b",
    r"\bidentify\b",
    r"\bprioritize\b",
    r"\bimprove\b",
    r"\bcalculate\b",
    r"\bdesign\b",
]


# ============================================================
# 4. TASK-SEMANTIC DIRECTNESS
#
# NOT A REJECTION.
# ============================================================

DIRECT_DOMAIN_TERMS = {
    "accessibility_review": [
        "accessibility",
        "accessible",
        "wcag",
    ],
    "code": [
        "code",
        "coding",
        "programming",
        "algorithm",
    ],
    "customer_support_triage": [
        "customer support",
        "customer service",
        "support ticket",
        "support tickets",
        "customer inquiries",
        "triage",
    ],
    "data": [
        "dataset",
        "data analysis",
        "analyze data",
        "data-driven",
    ],
    "database_design": [
        "database",
        "databases",
        "sql",
        "schema",
    ],
    "documentation": [
        "documentation",
        "documenting",
    ],
    "experiment_design": [
        "experiment",
        "experimental design",
        "a/b test",
        "ab test",
    ],
    "financial_analysis": [
        "financial",
        "finance",
        "revenue",
        "profit",
        "cash flow",
    ],
    "incident_management": [
        "incident management",
        "incident response",
    ],
    "math": [
        "mathematical",
        "mathematics",
        "statistical",
        "statistics",
    ],
    "qa_testing": [
        "quality assurance",
        "qa testing",
        "software testing",
        "test cases",
    ],
    "requirements_analysis": [
        "requirements analysis",
        "requirements",
    ],
    "risk_assessment": [
        "risk assessment",
        "risk analysis",
    ],
    "translation": [
        "translate",
        "translation",
    ],
}


def norm(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def matches(patterns, text):
    return [
        p
        for p in patterns
        if re.search(
            p,
            text,
            flags=re.I,
        )
    ]


def domain_hits(text):
    out = []

    for role, terms in DIRECT_DOMAIN_TERMS.items():
        if any(
            term in text
            for term in terms
        ):
            out.append(role)

    return sorted(
        set(out)
    )


def classify(q):
    t = norm(q)

    reverse = matches(
        REVERSED_DIRECTION,
        t,
    )

    self_q = matches(
        SYSTEM_SELF,
        t,
    )

    routing = matches(
        CAPABILITY_ROUTING,
        t,
    )

    dangling = matches(
        DANGLING,
        t,
    )

    unnatural = matches(
        UNNATURAL_USER_ACT,
        t,
    )

    meta_help = matches(
        META_HELP_SELECTION,
        t,
    )

    clarification = matches(
        GENERIC_CLARIFICATION,
        t,
    )

    substantive_hits = matches(
        SUBSTANTIVE_MARKERS,
        t,
    )

    benign_failures = []

    if reverse:
        benign_failures.append(
            "REVERSED_DIRECTION"
        )

    if self_q:
        benign_failures.append(
            "SYSTEM_SELF"
        )

    if routing:
        benign_failures.append(
            "CAPABILITY_ROUTING"
        )

    if dangling:
        benign_failures.append(
            "DANGLING"
        )

    userlike_failures = []

    if unnatural:
        userlike_failures.append(
            "UNNATURAL_USER_ACT"
        )

    if meta_help:
        userlike_failures.append(
            "META_HELP_SELECTION"
        )

    benign = (
        len(benign_failures) == 0
    )

    user_like = (
        benign
        and len(
            userlike_failures
        ) == 0
    )

    generic_clarification = bool(
        clarification
    )

    substantive = (
        user_like
        and not generic_clarification
        and bool(
            substantive_hits
        )
    )

    direct = domain_hits(
        t
    )

    sensor_candidate = (
        benign
        and user_like
        and substantive
    )

    return {
        "benign":
            benign,

        "user_like":
            user_like,

        "substantive":
            substantive,

        "sensor_candidate":
            sensor_candidate,

        "benign_failures":
            benign_failures,

        "userlike_failures":
            userlike_failures,

        "generic_clarification":
            generic_clarification,

        "task_semantic_direct":
            bool(direct),

        "direct_domains":
            direct,

        "debug_matches": {
            "reverse":
                reverse,
            "system_self":
                self_q,
            "routing":
                routing,
            "dangling":
                dangling,
            "unnatural":
                unnatural,
            "meta_help":
                meta_help,
            "clarification":
                clarification,
            "substantive":
                substantive_hits,
        },
    }


rows = []

with SRC.open(
    encoding="utf-8"
) as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(
            line
        )

        audit = classify(
            x["query"]
        )

        rows.append({
            **x,
            "audit_v2":
                audit,
        })


counts = Counter()

failure_counts = Counter()
userlike_failure_counts = Counter()
role_counts = Counter()

sessions = set()
candidate_sessions = set()

for x in rows:
    a = x["audit_v2"]

    counts["rows"] += 1
    counts["benign"] += int(
        a["benign"]
    )
    counts["user_like"] += int(
        a["user_like"]
    )
    counts["substantive"] += int(
        a["substantive"]
    )
    counts["sensor_candidate"] += int(
        a["sensor_candidate"]
    )
    counts["generic_clarification"] += int(
        a["generic_clarification"]
    )
    counts["task_semantic_direct"] += int(
        a["task_semantic_direct"]
    )

    sessions.add(
        x["session_id"]
    )

    for f in a[
        "benign_failures"
    ]:
        failure_counts[f] += 1

    for f in a[
        "userlike_failures"
    ]:
        userlike_failure_counts[f] += 1

    if a[
        "sensor_candidate"
    ]:
        candidate_sessions.add(
            x["session_id"]
        )

        for r in x[
            "selected_roles"
        ]:
            role_counts[r] += 1


candidate = [
    x
    for x in rows
    if x[
        "audit_v2"
    ][
        "sensor_candidate"
    ]
]

rejected = [
    x
    for x in rows
    if not x[
        "audit_v2"
    ][
        "sensor_candidate"
    ]
]


def sample(xs, n):
    if len(xs) <= n:
        return list(xs)

    return rng.sample(
        xs,
        n,
    )


summary = {
    "input_rows":
        len(rows),

    "benign":
        counts["benign"],

    "benign_rate":
        counts["benign"]
        / len(rows),

    "user_like":
        counts["user_like"],

    "user_like_rate":
        counts["user_like"]
        / len(rows),

    "substantive":
        counts["substantive"],

    "substantive_rate":
        counts["substantive"]
        / len(rows),

    "sensor_candidate":
        counts["sensor_candidate"],

    "sensor_candidate_rate":
        counts["sensor_candidate"]
        / len(rows),

    "generic_clarification":
        counts[
            "generic_clarification"
        ],

    "task_semantic_direct":
        counts[
            "task_semantic_direct"
        ],

    "benign_failure_counts":
        dict(
            failure_counts
        ),

    "userlike_failure_counts":
        dict(
            userlike_failure_counts
        ),

    "unique_sessions_input":
        len(sessions),

    "unique_sessions_candidate":
        len(candidate_sessions),

    "selected_role_positive_counts_candidate":
        dict(
            sorted(
                role_counts.items()
            )
        ),
}


with (
    OUT / "sensor_candidates.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in candidate:
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


audit = {
    "candidate_random_60":
        sample(
            candidate,
            60,
        ),

    "rejected_random_40":
        sample(
            rejected,
            40,
        ),
}


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


(
    OUT / "human_audit.json"
).write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "\n===== V2 SUMMARY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 30 RANDOM SENSOR CANDIDATES ====="
)

for i, x in enumerate(
    sample(
        candidate,
        30,
    ),
    1,
):
    a = x[
        "audit_v2"
    ]

    print(
        f"\n[{i:02d}] "
        f"selected={x['selected_roles']} "
        f"direct={a['direct_domains']}"
    )

    print(
        "Q:",
        x["query"],
    )


print(
    "\n===== 20 RANDOM REJECTED ====="
)

for i, x in enumerate(
    sample(
        rejected,
        20,
    ),
    1,
):
    a = x[
        "audit_v2"
    ]

    print(
        f"\n[{i:02d}] "
        f"benign_fail={a['benign_failures']} "
        f"userlike_fail={a['userlike_failures']} "
        f"clarification={a['generic_clarification']}"
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
    OUT / "sensor_candidates.jsonl",
)

print(
    " ",
    OUT / "human_audit.json",
)
