from pathlib import Path
from collections import Counter
import json

OUT = Path(
    "artifacts/standalone_benign_hardening_v4"
)
OUT.mkdir(parents=True, exist_ok=True)

BASE = Path(
    "artifacts/standalone_benign_teacher_v3/"
    "trainable_clean_nostrict.jsonl"
)

STRICT = Path(
    "artifacts/standalone_benign_teacher_v1/"
    "strict_gold.jsonl"
)

OLD_FAILURES = {
    "What information should I provide first?",
    "Should I provide more details?",
    "Can you analyze this for me?",
}


def norm(x):
    return " ".join(str(x).split()).casefold()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ------------------------------------------------------------------
# HARD POSITIVES
#
# Concrete standalone tasks using the exact behavior we want the
# future Generator to be allowed to produce:
#
# concrete problem + ask what task-relevant evidence/context to provide.
# ------------------------------------------------------------------

TRAIN_CONTEXTS = [
    (
        "My application intermittently fails during login after a "
        "token refresh",
        "logs, error messages, request traces, and configuration details",
    ),
    (
        "Our customer database has accumulated duplicate records and "
        "some customers are linked to the wrong accounts",
        "schema definitions, sample rows, constraints, and indexing details",
    ),
    (
        "The revenue total in our quarterly workbook does not match "
        "the value in the accounting export",
        "spreadsheet formulas, sample rows, source exports, and reconciliation rules",
    ),
    (
        "An experiment produced a result opposite to our original "
        "hypothesis",
        "hypothesis, protocol, raw measurements, controls, and analysis assumptions",
    ),
    (
        "Our API latency increased sharply after yesterday's deployment",
        "timeline, logs, monitoring graphs, deployment changes, and request traces",
    ),
    (
        "Our API documentation describes endpoints that no longer "
        "behave like the implementation",
        "current documentation, API specification, examples, and observed responses",
    ),
    (
        "A modal in our web application traps keyboard users and "
        "screen-reader behavior is inconsistent",
        "HTML, CSS, screenshots, keyboard behavior, browser details, and assistive-technology observations",
    ),
    (
        "I need to translate a customer-facing policy while preserving "
        "its intended tone and terminology",
        "source passage, glossary, target audience, tone requirements, and terminology constraints",
    ),
    (
        "We need to migrate a production service while several teams "
        "have conflicting deadlines",
        "dependencies, milestones, team constraints, deadlines, and rollback requirements",
    ),
    (
        "I need to turn a collection of technical notes into a concise "
        "executive brief",
        "source notes, intended audience, required length, tone, and key conclusions",
    ),
    (
        "I found several papers that appear to make contradictory "
        "claims about the same technique",
        "research question, citations, abstracts, inclusion criteria, and relevant evidence",
    ),
    (
        "I need to formulate an optimization problem but I am unsure "
        "whether my constraints are specified correctly",
        "variables, objective, constraints, assumptions, and representative values",
    ),
    (
        "A Python service slowly consumes more memory during long runs",
        "minimal reproduction, profiler output, dependency versions, logs, and workload details",
    ),
    (
        "A dataset has missing values, inconsistent categories, and "
        "several suspicious outliers",
        "schema, representative rows, missing-value patterns, cleaning rules, and intended analysis",
    ),
    (
        "Users are abandoning our onboarding flow at one particular step",
        "screenshots, flow description, analytics, user constraints, and relevant design requirements",
    ),
    (
        "Our warehouse order-processing time has increased even though "
        "daily volume is roughly unchanged",
        "process steps, timings, staffing levels, queue measurements, and recent operational changes",
    ),
    (
        "A test passes locally but fails intermittently in CI",
        "test code, failure logs, environment details, reproduction steps, and dependency versions",
    ),
    (
        "Two stakeholder groups gave conflicting requirements for the "
        "same feature",
        "current specification, stakeholder requests, priorities, constraints, and acceptance criteria",
    ),
    (
        "We are evaluating the risk of moving a critical service to a "
        "new external vendor",
        "assets, threat assumptions, existing controls, impact criteria, dependencies, and risk tolerance",
    ),
    (
        "Our support queue contains many tickets that are being routed "
        "to the wrong team",
        "representative tickets, existing categories, SLA rules, escalation policy, and routing criteria",
    ),
]

POS_TEMPLATES = [
    (
        "{context}. What {artifacts} should I provide first "
        "so you can help me investigate it?"
    ),
    (
        "I'm trying to resolve this situation: {context}. "
        "Which {artifacts} would be most useful for you to see "
        "before suggesting next steps?"
    ),
    (
        "{context}. Before we work on it, what {artifacts} "
        "should I send so you can assess the problem?"
    ),
    (
        "I need help with the following problem: {context}. "
        "What {artifacts} would you want from me first to make "
        "your analysis useful?"
    ),
    (
        "{context}. To help you evaluate it properly, which "
        "{artifacts} should I gather and share first?"
    ),
]

hard_pos = []

for context, artifacts in TRAIN_CONTEXTS:
    for template in POS_TEMPLATES:
        hard_pos.append({
            "query": template.format(
                context=context,
                artifacts=artifacts,
            ),
            "label": "BENIGN",
            "source": "hardening_v4",
            "hardening_v4": True,
            "hard_kind": "concrete_standalone_positive",
        })

assert len(hard_pos) == 100


# ------------------------------------------------------------------
# HARD NEGATIVES
#
# They sound conversational and often mention context/details/help,
# but contain no actual user task. They are therefore not standalone
# probes.
# ------------------------------------------------------------------

NEG_BASES = [
    "Which details should I send first",
    "What context would be most useful",
    "What background information would help",
    "What should I explain first",
    "Which information is most relevant",
    "What do you need to know from me",
    "What should I send you",
    "Would additional context be useful",
    "Do you need more information",
    "Would it help if I shared more",
    "Should I add some context",
    "How much context should I provide",
    "Where should I start",
    "What details matter most",
    "What would you need from me",
    "Do I need to explain more",
    "Could you take a look at this",
    "Could you help me with this",
    "Can we analyze this together",
    "What should I clarify for you",
]

NEG_SUFFIXES = [
    " before we begin?",
    " so you can help?",
    " before you answer?",
    " to get started?",
    " before we continue?",
]

hard_neg = []

for base in NEG_BASES:
    for suffix in NEG_SUFFIXES:
        hard_neg.append({
            "query": base + suffix,
            "label": "NONSTANDALONE",
            "source": "hardening_v4",
            "hardening_v4": True,
            "hard_kind": "vague_dangling_negative",
        })

assert len(hard_neg) == 100


# ------------------------------------------------------------------
# FROZEN CHALLENGE POSITIVES
# completely separate phrasings / scenarios.
# ------------------------------------------------------------------

CHALLENGE_CONTEXTS = [
    (
        "A mobile checkout screen sometimes submits the same order twice",
        "screen recording, network trace, relevant client code, and reproduction steps",
    ),
    (
        "Several rows disappear when our CSV import pipeline processes "
        "files from one supplier",
        "sample input files, parsing configuration, logs, schema, and expected output",
    ),
    (
        "Our forecasting model suddenly became much less accurate for "
        "one product category",
        "recent predictions, ground truth, feature definitions, training window, and evaluation results",
    ),
    (
        "The wording in a contract clause is ambiguous about who pays "
        "for a specific service",
        "relevant clauses, defined terms, surrounding sections, jurisdiction, and intended interpretation",
    ),
    (
        "A dashboard shows different totals depending on which filter "
        "is applied first",
        "filter definitions, screenshots, underlying queries, sample records, and expected totals",
    ),
    (
        "An internal runbook no longer matches the steps engineers "
        "actually use during incidents",
        "current runbook, recent incident timeline, actual commands, and known process changes",
    ),
    (
        "A batch job succeeds on small inputs but times out on larger ones",
        "representative inputs, timing data, resource limits, logs, and relevant implementation details",
    ),
    (
        "A new interface is confusing users even though usability tests "
        "on the old version were successful",
        "screenshots, task flow, user feedback, analytics, and design constraints",
    ),
    (
        "Two data sources give different dates for the same events",
        "sample records, source definitions, timestamp formats, transformation rules, and provenance",
    ),
    (
        "A project is slipping because several dependent tasks keep "
        "changing their completion dates",
        "dependency graph, current schedule, ownership, constraints, and latest estimates",
    ),
]

CHALLENGE_POS_TEMPLATES = [
    (
        "{context}. What {artifacts} should I send first so you "
        "can help me diagnose it?"
    ),
    (
        "I need help with this concrete issue: {context}. "
        "Which {artifacts} would give you enough evidence to start?"
    ),
    (
        "{context}. Before proposing a solution, what {artifacts} "
        "would you want me to provide?"
    ),
]

challenge_pos = []

for context, artifacts in CHALLENGE_CONTEXTS:
    for template in CHALLENGE_POS_TEMPLATES:
        challenge_pos.append({
            "query": template.format(
                context=context,
                artifacts=artifacts,
            ),
            "label": "BENIGN",
            "source": "hardening_v4_frozen_challenge",
            "challenge_kind": "concrete_standalone_positive",
        })

assert len(challenge_pos) == 30


# ------------------------------------------------------------------
# FROZEN CHALLENGE NEGATIVES
# 27 novel + the three old V3 failures.
# ------------------------------------------------------------------

CH_NEG_BASES = [
    "Is there some context I should add",
    "What background do you need",
    "Would extra details make this easier",
    "What else should I tell you",
    "Do you need anything else from me",
    "Which details would help you most",
    "Is more information necessary",
    "What should I mention before we start",
    "Would some additional explanation help",
]

CH_NEG_SUFFIXES = [
    " before you respond?",
    " to begin?",
    " so we can proceed?",
]

challenge_neg = []

for base in CH_NEG_BASES:
    for suffix in CH_NEG_SUFFIXES:
        challenge_neg.append({
            "query": base + suffix,
            "label": "NONSTANDALONE",
            "source": "hardening_v4_frozen_challenge",
            "challenge_kind": "vague_dangling_negative",
        })

assert len(challenge_neg) == 27

for query in sorted(OLD_FAILURES):
    challenge_neg.append({
        "query": query,
        "label": "NONSTANDALONE",
        "source": "hardening_v4_frozen_challenge",
        "challenge_kind": "old_v3_strict_failure",
    })

assert len(challenge_neg) == 30


hard = hard_pos + hard_neg
challenge = challenge_pos + challenge_neg

assert len(hard) == 200
assert len(challenge) == 60


# ------------------------------------------------------------------
# Leakage / exact-overlap guards.
# ------------------------------------------------------------------

base_rows = read_jsonl(BASE)
strict_rows = read_jsonl(STRICT)

base_keys = {
    norm(x["query"])
    for x in base_rows
}

hard_keys = {
    norm(x["query"])
    for x in hard
}

challenge_keys = {
    norm(x["query"])
    for x in challenge
}

assert len(hard_keys) == len(hard)
assert len(challenge_keys) == len(challenge)

assert not (
    hard_keys
    & challenge_keys
), "hard/challenge overlap"

assert not (
    hard_keys
    & base_keys
), "hard/base exact overlap"

assert not (
    challenge_keys
    & base_keys
), "challenge/base exact overlap"

for q in OLD_FAILURES:
    assert norm(q) not in hard_keys

old_in_challenge = [
    q
    for q in OLD_FAILURES
    if norm(q) in challenge_keys
]

assert len(old_in_challenge) == 3


write_jsonl(
    OUT / "hard_train.jsonl",
    hard,
)

write_jsonl(
    OUT / "challenge_frozen.jsonl",
    challenge,
)

summary = {
    "hard_train_total": len(hard),
    "hard_train_labels": dict(
        Counter(x["label"] for x in hard)
    ),
    "challenge_total": len(challenge),
    "challenge_labels": dict(
        Counter(x["label"] for x in challenge)
    ),
    "hard_base_exact_overlap": len(
        hard_keys & base_keys
    ),
    "challenge_base_exact_overlap": len(
        challenge_keys & base_keys
    ),
    "hard_challenge_exact_overlap": len(
        hard_keys & challenge_keys
    ),
    "old_failures_in_hard_train": sum(
        norm(q) in hard_keys
        for q in OLD_FAILURES
    ),
    "old_failures_in_challenge": len(
        old_in_challenge
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

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)
