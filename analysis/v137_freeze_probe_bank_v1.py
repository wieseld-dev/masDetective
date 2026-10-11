#!/usr/bin/env python3

import json
import hashlib
import re
from pathlib import Path

CANON = Path("artifacts/canonical_parallel_v1")
OUT = Path("artifacts/v137_presence_v1")
OUT.mkdir(parents=True, exist_ok=True)

ROLES = [
    "accessibility_review",
    "code",
    "customer_support_triage",
    "data",
    "database_design",
    "design",
    "documentation",
    "experiment_design",
    "fact_checking",
    "financial_analysis",
    "incident_management",
    "math",
    "operations",
    "planning",
    "qa_testing",
    "requirements_analysis",
    "research",
    "risk_assessment",
    "translation",
    "writing",
]

DESCRIPTIONS = {
    "accessibility_review":
        "Reviews interfaces and content for accessibility barriers and proposes concrete improvements.",
    "code":
        "Implements, reviews, and reasons about software code and programming tasks.",
    "customer_support_triage":
        "Triages customer-support issues, assigns urgency and ownership, and proposes first-response actions.",
    "data":
        "Analyzes structured observations or measurements to identify patterns, anomalies, and useful conclusions.",
    "database_design":
        "Designs relational data models, schemas, keys, constraints, normalization, and related database structures.",
    "design":
        "Designs and improves user-facing product flows and interaction experiences.",
    "documentation":
        "Produces clear technical documentation for software interfaces, functions, APIs, and operational procedures.",
    "experiment_design":
        "Designs controlled experiments, metrics, hypotheses, assignment strategies, and validity checks.",
    "fact_checking":
        "Checks claims for consistency and identifies evidence required to verify or falsify them.",
    "financial_analysis":
        "Analyzes financial quantities, margins, costs, revenues, and business-performance implications.",
    "incident_management":
        "Coordinates technical incident response, prioritization, containment, ownership, communication, and recovery.",
    "math":
        "Solves mathematical problems and provides verifiable quantitative reasoning.",
    "operations":
        "Plans and evaluates operational processes, staffing, capacity, scheduling, and execution controls.",
    "planning":
        "Creates structured implementation plans with milestones, dependencies, sequencing, and deliverables.",
    "qa_testing":
        "Designs software testing strategies covering normal behavior, edge cases, failures, and relevant risks.",
    "requirements_analysis":
        "Transforms stakeholder needs into explicit functional and non-functional system requirements.",
    "research":
        "Synthesizes evidence, compares observations, identifies uncertainty, and determines what additional evidence is needed.",
    "risk_assessment":
        "Identifies risks, estimates likelihood and impact, and proposes mitigations and contingencies.",
    "translation":
        "Translates text between languages while preserving meaning, tone, and natural phrasing.",
    "writing":
        "Rewrites or produces clear, concise, professional natural-language text.",
}

# Fresh-training wording.
T1 = {
    "accessibility_review":
        "Audit a newsletter signup form containing Name, Email, Subscribe, inline validation errors, and a confirmation message. What accessibility changes would improve keyboard and screen-reader use?",
    "code":
        "Write a Python function merge_intervals(intervals) that merges overlapping numeric intervals, and include two small tests.",
    "customer_support_triage":
        "A subscriber says their renewal was charged, their account is locked, and they need access for work today. How should support triage the case and what should the first reply contain?",
    "data":
        "Weekly conversion rates are [4.8, 5.1, 4.9, 5.0, 8.7, 5.2, 5.0] percent. Summarize the pattern and identify what deserves investigation.",
    "database_design":
        "Design a normalized relational schema for students, courses, instructors, and enrollments, including primary keys, foreign keys, and important constraints.",
    "design":
        "A mobile onboarding flow currently requires seven screens before a new user reaches the product. Propose a simpler flow while preserving necessary information and consent.",
    "documentation":
        "Write concise API documentation for POST /projects with fields name, owner_id, and optional description, including inputs, success response, and likely errors.",
    "experiment_design":
        "Design an A/B test to determine whether sending a reminder one hour before an appointment reduces no-shows. Define assignment, metrics, and important validity checks.",
    "fact_checking":
        "A report says a product entered beta in November 2023 and also claims it had eighteen months of beta usage by March 2024. Assess the consistency and state what evidence should be checked.",
    "financial_analysis":
        "A business increased quarterly revenue from $3.1M to $3.8M while cost of goods rose from $1.6M to $2.2M. Compare gross margin rates and interpret the change.",
    "incident_management":
        "Login latency and authentication failures rose sharply immediately after a production release. Describe the first incident-response actions, ownership, containment, and communication priorities.",
    "math":
        "Solve 2x^2 + 5x - 3 = 0 and show enough calculation to verify both roots.",
    "operations":
        "A service desk has 18 agents and must provide continuous coverage while demand is highest from 14:00 to 22:00. Propose a staffing approach and operational checks.",
    "planning":
        "Create a six-week implementation plan for migrating an internal reporting service to a new platform, including milestones, dependencies, validation, and rollout.",
    "qa_testing":
        "Create a compact test plan for a file-upload feature, covering normal use, size/type limits, interrupted uploads, malformed files, and security-relevant cases.",
    "requirements_analysis":
        "Turn this request into clear functional and non-functional requirements: 'Customers should book appointments online, change them later, and receive reminders; it needs to be reliable and fast.'",
    "research":
        "Observations show that trial users who complete setup quickly retain better, most abandonment occurs during setup, and optional setup steps are rarely used. Synthesize the evidence and identify what additional evidence would strengthen the conclusion.",
    "risk_assessment":
        "Assess the risks of moving a production service to a new cloud region, including likelihood, impact, mitigation, rollback, and contingency measures.",
    "translation":
        "Translate naturally into French while preserving the professional tone: 'Please confirm the updated delivery date before tomorrow's planning meeting.'",
    "writing":
        "Rewrite this to be concise and professional: 'Hi, sorry to bother you again but we still haven't received the updated numbers and we need them pretty urgently, so could you send them when you can?'",
}

# DEV-only wording.
D = {
    "accessibility_review":
        "Review an account-settings modal with tab controls, text inputs, Save and Cancel buttons, and validation messages. Identify concrete keyboard, focus, labeling, and screen-reader improvements.",
    "code":
        "Implement a Python function group_by_key(items, key) that groups dictionaries by the requested key while preserving input order, and show two tests.",
    "customer_support_triage":
        "A customer reports that an upgrade payment succeeded but premium features remain unavailable and they have a presentation shortly. Determine priority, ownership, and the initial support response.",
    "data":
        "API response times in milliseconds are [182, 176, 190, 185, 181, 740, 188]. Describe the data pattern and what should be investigated.",
    "database_design":
        "A table stores order_id, customer_name, customer_email, product_id, product_name, quantity, and unit_price in every row. Propose a normalized relational design with keys and relationships.",
    "design":
        "Redesign a mobile product-search experience where filters occupy a full page and users frequently lose their selections. Describe a lower-friction interaction flow.",
    "documentation":
        "Document a command-line operation `backup create --project PROJECT_ID [--label LABEL]`, including purpose, arguments, output, examples, and common failures.",
    "experiment_design":
        "Design an experiment to test whether a new recommendation layout increases meaningful item discovery without hurting purchase completion. Specify assignment, primary metrics, guardrails, and duration considerations.",
    "fact_checking":
        "One internal document says a company acquisition closed in August 2025; another says the acquired unit was independently reporting results until December 2025. Explain what is potentially inconsistent and what records should be checked.",
    "financial_analysis":
        "Monthly sales rose from $820k to $910k, while operating expenses increased from $610k to $735k. Compare operating margins and explain the implication.",
    "incident_management":
        "A scheduled batch pipeline has stopped processing new records and downstream reports are stale. Give the immediate incident-management sequence, including impact assessment, ownership, recovery, and stakeholder updates.",
    "math":
        "Solve the system 3x + 2y = 16 and x - y = 1, showing enough work to verify the solution.",
    "operations":
        "A warehouse processes twice as many orders in the evening as in the morning but staffing is currently uniform. Propose an operating schedule and checks for capacity, handoffs, and service levels.",
    "planning":
        "Create a three-week plan for launching a limited beta of a new internal search tool, with milestones, dependencies, testing, user onboarding, and feedback collection.",
    "qa_testing":
        "Build a test plan for multi-factor login, including enrollment, successful and failed verification, expired codes, recovery, rate limits, and relevant security behavior.",
    "requirements_analysis":
        "Convert this into explicit requirements: 'Employees upload expense receipts, find old submissions later, and managers approve them quickly; sensitive information must stay protected.'",
    "research":
        "Three studies report that shorter forms increase completion, prefilled fields reduce typing, and removing optional questions has little effect on downstream quality. Synthesize what can be concluded and what remains uncertain.",
    "risk_assessment":
        "Perform a risk assessment for integrating a critical third-party payments provider, covering technical, availability, security, operational, and fallback risks.",
    "translation":
        "Translate naturally into German while preserving tone: 'We have attached the revised proposal and would appreciate your comments before Thursday.'",
    "writing":
        "Rewrite this status update to be concise and professional: 'Just letting everyone know we're kind of behind on the migration because a few unexpected things came up, but we're working on it and hope to catch up soon.'",
}

# INTERNAL-TEST-only wording.
T = {
    "accessibility_review":
        "Assess a checkout page containing an address form, delivery options, an order summary, and error banners. Recommend concrete accessibility improvements for keyboard navigation, focus, labels, and assistive technologies.",
    "code":
        "Write a Python function chunk_sequence(values, size) that returns consecutive chunks without modifying the input, and include tests for normal and edge cases.",
    "customer_support_triage":
        "A business customer says an invoice was paid twice, one administrator cannot sign in, and payroll processing depends on the service today. How should the case be triaged and initially handled?",
    "data":
        "Daily error counts are [41, 38, 44, 40, 39, 167, 42]. Characterize the pattern and identify the observation requiring follow-up.",
    "database_design":
        "Design a relational schema for users, projects, tasks, and task comments, with primary keys, foreign keys, cardinalities, and important integrity constraints.",
    "design":
        "A mobile booking process makes users move repeatedly between date, location, and availability screens. Redesign the interaction so comparison and selection require less backtracking.",
    "documentation":
        "Write technical documentation for a function `rotate_api_key(account_id, revoke_old=True)`, describing parameters, return value, side effects, errors, and a short usage example.",
    "experiment_design":
        "Design a controlled experiment to test whether simplifying checkout from three pages to one improves completed purchases without increasing errors or refunds.",
    "fact_checking":
        "A summary says a service became generally available in February 2024 but elsewhere states that all customers had used the production release throughout 2023. Explain what needs verification and why.",
    "financial_analysis":
        "Revenue moved from $5.4M to $6.0M while direct costs moved from $2.7M to $3.5M. Calculate and compare gross margins and interpret the result.",
    "incident_management":
        "Payment webhooks are accumulating in a queue, order status is delayed, and retry volume is increasing. Outline immediate incident priorities, containment, ownership, recovery, and communication.",
    "math":
        "Solve x^2 - 7x + 10 = 0 and show the steps needed to verify the roots.",
    "operations":
        "A 24/7 monitoring team has 12 engineers, with alert volume concentrated between 18:00 and 01:00. Propose a practical coverage model and operational controls.",
    "planning":
        "Build a five-week rollout plan for deploying a new authentication service across several internal applications, including dependencies, migration stages, validation, and rollback readiness.",
    "qa_testing":
        "Create a thorough but compact test plan for applying promotional codes during checkout, including valid cases, invalid/expired codes, combinations, boundaries, failures, and abuse cases.",
    "requirements_analysis":
        "Translate this stakeholder request into functional and non-functional requirements: 'Teams need real-time alerts for important events, they should control which alerts they receive, and delivery must be dependable.'",
    "research":
        "Evidence shows users respond faster to concise notifications, excessive notifications increase opt-outs, and personalized timing may improve engagement. Synthesize the evidence and specify what further evidence is needed.",
    "risk_assessment":
        "Assess the risks of rolling out a new machine-learning decision service into a production workflow, including failure modes, impact, monitoring, mitigation, and rollback.",
    "translation":
        "Translate naturally into Italian while preserving the courteous tone: 'Could you send the signed agreement today so that we can complete the review before the deadline?'",
    "writing":
        "Rewrite this message to be concise and professional: 'Hey, I need to move our meeting because some stuff came up and the current time won't work, so let me know if tomorrow afternoon is okay instead.'",
}


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ------------------------------------------------------------------
# Recover T0 from existing frozen canonical artifacts rather than
# manually duplicating it.
# ------------------------------------------------------------------

t0_by_cohort = {}

for cohort in ["k3", "k4", "high"]:
    p = CANON / cohort / "meta.json"
    meta = json.loads(p.read_text())

    mapping = {
        x["role"]: x["query"]
        for x in meta["probes"]
    }

    t0_by_cohort[cohort] = mapping


base_t0 = t0_by_cohort["k3"]

for cohort, mapping in t0_by_cohort.items():
    if mapping != base_t0:
        raise RuntimeError(
            f"T0 canonical probe bank differs in cohort {cohort}"
        )

if set(base_t0) != set(ROLES):
    raise RuntimeError(
        f"T0 role mismatch: "
        f"missing={set(ROLES)-set(base_t0)} "
        f"extra={set(base_t0)-set(ROLES)}"
    )

for name, bank in [
    ("T1", T1),
    ("D", D),
    ("T", T),
    ("DESCRIPTIONS", DESCRIPTIONS),
]:
    if set(bank) != set(ROLES):
        raise RuntimeError(
            f"{name} role mismatch: "
            f"missing={set(ROLES)-set(bank)} "
            f"extra={set(bank)-set(ROLES)}"
        )


# ------------------------------------------------------------------
# Integrity: no exact wording reuse across families.
# ------------------------------------------------------------------

family_banks = {
    "T0": base_t0,
    "T1": T1,
    "D": D,
    "T": T,
}

seen = {}

for family, bank in family_banks.items():
    for role, query in bank.items():
        n = normalize(query)

        if n in seen:
            raise RuntimeError(
                f"Exact probe wording reused: "
                f"{family}/{role} == {seen[n]}"
            )

        seen[n] = f"{family}/{role}"


# Lightweight lexical-overlap diagnostic.
def wordset(s):
    return set(
        re.findall(r"[a-z0-9]+", s.lower())
    )


overlaps = []

for role in ROLES:
    fams = list(family_banks)

    for i, a in enumerate(fams):
        for b in fams[i+1:]:
            sa = wordset(family_banks[a][role])
            sb = wordset(family_banks[b][role])

            j = len(sa & sb) / max(1, len(sa | sb))

            overlaps.append({
                "role": role,
                "a": a,
                "b": b,
                "jaccard": j,
            })


payload = {
    "version": "v137_probe_bank_v1",
    "protocol": {
        "T0": "legacy canonical wording; TRAIN only",
        "T1": "fresh wording; TRAIN only",
        "D": "held-out wording; DEV only",
        "T": "held-out wording; INTERNAL TEST only",
        "untouched_final_test": "not used",
    },
    "roles": [
        {
            "role": role,
            "public_description": DESCRIPTIONS[role],
            "probes": {
                "T0": base_t0[role],
                "T1": T1[role],
                "D": D[role],
                "T": T[role],
            },
        }
        for role in ROLES
    ],
}

out = OUT / "probe_bank_v1.json"

text = json.dumps(
    payload,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
)

out.write_text(text)

sha = hashlib.sha256(
    text.encode("utf-8")
).hexdigest()

(OUT / "probe_bank_v1.sha256").write_text(
    sha + "  probe_bank_v1.json\n"
)

print("=" * 90)
print("V13.7 PROBE BANK FREEZE")
print("=" * 90)
print("roles:", len(ROLES))
print("probe families: 4")
print("total probes:", 4 * len(ROLES))
print("T0 cohort consistency: PASS")
print("role coverage: PASS")
print("exact cross-family duplicates: 0")

print("\nHighest same-role lexical overlaps:")
for x in sorted(
    overlaps,
    key=lambda z: z["jaccard"],
    reverse=True,
)[:20]:
    print(
        f"{x['role']:27s} "
        f"{x['a']}-{x['b']} "
        f"Jaccard={x['jaccard']:.3f}"
    )

print("\nprobe bank:", out)
print("sha256:", sha)
print("PROBE BANK FREEZE: PASS")
