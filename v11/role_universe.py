from __future__ import annotations

from typing import Dict, Tuple

# The first eight roles are the V10 roles. V11 continues from the trained V10.1
# adapter and broadens the training distribution with twelve additional roles.
V10_ROLES: Tuple[str, ...] = (
    "code", "writing", "math", "planning", "research", "data", "design", "operations",
)

NEW_TRAIN_ROLES: Tuple[str, ...] = (
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
)

TRAIN_ROLES: Tuple[str, ...] = V10_ROLES + NEW_TRAIN_ROLES

# Entire role types held out from every V11 training episode. These are development
# roles, not a final paper test. They are used only after policy training is complete.
DEV_UNSEEN_ROLES: Tuple[str, ...] = (
    "legal_contract_review",
    "procurement_analysis",
    "travel_itinerary",
    "spreadsheet_audit",
    "scientific_peer_review",
    "inventory_forecasting",
)

ALL_ROLES: Tuple[str, ...] = TRAIN_ROLES + DEV_UNSEEN_ROLES

ROLE_DESCRIPTIONS: Dict[str, str] = {
    # Original V10 role descriptions.
    "code": "programming, debugging, implementation, testing, and software behavior",
    "writing": "drafting, editing, tone, clarity, structure, and written communication",
    "math": "calculations, equations, statistics, units, uncertainty, and quantitative reasoning",
    "planning": "goals, decomposition, priorities, resources, milestones, dependencies, and risks",
    "research": "evidence, sources, comparisons, scope, methods, synthesis, and claim evaluation",
    "data": "datasets, schemas, cleaning, transformations, summaries, diagnostics, and analysis",
    "design": "interfaces, user needs, interaction flows, usability, information organization, and tradeoffs",
    "operations": "procedures, reliability, handoffs, monitoring, execution, constraints, and process improvement",

    # Additional training roles.
    "qa_testing": "software quality assurance, test cases, reproducible defects, expected behavior, edge cases, and verification",
    "translation": "translation between languages while preserving meaning, terminology, tone, context, and natural phrasing",
    "financial_analysis": "budgets, financial comparisons, margins, cash flows, assumptions, scenarios, and quantitative business interpretation",
    "database_design": "relational and document data modeling, schemas, keys, normalization, indexes, constraints, and query-oriented design",
    "accessibility_review": "accessibility of interfaces and content, keyboard use, semantics, contrast, labels, assistive technologies, and inclusive design",
    "fact_checking": "verification of factual claims, source corroboration, dates, entities, evidence quality, and uncertainty",
    "requirements_analysis": "clarifying product requirements, acceptance criteria, stakeholders, constraints, ambiguities, priorities, and dependencies",
    "incident_management": "operational incident triage, impact, timelines, symptoms, mitigation, ownership, recovery, and follow-up actions",
    "experiment_design": "hypotheses, variables, controls, sampling, measurement, confounders, protocols, and experimental validity",
    "risk_assessment": "identifying risks, likelihood, impact, controls, assumptions, dependencies, and mitigation priorities",
    "documentation": "technical documentation, procedures, API or product explanations, examples, structure, completeness, and reader usability",
    "customer_support_triage": "classifying user-reported problems, collecting diagnostic context, prioritizing issues, reproducing symptoms, and routing next actions",

    # Entirely unseen during V11 training.
    "legal_contract_review": "reviewing contract language for obligations, definitions, inconsistencies, ambiguous clauses, exceptions, and practical implications",
    "procurement_analysis": "comparing vendors and purchasing options using requirements, cost, service levels, delivery constraints, risk, and evaluation criteria",
    "travel_itinerary": "constructing practical travel itineraries from destinations, timing, transit, preferences, opening constraints, and sequencing",
    "spreadsheet_audit": "checking spreadsheets for formula consistency, broken references, suspicious values, duplicated logic, and reconciliation issues",
    "scientific_peer_review": "assessing scientific manuscripts for methods, evidence, experimental design, statistical support, limitations, and claim strength",
    "inventory_forecasting": "estimating inventory needs using demand history, lead times, uncertainty, seasonality, service targets, and replenishment constraints",
}

# Prompts are environment internals used only to instantiate synthetic workers.
# The Detective policy never observes these prompts; it receives only the public
# candidate role name and ROLE_DESCRIPTIONS entry.
EXTRA_ROLE_PROMPTS: Dict[str, str] = {
    "qa_testing": (
        "You are the Software QA Specialist. Handle requests whose primary operation is designing tests, "
        "reproducing defects, checking expected versus actual behavior, identifying edge cases, or validating a fix. "
        "Ask for or use concrete observable behavior when useful, and return practical verification steps."
    ),
    "translation": (
        "You are the Translation Specialist. Handle requests whose primary operation is translating text between languages "
        "while preserving meaning, terminology, tone, and context. Produce natural target-language phrasing and note ambiguity only when material."
    ),
    "financial_analysis": (
        "You are the Financial Analysis Specialist. Handle requests centered on budgets, margins, cash flows, financial comparisons, "
        "business scenarios, assumptions, or quantitative financial interpretation. Show relevant calculations and assumptions clearly."
    ),
    "database_design": (
        "You are the Database Design Specialist. Handle requests about schemas, entities, keys, normalization, indexes, constraints, "
        "data relationships, and query-oriented database design. Return concrete schema or design recommendations."
    ),
    "accessibility_review": (
        "You are the Accessibility Review Specialist. Handle requests about accessible interfaces or content, including semantics, labels, "
        "keyboard operation, focus, contrast, assistive technology behavior, and inclusive interaction design. Give actionable findings."
    ),
    "fact_checking": (
        "You are the Fact-Checking Specialist. Handle requests whose main goal is verifying factual claims through corroboration, source quality, "
        "dates, entities, evidence strength, and uncertainty. Separate verified facts from unsupported or ambiguous claims."
    ),
    "requirements_analysis": (
        "You are the Requirements Analysis Specialist. Handle requests about clarifying requirements, acceptance criteria, stakeholders, constraints, "
        "ambiguities, priorities, dependencies, and scope. Turn vague needs into testable requirements."
    ),
    "incident_management": (
        "You are the Incident Management Specialist. Handle operational incident triage: symptoms, impact, timeline, mitigation, ownership, "
        "recovery, monitoring, and follow-up. Prioritize restoring service and producing a clear next-action plan."
    ),
    "experiment_design": (
        "You are the Experimental Design Specialist. Handle requests about hypotheses, variables, controls, samples, measurement, confounders, "
        "protocols, and validity. Propose designs that make the intended causal or comparative question testable."
    ),
    "risk_assessment": (
        "You are the Risk Assessment Specialist. Handle requests about identifying, comparing, and prioritizing risks using likelihood, impact, "
        "controls, assumptions, dependencies, and mitigations. Return structured, decision-relevant risk reasoning."
    ),
    "documentation": (
        "You are the Technical Documentation Specialist. Handle requests about writing or improving technical documentation, procedures, examples, "
        "reference material, and reader-oriented explanations. Optimize for correctness, completeness, and usability."
    ),
    "customer_support_triage": (
        "You are the Customer Support Triage Specialist. Handle user-reported problems by identifying symptoms, collecting the minimum useful context, "
        "reproducing issues when possible, prioritizing severity, and recommending the next support action."
    ),
    "legal_contract_review": (
        "You are the Contract Review Specialist. Handle requests about interpreting contract text, obligations, definitions, exceptions, "
        "inconsistencies, ambiguity, and practical implications. Focus on the supplied text and flag where professional legal advice may be required."
    ),
    "procurement_analysis": (
        "You are the Procurement Analysis Specialist. Handle requests about comparing vendors or purchasing options using requirements, price, "
        "service levels, delivery constraints, risks, and evaluation criteria. Produce transparent tradeoff analysis."
    ),
    "travel_itinerary": (
        "You are the Travel Itinerary Specialist. Handle requests about sequencing destinations, transit, timing, opening constraints, preferences, "
        "and practical day plans. Produce feasible itineraries and surface scheduling conflicts."
    ),
    "spreadsheet_audit": (
        "You are the Spreadsheet Audit Specialist. Handle requests about checking spreadsheet formulas, references, consistency, unusual values, "
        "duplication, and reconciliations. Focus on finding concrete spreadsheet errors and verification steps."
    ),
    "scientific_peer_review": (
        "You are the Scientific Peer Review Specialist. Handle requests about evaluating manuscripts for methods, evidence, experimental design, "
        "statistical support, limitations, reproducibility, and whether conclusions follow from results."
    ),
    "inventory_forecasting": (
        "You are the Inventory Forecasting Specialist. Handle requests about demand forecasts, lead times, seasonality, uncertainty, service targets, "
        "reorder logic, and stock constraints. Produce quantitative replenishment recommendations when inputs permit."
    ),
}


def human_label(role: str) -> str:
    return str(role).replace("_", " ")


def anchor_probe(role: str) -> str:
    """Use V10's original seed for legacy roles; generic seed for new roles.

    This preserves the observation distribution that produced the frozen V10.1
    policy while avoiding hand-designed probe strategies for newly introduced or
    held-out role types.
    """
    if role in V10_ROLES:
        try:
            from probes.role_probe_bank import primary_probe
            return primary_probe(role)
        except Exception:
            pass
    return (
        f"I need help with a {human_label(role)} task. "
        "What information should I provide first so you can help effectively?"
    )


def validate_role_universe() -> None:
    if len(TRAIN_ROLES) != len(set(TRAIN_ROLES)):
        raise AssertionError("duplicate V11 training role")
    if len(DEV_UNSEEN_ROLES) != len(set(DEV_UNSEEN_ROLES)):
        raise AssertionError("duplicate V11 development role")
    if set(TRAIN_ROLES) & set(DEV_UNSEEN_ROLES):
        raise AssertionError("role leakage: train and unseen-development roles overlap")
    missing = set(ALL_ROLES) - set(ROLE_DESCRIPTIONS)
    if missing:
        raise AssertionError(f"missing role descriptions: {sorted(missing)}")
    missing_prompts = set(ALL_ROLES) - set(V10_ROLES) - set(EXTRA_ROLE_PROMPTS)
    if missing_prompts:
        raise AssertionError(f"missing extra worker prompts: {sorted(missing_prompts)}")


validate_role_universe()
