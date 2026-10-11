from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from v8.counterfactual import (
    CounterfactualPairV8,
    clone_spec,
    get_roles,
    get_system_id,
)


# Public capability-level confusions, not hidden-system information.  The lists
# deliberately target roles that can plausibly produce similar visible answers.
PREFERRED_CONFUSERS: dict[str, tuple[str, ...]] = {
    "code": ("qa_testing", "database_design", "documentation"),
    "writing": ("documentation", "translation", "customer_support_triage"),
    "math": ("data", "financial_analysis", "experiment_design"),
    "planning": ("operations", "requirements_analysis", "risk_assessment"),
    "research": ("fact_checking", "experiment_design", "data"),
    "data": ("database_design", "financial_analysis", "research"),
    "design": ("accessibility_review", "requirements_analysis", "planning"),
    "operations": ("incident_management", "planning", "customer_support_triage"),
    "qa_testing": ("code", "requirements_analysis", "documentation"),
    "translation": ("writing", "documentation", "customer_support_triage"),
    "financial_analysis": ("data", "risk_assessment", "math"),
    "database_design": ("data", "code", "requirements_analysis"),
    "accessibility_review": ("design", "qa_testing", "requirements_analysis"),
    "fact_checking": ("research", "documentation", "data"),
    "requirements_analysis": ("planning", "design", "qa_testing"),
    "incident_management": ("operations", "risk_assessment", "customer_support_triage"),
    "experiment_design": ("research", "data", "math"),
    "risk_assessment": ("planning", "financial_analysis", "incident_management"),
    "documentation": ("writing", "code", "requirements_analysis"),
    "customer_support_triage": ("incident_management", "operations", "documentation"),

    # Unseen-development roles. These are used only during dev evaluation if the
    # caller includes them in candidate_roles; they never enter V11.2 train specs.
    "legal_contract_review": ("requirements_analysis", "fact_checking", "risk_assessment"),
    "procurement_analysis": ("financial_analysis", "risk_assessment", "requirements_analysis"),
    "travel_itinerary": ("planning", "operations", "customer_support_triage"),
    "spreadsheet_audit": ("data", "financial_analysis", "qa_testing"),
    "scientific_peer_review": ("research", "fact_checking", "experiment_design"),
    "inventory_forecasting": ("data", "financial_analysis", "operations"),
}


def _stable_index(text: str, n: int) -> int:
    if n <= 0:
        return 0
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(h, 16) % n


def ranked_absent_confusers(
    target_role: str,
    plus_roles: Sequence[str],
    candidate_roles: Sequence[str],
) -> list[str]:
    """Return hard absent replacements, preferred first, with deterministic fallback."""
    present = set(map(str, plus_roles))
    target = str(target_role)
    absent = [r for r in map(str, candidate_roles) if r not in present]
    preferred = [r for r in PREFERRED_CONFUSERS.get(target, ()) if r in absent]
    remaining = [r for r in sorted(absent) if r not in set(preferred)]
    return preferred + remaining


def make_hard_counterfactual_pair(
    spec: object,
    target_role: str,
    candidate_roles: Sequence[str],
    *,
    include_sham: bool = True,
    salt: str = "v11_2_hard",
    top_k: int = 3,
) -> CounterfactualPairV8 | None:
    """Constant-K pair where MINUS replaces target with a hard confuser.

    Across base systems we deterministically rotate among the top-k available
    confusers.  This gives diversity while focusing the continuation objective on
    semantically/behaviorally adjacent alternatives rather than arbitrary roles.
    """
    target = str(target_role)
    plus_roles = list(get_roles(spec))
    if target not in plus_roles:
        return None

    ranked = ranked_absent_confusers(target, plus_roles, candidate_roles)
    if not ranked:
        return None

    base_id = get_system_id(spec)
    pool = ranked[: max(1, min(int(top_k), len(ranked)))]
    neg = pool[_stable_index(f"{salt}|{base_id}|{target}|neg", len(pool))]

    minus_roles = list(plus_roles)
    minus_roles[minus_roles.index(target)] = neg

    pair_id = hashlib.sha256(
        f"{salt}|{base_id}|{target}|{neg}".encode("utf-8")
    ).hexdigest()[:16]
    plus = clone_spec(
        spec,
        system_id=f"v112p_{pair_id}_plus",
        worker_roles=plus_roles,
    )
    minus = clone_spec(
        spec,
        system_id=f"v112p_{pair_id}_minus",
        worker_roles=minus_roles,
    )

    sham = None
    sham_removed = None
    sham_replacement = None
    if include_sham and len(plus_roles) >= 2:
        removable = [r for r in plus_roles if r != target]
        if removable:
            sham_removed = removable[
                _stable_index(f"{salt}|{base_id}|{target}|sham-remove", len(removable))
            ]
            absent = [r for r in map(str, candidate_roles) if r not in set(plus_roles)]
            sham_candidates = [r for r in absent if r != neg] or absent
            if sham_candidates:
                sham_replacement = sham_candidates[
                    _stable_index(
                        f"{salt}|{base_id}|{target}|sham-replace",
                        len(sham_candidates),
                    )
                ]
                sham_roles = list(plus_roles)
                sham_roles[sham_roles.index(sham_removed)] = sham_replacement
                if target in sham_roles and len(set(sham_roles)) == len(sham_roles):
                    sham = clone_spec(
                        spec,
                        system_id=f"v112p_{pair_id}_sham",
                        worker_roles=sham_roles,
                    )

    return CounterfactualPairV8(
        pair_id=pair_id,
        target_role=target,
        plus_spec=plus,
        minus_spec=minus,
        sham_spec=sham,
        negative_replacement=neg,
        sham_removed_role=sham_removed,
        sham_replacement=sham_replacement,
    )


def build_hard_counterfactual_pairs(
    specs: Iterable[object],
    candidate_roles: Sequence[str],
    *,
    include_sham: bool = True,
) -> list[CounterfactualPairV8]:
    out: list[CounterfactualPairV8] = []
    for spec in specs:
        for role in candidate_roles:
            pair = make_hard_counterfactual_pair(
                spec,
                role,
                candidate_roles,
                include_sham=include_sham,
            )
            if pair is not None:
                out.append(pair)
    return out
