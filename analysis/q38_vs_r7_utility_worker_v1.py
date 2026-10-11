from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

import ten_system_belief_trace_control_v1 as ctrl
import visible_adaptive_bridge_v1 as vb

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


PROBES = Path("artifacts/q38_vs_r7_ab_v1/probes.jsonl")
OUT = Path("artifacts/q38_vs_r7_utility_v1/parts")

ROLES = tuple(ctrl.ROLES)

CONFUSABLES = {
    "code": ["qa_testing", "incident_management", "database_design", "operations"],
    "writing": ["documentation", "research", "translation", "requirements_analysis"],
    "math": ["data", "financial_analysis", "research", "experiment_design"],
    "planning": ["requirements_analysis", "operations", "risk_assessment", "experiment_design"],
    "research": ["fact_checking", "experiment_design", "data", "writing"],
    "data": ["math", "financial_analysis", "research", "database_design"],
    "design": ["accessibility_review", "requirements_analysis", "qa_testing", "documentation"],
    "operations": ["incident_management", "planning", "customer_support_triage", "risk_assessment"],
    "qa_testing": ["code", "requirements_analysis", "accessibility_review", "documentation"],
    "translation": ["writing", "documentation", "accessibility_review", "customer_support_triage"],
    "financial_analysis": ["data", "math", "risk_assessment", "research"],
    "database_design": ["code", "data", "requirements_analysis", "operations"],
    "accessibility_review": ["design", "qa_testing", "documentation", "requirements_analysis"],
    "fact_checking": ["research", "writing", "data", "risk_assessment"],
    "requirements_analysis": ["planning", "design", "documentation", "customer_support_triage"],
    "incident_management": ["operations", "customer_support_triage", "risk_assessment", "code"],
    "experiment_design": ["research", "data", "math", "planning"],
    "risk_assessment": ["financial_analysis", "operations", "incident_management", "research"],
    "documentation": ["writing", "requirements_analysis", "qa_testing", "translation"],
    "customer_support_triage": ["incident_management", "operations", "requirements_analysis", "accessibility_review"],
}


def load_rows():
    return [
        json.loads(x)
        for x in PROBES.read_text().splitlines()
        if x.strip()
    ]


def replacement_for(role):
    for x in CONFUSABLES.get(role, []):
        if x in ROLES and x != role:
            return x
    return next(x for x in ROLES if x != role)


def fillers_for(role, replacement):
    i = ROLES.index(role)
    preferred = [
        ROLES[(i + 7) % len(ROLES)],
        ROLES[(i + 13) % len(ROLES)],
        ROLES[(i + 5) % len(ROLES)],
        ROLES[(i + 11) % len(ROLES)],
    ]

    out = []

    for x in preferred + list(ROLES):
        if x not in {role, replacement} and x not in out:
            out.append(x)
        if len(out) == 2:
            return out

    raise RuntimeError(role)


def make_target(system_id, roles):
    spec = SingleRouterSystemSpec(
        system_id=system_id,
        split="transparent_control",
        execution_mode="mixed",
        style="s1",
        worker_roles=tuple(roles),
    )
    return SingleRouterFanoutMAS(spec)


def score_visible(pipe, query, output, target_role):
    qemb = np.asarray(
        pipe["embedder"].encode(
            [query],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    oemb = np.asarray(
        pipe["embedder"].encode(
            [output],
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    X = np.concatenate([qemb, oemb], axis=1)

    P = vb.visible_probs(
        pipe["fp_models"],
        pipe["fp_cals"],
        X,
    )[0]

    idx = ROLES.index(target_role)

    return float(P[idx]), {
        r: float(p)
        for r, p in zip(ROLES, P)
    }


def selected_eval_only(target):
    tr = getattr(target, "_last_trace", None)
    if tr is None:
        return []

    return list(
        map(
            str,
            getattr(tr, "selected_roles", ()) or (),
        )
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker.py ROLE_INDEX")

    role_index = int(sys.argv[1])
    role = ROLES[role_index]

    rows = [
        x for x in load_rows()
        if x["role"] == role
    ]

    assert len(rows) == 10, (
        role,
        len(rows),
    )

    replacement = replacement_for(role)
    fillers = fillers_for(role, replacement)

    plus_roles = (role, *fillers)
    minus_roles = (replacement, *fillers)

    print("=" * 100)
    print("ROLE       :", role)
    print("REPLACEMENT:", replacement)
    print("PLUS       :", plus_roles)
    print("MINUS      :", minus_roles)
    print("PROBES     :", len(rows))
    print("=" * 100)

    # This experiment uses pre-generated probes, so the historical
    # Generator is not needed. load_pipeline() normally constructs it
    # and tries to place it on CUDA. Replace only that unused constructor
    # so the visible fingerprint pipeline can load on a CPU node.
    class _UnusedGenerator:
        def __init__(self, *args, **kwargs):
            pass

    ctrl.CurriculumAdaptiveProbePolicyV11 = _UnusedGenerator

    pipe = ctrl.load_pipeline()

    # Same hidden compositions for both Generator arms.
    plus = make_target(
        f"q38r7_{role}_plus",
        plus_roles,
    )

    minus = make_target(
        f"q38r7_{role}_minus",
        minus_roles,
    )

    results = []

    rows = sorted(
        rows,
        key=lambda x: (
            x["arm"],
            int(x["candidate_index"]),
        ),
    )

    for n, x in enumerate(rows, 1):
        arm = x["arm"]
        query = str(x["query"]).strip()

        print()
        print("-" * 100)
        print(
            f"[{n}/{len(rows)}] "
            f"{arm} / {role} / "
            f"candidate={x['candidate_index']}"
        )
        print("QUERY:", query)

        # -------------------------------------------------------
        # PLUS
        # -------------------------------------------------------
        plus_output = str(
            plus.query(query)
        )

        plus_score, plus_scores = score_visible(
            pipe,
            query,
            plus_output,
            role,
        )

        plus_selected = selected_eval_only(plus)

        # -------------------------------------------------------
        # MINUS
        # -------------------------------------------------------
        minus_output = str(
            minus.query(query)
        )

        minus_score, minus_scores = score_visible(
            pipe,
            query,
            minus_output,
            role,
        )

        minus_selected = selected_eval_only(minus)

        margin = plus_score - minus_score

        rec = {
            "arm": arm,
            "role": role,
            "candidate_index": int(x["candidate_index"]),
            "query": query,

            "plus_roles_eval_only": list(plus_roles),
            "minus_roles_eval_only": list(minus_roles),
            "replacement_role_eval_only": replacement,

            "plus_output": plus_output,
            "minus_output": minus_output,

            "plus_target_score": plus_score,
            "minus_target_score": minus_score,
            "pair_margin": margin,

            "plus_all_scores": plus_scores,
            "minus_all_scores": minus_scores,

            # Routing trace is read only AFTER visible scoring.
            # It is evaluation-only and never used in inference.
            "plus_selected_roles_eval_only": plus_selected,
            "minus_selected_roles_eval_only": minus_selected,

            "target_activated_plus_eval_only":
                role in plus_selected,

            "target_absent_from_minus_check":
                role not in minus_roles,
        }

        results.append(rec)

        print(
            f"TARGET SCORE "
            f"PLUS={plus_score:.5f} "
            f"MINUS={minus_score:.5f} "
            f"MARGIN={margin:+.5f}"
        )

        print(
            "PLUS routing eval-only:",
            plus_selected,
        )

    path = OUT / f"{role}.jsonl"

    with path.open("w") as f:
        for x in results:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("SAVED:", path)
    print("ROLE_DONE")


if __name__ == "__main__":
    main()
