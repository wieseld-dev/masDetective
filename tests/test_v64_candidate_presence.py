from __future__ import annotations

from estimator.candidate_presence_contrastive_v64 import (
    ContrastiveCandidateEstimatorV64,
    build_candidate_records,
)


def _fake_rows():
    public = []
    truth = []
    systems = [
        ("s1", ["math", "research"], "single"),
        ("s2", ["code", "writing"], "fanout2"),
        ("s3", ["math", "writing"], "fanout3"),
        ("s4", ["code", "research"], "mixed"),
    ]
    for sid, roles, mode in systems:
        public.append({"system_id": sid, "focus_role": "__anchor__", "output": "Please provide goals and context."})
        truth.append({"system_id": sid, "focus_role": "__anchor__", "true_roles": roles,
                      "selected_roles": [], "mode": mode})
        for role in ["math", "code", "writing", "research"]:
            if role in roles:
                out = {
                    "math": "Please provide equations variables units and the quantitative result.",
                    "code": "Please provide code error logs language and expected behavior.",
                    "writing": "Please provide the draft audience tone and desired structure.",
                    "research": "Please provide sources claims scope and evidence requirements.",
                }[role]
                selected = [role]
            else:
                out = "Please provide general context, goals, and constraints so I can help."
                selected = [roles[0]]
            public.append({"system_id": sid, "focus_role": role, "output": out})
            truth.append({"system_id": sid, "focus_role": role, "true_roles": roles,
                          "selected_roles": selected, "mode": mode})
    return public, truth


def test_build_records_is_candidate_by_candidate():
    public, truth = _fake_rows()
    recs = build_candidate_records(public, truth)
    assert len(recs) == 16
    math_s1 = next(r for r in recs if r.system_id == "s1" and r.role == "math")
    code_s1 = next(r for r in recs if r.system_id == "s1" and r.role == "code")
    assert math_s1.label == 1
    assert code_s1.label == 0
    assert math_s1.anchor_output == code_s1.anchor_output


def test_estimator_uses_external_pair_and_returns_probability():
    public, truth = _fake_rows()
    recs = build_candidate_records(public, truth)
    est = ContrastiveCandidateEstimatorV64(
        ["math", "code", "writing", "research"],
        {
            "math": "equations variables units quantitative reasoning",
            "code": "programming debugging code errors software",
            "writing": "draft tone audience clarity structure",
            "research": "sources evidence claims scope research",
        },
    ).fit(recs)
    p = est.predict_candidate_proba(
        role="math",
        anchor_output="Please provide goals and context.",
        target_output="Please provide equations variables units and the quantitative result.",
    )
    assert 0.0 <= p <= 1.0


def test_selected_roles_are_not_required_for_features():
    public, truth = _fake_rows()
    for row in truth:
        row.pop("selected_roles", None)
    recs = build_candidate_records(public, truth)
    assert len(recs) == 16
