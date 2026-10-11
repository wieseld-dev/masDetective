from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from analysis.train_v13_2_dual_judge import (
    BranchCase,
    attach_components,
    build_items_and_endpoints,
    collect_texts,
    combined_branch_metrics,
    endpoint_features,
    eval_branch_metrics,
    fit_calibrator,
    normalize_discovery_rows,
    quality_weight,
)
from v13_2.judge import Calibrator, attribution_text, evidence_prior_text, evidence_text


def mk_case(**kw):
    base = dict(
        source="v13",
        group_id="g1",
        row_index=0,
        target="data",
        target_desc="analyzes datasets",
        query="How would you inspect this dataset?",
        pair_id="p1",
        plus_output="I profile columns and compute missingness statistics.",
        minus_output="I design a normalized schema with keys.",
        sham_output="I profile distributions and check outliers.",
        plus_selected=("data",),
        minus_selected=("database_design",),
        sham_selected=("data",),
        plus_roles=("data", "writing"),
        minus_roles=("database_design", "writing"),
        sham_roles=("data", "planning"),
        hardneg="database_design",
        hardneg_desc="designs relational schemas",
        specific=0.12,
        strict=True,
        confirmed=True,
        elite=False,
        quality_weight=1.2,
        quality_tier="strong",
        component="C",
    )
    base.update(kw)
    return BranchCase(**base)


def test_prompt_separation():
    q = "How would you inspect this dataset?"
    a = "I compute missingness and distribution statistics."
    ev = evidence_text("data", "analyzes datasets", q, a)
    pr = evidence_prior_text("data", "analyzes datasets", q)
    at = attribution_text("data", "analyzes datasets", a)
    assert q in ev and a in ev
    assert q in pr and "[NO ANSWER OBSERVED]" in pr
    assert q not in at
    assert a in at


def test_quality_curriculum():
    w, tier = quality_weight(specific=-0.1, strict=False, confirmed=False, elite=False, source_weight=1.0)
    assert w == 0.0 and tier == "negative"
    w, tier = quality_weight(specific=0.02, strict=False, confirmed=False, elite=False, source_weight=1.0)
    assert 0.0 < w < 1.0 and tier == "moderate"
    w, tier = quality_weight(specific=-0.1, strict=False, confirmed=True, elite=False, source_weight=1.0)
    assert w > 1.0 and tier == "strong"


def test_normalize_v11_and_v13_fields():
    desc = {"data": "analyzes datasets", "database_design": "designs schemas"}
    v11 = [{
        "phase": "train",
        "target_role": "data",
        "candidate_description_public": "analyzes datasets",
        "query": "Q?",
        "elite": False,
        "confirmation_pass_training_only": False,
        "branches_training_only": [{
            "pair_id": "x",
            "plus_output": "P",
            "minus_output": "M",
            "sham_output": "S",
            "plus_selected": ["data"],
            "minus_selected": ["database_design"],
            "sham_selected": ["data"],
            "specific_pairwise": 0.1,
        }],
    }]
    c, sk = normalize_discovery_rows(rows=v11, source="v11", role_desc=desc, source_weight=0.7)
    assert len(c) == 1 and c[0].specific == 0.1 and c[0].hardneg is None

    v13 = [{
        "phase": "train",
        "status": "ok",
        "group_id": 1,
        "target_role": "data",
        "candidate_description_public": "analyzes datasets",
        "query": "Q?",
        "reward_breakdown_training_only": {"strict_specific_pass": True},
        "branches_training_only": [{
            "pair_id": "y",
            "plus_output": "P",
            "minus_output": "M",
            "sham_output": "S",
            "plus_selected": ["data"],
            "minus_selected": ["database_design"],
            "sham_selected": ["data"],
            "plus_roles_training_only": ["data"],
            "minus_roles_training_only": ["database_design"],
            "sham_roles_training_only": ["data"],
            "hard_negative_replacement_training_only": "database_design",
            "specific_pairwise_training_only": -0.2,
        }],
    }]
    c, sk = normalize_discovery_rows(rows=v13, source="v13", role_desc=desc, source_weight=1.0)
    assert len(c) == 1 and c[0].hardneg == "database_design"
    assert c[0].quality_weight > 0.0  # strict probe survives one noisy branch


def test_leakage_components_link_query_and_output_not_composition():
    a = mk_case(group_id="a", pair_id="pa", query="SAME?", component="")
    b = mk_case(group_id="b", pair_id="pb", query="SAME?", plus_output="different", component="")
    linked = attach_components([a, b])
    assert linked[0].component == linked[1].component

    c = mk_case(group_id="c", pair_id="pc", query="OTHER?", plus_output="unique-c", minus_output="unique-cm", sham_output="unique-cs", component="")
    d = mk_case(group_id="d", pair_id="pd", query="OTHER2?", plus_output="unique-d", minus_output="unique-dm", sham_output="unique-ds", component="")
    # Repeated unordered role composition is audit-only; it must not create a
    # transitive DSU edge that collapses independent counterfactual families.
    linked = attach_components([c, d])
    assert linked[0].component != linked[1].component


def test_objectives_include_explicit_minus_flip():
    c = mk_case()
    items, endpoints, counts = build_items_and_endpoints(
        [c], attr_target_weight=0.75, attr_hard_weight=1.0, minus_flip_weight=2.0
    )
    types = [x.item_type for x in items]
    for required in (
        "evidence_plus_minus",
        "evidence_sham_minus",
        "attr_target_plus_minus",
        "attr_target_sham_minus",
        "attr_hard_plus",
        "attr_hard_sham",
        "attr_hard_minus",
    ):
        assert required in types
    minus = [x for x in items if x.item_type == "attr_hard_minus"][0]
    assert minus.weight == c.quality_weight * 2.0
    assert minus.prior_text is None
    assert "BENIGN DIAGNOSTIC QUESTION" not in minus.pos_text
    labels = {(e.endpoint_type, e.label) for e in endpoints}
    assert ("hard_minus_selected", 1) in labels
    assert ("target_minus", 0) in labels



def test_collected_texts_cover_branch_metrics():
    c = mk_case()
    items, endpoints, _ = build_items_and_endpoints(
        [c], attr_target_weight=0.75, attr_hard_weight=1.0, minus_flip_weight=2.0
    )
    texts = collect_texts(items, endpoints, [c])
    # Give each text deterministic two-head logits; the point is complete key coverage.
    sc = {t: (float(i % 7), float((i * 3) % 11)) for i, t in enumerate(dict.fromkeys(texts))}
    metrics = eval_branch_metrics([c], sc)
    assert metrics["evidence_causal_plus_n"] == 1
    assert metrics["attribution_hard_minus_n"] == 1



def test_combined_calibrator_and_flip_path():
    cases = []
    for i in range(12):
        cases.append(mk_case(
            group_id=f"g{i}", pair_id=f"p{i}", query=f"Q{i}?",
            plus_output=f"plus{i}", minus_output=f"minus{i}", sham_output=f"sham{i}",
            component=f"C{i}",
        ))
    items, endpoints, _ = build_items_and_endpoints(
        cases, attr_target_weight=0.75, attr_hard_weight=1.0, minus_flip_weight=2.0
    )
    texts = list(dict.fromkeys(collect_texts(items, endpoints, cases)))
    sc = {t: [0.0, 0.0] for t in texts}
    # Encode endpoint truth cleanly into both heads. This is only a plumbing test.
    for e in endpoints:
        ev_full = evidence_text(e.role, e.desc, e.query, e.output)
        ev_prior = evidence_prior_text(e.role, e.desc, e.query)
        at = attribution_text(e.role, e.desc, e.output)
        sc[ev_prior][0] = 0.0
        sc[ev_full][0] = 2.0 if e.label else -2.0
        sc[at][1] = 2.0 if e.label else -2.0
    sc = {k: tuple(v) for k, v in sc.items()}
    scaler, lr = fit_calibrator(endpoints, sc)
    m = combined_branch_metrics(cases, sc, scaler, lr)
    assert m["combined_hard_minus_accuracy"] == 1.0
    assert m["combined_hard_flip_accuracy"] == 1.0


def test_calibrator_math_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        p.write_text(json.dumps({
            "feature_names": ["evidence_delta", "answer_attribution"],
            "mean": [1.0, 2.0],
            "scale": [2.0, 4.0],
            "coef": [0.5, -0.25],
            "intercept": 0.1,
        }))
        c = Calibrator.from_json(p)
        got = c.raw_logit([3.0, 6.0])
        expected = 0.5 * ((3 - 1) / 2) - 0.25 * ((6 - 2) / 4) + 0.1
        assert abs(got - expected) < 1e-9
        assert 0.0 < c.probability([3.0, 6.0]) < 1.0


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in sorted(tests, key=lambda f: f.__name__):
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
