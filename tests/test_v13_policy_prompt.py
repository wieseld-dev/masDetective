from v13.policy import DisambiguationProbePolicyV13


def test_prompt_mentions_behavioral_disambiguation_without_truth_fields():
    class Tok:
        pass
    p = object.__new__(DisambiguationProbePolicyV13)
    p.tokenizer = Tok()
    text = p.format_discovery_prompt({
        "candidate_role":"requirements_analysis",
        "candidate_description":"clarifying requirements and acceptance criteria",
        "anchor_probe":"What information should I provide?",
        "avoid_probes":["What are the constraints?"],
        "confusable_roles":[{"role":"customer_support_triage","description":"triage, severity, routing"}],
        "public_history":[{"query":"What failed?","output":"Please clarify expected behavior and acceptance criteria."}],
    })
    low=text.lower()
    assert "confusable alternative" in low
    assert "characteristic method" in low or "professional instinct" in low
    assert "true roles" not in low
    assert "k_true" not in low
