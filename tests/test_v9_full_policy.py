from v9.policy import SeededProbePolicyV9

def test_full_prompt_has_user_perspective_and_no_internal_targeting():
    p=object.__new__(SeededProbePolicyV9)
    p.tokenizer=object()
    txt=SeededProbePolicyV9.format_discovery_prompt(p,{"candidate_role":"math","candidate_description":"quantitative reasoning","anchor_probe":"What inputs are needed?"})
    assert 'USER perspective' in txt
    assert 'Never reverse the speaker roles' in txt
    assert 'system prompts' in txt
