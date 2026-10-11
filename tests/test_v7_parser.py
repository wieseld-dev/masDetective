from detective.generative_probe_policy_v7 import build_probe_policy_instruction_v7, parse_probe_action_v7


def test_parse_ask_and_stop():
    a=parse_probe_action_v7('{"action":"ask","query":"What context would help you check this result?"}')
    assert a.action == "ask" and a.query.endswith("?") and a.parse_ok
    s=parse_probe_action_v7('{"action":"stop"}')
    assert s.is_stop and s.parse_ok


def test_malformed_is_invalid():
    assert not parse_probe_action_v7("I think maybe ask something").parse_ok


def test_policy_prompt_has_no_privileged_fields():
    obs={"candidate_role":"math","candidate_description":"quantitative reasoning","belief":0.61,"remaining_budget":2,"remaining_attempts":4,"history":[{"query":"Q?","output":"A"}]}
    txt=build_probe_policy_instruction_v7(obs)
    assert "true_roles" not in txt and "K_true" not in txt and "system_id" not in txt
    assert "0.610" in txt and "math" in txt

def test_json_scalar_or_array_is_invalid_not_crash():
    bad = [
        '"just a string"',
        '["action", "ask"]',
        '123',
        'true',
        'null',
    ]
    for text in bad:
        parsed = parse_probe_action_v7(text)
        assert parsed.action == "invalid"
        assert not parsed.parse_ok
