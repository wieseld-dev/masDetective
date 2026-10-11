from detective.role_choice_policy_v63 import available_roles, build_role_choice_prompt, fixed_query_for_role


def test_policy_state_is_compact_and_no_raw_history():
    obs = {
        "app_name":"General multi-skill assistant",
        "candidate_roles":["code","math","writing"],
        "tested_roles":["writing"],
        "role_descriptions":{"code":"coding","math":"quantitative","writing":"text"},
        "role_belief":{"code":.2,"math":.5,"writing":.9},
        "history":[{"output":"RAW HISTORY SHOULD NOT APPEAR"}],
        "remaining_budget":2,
    }
    assert available_roles(obs) == ("code","math")
    p = build_role_choice_prompt(obs)
    assert "P_present" in p and "uncertainty" in p
    assert "RAW HISTORY SHOULD NOT APPEAR" not in p
    assert fixed_query_for_role("math").endswith("?")
