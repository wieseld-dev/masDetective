from detective.role_policy_model import parse_policy_action


def test_parse_role_action():
    focus, q = parse_policy_action(
        "FOCUS: planning\nQUESTION: What context would be most useful before we start?",
        ["code", "planning"],
    )
    assert focus == "planning"
    assert q.endswith("?")


def test_invalid_focus_is_rejected():
    focus, _ = parse_policy_action("FOCUS: hidden\nQUESTION: What context is useful?", ["code"])
    assert focus == ""
