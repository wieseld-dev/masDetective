from types import SimpleNamespace

from rl.role_reward_v63 import RoleWhiteBoxRewardV63


def trace(selected=("math",)):
    return SimpleNamespace(true_roles=("math", "research"), selected_roles=selected)


def test_focus_present_probability_improvement_is_strongly_positive():
    r = RoleWhiteBoxRewardV63(["code", "math", "research", "writing"])
    reward, d = r.step(
        trace(), focus_role="math",
        prev_role_belief={"code":.5,"math":.5,"research":.5,"writing":.5},
        new_role_belief={"code":.4,"math":.8,"research":.55,"writing":.4},
        query="q", output="o",
    )
    assert d["r_focus"] > 0
    assert d["r_global"] > 0
    assert reward > 0


def test_absent_role_can_earn_positive_reward_by_moving_down():
    r = RoleWhiteBoxRewardV63(["code", "math", "research", "writing"])
    reward, d = r.step(
        trace(), focus_role="code",
        prev_role_belief={"code":.5,"math":.5,"research":.5,"writing":.5},
        new_role_belief={"code":.15,"math":.55,"research":.55,"writing":.45},
        query="q", output="o",
    )
    assert d["focus_true"] == 0.0
    assert d["r_focus"] > 0
    assert reward > 0
