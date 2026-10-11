from dataclasses import replace

from rl.role_reward import RoleWhiteBoxReward
from targets.hierarchical_mas import PrivilegedExecutionTrace


def make_trace():
    return PrivilegedExecutionTrace(
        system_id="x", k_true=2, family="flat", depth=1,
        selected_worker="w0", selected_role="code",
        path_nodes=("r0", "w0"), path_edges=(("r0", "w0"),),
        activated_nodes=("r0", "w0"), all_nodes=("r0", "w0", "w1"),
        all_edges=(("r0", "w0"), ("r0", "w1")),
        true_roles=("code", "math"),
    )


def test_brier_improvement_is_rewarded():
    r = RoleWhiteBoxReward(["code", "math", "writing"])
    reward, diag = r.step(
        make_trace(),
        prev_role_belief={"code": .4, "math": .4, "writing": .6},
        new_role_belief={"code": .8, "math": .7, "writing": .2},
        query="What information should I provide?",
        output="Please provide the relevant material.",
    )
    assert diag["r_role_raw"] > 0
    assert reward > -1
