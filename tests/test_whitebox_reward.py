from rl.whitebox_reward import CoverageRewardConfig, WhiteBoxCoverageReward
from targets.role_presence_mas import PrivilegedTrace


def trace(worker_id: str, k: int = 3):
    return PrivilegedTrace(
        system_id="X",
        worker_id=worker_id,
        selected_role="dummy",
        k_true=k,
        comp_id="X",
        style="s1",
        roles=("a", "b", "c"),
    )


def test_new_worker_is_positive_and_repeat_is_negative():
    rewarder = WhiteBoxCoverageReward(
        CoverageRewardConfig(
            coverage_weight=1.0,
            query_cost=0.02,
            completion_bonus=0.20,
        )
    )
    rewarder.reset(3)

    r1, d1 = rewarder.step(trace("w0"))
    r2, d2 = rewarder.step(trace("w0"))

    assert r1 > 0
    assert d1["new_worker"] == 1.0
    assert r2 < 0
    assert d2["new_worker"] == 0.0


def test_total_coverage_is_normalized():
    rewarder = WhiteBoxCoverageReward()
    rewarder.reset(3)

    rewarder.step(trace("w0"))
    rewarder.step(trace("w1"))
    _, d = rewarder.step(trace("w2"))

    assert d["coverage"] == 1.0
    assert d["completed"] == 1.0
