from rl.reward import FullWhiteBoxReward, RewardConfig
from targets.hierarchical_mas import PrivilegedExecutionTrace


def mk(worker="w0", path=("r0", "w0")):
    return PrivilegedExecutionTrace(
        system_id="x", k_true=3, family="flat", depth=1,
        selected_worker=worker, selected_role="code",
        path_nodes=tuple(path), path_edges=tuple(zip(path[:-1], path[1:])),
        activated_nodes=tuple(path),
        all_nodes=("r0", "w0", "w1", "w2"),
        all_edges=(("r0", "w0"), ("r0", "w1"), ("r0", "w2")),
    )


def test_k_information_gain_increases_reward():
    r = FullWhiteBoxReward(RewardConfig(query_cost=0.0, token_cost_per_1k=0.0))
    score, d = r.step(
        mk(), prev_k_belief={3: 0.2}, new_k_belief={3: 0.4}, query="q", output="o"
    )
    assert d["r_k"] > 0
    assert score > 0


def test_repeat_path_is_penalized():
    cfg = RewardConfig(k_information_weight=0.0, worker_coverage_weight=0.0, edge_coverage_weight=0.0,
                       path_novelty_weight=0.1, repeat_path_penalty=0.1, query_cost=0.0,
                       token_cost_per_1k=0.0, completion_bonus=0.0)
    r = FullWhiteBoxReward(cfg)
    first, _ = r.step(mk(), prev_k_belief={3:1}, new_k_belief={3:1}, query="q", output="o")
    second, _ = r.step(mk(), prev_k_belief={3:1}, new_k_belief={3:1}, query="q", output="o")
    assert first > second
