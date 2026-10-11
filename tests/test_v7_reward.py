from rl.generative_reward_v7 import GenerativeProbeRewardV7


def test_information_gain_rewards_correct_direction():
    r=GenerativeProbeRewardV7(activation_weight=0,novelty_weight=0,query_cost=0)
    good=r.ask(p_before=.55,p_after=.85,truth_present=True,target_selected=None,query="What details should I provide?",previous_queries=[])
    bad=r.ask(p_before=.55,p_after=.25,truth_present=True,target_selected=None,query="What details should I provide?",previous_queries=[])
    assert good.total > 0 and bad.total < 0


def test_stop_rewards_correct_classification():
    r=GenerativeProbeRewardV7()
    assert r.stop(belief=.9,truth_present=True,remaining_budget=2).total > 0
    assert r.stop(belief=.9,truth_present=False,remaining_budget=2).total < 0
