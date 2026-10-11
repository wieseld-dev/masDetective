import numpy as np
from v8.reward import CounterfactualRewardV8,group_relative_advantages

def test_reward_prefers_target_specific_separation():
 r=CounterfactualRewardV8(query_cost=0)
 a=np.array([1.,0.]);b=np.array([.98,.02]);c=np.array([0.,1.]);sh=np.array([.99,.01])
 good=r.score(target_selected_in_plus=True,plus_seed_emb=a,minus_seed_emb=b,plus_query_emb=a,minus_query_emb=c,sham_seed_emb=a,sham_query_emb=sh)
 bad=r.score(target_selected_in_plus=False,plus_seed_emb=a,minus_seed_emb=b,plus_query_emb=a,minus_query_emb=b,sham_seed_emb=a,sham_query_emb=sh)
 assert good.total>bad.total and good.incremental_separability>0

def test_group_advantages_center():
 a=group_relative_advantages([1,2,3,4]);assert abs(float(a.mean()))<1e-5 and a[-1]>a[0]
