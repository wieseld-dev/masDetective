from dataclasses import dataclass
from v8.counterfactual import make_counterfactual_pair,get_roles,get_k
@dataclass(frozen=True)
class S:
 system_id:str;worker_roles:tuple;execution_mode:str='fanout2';k_true:int=4

def test_pair_preserves_k_and_only_swaps_target():
 s=S('x',('code','math','planning','research'))
 p=make_counterfactual_pair(s,'math',['code','writing','math','planning','research','data','design','operations'],include_sham=True)
 assert p is not None
 assert get_k(p.plus_spec)==get_k(p.minus_spec)==4
 assert 'math' in get_roles(p.plus_spec) and 'math' not in get_roles(p.minus_spec)
 assert len(set(get_roles(p.minus_spec)))==4
 assert p.plus_spec.execution_mode==p.minus_spec.execution_mode=='fanout2'
