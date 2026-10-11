from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
from adaptive_v1.bank import load_bank,split_bank
from adaptive_v1.belief import PrefixBeliefModel
from adaptive_v1.imitation import OracleImitationScorer,action_features
from adaptive_v1.metrics import role_metrics
from adaptive_v1.replay import ReplayEnv
from v11.role_universe import TRAIN_ROLES


def main():
    ap=argparse.ArgumentParser(description='Train pointwise controller initialization from privileged one-step oracle values on training systems only')
    ap.add_argument('--bank',default='artifacts/adaptive_v1/replay_bank.jsonl')
    ap.add_argument('--belief',default='artifacts/adaptive_v1/prefix_belief.joblib')
    ap.add_argument('--output',default='artifacts/adaptive_v1/oracle_imitation.joblib')
    ap.add_argument('--states-per-system',type=int,default=24)
    ap.add_argument('--max-prefix',type=int,default=10)
    ap.add_argument('--query-cost',type=float,default=.01)
    ap.add_argument('--seed',type=int,default=1717)
    args=ap.parse_args()
    systems=load_bank(args.bank,TRAIN_ROLES);train,_=split_bank(systems);belief=PrefixBeliefModel.load(args.belief);rng=random.Random(args.seed);X=[];y=[]
    for si,s in enumerate(train):
        for rep in range(args.states_per_system):
            env=ReplayEnv(s,belief,budget=20,seed=args.seed+si*1000+rep);depth=rng.randint(0,min(args.max_prefix,len(env.available_actions())))
            for _ in range(depth):
                av=env.available_actions()
                if not av:break
                env.step(rng.choice(av))
            state=env.state();av=env.available_actions();m0=role_metrics(state.belief,s.truth_roles,state.roles);b0=m0['brier']
            for a in av:
                o=env.peek(a);b1=belief.predict(state.history+[o]);b1r=role_metrics(b1,s.truth_roles,state.roles)['brier'];util=float(b0-b1r-args.query_cost)
                X.append(action_features(state,a,TRAIN_ROLES));y.append(util)
    model=OracleImitationScorer(TRAIN_ROLES);stat=model.fit(X,y);Path(args.output).parent.mkdir(parents=True,exist_ok=True);model.save(args.output);print('[imitation]',json.dumps(stat,sort_keys=True));print('[done]',args.output)
if __name__=='__main__':main()
