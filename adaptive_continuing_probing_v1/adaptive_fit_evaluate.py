from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
from adaptive_v1.bank import load_bank,split_bank
from adaptive_v1.belief import PrefixBeliefModel
from adaptive_v1.controllers import RandomController,MaxUncertaintyController,EmpiricalVOIController
from adaptive_v1.evaluate import run_controller
from adaptive_v1.outcome import EmpiricalOutcomeModel
from adaptive_v1.reporting import write_report
from v11.role_universe import TRAIN_ROLES


def main():
    ap=argparse.ArgumentParser(description='Fit learned prefix aggregator on train systems and evaluate adaptive baselines on held-out dev systems')
    ap.add_argument('--bank',default='artifacts/adaptive_v1/replay_bank.jsonl')
    ap.add_argument('--belief-out',default='artifacts/adaptive_v1/prefix_belief.joblib')
    ap.add_argument('--report',default='artifacts/adaptive_v1/controller_report.json')
    ap.add_argument('--episodes-out',default='artifacts/adaptive_v1/controller_episodes.jsonl')
    ap.add_argument('--budgets',default='3,5,8,12,20')
    ap.add_argument('--voi-costs',default='.01,.02,.04,.08')
    ap.add_argument('--seed',type=int,default=4242)
    args=ap.parse_args()
    systems=load_bank(args.bank,TRAIN_ROLES);train,dev=split_bank(systems)
    if len(train)<4 or len(dev)<2:raise SystemExit(f'need train/dev bank; got train={len(train)} dev={len(dev)}')
    belief=PrefixBeliefModel(TRAIN_ROLES,C=.30);fit=belief.fit(train,permutations_per_system=6,max_prefix=20,seed=args.seed);belief.save(args.belief_out)
    outcome=EmpiricalOutcomeModel(TRAIN_ROLES);outfit=outcome.fit(train)
    print('[belief-fit]',json.dumps(fit,sort_keys=True));print('[outcome-fit]',json.dumps(outfit,sort_keys=True))
    budgets=[int(x) for x in args.budgets.split(',') if x.strip()];costs=[float(x) for x in args.voi_costs.split(',') if x.strip()]
    rows=[]
    for budget in budgets:
        ctrls=[('random',RandomController(args.seed+budget),False),('max_uncertainty',MaxUncertaintyController(),False),('oracle_upper_bound',None,True)]
        for cost in costs:ctrls.append((f'voi_c{cost:g}',EmpiricalVOIController(belief,outcome,query_cost=cost,mc_samples=6,min_queries=2,uncertain_scope=4,max_candidate_actions=6,seed=args.seed),False))
        for name,ctrl,oracle in ctrls:
            for i,s in enumerate(dev):
                m=run_controller(s,belief,ctrl,budget=budget,seed=args.seed+i+budget*100,oracle=oracle,oracle_cost=.01);m.update({'controller':name,'budget':budget});rows.append(m)
            vals=[r for r in rows if r['controller']==name and r['budget']==budget]
            print(f"[eval] budget={budget:2d} ctrl={name:<18} n={len(vals)} q={np.mean([x['queries'] for x in vals]):.2f} F1={np.mean([x['f1_blind'] for x in vals]):.3f} exact={np.mean([x['exact_blind'] for x in vals]):.3f} R@K={np.mean([x['known_k_recall'] for x in vals]):.3f} rank={np.mean([x['mean_true_rank'] for x in vals]):.2f}",flush=True)
    ep=Path(args.episodes_out);ep.parent.mkdir(parents=True,exist_ok=True)
    with ep.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rep=write_report(args.report,rows,{'belief_fit':fit,'outcome_fit':outfit,'train_systems':len(train),'dev_systems':len(dev),'warning':'controller-development benchmark only; oracle and known-K metrics are upper-bound diagnostics'})
    print(f'[done] report={args.report} episodes={args.episodes_out}')
if __name__=='__main__':main()
