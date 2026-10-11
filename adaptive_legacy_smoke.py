from __future__ import annotations
import argparse,json
from adaptive_v1.legacy import convert_cross_role_matrices
from adaptive_v1.bank import load_bank
from adaptive_v1.belief import PrefixBeliefModel
from v11.role_universe import TRAIN_ROLES


def main():
    ap=argparse.ArgumentParser(description='Smoke-test adaptive aggregation mechanics using heavily-inspected V13.2 development matrices only')
    ap.add_argument('--inputs',nargs='+',default=['artifacts/v13_3c_cross_role/cross_role_matrix.json','artifacts/v13_3c_cross_role_v12/cross_role_matrix.json'])
    ap.add_argument('--bank',default='artifacts/adaptive_v1_legacy/replay_bank.jsonl')
    ap.add_argument('--belief',default='artifacts/adaptive_v1_legacy/prefix_belief.joblib')
    args=ap.parse_args();n=convert_cross_role_matrices(args.inputs,TRAIN_ROLES,args.bank);systems=load_bank(args.bank,TRAIN_ROLES);print(f'[legacy] rows={n} systems={len(systems)}')
    if len(systems)<2:raise SystemExit('need >=2 legacy systems')
    m=PrefixBeliefModel(TRAIN_ROLES);stat=m.fit(systems,permutations_per_system=3,max_prefix=20,seed=7);m.save(args.belief);print(json.dumps(stat,indent=2));print('WARNING: this proves code mechanics only, NOT controller quality.')
if __name__=='__main__':main()
