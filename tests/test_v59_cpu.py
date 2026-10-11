#!/usr/bin/env python3
"""CPU end-to-end mock of V59 preparation and evaluation. No MAS/API/GPU."""
import json, tempfile
from pathlib import Path
from analysis.role_direct_within_v4 import v59_fresh_same_mas as m
from analysis.role_direct_within_v4 import v56_nonadaptive as v56
from analysis.role_direct_within_v4 import v57_sensor_night as v57

def jl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def put(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(''.join(json.dumps(x)+'\n' for x in rows))

def run():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);proj=root/'proj';roles=['role'+str(i) for i in range(20)];sids=['mas'+str(i) for i in range(5)]
        d=proj/'artifacts/role_direct_within_v4'
        old=d/'checkpoints/sensor';(old).mkdir(parents=True);(old/'adapter_config.json').write_text('{}')
        (proj/'scripts').mkdir(parents=True);(proj/'scripts/adaptive_v11_execute_round.py').touch()
        manifest=proj/'manifest.json'
        manifest.write_text(json.dumps({'systems':[{'session_id':sid,'split':'dev','roles':roles[i*3:i*3+3]} for i,sid in enumerate(sids)]}))
        source=d/'nonadaptive_v56/mas5_old';source.mkdir(parents=True)
        bankfile=d/'bank.jsonl';approved=d/'approved.jsonl'
        bank=[];ap=[]
        scenarios=['client invoices','load latency','deployment regressions','supplier contracts','API permissions','quality standards','workflow failures','environment risks']
        evidence=['sample records','timing metrics','build traces','policy documents','access logs','test fixtures','error reports','monitoring charts']
        for ri,role in enumerate(roles):
            for slot in range(8):
                question=(f'I am investigating {role} for a project using {scenarios[slot]}. ' + f'What {evidence[slot]} should I provide to help you diagnose constraints and assess this case?')
                row={'target_role':role,'strategy_index':slot,'probe_family':'help_me_help_you','query':question}
                bank.append(row);ap.append(row)
        put(bankfile,bank);put(approved,ap)
        (source/'public_preflight.json').write_text(json.dumps({'session_ids':sids,'source_bank':str(bankfile),'bank_sha256':v56.sha(bankfile)}))
        for rd,slot in ((1,0),(2,1)):
            resp=[]
            for sid in sids:
                for role in roles:
                    q=next(x['query'] for x in bank if x['target_role']==role and x['strategy_index']==slot)
                    resp.append({'session_id':sid,'target_role':role,'query':q,'final_output':f'Answer from {sid} role {role} original'})
            put(source/f'responses_public_dev_round_{rd:02}.jsonl',resp)
        v57root=d/'sensor_v57/night_train';v57root.mkdir(parents=True);(v57root/'jobs.tsv').write_text('123\tv57_sensor\n')
        # Fake completed Sensor checkpoint for REPORT only; GPU scoring is not tested here.
        selected=v57root/'checkpoints/sensor_step_00042';selected.mkdir(parents=True);(selected/'adapter_config.json').write_text('{}')
        (v57root/'training_report.json').write_text(json.dumps({'status':'FINISHED','selected_adapter':str(selected)}))
        # Patch runtime paths and discovery functions to emulate server project.
        m.P=proj;m.AREA=d/'fresh_test_v59';m.SENSOR_OLD=old
        v56.PROJECT=proj;v56.SIMPLE=manifest;v56.APPROVED_134=approved
        v56.manifests=lambda unused:(roles,sids)
        v57.historical_v137=lambda:([],[])
        v57.locate_matched=lambda:None
        v57.v54_matched=lambda:[]
        v56.choose_bank=lambda:bankfile
        out=m.AREA/'fresh_mock'
        m.prepare(out,source,v57root)
        selected_questions=jl(out/'review_40_fresh_questions.jsonl')
        assert len(selected_questions)==40 and {x['strategy_index'] for x in selected_questions}.isdisjoint({0,1})
        assert len({m.hashq(x['query']) for x in selected_questions})==40
        for rd in (1,2):
            responses=[]
            for cand in jl(out/f'candidates_dev_round_{rd:02}.jsonl'):
                responses.append({**cand,'final_output':'NEW ANSWER '+cand['query'],'execution_status':'ok'})
            put(out/f'responses_public_dev_round_{rd:02}.jsonl',responses)
        fake=m.rows_for_sensor(out);assert len(fake)==300
        for name in ('old','v57'):
            scored=[]
            for x in fake:
                truth=json.loads(manifest.read_text())
                present=any(s['session_id']==x['system_id'] and x['role'] in s['roles'] for s in truth['systems'])
                val=float(1 if present else -1) if name=='v57' else float(0)
                scored.append({'system_id':x['system_id'],'target_role':x['role'],'round':x['round'],'logit':val,'token_count':200,'truncated':False})
            put(out/f'fresh_scores_{name}.jsonl',scored)
        m.evaluate(out)
        rep=json.loads((out/'fresh_MAS5_comparison.json').read_text())
        assert rep['methods']['v57']['mean']['mean_within_mas_auc']==1
        assert rep['methods']['v57']['mean']['exact_top3_count']==5
        assert rep['methods']['old']['mean']['mean_within_mas_auc']==.5
        assert rep['training_queries_exact_overlap']==0
        print('MAS5_V59_MOCK_END_TO_END_PASS; fresh questions, 300 model rows, old/new compare, truth metrics')

if __name__=='__main__':run()
