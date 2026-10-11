"""No-network V58 replay check with injected synthetic manifest/responses."""
import tempfile,json,joblib,os
from pathlib import Path
from tests.test_v511_cpu import fixture
from analysis.role_direct_within_v4 import v511_rolewise_verifier as v

def main():
    with tempfile.TemporaryDirectory() as t:
        d=Path(t);roles,train,val,test,stat,priority=fixture()
        sid='mas_058';truth=set(test[sid][1]);data=test[sid][0]
        v58=d/'v58';v58.mkdir();(v58/'public_preflight.json').write_text(json.dumps({'session_id':sid,'roles':roles}))
        manifest=d/'manifest.json';manifest.write_text(json.dumps({'systems':[{'session_id':sid,'roles':list(truth),'split':'dev'}]}))
        for rd in range(1,4):
            with (v58/f'responses_public_dev_round_{rd:02}.jsonl').open('w') as f:
                for role in roles:
                    row=data[role]
                    f.write(json.dumps({'session_id':sid,'target_role':role,'query':row['query'],
                           'final_output':row['answer']+' measured round '+str(rd),'execution_status':'ok'})+'\n')
        models_by={};results={}
        old_v58=os.environ.get('V511_V58_ROOT');old_man=v.v510.v57.MANIFEST
        try:
            v.v510.v57.MANIFEST=manifest;os.environ['V511_V58_ROOT']=str(v58)
            for arm in v.ARMS:
                models_by[arm]={role:v.fit_one(role,train,arm,priority) for role in roles}
                result,_=v.eval_arm(models_by[arm],arm,val,test,roles)
                results[arm]=result
            out=d/'out';out.mkdir()
            replay=v.replay_existing_v58(models_by,results,roles,train,out)
            assert replay['status']=='REPLAY_COMPLETE' and replay['rounds']==3,replay
            r=json.loads((out/'v58_offline_replay.json').read_text())
            assert r['rounds'][0]['models']['answer_only']['within_mas_auc']>=0
            print('V511_REPLAY_SELFTEST_OK (3 synthetic rounds, no API)')
        finally:
            v.v510.v57.MANIFEST=old_man
            if old_v58 is None:os.environ.pop('V511_V58_ROOT',None)
            else:os.environ['V511_V58_ROOT']=old_v58
if __name__=='__main__':main()
