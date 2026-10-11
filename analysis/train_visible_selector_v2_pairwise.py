#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path('analysis').resolve()))
import learned_visible_probe_ranker_v1 as v1


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class ResidualRanker(nn.Module):
    def __init__(self, d:int, hidden:int=128, dropout:float=.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
        # Start exactly from analytic Visible-MI ordering.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_pairs(groups, y, gap_quantile:float):
    raw=[]; gaps=[]
    for g in groups:
        for a in range(len(g)):
            for b in range(a+1, len(g)):
                i,j=g[a],g[b]
                d=float(y[i]-y[j])
                if abs(d)>1e-12:
                    raw.append((i,j,d)); gaps.append(abs(d))
    if not gaps:
        raise RuntimeError('No non-tied pairs')
    thr=float(np.quantile(gaps, gap_quantile))
    med=max(float(np.median(gaps)), 1e-6)
    pairs=[]
    for i,j,d in raw:
        if abs(d)<thr:
            continue
        if d>0: better,worse=i,j
        else: better,worse=j,i
        weight=float(np.clip(abs(d)/med, .5, 4.0))
        pairs.append((better,worse,weight))
    return pairs,thr


def fit_ranker(X, base, y, groups, cfg, seed):
    seed_all(seed)
    pairs,gap=build_pairs(groups,y,cfg['gap_quantile'])
    model=ResidualRanker(X.shape[1],cfg['hidden'],cfg['dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    Xt=torch.tensor(X,dtype=torch.float32)
    bt=torch.tensor(base,dtype=torch.float32)
    rng=np.random.default_rng(seed)

    for ep in range(cfg['epochs']):
        order=rng.permutation(len(pairs)); losses=[]
        model.train()
        for st in range(0,len(order),cfg['pair_batch']):
            sel=order[st:st+cfg['pair_batch']]
            bi=torch.tensor([pairs[t][0] for t in sel],dtype=torch.long)
            wi=torch.tensor([pairs[t][1] for t in sel],dtype=torch.long)
            wt=torch.tensor([pairs[t][2] for t in sel],dtype=torch.float32)
            rb=model(Xt[bi]); rw=model(Xt[wi])
            sb=bt[bi] + cfg['residual_scale']*rb
            sw=bt[wi] + cfg['residual_scale']*rw
            rank=(torch.nn.functional.softplus(-(sb-sw))*wt).sum()/wt.sum().clamp_min(1e-6)
            reg=cfg['residual_l2']*(rb.square().mean()+rw.square().mean())
            loss=rank+reg
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            losses.append(float(loss.detach()))
        if ep in (0,cfg['epochs']-1):
            print(f"epoch={ep+1}/{cfg['epochs']} loss={np.mean(losses):.6f} pairs={len(pairs)} gap={gap:.6f}",flush=True)

    model.eval()
    with torch.no_grad():
        residual=model(Xt).cpu().numpy()
    score=base + cfg['residual_scale']*residual
    return model,score,gap,len(pairs)


def offset_groups(groups, off):
    return [[off+i for i in g] for g in groups]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='artifacts/adaptive_v11_curriculum_v2')
    ap.add_argument('--obs-dir',default='artifacts/observation_model_v1')
    ap.add_argument('--vis-dir',default='artifacts/visible_output_fingerprint_v1')
    ap.add_argument('--cal-dir',default='artifacts/calibrated_bayes_diag_v1')
    ap.add_argument('--out',default='artifacts/visible_selector_v2_pairwise')
    ap.add_argument('--particles-per-k',type=int,default=8)
    ap.add_argument('--pca-dim',type=int,default=24)
    ap.add_argument('--batch-size',type=int,default=4096)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--seed',type=int,default=117)
    args=ap.parse_args(); seed_all(args.seed)

    om,vb=v1.load_helpers()
    root=Path(args.root); obsdir=Path(args.obs_dir); visdir=Path(args.vis_dir); caldir=Path(args.cal_dir)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    print('loading examples/embeddings...',flush=True)
    examples,roles,_,_=om.collect_examples(root)
    split=json.loads((obsdir/'split_systems.json').read_text())
    trsys,vasys,tesys=set(split['train']),set(split['val']),set(split['test'])
    Q=np.load(obsdir/'query_embeddings.npy').astype(np.float32)
    O=np.load(visdir/'output_embeddings.npy').astype(np.float32)
    Xtxt=np.concatenate([Q,O],axis=1)
    tri=np.array([i for i,x in enumerate(examples) if x['system_id'] in trsys],int)
    vai=np.array([i for i,x in enumerate(examples) if x['system_id'] in vasys],int)

    Y=vb.labels(examples,roles,'selected_roles')
    print('fitting visible fingerprint detector TRAIN / calibrating VAL...',flush=True)
    fp_models,fp_cals,thr,tpr,fpr,_=vb.fit_visible_detector(Xtxt,Y,tri,vai)
    Pall=vb.visible_probs(fp_models,fp_cals,Xtxt)
    Eall=(Pall>=thr[None,:]).astype(np.float64)

    pca=PCA(n_components=args.pca_dim,svd_solver='randomized',random_state=args.seed)
    pca.fit(Q[tri]); Qpca=pca.transform(Q).astype(np.float32)

    ckpt=torch.load(obsdir/'model.pt',map_location='cpu')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    fwd=om.ObservationNet(qdim=ckpt['query_embedding_dim'],n_roles=len(roles),hidden=ckpt['hidden'],dropout=ckpt['dropout']).to(device)
    fwd.load_state_dict(ckpt['state_dict']); fwd.eval()

    ri={r:i for i,r in enumerate(roles)}
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]['selected_roles']: pos[ri[r]]+=1
    posw=np.clip((len(tri)-pos)/np.maximum(pos,1.0),1.0,20.0); log_posw=np.log(posw)
    cs=json.loads((caldir/'summary.json').read_text())
    T=float(cs['true_Z_probability_calibration']['corrected_plus_temperature']['best_T'])
    beta=float(cs['validation_posterior_update']['selected_beta_by_validation_log_gain'])
    max_round=max(x['round'] for x in examples)

    common=dict(examples=examples,Q=Q,Qpca=Qpca,Eall=Eall,roles=roles,model=fwd,device=device,
                log_posw=log_posw,T=T,beta=beta,tpr=tpr,fpr=fpr,vb=vb,
                particles_per_k=args.particles_per_k,batch_size=args.batch_size,max_round=max_round,seed=args.seed)
    print('building the exact V1 feature/target dataset...',flush=True)
    Xtr,ytr,mtr,gtr=v1.collect_split('train',trsys,**common)
    Xva,yva,mva,gva=v1.collect_split('val',vasys,**common)
    Xte,yte,mte,gte=v1.collect_split('test',tesys,**common)
    print({'train':(len(Xtr),len(gtr)),'val':(len(Xva),len(gva)),'test':(len(Xte),len(gte))},flush=True)

    scaler=StandardScaler().fit(Xtr)
    Ztr=scaler.transform(Xtr).astype(np.float32)
    Zva=scaler.transform(Xva).astype(np.float32)
    Zte=scaler.transform(Xte).astype(np.float32)

    vtr=np.array([m['visible_mi'] for m in mtr],np.float32)
    vva=np.array([m['visible_mi'] for m in mva],np.float32)
    vte=np.array([m['visible_mi'] for m in mte],np.float32)
    vm=float(vtr.mean()); vs=max(float(vtr.std()),1e-6)
    btr=(vtr-vm)/vs; bva=(vva-vm)/vs; bte=(vte-vm)/vs

    baseline_val=v1.evaluate_scores(bva,mva,gva)
    baseline_test=v1.evaluate_scores(bte,mte,gte)
    print('VISIBLE-MI BASELINE VAL learned_log=',baseline_val.get('learned_log'),flush=True)

    configs=[]
    for gapq in [.35,.55,.70]:
        for rs in [.35,.70]:
            configs.append(dict(gap_quantile=gapq,residual_scale=rs,hidden=128,dropout=.10,
                                lr=7e-4,epochs=12,pair_batch=1024,residual_l2=2e-3))

    trials=[]; best=None
    for ci,cfg in enumerate(configs):
        print(f"\n===== CONFIG {ci+1}/{len(configs)} {cfg} =====",flush=True)
        mdl,strain,gap,npairs=fit_ranker(Ztr,btr,ytr,gtr,cfg,args.seed+ci)
        mdl.eval()
        with torch.no_grad():
            rva=mdl(torch.tensor(Zva,dtype=torch.float32)).numpy()
        sva=bva+cfg['residual_scale']*rva
        ev=v1.evaluate_scores(sva,mva,gva)
        rec={'config':cfg,'gap_threshold':gap,'train_pairs':npairs,'validation':ev}; trials.append(rec)
        print(json.dumps(ev,indent=2),flush=True)
        key=(ev.get('learned_log',-1e30), ev.get('learned_vs_log_gain_spearman') or -1e30)
        if best is None or key>best[0]: best=(key,cfg)

    chosen=best[1]
    print('\nCHOSEN=',json.dumps(chosen),flush=True)

    # Strict train-only model evaluated on untouched TEST.
    strict_model,_,gap,npairs=fit_ranker(Ztr,btr,ytr,gtr,chosen,args.seed+999)
    strict_model.eval()
    with torch.no_grad(): rte=strict_model(torch.tensor(Zte,dtype=torch.float32)).numpy()
    strict_score=bte+chosen['residual_scale']*rte
    strict_test=v1.evaluate_scores(strict_score,mte,gte)

    # Final refit TRAIN+VAL. Keep preprocessing/PCA fitted on TRAIN only; test untouched.
    Ztv=np.concatenate([Ztr,Zva]); ytv=np.concatenate([ytr,yva]); mtv=mtr+mva
    gtv=gtr+offset_groups(gva,len(Ztr))
    vtv=np.concatenate([vtr,vva]); vm2=float(vtv.mean()); vs2=max(float(vtv.std()),1e-6)
    btv=(vtv-vm2)/vs2; bte2=(vte-vm2)/vs2
    final_model,_,gap2,npairs2=fit_ranker(Ztv,btv,ytv,gtv,chosen,args.seed+2026)
    final_model.eval()
    with torch.no_grad(): rte2=final_model(torch.tensor(Zte,dtype=torch.float32)).numpy()
    final_score=bte2+chosen['residual_scale']*rte2
    final_test=v1.evaluate_scores(final_score,mte,gte)

    torch.save({'state_dict':final_model.state_dict(),'input_dim':Ztv.shape[1],'config':chosen,
                'feature_scaler_mean':scaler.mean_,'feature_scaler_scale':scaler.scale_,
                'visible_mi_mean':vm2,'visible_mi_std':vs2,'roles':roles},out/'selector_v2.pt')

    with (out/'test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,m in enumerate(mte):
            z=dict(m); z['selector_v2_score']=float(final_score[i]); f.write(json.dumps(z)+'\n')

    rep=Path('artifacts/repeat_probe_stability_v1/fresh_repeats.jsonl')
    partial=sum(1 for line in rep.open(encoding='utf-8') if line.strip()) if rep.exists() else 0
    summary={
        'setting':{
            'selector_inputs':'same deployable pre-query features as V1; no actual output/truth at selection',
            'training_target':'single-execution realized visible-output delta log P(K_true), group-centered',
            'objective':'confidence-filtered pairwise ranking + residual correction to analytic Visible-MI',
            'split_by_system':True,'particles_per_K':args.particles_per_k,
            'partial_repeat_rows_seen_but_not_used':partial,
        },
        'visible_mi_baseline_validation':baseline_val,
        'visible_mi_baseline_test':baseline_test,
        'validation_trials':trials,
        'chosen_config':chosen,
        'strict_train_only_test':strict_test,
        'final_refit_train_plus_val_test':final_test,
        'decision':{
            'success':'final learned_log > visible_mi baseline learned_log on test, preferably with higher Spearman',
            'failure':'do not tune V2 further; finish repeated execution collection and train on multi-execution mean utility labels'
        }
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+'='*100+'\nSELECTOR V2 FINAL TEST\n'+'='*100)
    print(json.dumps(final_test,indent=2))
    print('[saved]',out/'summary.json',flush=True)

if __name__=='__main__':
    main()
