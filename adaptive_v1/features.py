from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .schema import EvidenceObservation

EPS = 1e-6


def _safe_stats(xs: Sequence[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "last": 0.0, "trend": 0.0, "pos_frac": 0.0}
    a = np.asarray(xs, dtype=np.float64)
    return {"mean":float(np.mean(a)),"median":float(np.median(a)),"max":float(np.max(a)),"min":float(np.min(a)),"std":float(np.std(a)),"last":float(a[-1]),"trend":float(a[-1]-a[0]) if len(a)>1 else 0.0,"pos_frac":float(np.mean(a>0.0))}


def robust_event_z(evidence: Mapping[str, float], roles: Sequence[str]) -> dict[str, float]:
    vals=np.asarray([float(evidence.get(r,0.0)) for r in roles],dtype=np.float64); med=float(np.median(vals)); mad=float(np.median(np.abs(vals-med))); scale=max(1.4826*mad,0.25)
    return {r:float(np.clip((float(evidence.get(r,0.0))-med)/scale,-8.0,8.0)) for r in roles}


def event_rank_percentile(evidence: Mapping[str,float], roles: Sequence[str]) -> dict[str,float]:
    order=sorted(roles,key=lambda r:float(evidence.get(r,0.0)),reverse=True); n=max(len(order)-1,1); rank={r:i for i,r in enumerate(order)}
    return {r:float(1.0-rank[r]/n) for r in roles}


V135_FEATURE_NAMES=(
    "n_history",
    "v135_raw_mean","v135_raw_median","v135_raw_max","v135_raw_min","v135_raw_std","v135_raw_last","v135_raw_trend","v135_raw_pos_frac",
    "v135_z_mean","v135_z_median","v135_z_max","v135_z_std","v135_z_last","v135_z_pos_frac",
    "v135_rank_mean","v135_rank_max","v135_rank_last","v135_top3_frac",
    "target_count","v135_target_raw_mean","v135_target_raw_last","v135_target_z_mean","v135_target_z_last","v135_target_rank_mean",
    "cross_count","v135_cross_raw_mean","v135_cross_z_mean","v135_cross_rank_mean",
    "competitor_count","v135_competitor_raw_mean","v135_competitor_z_mean",
    "last_action_is_target","last_action_is_competitor",
)
V134_FEATURE_NAMES=(
    "v134_available_frac",
    "v134_raw_mean","v134_raw_median","v134_raw_max","v134_raw_min","v134_raw_std","v134_raw_last","v134_raw_trend","v134_raw_pos_frac",
    "v134_z_mean","v134_z_last","v134_rank_mean","v134_rank_last","v134_top3_frac",
    "v134_target_raw_mean","v134_target_z_mean","v134_target_rank_mean",
    "v134_cross_raw_mean","v134_cross_z_mean","v134_cross_rank_mean",
    "dual_last_z_gap","dual_mean_z_gap","dual_last_rank_gap","dual_both_top3_frac",
)
BASE_FEATURE_NAMES=V135_FEATURE_NAMES+V134_FEATURE_NAMES


def _v134_map(o: EvidenceObservation) -> Mapping[str,float]:
    x=o.metadata.get('evidence_v134',{}) if o.metadata else {}
    return x if isinstance(x,Mapping) else {}


def role_history_features(history: Sequence[EvidenceObservation], role: str, roles: Sequence[str]) -> np.ndarray:
    if not history: return np.zeros(len(BASE_FEATURE_NAMES),dtype=np.float32)

    # V13.5 causal-participation channel (backward-compatible `evidence`).
    raw=[o.score(role) for o in history]; z_maps=[robust_event_z(o.evidence,roles) for o in history]; rank_maps=[event_rank_percentile(o.evidence,roles) for o in history]
    zs=[m[role] for m in z_maps]; ranks=[m[role] for m in rank_maps]; top3=[1.0 if sorted(roles,key=lambda r:float(o.evidence.get(r,0.0)),reverse=True).index(role)<3 else 0.0 for o in history]
    t_idx=[i for i,o in enumerate(history) if o.action.target==role]; c_idx=[i for i,o in enumerate(history) if o.action.competitor==role]; x_idx=[i for i,o in enumerate(history) if o.action.target!=role]
    def pick(vals,idx): return [vals[i] for i in idx]
    rs=_safe_stats(raw); zz=_safe_stats(zs); rr=_safe_stats(ranks); trs=_safe_stats(pick(raw,t_idx)); tzs=_safe_stats(pick(zs,t_idx)); trr=_safe_stats(pick(ranks,t_idx)); crs=_safe_stats(pick(raw,c_idx)); czs=_safe_stats(pick(zs,c_idx)); xrs=_safe_stats(pick(raw,x_idx)); xzs=_safe_stats(pick(zs,x_idx)); xrr=_safe_stats(pick(ranks,x_idx))
    vals=[float(len(history)),rs['mean'],rs['median'],rs['max'],rs['min'],rs['std'],rs['last'],rs['trend'],rs['pos_frac'],zz['mean'],zz['median'],zz['max'],zz['std'],zz['last'],zz['pos_frac'],rr['mean'],rr['max'],rr['last'],float(np.mean(top3)) if top3 else 0.0,float(len(t_idx)),trs['mean'],trs['last'],tzs['mean'],tzs['last'],trr['mean'],float(len(x_idx)),xrs['mean'],xzs['mean'],xrr['mean'],float(len(c_idx)),crs['mean'],czs['mean'],1.0 if history[-1].action.target==role else 0.0,1.0 if history[-1].action.competitor==role else 0.0]

    # V13.4 absolute-composition channel stored in metadata to preserve schema compatibility.
    maps=[_v134_map(o) for o in history]; avail=[1.0 if role in m else 0.0 for m in maps]; p_raw=[float(m.get(role,0.0)) for m in maps]
    p_zmaps=[robust_event_z(m,roles) if m else {r:0.0 for r in roles} for m in maps]; p_rmaps=[event_rank_percentile(m,roles) if m else {r:0.0 for r in roles} for m in maps]
    p_z=[m[role] for m in p_zmaps]; p_rank=[m[role] for m in p_rmaps]
    p_top3=[]
    for m in maps:
        if not m: p_top3.append(0.0)
        else: p_top3.append(1.0 if sorted(roles,key=lambda r:float(m.get(r,0.0)),reverse=True).index(role)<3 else 0.0)
    prs=_safe_stats(p_raw); pzs=_safe_stats(p_z); prr=_safe_stats(p_rank); ptrs=_safe_stats(pick(p_raw,t_idx)); ptzs=_safe_stats(pick(p_z,t_idx)); ptrr=_safe_stats(pick(p_rank,t_idx)); pxrs=_safe_stats(pick(p_raw,x_idx)); pxzs=_safe_stats(pick(p_z,x_idx)); pxrr=_safe_stats(pick(p_rank,x_idx))
    both_top3=[a*b for a,b in zip(top3,p_top3)]
    vals.extend([float(np.mean(avail)),prs['mean'],prs['median'],prs['max'],prs['min'],prs['std'],prs['last'],prs['trend'],prs['pos_frac'],pzs['mean'],pzs['last'],prr['mean'],prr['last'],float(np.mean(p_top3)),ptrs['mean'],ptzs['mean'],ptrr['mean'],pxrs['mean'],pxzs['mean'],pxrr['mean'],float(p_z[-1]-zs[-1]),float(pzs['mean']-zz['mean']),float(p_rank[-1]-ranks[-1]),float(np.mean(both_top3))])
    if len(vals)!=len(BASE_FEATURE_NAMES): raise RuntimeError(f'feature length mismatch {len(vals)} != {len(BASE_FEATURE_NAMES)}')
    return np.asarray(vals,dtype=np.float32)


def all_role_feature_matrix(history: Sequence[EvidenceObservation], roles: Sequence[str]) -> np.ndarray:
    return np.vstack([role_history_features(history,r,roles) for r in roles]).astype(np.float32)


def binary_entropy(p: float) -> float:
    p=float(np.clip(p,EPS,1.0-EPS)); return float(-(p*math.log(p)+(1.0-p)*math.log(1.0-p)))


def total_entropy(belief: Mapping[str,float], roles: Sequence[str]) -> float:
    return float(sum(binary_entropy(float(belief.get(r,0.5))) for r in roles))
