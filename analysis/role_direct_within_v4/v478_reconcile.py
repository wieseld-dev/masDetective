#!/usr/bin/env python3
"""Pure, pre-GPU resume reconciliation for MAS5 V4.7.8.

All human-approved queries remain unchanged. Near-identical generated queries
are quarantined by role-slot and regenerated rather than stopping execution.
"""
from __future__ import annotations
from collections import defaultdict

def reconcile_overlap(merged, approved, jaccard, threshold=.62):
    approved_keys=set(approved)
    if not approved_keys.issubset(merged):
        raise RuntimeError('Resume lacks human-approved keys')
    if any(merged[k]!=approved[k] for k in approved_keys):
        raise RuntimeError('Resume changed human-approved query')
    kept={k:approved[k] for k in approved}
    approved_warnings=[]
    # REVIEW ONLY: the accepted training inputs are immutable.
    by_role=defaultdict(list)
    for k in sorted(approved_keys):
        by_role[k[0]].append(k)
    for role,keys in by_role.items():
        for i,k in enumerate(keys):
            for prior in keys[:i]:
                similarity=jaccard(approved[k]['query'],approved[prior]['query'])
                if similarity>threshold:
                    approved_warnings.append({'slot':list(k),'other_slot':list(prior),'jaccard':round(similarity,4)})
    dropped=[]
    for k in sorted(set(merged)-approved_keys):
        r=merged[k]
        conflicts=[]
        for other,other_row in kept.items():
            if other[0]!=k[0]:continue
            sim=jaccard(r['query'],other_row['query'])
            if sim>threshold:conflicts.append({'slot':list(other),'jaccard':round(sim,4),'approved':other in approved_keys})
        if conflicts:
            dropped.append({'slot':list(k),'query':r['query'],'conflicts':conflicts,'decision':'REGENERATE_UNAPPROVED_ONLY'})
        else:
            kept[k]=r
    audit={'summary':{'approved_unchanged':len(approved),'input_unique':len(merged),
          'resume_after_reconcile':len(kept),'dropped_new':len(dropped),
          'remaining':160-len(kept),'approved_pair_warnings':len(approved_warnings)},
          'dropped_unapproved':dropped,'approved_overlap_warnings_for_human_review':approved_warnings}
    return kept,audit
