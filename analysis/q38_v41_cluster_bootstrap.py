from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT=Path(
    "artifacts/q38_selector_v41_permutation_audit"
)

rows=[
    json.loads(x)
    for x in (
        ROOT/"records.jsonl"
    ).read_text().splitlines()
    if x.strip()
]


rng=np.random.default_rng(
    20261008
)

B=50000


def run(split,tag):

    rr=[
        x for x in rows
        if x["split"]==split
        and x["tag"]==tag
    ]

    by=defaultdict(list)

    for x in rr:
        by[
            str(x["system_id"])
        ].append(
            float(
                x["selected_gain"]
                - x["random_gain"]
            )
        )

    systems=sorted(by)

    observed=np.mean([
        z
        for sid in systems
        for z in by[sid]
    ])

    boot=np.empty(
        B,
        dtype=float,
    )

    for b in range(B):

        sample=rng.choice(
            systems,
            size=len(systems),
            replace=True,
        )

        vals=[]

        for sid in sample:
            vals.extend(
                by[sid]
            )

        boot[b]=np.mean(
            vals
        )

    lo,hi=np.quantile(
        boot,
        [.025,.975],
    )

    p_gt=float(
        np.mean(
            boot>0
        )
    )

    p_two=min(
        1.0,
        2*min(
            float(np.mean(boot<=0)),
            float(np.mean(boot>=0)),
        )
    )

    print(
        f"{split:10} "
        f"{tag:20} "
        f"Nsys={len(systems):3d} "
        f"delta={observed:+.6f} "
        f"CI95=[{lo:+.6f},{hi:+.6f}] "
        f"P(delta>0)={p_gt:.4f} "
        f"p2={p_two:.4f}"
    )


for split in (
    "val",
    "challenge",
):
    for tag in (
        "VAL_E1",
        "CHALLENGE",
        "PERM_A",
        "PERM_B",
        "AGGREGATE_9_PERM",
    ):
        if any(
            x["split"]==split
            and x["tag"]==tag
            for x in rows
        ):
            run(
                split,
                tag,
            )
