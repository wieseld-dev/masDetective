#!/usr/bin/env python3

import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

from adaptive_v11_active.io import sha256_json


OLD = Path(
    "artifacts/role_direct_matched_cf_v1"
)

ROOT = Path(
    "artifacts/role_direct_sealed_v1"
)

ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
]

SUBSTITUTE = {
    "code": "qa_testing",
    "writing": "documentation",
    "math": "data",
    "planning": "operations",
    "research": "fact_checking",
    "data": "financial_analysis",
    "design": "accessibility_review",
    "operations": "incident_management",
    "qa_testing": "code",
    "translation": "writing",
    "financial_analysis": "risk_assessment",
    "database_design": "requirements_analysis",
    "accessibility_review": "design",
    "fact_checking": "research",
    "requirements_analysis": "planning",
    "incident_management": "operations",
    "experiment_design": "research",
    "risk_assessment": "financial_analysis",
    "documentation": "writing",
    "customer_support_triage": "incident_management",
}


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


old_manifest = json.loads(
    (OLD / "manifest_privileged.json").read_text()
)

old_candidates = read_jsonl(
    OLD / "candidates_dev_round_01.jsonl"
)

fresh = read_jsonl(
    ROOT / "fresh_panels.jsonl"
)

assert len(fresh) == 120


# Preserve a proven-valid candidate schema.
candidate_template = copy.deepcopy(
    old_candidates[0]
)


# Valid mode/style values are borrowed from the already executed manifest.
templates = []

seen_ms = set()

for s in old_manifest["systems"]:
    key = (
        str(s["mode"]),
        str(s["style"]),
    )

    if key not in seen_ms:
        seen_ms.add(key)
        templates.append(key)

assert templates


# Do not reuse any exact old role composition.
old_compositions = {
    tuple(sorted(map(str, s["roles"])))
    for s in old_manifest["systems"]
}


panel_map = {}

for x in fresh:
    panel_map[
        (
            x["target_role"],
            x["panel"],
        )
    ] = panel_map.get(
        (
            x["target_role"],
            x["panel"],
        ),
        [],
    ) + [x]


for key in panel_map:
    panel_map[key] = sorted(
        panel_map[key],
        key=lambda x:
            x["probe_rank"],
    )

    assert len(panel_map[key]) == 3


systems = []
pairs = []
candidates = []

used_compositions = set(
    old_compositions
)


for ridx, role in enumerate(ROLES):

    substitute = SUBSTITUTE[role]

    allowed_background = [
        x
        for x in ROLES
        if x not in {
            role,
            substitute,
        }
    ]

    bg_pairs = list(
        itertools.combinations(
            allowed_background,
            2,
        )
    )

    rng = random.Random(
        20261009
        + ridx * 1009
    )

    rng.shuffle(bg_pairs)

    chosen = []

    for a, b in bg_pairs:

        pos_roles = tuple(
            sorted(
                [
                    a,
                    b,
                    role,
                ]
            )
        )

        neg_roles = tuple(
            sorted(
                [
                    a,
                    b,
                    substitute,
                ]
            )
        )

        if pos_roles in used_compositions:
            continue

        if neg_roles in used_compositions:
            continue

        chosen.append(
            (
                a,
                b,
                pos_roles,
                neg_roles,
            )
        )

        used_compositions.add(
            pos_roles
        )

        used_compositions.add(
            neg_roles
        )

        if len(chosen) == 10:
            break

    assert len(chosen) == 10, role

    for j, (
        bg1,
        bg2,
        pos_roles,
        neg_roles,
    ) in enumerate(
        chosen,
        1,
    ):

        panel = (
            "calibration"
            if j <= 5
            else "sealed_test"
        )

        panel_pair_index = (
            j
            if panel == "calibration"
            else j - 5
        )

        pair_id = (
            f"sealed_{role}_"
            f"{panel}_"
            f"{panel_pair_index:02d}"
        )

        mode, style = templates[
            (
                ridx * 10
                + j - 1
            )
            % len(templates)
        ]

        digest = hashlib.sha256(
            pair_id.encode()
        ).hexdigest()[:12]

        pos_sid = (
            f"rdseal_{digest}_POS"
        )

        neg_sid = (
            f"rdseal_{digest}_NEG"
        )

        common = {
            "split": "dev",
            "mode": mode,
            "style": style,
            "target_role": role,
            "substitute_role": substitute,
            "background_roles": [
                bg1,
                bg2,
            ],
            "panel": panel,
            "pair_id": pair_id,
            "pair_index": panel_pair_index,
        }

        systems.append({
            **common,
            "session_id": pos_sid,
            "system_id": pos_sid,
            "condition": "TARGET_PRESENT",
            "roles": list(pos_roles),
        })

        systems.append({
            **common,
            "session_id": neg_sid,
            "system_id": neg_sid,
            "condition": "TARGET_ABSENT",
            "roles": list(neg_roles),
        })

        pairs.append({
            **common,
            "positive_session_id": pos_sid,
            "negative_session_id": neg_sid,
            "positive_roles": list(pos_roles),
            "negative_roles": list(neg_roles),
        })

        probes = panel_map[
            (
                role,
                panel,
            )
        ]

        for sid, condition in [
            (
                pos_sid,
                "TARGET_PRESENT",
            ),
            (
                neg_sid,
                "TARGET_ABSENT",
            ),
        ]:

            for qidx, p in enumerate(
                probes
            ):

                row = copy.deepcopy(
                    candidate_template
                )

                row.update({
                    "round": 1,
                    "split": "dev",
                    "session_id": sid,
                    "system_id": sid,
                    "candidate_index": qidx,
                    "query": p["query"],
                    "valid": True,

                    "target_role": role,
                    "substitute_role": substitute,
                    "condition": condition,
                    "panel": panel,
                    "pair_id": pair_id,
                    "pair_index": panel_pair_index,
                    "direct_probe_rank": p["probe_rank"],
                    "probe_strategy": p["probe_strategy"],
                })

                candidates.append(
                    row
                )


assert len(pairs) == 200
assert len(systems) == 400
assert len(candidates) == 1200

assert sum(
    x["panel"] == "calibration"
    for x in pairs
) == 100

assert sum(
    x["panel"] == "sealed_test"
    for x in pairs
) == 100


# CAL and TEST role compositions must be disjoint.
cal_comps = {
    tuple(sorted(x["roles"]))
    for x in systems
    if x["panel"] == "calibration"
}

test_comps = {
    tuple(sorted(x["roles"]))
    for x in systems
    if x["panel"] == "sealed_test"
}

assert not (
    cal_comps
    & test_comps
)


manifest = copy.deepcopy(
    old_manifest
)

manifest["systems"] = systems
manifest["sha256"] = sha256_json(
    systems
)

manifest[
    "experiment"
] = "role_direct_sealed_v1"


(ROOT / "manifest_privileged.json").write_text(
    json.dumps(
        manifest,
        indent=2,
    )
)

write_jsonl(
    ROOT / "matched_pairs.jsonl",
    pairs,
)

write_jsonl(
    ROOT / "candidates_dev_round_01.jsonl",
    candidates,
)


print("SEALED_BUILD_PASS")
print("systems        =", len(systems))
print("matched pairs  =", len(pairs))
print("candidates     =", len(candidates))
print(
    "cal pairs      =",
    sum(
        x["panel"] == "calibration"
        for x in pairs
    ),
)
print(
    "sealed pairs   =",
    sum(
        x["panel"] == "sealed_test"
        for x in pairs
    ),
)
print(
    "old composition overlap =",
    len(
        {
            tuple(sorted(x["roles"]))
            for x in systems
        }
        & old_compositions
    ),
)
print(
    "CAL/TEST composition overlap =",
    len(
        cal_comps
        & test_comps
    ),
)
