#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.io import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from v11.role_universe import TRAIN_ROLES


def safe_auc(
    y,
    s,
):
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    s = np.asarray(
        s,
        dtype=np.float64,
    )

    if len(
        np.unique(
            y
        )
    ) < 2:
        return None

    return float(
        roc_auc_score(
            y,
            s,
        )
    )


def mean_or_none(
    xs,
):
    xs = [
        float(x)
        for x in xs
    ]

    if not xs:
        return None

    return float(
        np.mean(
            xs
        )
    )


def pair_aggregate(
    probe_rows,
    *,
    max_probe=None,
):

    grouped = defaultdict(
        list
    )

    for r in probe_rows:

        if (
            max_probe is not None
            and
            int(
                r[
                    "probe_index"
                ]
            )
            > int(
                max_probe
            )
        ):
            continue

        key = (
            str(
                r[
                    "system_id"
                ]
            ),
            str(
                r[
                    "target_role"
                ]
            ),
        )

        grouped[
            key
        ].append(
            r
        )

    out = []

    for (
        sid,
        role,
    ), rows in sorted(
        grouped.items()
    ):

        rows = sorted(
            rows,
            key=lambda x:
                int(
                    x[
                        "probe_index"
                    ]
                ),
        )

        present_values = {
            bool(
                x[
                    "present"
                ]
            )
            for x in rows
        }

        if len(
            present_values
        ) != 1:
            raise RuntimeError(
                f"Truth mismatch inside pair {(sid, role)}"
            )

        out.append(
            {
                "system_id":
                    sid,

                "target_role":
                    role,

                "present":
                    bool(
                        rows[0][
                            "present"
                        ]
                    ),

                "n_probes":
                    len(
                        rows
                    ),

                "target_activation_rate":
                    float(
                        np.mean(
                            [
                                float(
                                    x[
                                        "target_activated"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),

                "exclusive_target_rate":
                    float(
                        np.mean(
                            [
                                float(
                                    x[
                                        "exclusive_target"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),

                "mean_v135_target":
                    float(
                        np.mean(
                            [
                                float(
                                    x[
                                        "v135_target"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),

                "median_v135_target":
                    float(
                        np.median(
                            [
                                float(
                                    x[
                                        "v135_target"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),

                "mean_v134_target":
                    float(
                        np.mean(
                            [
                                float(
                                    x[
                                        "v134_target"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),

                "median_v134_target":
                    float(
                        np.median(
                            [
                                float(
                                    x[
                                        "v134_target"
                                    ]
                                )
                                for x in rows
                            ]
                        )
                    ),
            }
        )

    return out


def per_role_metrics(
    pairs,
):

    out = {}

    for role in TRAIN_ROLES:

        rows = [
            x
            for x in pairs
            if str(
                x[
                    "target_role"
                ]
            )
            == str(
                role
            )
        ]

        y = [
            int(
                x[
                    "present"
                ]
            )
            for x in rows
        ]

        v135 = [
            float(
                x[
                    "mean_v135_target"
                ]
            )
            for x in rows
        ]

        v134 = [
            float(
                x[
                    "mean_v134_target"
                ]
            )
            for x in rows
        ]

        positives = [
            x
            for x in rows
            if x[
                "present"
            ]
        ]

        out[
            str(
                role
            )
        ] = {
            "systems":
                len(
                    rows
                ),

            "positive_systems":
                int(
                    sum(
                        y
                    )
                ),

            "negative_systems":
                int(
                    len(
                        y
                    )
                    - sum(
                        y
                    )
                ),

            "auc_v135":
                safe_auc(
                    y,
                    v135,
                ),

            "auc_v134":
                safe_auc(
                    y,
                    v134,
                ),

            "mean_target_routing_when_present":
                mean_or_none(
                    [
                        x[
                            "target_activation_rate"
                        ]
                        for x in positives
                    ]
                ),

            "mean_exclusive_routing_when_present":
                mean_or_none(
                    [
                        x[
                            "exclusive_target_rate"
                        ]
                        for x in positives
                    ]
                ),
        }

    return out


def macro_auc(
    per_role,
    key,
):

    vals = [
        float(
            x[
                key
            ]
        )
        for x in per_role.values()
        if x[
            key
        ]
        is not None
    ]

    if not vals:
        return None

    return float(
        np.mean(
            vals
        )
    )


def system_auc(
    pairs,
    score_key,
):

    grouped = defaultdict(
        list
    )

    for r in pairs:
        grouped[
            str(
                r[
                    "system_id"
                ]
            )
        ].append(
            r
        )

    out = {}

    for sid, rows in sorted(
        grouped.items()
    ):

        y = [
            int(
                x[
                    "present"
                ]
            )
            for x in rows
        ]

        s = [
            float(
                x[
                    score_key
                ]
            )
            for x in rows
        ]

        out[
            sid
        ] = safe_auc(
            y,
            s,
        )

    values = [
        float(x)
        for x in out.values()
        if x is not None
    ]

    return {
        "per_system":
            out,

        "mean":
            (
                float(
                    np.mean(
                        values
                    )
                )
                if values
                else None
            ),
    }


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/role_direct_approach_v1",
    )

    ap.add_argument(
        "--split",
        default="dev",
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    candidates = read_jsonl(
        root
        / f"candidates_{args.split}_round_01.jsonl"
    )

    responses = read_jsonl(
        root
        / f"responses_{args.split}_round_01.jsonl"
    )

    scored = read_jsonl(
        root
        / f"scored_{args.split}_round_01.jsonl"
    )

    manifest = json.loads(
        (
            root
            / "manifest_privileged.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    truth = {
        str(
            x[
                "session_id"
            ]
        ):
            set(
                map(
                    str,
                    x[
                        "roles"
                    ],
                )
            )
        for x in manifest[
            "systems"
        ]
        if str(
            x[
                "split"
            ]
        )
        == args.split
    }

    response_by = {}

    for r in responses:

        key = (
            str(
                r[
                    "session_id"
                ]
            ),
            str(
                r[
                    "query"
                ]
            ),
        )

        if key in response_by:
            raise RuntimeError(
                f"Duplicate response key: {key}"
            )

        response_by[
            key
        ] = r

    score_by = {}

    for r in scored:

        key = (
            str(
                r.get(
                    "system_id",
                    r.get(
                        "session_id",
                        "",
                    ),
                )
            ),
            str(
                r[
                    "query"
                ]
            ),
        )

        if key in score_by:
            raise RuntimeError(
                f"Duplicate score key: {key}"
            )

        score_by[
            key
        ] = r

    probe_rows = []

    for c in candidates:

        sid = str(
            c[
                "session_id"
            ]
        )

        q = str(
            c[
                "query"
            ]
        )

        role = str(
            c[
                "target_role"
            ]
        )

        key = (
            sid,
            q,
        )

        if key not in response_by:
            raise RuntimeError(
                f"Missing response: {key}"
            )

        if key not in score_by:
            raise RuntimeError(
                f"Missing score: {key}"
            )

        r = response_by[
            key
        ]

        sc = score_by[
            key
        ]

        if (
            r.get(
                "execution_status"
            )
            != "ok"
        ):
            raise RuntimeError(
                f"Execution failure: {key}"
            )

        selected = list(
            map(
                str,
                r.get(
                    "selected_roles_training_only",
                    [],
                ),
            )
        )

        selected_set = set(
            selected
        )

        evidence135 = dict(
            sc.get(
                "evidence",
                {},
            )
        )

        evidence134 = dict(
            sc.get(
                "metadata",
                {},
            ).get(
                "evidence_v134",
                {},
            )
        )

        if role not in evidence135:
            raise RuntimeError(
                f"V13.5 missing role={role}: {key}"
            )

        if role not in evidence134:
            raise RuntimeError(
                f"V13.4 missing role={role}: {key}"
            )

        true_roles = truth[
            sid
        ]

        probe_rows.append(
            {
                "system_id":
                    sid,

                "target_role":
                    role,

                "probe_index":
                    int(
                        c[
                            "probe_index"
                        ]
                    ),

                "query":
                    q,

                "present":
                    bool(
                        role
                        in true_roles
                    ),

                "selected_roles_eval_only":
                    selected,

                "target_activated":
                    bool(
                        role
                        in selected_set
                    ),

                "exclusive_target":
                    bool(
                        selected_set
                        == {
                            role
                        }
                    ),

                "v135_target":
                    float(
                        evidence135[
                            role
                        ]
                    ),

                "v134_target":
                    float(
                        evidence134[
                            role
                        ]
                    ),
            }
        )

    write_jsonl(
        root
        / "role_direct_probe_level.jsonl",
        probe_rows,
    )

    full_pairs = pair_aggregate(
        probe_rows
    )

    write_jsonl(
        root
        / "role_direct_pair_level.jsonl",
        full_pairs,
    )

    per_role = per_role_metrics(
        full_pairs
    )

    max_probe = max(
        int(
            x[
                "probe_index"
            ]
        )
        for x in probe_rows
    )

    curve = []

    for k in range(
        1,
        max_probe + 1,
    ):

        pairs_k = pair_aggregate(
            probe_rows,
            max_probe=k,
        )

        role_k = per_role_metrics(
            pairs_k
        )

        positive_probe_rows = [
            x
            for x in probe_rows
            if (
                x[
                    "present"
                ]
                and
                int(
                    x[
                        "probe_index"
                    ]
                )
                <= k
            )
        ]

        curve.append(
            {
                "probes_per_role":
                    k,

                "macro_role_auc_v135":
                    macro_auc(
                        role_k,
                        "auc_v135",
                    ),

                "macro_role_auc_v134":
                    macro_auc(
                        role_k,
                        "auc_v134",
                    ),

                "positive_target_activation_rate":
                    mean_or_none(
                        [
                            x[
                                "target_activated"
                            ]
                            for x
                            in positive_probe_rows
                        ]
                    ),

                "positive_exclusive_target_rate":
                    mean_or_none(
                        [
                            x[
                                "exclusive_target"
                            ]
                            for x
                            in positive_probe_rows
                        ]
                    ),
            }
        )

    positive_rows = [
        x
        for x in probe_rows
        if x[
            "present"
        ]
    ]

    activated_positive = [
        x[
            "v135_target"
        ]
        for x in positive_rows
        if x[
            "target_activated"
        ]
    ]

    missed_positive = [
        x[
            "v135_target"
        ]
        for x in positive_rows
        if not x[
            "target_activated"
        ]
    ]

    confusion = {}

    for role in TRAIN_ROLES:

        ctr = Counter()

        for x in probe_rows:

            if (
                str(
                    x[
                        "target_role"
                    ]
                )
                != str(
                    role
                )
            ):
                continue

            if x[
                "present"
            ]:
                continue

            ctr.update(
                x[
                    "selected_roles_eval_only"
                ]
            )

        confusion[
            str(
                role
            )
        ] = [
            {
                "selected_role":
                    r,

                "count":
                    int(
                        n
                    ),
            }
            for r, n in ctr.most_common(
                5
            )
        ]

    role_aucs_135 = [
        float(
            x[
                "auc_v135"
            ]
        )
        for x in per_role.values()
        if x[
            "auc_v135"
        ]
        is not None
    ]

    role_aucs_134 = [
        float(
            x[
                "auc_v134"
            ]
        )
        for x in per_role.values()
        if x[
            "auc_v134"
        ]
        is not None
    ]

    covered_roles = [
        r
        for r, x in per_role.items()
        if int(
            x[
                "positive_systems"
            ]
        )
        > 0
    ]

    final = {
        "experiment":
            "ROLE DIRECT APPROACH V1",

        "systems":
            len(
                truth
            ),

        "roles":
            len(
                TRAIN_ROLES
            ),

        "roles_with_positive_example":
            len(
                covered_roles
            ),

        "covered_roles":
            covered_roles,

        "probe_rows":
            len(
                probe_rows
            ),

        "probes_per_role":
            max_probe,

        "routing_eval_only":
            {
                "target_activation_rate_when_present":
                    mean_or_none(
                        [
                            x[
                                "target_activated"
                            ]
                            for x in positive_rows
                        ]
                    ),

                "exclusive_target_rate_when_present":
                    mean_or_none(
                        [
                            x[
                                "exclusive_target"
                            ]
                            for x in positive_rows
                        ]
                    ),
            },

        "sensor_target_separation":
            {
                "macro_role_auc_v135":
                    mean_or_none(
                        role_aucs_135
                    ),

                "macro_role_auc_v134":
                    mean_or_none(
                        role_aucs_134
                    ),

                "roles_in_macro_auc":
                    len(
                        role_aucs_135
                    ),

                "mean_v135_when_present_and_target_activated":
                    mean_or_none(
                        activated_positive
                    ),

                "mean_v135_when_present_but_target_not_activated":
                    mean_or_none(
                        missed_positive
                    ),
            },

        # Diagnostic only.
        # Raw evidence scales across role heads are not guaranteed calibrated,
        # so macro per-role AUC above is the primary Phase-1 sensor metric.
        "naive_cross_role_system_auc":
            {
                "v135":
                    system_auc(
                        full_pairs,
                        "mean_v135_target",
                    ),

                "v134":
                    system_auc(
                        full_pairs,
                        "mean_v134_target",
                    ),
            },

        "accumulation_curve":
            curve,

        "per_role":
            per_role,

        "absent_target_routing_confusions":
            confusion,

        "interpretation_note":
            (
                "Routing trace and truth are evaluation-only. "
                "Primary Phase-1 question is whether repeated targeted "
                "black-box probes produce separable V13.5/V13.4 evidence. "
                "No belief updater or selector is used."
            ),
    }

    write_json(
        root
        / "role_direct_final_report.json",
        final,
    )

    print(
        "=" * 90
    )

    print(
        "ROLE DIRECT APPROACH V1"
    )

    print(
        "=" * 90
    )

    print(
        "systems =",
        final[
            "systems"
        ]
    )

    print(
        "roles with positive example =",
        final[
            "roles_with_positive_example"
        ],
        "/",
        final[
            "roles"
        ],
    )

    print(
        "probe rows =",
        final[
            "probe_rows"
        ],
    )

    print()

    print(
        "TARGET ROUTING WHEN PRESENT =",
        final[
            "routing_eval_only"
        ][
            "target_activation_rate_when_present"
        ],
    )

    print(
        "EXCLUSIVE TARGET ROUTING =",
        final[
            "routing_eval_only"
        ][
            "exclusive_target_rate_when_present"
        ],
    )

    print()

    print(
        "MACRO ROLE AUC V13.5 =",
        final[
            "sensor_target_separation"
        ][
            "macro_role_auc_v135"
        ],
    )

    print(
        "MACRO ROLE AUC V13.4 =",
        final[
            "sensor_target_separation"
        ][
            "macro_role_auc_v134"
        ],
    )

    print()

    print(
        "ACCUMULATION CURVE"
    )

    for x in curve:

        print(
            "k=",
            x[
                "probes_per_role"
            ],
            "v135=",
            x[
                "macro_role_auc_v135"
            ],
            "v134=",
            x[
                "macro_role_auc_v134"
            ],
            "routing=",
            x[
                "positive_target_activation_rate"
            ],
        )

    print()

    print(
        "REPORT =",
        root
        / "role_direct_final_report.json",
    )


if __name__ == "__main__":
    main()
