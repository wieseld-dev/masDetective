from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from targets.hierarchical_mas import ROLE_PROMPTS
from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
from v8.embedding import FrozenOutputEmbedderV8


BANK_PATH = Path(
    "artifacts/general_role_bank_v1/role_bank_v1.json"
)

OUT = Path(
    "artifacts/general_role_pilot_v1"
)

SEED = 20260929
PAIRS_PER_ROLE = 2

EXPECTED = {
    "roles_total": 72,
    "train_roles": 60,
    "val_roles": 12,
    "pairs_per_role": 2,
    "probes_per_role": 3,
    "systems_total": 288,
    "dataset_rows": 864,
    "train_rows": 720,
    "val_rows": 144,
    "positive_rows": 432,
    "negative_rows": 432,
    "estimated_mas_calls": 3456,
}


def stable_int(text: str) -> int:
    return int(
        hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


def dump(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / (na * nb)
    )


def main():
    if not BANK_PATH.exists():
        raise SystemExit(
            f"missing role bank: {BANK_PATH}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    bank = json.loads(
        BANK_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert len(bank) == 72

    by_role = {
        str(x["role"]): x
        for x in bank
    }

    assert len(by_role) == 72

    # --------------------------------------------------------
    # REGISTER HIDDEN WORKER PROMPTS
    #
    # Same mechanism used by V11:
    # SingleRouterFanoutMAS imported the same ROLE_PROMPTS
    # dictionary object.
    # --------------------------------------------------------

    new_prompts = {
        str(x["role"]):
            str(x["hidden_worker_prompt"])
        for x in bank
    }

    ROLE_PROMPTS.update(
        new_prompts
    )

    missing_prompts = sorted(
        set(by_role)
        - set(ROLE_PROMPTS)
    )

    assert not missing_prompts, missing_prompts

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    train_roles = sorted(
        x["role"]
        for x in bank
        if x["split"] == "train"
    )

    val_roles = sorted(
        x["role"]
        for x in bank
        if x["split"] == "val"
    )

    assert len(train_roles) == 60
    assert len(val_roles) == 12
    assert not (
        set(train_roles)
        & set(val_roles)
    )

    # --------------------------------------------------------
    # EMBEDDING-BASED IN-SPLIT NEIGHBORS
    #
    # IMPORTANT:
    # Uses PUBLIC role descriptions only.
    # No hidden prompt / routing / labels.
    # --------------------------------------------------------

    print(
        "Loading frozen public-description embedder...",
        flush=True,
    )

    embedder = FrozenOutputEmbedderV8(
        device="cpu"
    )

    descs = {
        r: str(
            by_role[r]["public_role_description"]
        )
        for r in by_role
    }

    vecs = {
        r: np.asarray(
            embedder.encode(descs[r]),
            dtype=np.float32,
        )
        for r in by_role
    }

    neighbor_map = {}

    for split_roles in (
        train_roles,
        val_roles,
    ):
        for role in split_roles:
            vals = []

            for other in split_roles:
                if other == role:
                    continue

                vals.append(
                    (
                        cosine(
                            vecs[role],
                            vecs[other],
                        ),
                        other,
                    )
                )

            vals.sort(
                key=lambda z: (
                    -z[0],
                    z[1],
                )
            )

            neighbor_map[role] = [
                {
                    "role": other,
                    "similarity": sim,
                }
                for sim, other in vals[:5]
            ]

    # --------------------------------------------------------
    # NESTED LEARNING-CURVE ROLE SUBSETS
    #
    # Order is deterministic and approximately family-stratified.
    # --------------------------------------------------------

    fam_train = defaultdict(list)

    for x in bank:
        if x["split"] == "train":
            fam_train[
                str(x["family"])
            ].append(
                str(x["role"])
            )

    families = sorted(
        fam_train
    )

    for fam in families:
        fam_train[fam].sort(
            key=lambda r: stable_int(
                f"{SEED}|{fam}|{r}"
            )
        )

    interleaved = []

    max_len = max(
        len(v)
        for v in fam_train.values()
    )

    for depth in range(max_len):
        fam_order = sorted(
            families,
            key=lambda f: stable_int(
                f"{SEED}|depth={depth}|{f}"
            ),
        )

        for fam in fam_order:
            if depth < len(
                fam_train[fam]
            ):
                interleaved.append(
                    fam_train[fam][depth]
                )

    assert len(interleaved) == 60
    assert len(set(interleaved)) == 60

    original20 = {
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
    }

    assert original20 <= set(
        train_roles
    )

    learning_curve = {
        "lc20": interleaved[:20],
        "lc40": interleaved[:40],
        "lc60": interleaved[:60],
        "legacy20_reference":
            sorted(original20),
        "val12": val_roles,
    }

    assert set(
        learning_curve["lc20"]
    ) <= set(
        learning_curve["lc40"]
    )

    assert set(
        learning_curve["lc40"]
    ) <= set(
        learning_curve["lc60"]
    )

    # --------------------------------------------------------
    # BUILD MATCHED COUNTERFACTUAL SYSTEM PAIRS
    # --------------------------------------------------------

    pairs = []

    role_context_count = {
        "train": Counter(),
        "val": Counter(),
    }

    seen_compositions = {
        "train": set(),
        "val": set(),
    }

    for split, split_roles in [
        ("train", train_roles),
        ("val", val_roles),
    ]:
        pool = list(
            split_roles
        )

        for target in split_roles:
            # Two nearest in-split public semantic neighbors.
            replacements = [
                x["role"]
                for x in neighbor_map[target]
            ]

            assert len(
                replacements
            ) >= PAIRS_PER_ROLE

            for pair_index in range(
                PAIRS_PER_ROLE
            ):
                replacement = replacements[
                    pair_index
                ]

                # Alternate K=3 / K=4.
                k = (
                    3
                    if pair_index == 0
                    else 4
                )

                n_context = k - 1

                # Deterministic random tie-breaking while preferring
                # roles used less often as context.
                rng = random.Random(
                    stable_int(
                        f"{SEED}|{split}|"
                        f"{target}|{pair_index}"
                    )
                )

                candidates = [
                    r for r in pool
                    if r not in {
                        target,
                        replacement,
                    }
                ]

                rng.shuffle(
                    candidates
                )

                candidates.sort(
                    key=lambda r: (
                        role_context_count[
                            split
                        ][r],
                        stable_int(
                            f"{SEED}|context|"
                            f"{target}|{pair_index}|{r}"
                        ),
                    )
                )

                contexts = candidates[
                    :n_context
                ]

                assert len(
                    contexts
                ) == n_context

                for r in contexts:
                    role_context_count[
                        split
                    ][r] += 1

                # Keep target/replacement in identical worker position.
                pos_rng = random.Random(
                    stable_int(
                        f"{SEED}|position|"
                        f"{split}|{target}|{pair_index}"
                    )
                )

                insert_pos = pos_rng.randrange(
                    k
                )

                plus_roles = list(
                    contexts
                )

                minus_roles = list(
                    contexts
                )

                plus_roles.insert(
                    insert_pos,
                    target,
                )

                minus_roles.insert(
                    insert_pos,
                    replacement,
                )

                assert len(
                    plus_roles
                ) == k

                assert len(
                    minus_roles
                ) == k

                assert len(
                    set(plus_roles)
                ) == k

                assert len(
                    set(minus_roles)
                ) == k

                assert target in plus_roles
                assert target not in minus_roles

                assert replacement not in plus_roles
                assert replacement in minus_roles

                plus_key = tuple(
                    sorted(plus_roles)
                )

                minus_key = tuple(
                    sorted(minus_roles)
                )

                # Avoid accidental duplicate hidden systems.
                if plus_key in seen_compositions[
                    split
                ]:
                    raise RuntimeError(
                        f"duplicate PLUS composition: "
                        f"{split} {target} {pair_index} "
                        f"{plus_key}"
                    )

                if minus_key in seen_compositions[
                    split
                ]:
                    raise RuntimeError(
                        f"duplicate MINUS composition: "
                        f"{split} {target} {pair_index} "
                        f"{minus_key}"
                    )

                seen_compositions[
                    split
                ].add(
                    plus_key
                )

                seen_compositions[
                    split
                ].add(
                    minus_key
                )

                style = (
                    "s1"
                    if pair_index == 0
                    else "s2"
                )

                pair_id = (
                    f"GRPV1::{split}::"
                    f"{target}::P{pair_index}"
                )

                plus_id = (
                    f"grpv1_{split}_"
                    f"{target}_p{pair_index}_plus"
                )

                minus_id = (
                    f"grpv1_{split}_"
                    f"{target}_p{pair_index}_minus"
                )

                sim = next(
                    x["similarity"]
                    for x in neighbor_map[
                        target
                    ]
                    if x["role"]
                    == replacement
                )

                probes = []

                for p in by_role[
                    target
                ]["canonical_probes"]:
                    probes.append({
                        "probe_family":
                            str(
                                p["probe_family"]
                            ),
                        "query":
                            str(
                                p["query"]
                            ),
                    })

                assert len(
                    probes
                ) == 3

                pair = {
                    "pair_id": pair_id,
                    "split": split,
                    "target_role": target,
                    "public_role_description":
                        descs[target],
                    "pair_index": pair_index,
                    "k": k,
                    "execution_mode":
                        "fanout2",
                    "style": style,
                    "replacement_role":
                        replacement,
                    "replacement_public_similarity":
                        sim,
                    "insert_position":
                        insert_pos,
                    "context_roles":
                        contexts,
                    "plus_spec": {
                        "system_id":
                            plus_id,
                        "split":
                            split,
                        "execution_mode":
                            "fanout2",
                        "style":
                            style,
                        "worker_roles":
                            plus_roles,
                    },
                    "minus_spec": {
                        "system_id":
                            minus_id,
                        "split":
                            split,
                        "execution_mode":
                            "fanout2",
                        "style":
                            style,
                        "worker_roles":
                            minus_roles,
                    },
                    "probes":
                        probes,
                }

                pairs.append(
                    pair
                )

    # --------------------------------------------------------
    # AUDIT COUNTS
    # --------------------------------------------------------

    assert len(pairs) == (
        72 * PAIRS_PER_ROLE
    )

    train_pairs = [
        x for x in pairs
        if x["split"] == "train"
    ]

    val_pairs = [
        x for x in pairs
        if x["split"] == "val"
    ]

    assert len(train_pairs) == 120
    assert len(val_pairs) == 24

    system_ids = []

    rows_expected = []

    for pair in pairs:
        for branch, label in [
            ("plus", 1),
            ("minus", 0),
        ]:
            spec = pair[
                f"{branch}_spec"
            ]

            system_ids.append(
                spec["system_id"]
            )

            for probe in pair["probes"]:
                rows_expected.append({
                    "split":
                        pair["split"],
                    "target_role":
                        pair["target_role"],
                    "pair_id":
                        pair["pair_id"],
                    "branch":
                        branch,
                    "label":
                        label,
                    "probe_family":
                        probe["probe_family"],
                })

    assert len(system_ids) == 288
    assert len(set(system_ids)) == 288
    assert len(rows_expected) == 864

    count_split = Counter(
        x["split"]
        for x in rows_expected
    )

    count_label = Counter(
        x["label"]
        for x in rows_expected
    )

    assert count_split == {
        "train": 720,
        "val": 144,
    }

    assert count_label == {
        1: 432,
        0: 432,
    }

    # Exactly 12 rows per role:
    # 2 pairs × 2 branches × 3 probes.
    role_rows = Counter(
        x["target_role"]
        for x in rows_expected
    )

    assert set(
        role_rows.values()
    ) == {12}

    # --------------------------------------------------------
    # TARGET CONSTRUCTION PREFLIGHT
    #
    # client=object() prevents OpenAI client construction.
    # No query() call is made.
    # --------------------------------------------------------

    construction_ok = 0

    for pair in pairs:
        for branch in (
            "plus",
            "minus",
        ):
            s = pair[
                f"{branch}_spec"
            ]

            spec = SingleRouterSystemSpec(
                system_id=
                    s["system_id"],
                split=
                    s["split"],
                execution_mode=
                    s["execution_mode"],
                style=
                    s["style"],
                worker_roles=
                    tuple(
                        s["worker_roles"]
                    ),
            )

            SingleRouterFanoutMAS(
                spec,
                client=object(),
            )

            construction_ok += 1

    assert construction_ok == 288

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    private_manifest = {
        "version":
            "general_role_pilot_v1",
        "seed":
            SEED,
        "pairs_per_role":
            PAIRS_PER_ROLE,
        "pairs":
            pairs,
    }

    private_path = (
        OUT
        / "manifest_private.json"
    )

    dump(
        private_path,
        private_manifest,
    )

    dump(
        OUT
        / "neighbor_map_public.json",
        neighbor_map,
    )

    dump(
        OUT
        / "learning_curve_subsets.json",
        learning_curve,
    )

    manifest_sha = hashlib.sha256(
        private_path.read_bytes()
    ).hexdigest()

    summary = {
        "version":
            "general_role_pilot_v1",
        "role_bank":
            str(BANK_PATH),
        "old_internal_test_used":
            False,
        "openai_api_calls":
            0,
        "target_queries":
            0,
        "roles_total":
            72,
        "train_roles":
            60,
        "val_roles":
            12,
        "pairs_per_role":
            PAIRS_PER_ROLE,
        "train_pairs":
            120,
        "val_pairs":
            24,
        "systems_total":
            288,
        "probes_per_role":
            3,
        "dataset_rows_planned":
            864,
        "train_rows_planned":
            720,
        "val_rows_planned":
            144,
        "positive_rows_planned":
            432,
        "negative_rows_planned":
            432,
        "execution_mode":
            "fanout2",
        "k_values":
            [3, 4],
        "router_styles":
            ["s1", "s2"],
        "estimated_model_calls":
            3456,
        "target_construction_preflight":
            construction_ok,
        "private_manifest_sha256":
            manifest_sha,
        "learning_curve_sizes": {
            "lc20":
                len(
                    learning_curve[
                        "lc20"
                    ]
                ),
            "lc40":
                len(
                    learning_curve[
                        "lc40"
                    ]
                ),
            "lc60":
                len(
                    learning_curve[
                        "lc60"
                    ]
                ),
            "val12":
                len(
                    learning_curve[
                        "val12"
                    ]
                ),
        },
    }

    assert summary[
        "dataset_rows_planned"
    ] == EXPECTED[
        "dataset_rows"
    ]

    dump(
        OUT / "build_summary.json",
        summary,
    )

    print()
    print("=" * 92)
    print("GENERAL ROLE PILOT V1 — BUILD AUDIT")
    print("=" * 92)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("TOP SEMANTIC REPLACEMENTS:")

    for role in (
        train_roles[:5]
        + val_roles
    ):
        top = neighbor_map[
            role
        ][0]

        print(
            f"{role:<30} -> "
            f"{top['role']:<30} "
            f"sim={top['similarity']:.4f}"
        )

    print()
    print("NO MAS QUERIES EXECUTED.")
    print("NO OPENAI API CALLS.")
    print("OLD INTERNAL TEST UNTOUCHED.")


if __name__ == "__main__":
    main()
