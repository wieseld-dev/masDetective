from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROOT = Path(
    "artifacts/general_role_sensor_v1"
)

ROLE_BANK = Path(
    "artifacts/general_role_bank_v1/"
    "role_bank_v1.json"
)

GENERATED = Path(
    "artifacts/general_role_bank_v1/"
    "v14_generated_probes_v3.jsonl"
)

SEED = 20260929


PAIR_CONFIGS = [
    {
        "pair": 0,
        "k": 3,
        "mode": "fanout2",
        "style": "s1",
    },
    {
        "pair": 1,
        "k": 4,
        "mode": "fanout2",
        "style": "s2",
    },
    {
        "pair": 2,
        "k": 5,
        "mode": "fanout3",
        "style": "s3",
    },
]


def stable_int(*parts) -> int:
    s = "\0".join(
        map(str, parts)
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(s).digest()[:8],
        "big",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def read_jsonl(path: Path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_json(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]

    return default


def load_role_bank():
    x = json.loads(
        ROLE_BANK.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(x, dict):
        rows = (
            x.get("roles")
            or x.get("role_bank")
            or x.get("items")
        )
    else:
        rows = x

    if not isinstance(rows, list):
        raise RuntimeError(
            "Could not parse role bank"
        )

    out = []

    for x in rows:
        role = str(
            pick(
                x,
                "role",
                "name",
                "role_name",
                default="",
            )
        ).strip()

        split = str(
            pick(
                x,
                "split",
                default="",
            )
        ).strip().lower()

        family = str(
            pick(
                x,
                "family",
                "role_family",
                default="",
            )
        ).strip()

        desc = str(
            pick(
                x,
                "public_description",
                "public_role_description",
                "description",
                default="",
            )
        ).strip()

        hidden = str(
            pick(
                x,
                "hidden_worker_prompt",
                default="",
            )
        ).strip()

        canon_raw = x.get(
            "canonical_probes",
            []
        )

        canonical = []

        for q in canon_raw:
            if isinstance(q, str):
                text = q.strip()

            elif isinstance(q, dict):
                text = str(
                    pick(
                        q,
                        "query",
                        "probe",
                        "text",
                        "prompt",
                        default="",
                    )
                ).strip()

            else:
                text = ""

            if text:
                canonical.append(text)

        if not role:
            raise RuntimeError(
                f"invalid role row: {x}"
            )

        if split not in {
            "train",
            "val",
        }:
            raise RuntimeError(
                f"{role}: split={split!r}"
            )

        if not family:
            raise RuntimeError(
                f"{role}: missing family"
            )

        if not desc:
            raise RuntimeError(
                f"{role}: missing public description"
            )

        if not hidden:
            raise RuntimeError(
                f"{role}: missing hidden_worker_prompt"
            )

        if len(canonical) != 3:
            raise RuntimeError(
                f"{role}: expected 3 canonical probes, "
                f"got {len(canonical)}"
            )

        out.append(
            {
                "role": role,
                "split": split,
                "family": family,
                "description": desc,
                "hidden_worker_prompt":
                    hidden,
                "canonical_probes":
                    canonical,
                "hard_neighbors":
                    x.get(
                        "hard_neighbors",
                        [],
                    ),
            }
        )

    return out


def normalize_neighbors(raw):
    out = []

    for x in raw or []:
        if isinstance(x, str):
            r = x

        elif isinstance(x, dict):
            r = str(
                pick(
                    x,
                    "role",
                    "name",
                    "role_name",
                    default="",
                )
            )

        else:
            continue

        r = r.strip()

        if r:
            out.append(r)

    return out


def composition_key(roles):
    return tuple(
        sorted(
            map(str, roles)
        )
    )


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Register/validate roles.
    # -------------------------

    from targets.general_role_bank_v1 import (
        register_general_role_bank,
    )

    prompt_audit = (
        register_general_role_bank(
            ROLE_BANK
        )
    )

    from targets.hierarchical_mas import (
        ROLE_PROMPTS,
    )

    roles = load_role_bank()

    by_name = {
        x["role"]: x
        for x in roles
    }

    train_roles = [
        x["role"]
        for x in roles
        if x["split"] == "train"
    ]

    val_roles = [
        x["role"]
        for x in roles
        if x["split"] == "val"
    ]

    if len(train_roles) != 60:
        raise RuntimeError(
            f"TRAIN={len(train_roles)}, expected 60"
        )

    if len(val_roles) != 12:
        raise RuntimeError(
            f"VAL={len(val_roles)}, expected 12"
        )

    if not set(
        train_roles + val_roles
    ).issubset(ROLE_PROMPTS):
        raise RuntimeError(
            "not all roles registered"
        )

    # -------------------------
    # Family structure.
    # -------------------------

    family_train = defaultdict(list)

    for x in roles:
        if x["split"] == "train":
            family_train[
                x["family"]
            ].append(
                x["role"]
            )

    for family, rr in family_train.items():
        if len(rr) != 5:
            raise RuntimeError(
                f"{family}: TRAIN={len(rr)}, expected 5"
            )

    # -------------------------
    # Generated V14 probes.
    # -------------------------

    gen_rows = read_jsonl(
        GENERATED
    )

    generated = defaultdict(list)

    for row in gen_rows:
        role = str(
            row["role"]
        )

        q = str(
            row["query"]
        ).strip()

        generated[role].append(
            {
                "query": q,
                "probe_source":
                    "v14_generated",
                "probe_id":
                    f"G{int(row['probe_index'])}",
                "selection_type":
                    row.get(
                        "selection_type"
                    ),
                "canonical_anchor_index":
                    row.get(
                        "canonical_anchor_index"
                    ),
            }
        )

    for role in by_name:
        if len(generated[role]) != 6:
            raise RuntimeError(
                f"{role}: V14 generated="
                f"{len(generated[role])}, expected 6"
            )

    # -------------------------
    # Unified 9-probe bank.
    # -------------------------

    probes = {}

    for role, info in by_name.items():
        pp = []

        for i, q in enumerate(
            info["canonical_probes"],
            1,
        ):
            pp.append(
                {
                    "query": q,
                    "probe_source":
                        "canonical",
                    "probe_id":
                        f"C{i}",
                    "selection_type":
                        None,
                    "canonical_anchor_index":
                        i,
                }
            )

        pp.extend(
            sorted(
                generated[role],
                key=lambda x:
                    int(
                        x["probe_id"][1:]
                    ),
            )
        )

        if len(pp) != 9:
            raise RuntimeError(
                f"{role}: probes={len(pp)}"
            )

        if len(
            {
                x["query"].strip().lower()
                for x in pp
            }
        ) != 9:
            raise RuntimeError(
                f"{role}: duplicate probes"
            )

        probes[role] = pp

    # -------------------------
    # Replacement candidates.
    # -------------------------

    def replacement_order(target):
        info = by_name[target]

        same_family = [
            r
            for r in family_train[
                info["family"]
            ]
            if r != target
        ]

        hard = [
            r
            for r in normalize_neighbors(
                info["hard_neighbors"]
            )
            if r in same_family
        ]

        # hard neighbors first,
        # then remaining same-family TRAIN roles.
        ordered = []

        for r in hard + sorted(
            same_family
        ):
            if r not in ordered:
                ordered.append(r)

        if len(ordered) < 3:
            raise RuntimeError(
                f"{target}: only {len(ordered)} "
                "same-family TRAIN replacements"
            )

        return ordered

    # -------------------------
    # Build matched systems.
    # -------------------------

    systems = []
    executions = []

    used_compositions = set()

    expected_val_set = set(
        val_roles
    )

    for target_idx, target in enumerate(
        train_roles + val_roles
    ):
        info = by_name[target]

        split = info["split"]

        replacement_candidates = (
            replacement_order(target)
        )

        for pcfg in PAIR_CONFIGS:
            pair_idx = int(
                pcfg["pair"]
            )

            replacement = (
                replacement_candidates[
                    pair_idx
                    % len(
                        replacement_candidates
                    )
                ]
            )

            # At least one additional same-family TRAIN confuser
            # where possible.
            same_family_confusers = [
                r
                for r in family_train[
                    info["family"]
                ]
                if r not in {
                    target,
                    replacement,
                }
            ]

            other_train = [
                r
                for r in train_roles
                if r not in {
                    target,
                    replacement,
                }
                and by_name[r][
                    "family"
                ] != info[
                    "family"
                ]
            ]

            k = int(
                pcfg["k"]
            )

            selected = None

            for attempt in range(
                10000
            ):
                rng = random.Random(
                    stable_int(
                        SEED,
                        target,
                        pair_idx,
                        attempt,
                    )
                )

                distractors = []

                if same_family_confusers:
                    distractors.append(
                        rng.choice(
                            same_family_confusers
                        )
                    )

                candidates = [
                    r
                    for r in other_train
                    if r not in distractors
                ]

                need = (
                    k
                    - 1
                    - len(distractors)
                )

                distractors += (
                    rng.sample(
                        candidates,
                        need,
                    )
                )

                rng.shuffle(
                    distractors
                )

                plus_roles = tuple(
                    [target]
                    + distractors
                )

                minus_roles = tuple(
                    [replacement]
                    + distractors
                )

                plus_key = (
                    composition_key(
                        plus_roles
                    )
                )

                minus_key = (
                    composition_key(
                        minus_roles
                    )
                )

                if plus_key == minus_key:
                    continue

                if (
                    plus_key
                    in used_compositions
                    or minus_key
                    in used_compositions
                ):
                    continue

                selected = (
                    plus_roles,
                    minus_roles,
                    distractors,
                )

                break

            if selected is None:
                raise RuntimeError(
                    f"could not create unique matched "
                    f"pair for {target} pair={pair_idx}"
                )

            (
                plus_roles,
                minus_roles,
                distractors,
            ) = selected

            used_compositions.add(
                composition_key(
                    plus_roles
                )
            )

            used_compositions.add(
                composition_key(
                    minus_roles
                )
            )

            base_id = (
                f"grs1_{split}_"
                f"{target}_p{pair_idx}"
            )

            plus_id = (
                base_id + "_plus"
            )

            minus_id = (
                base_id + "_minus"
            )

            common = {
                "dataset_version":
                    "general_role_sensor_v1",

                "split":
                    split,

                "target_role":
                    target,

                "target_family":
                    info["family"],

                "pair_index":
                    pair_idx,

                "k":
                    k,

                "execution_mode":
                    pcfg["mode"],

                "style":
                    pcfg["style"],

                "replacement_role":
                    replacement,

                "distractor_roles":
                    list(
                        distractors
                    ),
            }

            systems.append(
                {
                    **common,
                    "system_id":
                        plus_id,
                    "branch":
                        "PLUS",
                    "installed_label":
                        1,
                    "worker_roles":
                        list(
                            plus_roles
                        ),
                }
            )

            systems.append(
                {
                    **common,
                    "system_id":
                        minus_id,
                    "branch":
                        "MINUS",
                    "installed_label":
                        0,
                    "worker_roles":
                        list(
                            minus_roles
                        ),
                }
            )

            for p in probes[target]:
                common_exec = {
                    "dataset_version":
                        "general_role_sensor_v1",

                    "split":
                        split,

                    "target_role":
                        target,

                    "target_family":
                        info["family"],

                    "pair_index":
                        pair_idx,

                    "probe_id":
                        p["probe_id"],

                    "probe_source":
                        p["probe_source"],

                    "selection_type":
                        p[
                            "selection_type"
                        ],

                    "canonical_anchor_index":
                        p[
                            "canonical_anchor_index"
                        ],

                    "query":
                        p["query"],

                    "k":
                        k,

                    "execution_mode":
                        pcfg["mode"],

                    "style":
                        pcfg["style"],
                }

                executions.append(
                    {
                        **common_exec,
                        "branch":
                            "PLUS",
                        "system_id":
                            plus_id,
                        "installed_label":
                            1,
                    }
                )

                executions.append(
                    {
                        **common_exec,
                        "branch":
                            "MINUS",
                        "system_id":
                            minus_id,
                        "installed_label":
                            0,
                    }
                )

    # -------------------------
    # Audits / assertions.
    # -------------------------

    train_exec = [
        x
        for x in executions
        if x["split"] == "train"
    ]

    val_exec = [
        x
        for x in executions
        if x["split"] == "val"
    ]

    if len(train_exec) != 3240:
        raise RuntimeError(
            f"TRAIN executions={len(train_exec)}, "
            "expected 3240"
        )

    if len(val_exec) != 648:
        raise RuntimeError(
            f"VAL executions={len(val_exec)}, "
            "expected 648"
        )

    if len(executions) != 3888:
        raise RuntimeError(
            f"total executions={len(executions)}, "
            "expected 3888"
        )

    if len(systems) != 432:
        raise RuntimeError(
            f"systems={len(systems)}, expected 432"
        )

    if len(
        {
            composition_key(
                x["worker_roles"]
            )
            for x in systems
        }
    ) != len(systems):
        raise RuntimeError(
            "duplicate system composition detected"
        )

    # Exact query must be perfectly balanced PLUS/MINUS
    # within each target/probe.
    qbalance = defaultdict(
        lambda: Counter()
    )

    for x in executions:
        key = (
            x["split"],
            x["target_role"],
            x["probe_id"],
            x["query"],
        )

        qbalance[key][
            x["installed_label"]
        ] += 1

    bad_balance = {
        k: dict(v)
        for k, v in qbalance.items()
        if v[0] != 3
        or v[1] != 3
    }

    if bad_balance:
        raise RuntimeError(
            "conditional query balance failure: "
            + repr(
                list(
                    bad_balance.items()
                )[:10]
            )
        )

    # TRAIN must contain no VAL roles anywhere.
    train_systems = [
        x
        for x in systems
        if x["split"] == "train"
    ]

    leaked_val_into_train = []

    for s in train_systems:
        hit = (
            set(
                s["worker_roles"]
            )
            & expected_val_set
        )

        if hit:
            leaked_val_into_train.append(
                (
                    s["system_id"],
                    sorted(hit),
                )
            )

    if leaked_val_into_train:
        raise RuntimeError(
            "VAL role leaked into TRAIN systems: "
            + repr(
                leaked_val_into_train[:10]
            )
        )

    # VAL PLUS contains exactly one VAL role:
    # the target. VAL MINUS contains zero VAL roles.
    for s in systems:
        if s["split"] != "val":
            continue

        unseen = (
            set(
                s["worker_roles"]
            )
            & expected_val_set
        )

        if s["branch"] == "PLUS":
            if unseen != {
                s["target_role"]
            }:
                raise RuntimeError(
                    f"{s['system_id']}: "
                    f"unexpected VAL roles {unseen}"
                )

        else:
            if unseen:
                raise RuntimeError(
                    f"{s['system_id']}: "
                    f"VAL role in MINUS {unseen}"
                )

    # Same-family replacement assertion.
    for s in systems:
        target = s["target_role"]
        repl = s[
            "replacement_role"
        ]

        if (
            by_name[target]["family"]
            != by_name[repl]["family"]
        ):
            raise RuntimeError(
                f"{target}: replacement {repl} "
                "is not same family"
            )

        if by_name[repl][
            "split"
        ] != "train":
            raise RuntimeError(
                f"{target}: replacement {repl} "
                "is not TRAIN"
            )

    # -------------------------
    # Save.
    # -------------------------

    manifest = {
        "version":
            "general_role_sensor_v1",

        "seed":
            SEED,

        "old_internal_test_used":
            False,

        "role_bank_sha256":
            sha256_file(
                ROLE_BANK
            ),

        "v14_probe_bank_sha256":
            sha256_file(
                GENERATED
            ),

        "pair_configs":
            PAIR_CONFIGS,

        "train_roles":
            train_roles,

        "val_roles":
            val_roles,

        "systems":
            systems,
    }

    audit = {
        "version":
            "general_role_sensor_v1",

        "roles_total":
            len(roles),

        "train_roles":
            len(train_roles),

        "val_roles":
            len(val_roles),

        "probes_per_role":
            9,

        "canonical_per_role":
            3,

        "v14_per_role":
            6,

        "matched_pairs_per_role":
            3,

        "systems_total":
            len(systems),

        "unique_compositions":
            len(
                {
                    composition_key(
                        x["worker_roles"]
                    )
                    for x in systems
                }
            ),

        "train_executions":
            len(train_exec),

        "val_executions":
            len(val_exec),

        "total_executions":
            len(executions),

        "label_counts_total":
            dict(
                Counter(
                    x["installed_label"]
                    for x in executions
                )
            ),

        "query_conditional_balance":
            "3 PLUS / 3 MINUS for every exact target-role probe",

        "val_roles_in_train_systems":
            0,

        "old_internal_test_used":
            False,

        "replacement_rule":
            "same-family TRAIN role",

        "val_negative_caveat":
            (
                "VAL target is unseen, but MINUS replacement "
                "is a same-family TRAIN role."
            ),

        "role_prompt_registration":
            prompt_audit,
    }

    write_json(
        ROOT
        / "manifest_privileged.json",
        manifest,
    )

    write_jsonl(
        ROOT
        / "executions_manifest.jsonl",
        executions,
    )

    write_json(
        ROOT
        / "audit.json",
        audit,
    )

    # Public probe-only view contains no system truth.
    public_probe_bank = []

    for role in (
        train_roles
        + val_roles
    ):
        for p in probes[role]:
            public_probe_bank.append(
                {
                    "split":
                        by_name[role][
                            "split"
                        ],

                    "role":
                        role,

                    "family":
                        by_name[role][
                            "family"
                        ],

                    "probe_id":
                        p[
                            "probe_id"
                        ],

                    "probe_source":
                        p[
                            "probe_source"
                        ],

                    "query":
                        p["query"],
                }
            )

    write_jsonl(
        ROOT
        / "probe_bank_public.jsonl",
        public_probe_bank,
    )

    print("=" * 80)
    print("GENERAL SENSOR MANIFEST V1")
    print("=" * 80)
    print(
        "TRAIN roles:",
        len(train_roles),
    )
    print(
        "VAL roles:",
        len(val_roles),
    )
    print(
        "probes/role: 9"
    )
    print(
        "pairs/role: 3"
    )
    print(
        "systems:",
        len(systems),
    )
    print(
        "unique compositions:",
        len(
            {
                composition_key(
                    x["worker_roles"]
                )
                for x in systems
            }
        ),
    )
    print(
        "TRAIN executions:",
        len(train_exec),
    )
    print(
        "VAL executions:",
        len(val_exec),
    )
    print(
        "TOTAL executions:",
        len(executions),
    )
    print(
        "labels:",
        Counter(
            x["installed_label"]
            for x in executions
        ),
    )
    print(
        "role prompts preserved:",
        prompt_audit[
            "historical_roles_preserved"
        ],
    )
    print(
        "new role prompts added:",
        prompt_audit[
            "new_roles_added"
        ],
    )
    print(
        "missing after registration:",
        prompt_audit[
            "missing_after"
        ],
    )
    print(
        "VAL leakage into TRAIN: 0"
    )
    print(
        "OLD INTERNAL TEST USED: FALSE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
