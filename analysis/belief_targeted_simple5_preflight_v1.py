#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES

import analysis.belief_targeted_selector_v1 as bt
import analysis.select_belief_targeted_v1_online as online


SOURCE_ROOT = Path(
    "artifacts/v14_R7_q38selectorV41_G8_simple5_v1"
)

MODEL_ROOT = Path(
    "artifacts/belief_targeted_selector_v1"
)

DATA_ROOT = Path(
    "artifacts/q38_selector_v41_data"
)

V14_BELIEF = Path(
    "artifacts/v14_adaptive_role_v1/frozen_belief.joblib"
)

GENERATOR = Path(
    "artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy"
)

BUILDER = Path(
    "analysis/build_llm_selector_v3_auc_dataset.py"
)

BUILDER_AUDITED_COPY = Path(
    "artifacts/q38_selector_v41_integrity_audit/builder_source.py"
)

TRAINING_SOURCE = Path(
    "analysis/belief_targeted_selector_v1.py"
)

ONLINE_SOURCE = Path(
    "analysis/select_belief_targeted_v1_online.py"
)

REQUIRED_SCRIPTS = [
    Path(
        "scripts/v14_adaptive_generate_round_v1.py"
    ),
    Path(
        "scripts/adaptive_v11_execute_round.py"
    ),
    Path(
        "scripts/adaptive_v11_score_round.py"
    ),
    Path(
        "scripts/kboot_advance_role_only.py"
    ),
]


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def fail(msg):
    raise RuntimeError(
        "PREFLIGHT FAIL: "
        + msg
    )


def check_exists():
    required = [
        SOURCE_ROOT
        / "states_dev_round_00.jsonl",

        SOURCE_ROOT
        / "manifest_privileged.json",

        SOURCE_ROOT
        / "frozen_belief.joblib",

        V14_BELIEF,

        GENERATOR
        / "adapter_config.json",

        BUILDER,

        BUILDER_AUDITED_COPY,

        TRAINING_SOURCE,

        ONLINE_SOURCE,

        MODEL_ROOT
        / "summary.json",
    ]

    required += REQUIRED_SCRIPTS

    for p in required:
        if not p.exists():
            fail(
                f"missing required path: {p}"
            )


def check_builder_identity():

    live = sha256(
        BUILDER
    )

    audited = sha256(
        BUILDER_AUDITED_COPY
    )

    print(
        "BUILDER LIVE  =",
        live,
    )

    print(
        "BUILDER AUDIT =",
        audited,
    )

    if live != audited:
        fail(
            "dataset builder changed since "
            "the Q38 integrity audit"
        )

    print(
        "BUILDER_HASH_MATCH: PASS"
    )


def check_role_contract():

    if list(
        bt.ROLES
    ) != list(
        TRAIN_ROLES
    ):
        fail(
            "bt.ROLES != TRAIN_ROLES"
        )

    if set(
        bt.ROLE_DESCRIPTIONS
    ) != set(
        bt.ROLES
    ):
        fail(
            "role descriptions do not "
            "cover exact role universe"
        )

    print(
        "ROLE_ORDER_MATCH: PASS"
    )


def check_checkpoints():

    if len(
        bt.ENSEMBLE_SEEDS
    ) != 5:
        fail(
            "expected 5 ensemble seeds"
        )

    hashes = {}

    dims = set()

    for seed in bt.ENSEMBLE_SEEDS:

        p = (
            MODEL_ROOT
            / f"selector_seed_{seed}.pt"
        )

        if not p.exists():
            fail(
                f"missing selector checkpoint {p}"
            )

        ck = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        if list(
            ck["roles"]
        ) != list(
            bt.ROLES
        ):
            fail(
                f"checkpoint role mismatch seed={seed}"
            )

        if (
            ck["encoder"]
            != bt.ENCODER
        ):
            fail(
                f"checkpoint encoder mismatch seed={seed}"
            )

        if (
            ck.get(
                "role_descriptions"
            )
            != bt.ROLE_DESCRIPTIONS
        ):
            fail(
                f"checkpoint role descriptions mismatch "
                f"seed={seed}"
            )

        if int(
            ck["seed"]
        ) != int(
            seed
        ):
            fail(
                f"checkpoint seed mismatch {seed}"
            )

        dims.add(
            int(
                ck[
                    "input_dim"
                ]
            )
        )

        hashes[
            str(seed)
        ] = sha256(
            p
        )

    if len(
        dims
    ) != 1:
        fail(
            f"checkpoint input dims disagree: {dims}"
        )

    print(
        "CHECKPOINTS: PASS"
    )

    print(
        "INPUT_DIM =",
        next(
            iter(
                dims
            )
        ),
    )

    return hashes


def check_offline_dataset():

    all_ids = set()

    for split in [
        "train",
        "val",
        "challenge",
    ]:

        path = (
            DATA_ROOT
            / f"{split}_groups.jsonl"
        )

        groups = read_jsonl(
            path
        )

        counts = Counter(
            len(
                g[
                    "queries"
                ]
            )
            for g in groups
        )

        rounds = Counter(
            int(
                g[
                    "round"
                ]
            )
            for g in groups
        )

        bad = (
            set(
                counts
            )
            - {
                6,
                8,
            }
        )

        if bad:
            fail(
                f"{split}: candidate counts "
                f"outside {{6,8}}: {bad}"
            )

        print(
            f"{split.upper()} "
            f"groups={len(groups)} "
            f"candidate_counts={dict(sorted(counts.items()))} "
            f"rounds={dict(sorted(rounds.items()))}"
        )

        all_ids |= {
            str(
                g[
                    "system_id"
                ]
            )
            for g in groups
        }

    return all_ids


def check_simple5_start(
    offline_ids,
):

    state_rows = read_jsonl(
        SOURCE_ROOT
        / "states_dev_round_00.jsonl"
    )

    if len(
        state_rows
    ) != 5:
        fail(
            f"Simple5 R0 has {len(state_rows)} states, not 5"
        )

    ids = []

    for x in state_rows:

        st = (
            PublicBeliefState
            .from_public_dict(
                x
            )
        )

        sid = str(
            st.session_id
        )

        ids.append(
            sid
        )

        if int(
            st.round_index
        ) != 0:
            fail(
                f"{sid}: initial round != 0"
            )

        if len(
            st.history
        ) != 0:
            fail(
                f"{sid}: initial history not empty"
            )

        if set(
            st.role_probs
        ) != set(
            bt.ROLES
        ):
            fail(
                f"{sid}: role universe mismatch"
            )

    if len(
        set(
            ids
        )
    ) != 5:
        fail(
            "duplicate Simple5 IDs"
        )

    manifest = json.loads(
        (
            SOURCE_ROOT
            / "manifest_privileged.json"
        ).read_text()
    )

    dev_manifest_ids = {
        str(
            x[
                "session_id"
            ]
        )
        for x in manifest[
            "systems"
        ]
        if x[
            "split"
        ] == "dev"
    }

    if (
        set(
            ids
        )
        != dev_manifest_ids
    ):
        fail(
            "R0 states != DEV manifest IDs"
        )

    overlap = (
        set(
            ids
        )
        & offline_ids
    )

    if overlap:
        fail(
            "Simple5 IDs overlap offline selector systems: "
            + str(
                sorted(
                    overlap
                )
            )
        )

    print(
        "SIMPLE5_R0: PASS"
    )

    print(
        "SIMPLE5_IDS =",
        ids,
    )


def check_belief_identity():

    a = sha256(
        SOURCE_ROOT
        / "frozen_belief.joblib"
    )

    b = sha256(
        V14_BELIEF
    )

    print(
        "SOURCE BELIEF HASH =",
        a,
    )

    print(
        "V14 BELIEF HASH    =",
        b,
    )

    if a != b:
        fail(
            "Simple5 frozen belief != "
            "V14 frozen belief"
        )

    print(
        "FROZEN_BELIEF_MATCH: PASS"
    )


def score_offline_split(
    runtime,
    split,
):

    groups = read_jsonl(
        DATA_ROOT
        / f"{split}_groups.jsonl"
    )

    all_queries = [
        str(q)
        for g in groups
        for q in g[
            "queries"
        ]
    ]

    histories = [
        str(
            g[
                "history_text"
            ]
        )
        for g in groups
    ]

    qmap = bt.encode_unique(
        runtime.encoder,
        all_queries,
    )

    hmap = bt.encode_unique(
        runtime.encoder,
        histories,
    )

    selected = []

    for n, g in enumerate(
        groups,
        1,
    ):

        X = np.stack([
            bt.build_feature(
                g,
                str(q),
                qmap,
                hmap,
                runtime.role_matrix,
            )
            for q in g[
                "queries"
            ]
        ]).astype(
            np.float32
        )

        scores = (
            runtime
            .score_features(
                X
            )
        )

        chosen = int(
            np.argmax(
                scores
            )
        )

        selected.append(
            float(
                g[
                    "auc_gains"
                ][
                    chosen
                ]
            )
        )

        if n % 200 == 0:
            print(
                f"[PARITY {split}] "
                f"{n}/{len(groups)}",
                flush=True,
            )

    return float(
        np.mean(
            selected
        )
    )


def check_offline_online_parity():

    summary = json.loads(
        (
            MODEL_ROOT
            / "summary.json"
        ).read_text()
    )

    # During preflight there is not yet a contract.
    runtime = online.SelectorRuntime(
        MODEL_ROOT,
        require_contract=False,
    )

    result = {}

    for split in [
        "val",
        "challenge",
    ]:

        expected = float(
            summary[
                "learned_selector"
            ][split][
                "mean_selected_gain"
            ]
        )

        got = score_offline_split(
            runtime,
            split,
        )

        diff = abs(
            got
            - expected
        )

        result[
            split
        ] = {
            "expected":
                expected,

            "recomputed":
                got,

            "abs_diff":
                diff,
        }

        print()
        print(
            f"{split.upper()} expected={expected:.12f}"
        )

        print(
            f"{split.upper()} online  ={got:.12f}"
        )

        print(
            f"{split.upper()} diff    ={diff:.12g}"
        )

        if diff > 1e-6:
            fail(
                f"{split}: offline/online "
                f"selector parity mismatch"
            )

    print(
        "OFFLINE_ONLINE_PARITY: PASS"
    )

    return result



def check_pipeline_sources():

    exec_src = Path(
        "scripts/adaptive_v11_execute_round.py"
    ).read_text()

    score_src = Path(
        "scripts/adaptive_v11_score_round.py"
    ).read_text()

    advance_src = Path(
        "scripts/kboot_advance_role_only.py"
    ).read_text()

    generator_src = Path(
        "scripts/v14_adaptive_generate_round_v1.py"
    ).read_text()

    online_src = Path(
        "analysis/select_belief_targeted_v1_online.py"
    ).read_text()

    # --------------------------------------------------------
    # GENERATOR
    # Current V14 generator writes raw candidate pool first
    # as candidates_*.
    # Driver later moves it to candidate_pool_* before selector.
    # --------------------------------------------------------

    generator_required = [
        "candidates_",
        "write_jsonl",
        "group_size",
        "PublicBeliefState",
    ]

    for token in generator_required:
        if token not in generator_src:
            fail(
                "generator missing required behavior: "
                + token
            )

    # --------------------------------------------------------
    # ONLINE SELECTOR
    #
    # Must read pool and write selected candidates only.
    # Must use EXACT training feature/history implementation.
    # --------------------------------------------------------

    selector_required = [
        "candidate_pool_",
        "candidates_",
        "dsbuilder.history_text",
        "bt.build_feature",
        "role_probs_before",
        "chosen_for_continuation",
        "selector_selected",
    ]

    for token in selector_required:
        if token not in online_src:
            fail(
                "online selector missing required behavior: "
                + token
            )

    # Absolutely forbidden at selector decision time.
    selector_forbidden = [
        "truth_roles_training_only",
        "selected_roles_training_only",
        "auc_gains",
        "pair_gains",
    ]

    for token in selector_forbidden:
        if token in online_src:
            fail(
                "online selector contains forbidden "
                "truth/hindsight token: "
                + token
            )

    # --------------------------------------------------------
    # EXECUTOR
    # --------------------------------------------------------

    executor_required = [
        "candidates_",
        "responses_",
        "final_output",
    ]

    for token in executor_required:
        if token not in exec_src:
            fail(
                "executor missing required behavior: "
                + token
            )

    # --------------------------------------------------------
    # SENSOR BRIDGE
    # --------------------------------------------------------

    sensor_ok = (
        "adaptive_score_v135.py"
        in score_src
        or
        "v135"
        in score_src.lower()
    )

    if not sensor_ok:
        fail(
            "score bridge does not appear to invoke "
            "the current dual sensor"
        )

    # --------------------------------------------------------
    # CURRENT KBOOT ROLE-ONLY ADVANCE
    #
    # IMPORTANT:
    # Current implementation is role_only_probs(),
    # NOT role_only_predict().
    # Exact numerical equivalence is checked separately below.
    # --------------------------------------------------------

    advance_required = [
        "FrozenBeliefBundle",
        "EvidenceFrame",
        "role_only_probs",
        "v135",
        "v134",
        "scored_",
        "responses_",
        "candidates_",
        "uniform_k",
        "PublicBeliefState",
    ]

    for token in advance_required:
        if token not in advance_src:
            fail(
                "advance missing required behavior: "
                + token
            )

    print(
        "GENERATOR_OUTPUT_CONTRACT: PASS"
    )

    print(
        "POOL_STAGING_CONTRACT: PASS"
    )

    print(
        "SELECTOR_IO_CONTRACT: PASS"
    )

    print(
        "SELECTOR_LEAKAGE_GUARD: PASS"
    )

    print(
        "EXECUTOR_SELECTED_ONLY_CONTRACT: PASS"
    )

    print(
        "DUAL_SENSOR_CONTRACT: PASS"
    )

    print(
        "KBOOT_ROLE_ONLY_SOURCE_CONTRACT: PASS"
    )

    print(
        "PIPELINE_SOURCE_CONTRACT: PASS"
    )



def check_updater_semantic_parity():

    import importlib.util

    from adaptive_v11_active.belief import (
        FrozenBeliefBundle,
    )

    from adaptive_v11_curriculum.belief_utils import (
        role_only_predict,
    )

    # --------------------------------------------------------
    # Load EXACT prospective updater implementation.
    # --------------------------------------------------------

    spec = importlib.util.spec_from_file_location(
        "kboot_runtime_for_parity",
        "scripts/kboot_advance_role_only.py",
    )

    kb = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        kb
    )

    if not hasattr(
        kb,
        "role_only_probs",
    ):
        fail(
            "kboot runtime has no role_only_probs()"
        )

    bundle = FrozenBeliefBundle.load(
        V14_BELIEF
    )

    # --------------------------------------------------------
    # role_only_predict has existed in more than one API form:
    #
    #   current:
    #       role_probs_dict
    #
    #   older:
    #       (k_probs_dict, role_probs_dict)
    #
    # Never assume the return arity.
    # Identify the role map structurally instead.
    # --------------------------------------------------------

    def extract_role_map(
        result,
    ):

        def is_role_map(x):
            return (
                isinstance(
                    x,
                    dict,
                )
                and
                all(
                    r in x
                    for r in bt.ROLES
                )
            )

        if is_role_map(
            result
        ):
            return {
                r:
                    float(
                        result[r]
                    )
                for r in bt.ROLES
            }

        if isinstance(
            result,
            (
                tuple,
                list,
            ),
        ):
            matches = [
                x
                for x in result
                if is_role_map(
                    x
                )
            ]

            if len(
                matches
            ) == 1:
                return {
                    r:
                        float(
                            matches[0][r]
                        )
                    for r in bt.ROLES
                }

            if len(
                matches
            ) > 1:
                fail(
                    "role_only_predict returned "
                    "multiple role-like dictionaries"
                )

        fail(
            "cannot identify role-probability map "
            "from role_only_predict result "
            f"type={type(result)}"
        )

    # --------------------------------------------------------
    # KBOOT explicitly keeps K fixed at bundle.initial().
    # Verify that distribution is truly uniform.
    # --------------------------------------------------------

    kboot_k, _ = bundle.initial()

    kvals = np.asarray(
        [
            float(
                kboot_k[k]
            )
            for k in sorted(
                kboot_k
            )
        ],
        dtype=float,
    )

    if abs(
        float(
            kvals.sum()
        )
        - 1.0
    ) > 1e-12:
        fail(
            "initial K distribution does not sum to 1"
        )

    if not np.allclose(
        kvals,
        kvals[0],
        atol=1e-12,
        rtol=0.0,
    ):
        fail(
            "KBOOT initial K is not uniform"
        )

    print(
        "KBOOT_UNIFORM_K: PASS",
        {
            int(k):
                float(
                    kboot_k[k]
                )
            for k in sorted(
                kboot_k
            )
        },
    )

    # --------------------------------------------------------
    # Compare BOTH role-updater paths on every available
    # historical Simple5 public state R0..R8.
    #
    # No truth used.
    # No API calls.
    # --------------------------------------------------------

    checked = 0

    max_diff = 0.0

    worst = None

    return_types = set()

    for rnd in range(
        0,
        9,
    ):

        state_path = (
            SOURCE_ROOT
            / f"states_dev_round_{rnd:02d}.jsonl"
        )

        if not state_path.exists():
            continue

        for raw in read_jsonl(
            state_path
        ):

            st = (
                PublicBeliefState
                .from_public_dict(
                    raw
                )
            )

            history = list(
                st.history
            )

            result = role_only_predict(
                bundle,
                history,
            )

            return_types.add(
                type(
                    result
                ).__name__
            )

            ref_roles = extract_role_map(
                result
            )

            prospective_roles = (
                kb.role_only_probs(
                    bundle,
                    history,
                )
            )

            if set(
                prospective_roles
            ) != set(
                bt.ROLES
            ):
                fail(
                    "kboot role universe mismatch "
                    f"sid={st.session_id} "
                    f"round={rnd}"
                )

            diff = max(
                abs(
                    float(
                        ref_roles[r]
                    )
                    -
                    float(
                        prospective_roles[r]
                    )
                )
                for r in bt.ROLES
            )

            if diff > max_diff:

                max_diff = diff

                worst = {
                    "session_id":
                        st.session_id,

                    "round":
                        rnd,
                }

            checked += 1

    if checked == 0:
        fail(
            "no Simple5 histories available "
            "for updater parity test"
        )

    print(
        "UPDATER_PARITY histories_checked =",
        checked,
    )

    print(
        "UPDATER_PARITY role_only_predict_return_types =",
        sorted(
            return_types
        ),
    )

    print(
        "UPDATER_PARITY max_role_abs_diff =",
        f"{max_diff:.16g}",
    )

    print(
        "UPDATER_PARITY worst =",
        worst,
    )

    # kboot clamps only at EPS boundaries.
    # Anything beyond numerical/EPS-level behavior is a real
    # prospective/offline updater mismatch.
    if max_diff > 1.1e-6:
        fail(
            "offline/prospective role updater mismatch: "
            f"max_abs_diff={max_diff}"
        )

    print(
        "UPDATER_SEMANTIC_PARITY: PASS"
    )


def main():

    print(
        "=" * 100
    )

    print(
        "BELIEF-TARGETED SIMPLE5 "
        "PROSPECTIVE PREFLIGHT"
    )

    print(
        "=" * 100
    )

    check_exists()

    check_builder_identity()

    check_role_contract()

    checkpoint_hashes = (
        check_checkpoints()
    )

    offline_ids = (
        check_offline_dataset()
    )

    check_simple5_start(
        offline_ids
    )

    check_belief_identity()

    check_pipeline_sources()

    check_updater_semantic_parity()

    parity = (
        check_offline_online_parity()
    )

    contract = {
        "experiment":
            "belief_targeted_simple5_clean_v1",

        "decision_time_inputs":
            [
                "current role belief",
                "public history",
                "candidate query",
                "round",
            ],

        "source_hashes": {
            "training_source":
                sha256(
                    TRAINING_SOURCE
                ),

            "builder_source":
                sha256(
                    BUILDER
                ),

            "online_source":
                sha256(
                    ONLINE_SOURCE
                ),

            "generator_source":
                sha256(
                    Path(
                        "scripts/v14_adaptive_generate_round_v1.py"
                    )
                ),

            "executor_source":
                sha256(
                    Path(
                        "scripts/adaptive_v11_execute_round.py"
                    )
                ),

            "score_source":
                sha256(
                    Path(
                        "scripts/adaptive_v11_score_round.py"
                    )
                ),

            "advance_source":
                sha256(
                    Path(
                        "scripts/kboot_advance_role_only.py"
                    )
                ),
        },

        "checkpoint_hashes":
            checkpoint_hashes,

        "offline_online_parity":
            parity,

        "source_simple5_root":
            str(
                SOURCE_ROOT
            ),

        "source_initial_state_hash":
            sha256(
                SOURCE_ROOT
                / "states_dev_round_00.jsonl"
            ),

        "source_manifest_hash":
            sha256(
                SOURCE_ROOT
                / "manifest_privileged.json"
            ),

        "frozen_belief_hash":
            sha256(
                V14_BELIEF
            ),

        "generator_adapter":
            str(
                GENERATOR
            ),

        "generator_adapter_config_hash":
            sha256(
                GENERATOR
                / "adapter_config.json"
            ),
    }

    out = (
        MODEL_ROOT
        / "prospective_contract.json"
    )

    out.write_text(
        json.dumps(
            contract,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "CONTRACT WRITTEN =",
        out,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PREFLIGHT_PASS"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
