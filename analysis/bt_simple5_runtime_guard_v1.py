#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


MODEL_ROOT = Path(
    "artifacts/belief_targeted_selector_v1"
)

GENERATOR = Path(
    "artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy"
)

SOURCE_FILES = {
    "training_source":
        Path("analysis/belief_targeted_selector_v1.py"),

    "builder_source":
        Path("analysis/build_llm_selector_v3_auc_dataset.py"),

    "online_source":
        Path("analysis/select_belief_targeted_v1_online.py"),

    "generator_source":
        Path("scripts/v14_adaptive_generate_round_v1.py"),

    "executor_source":
        Path("scripts/adaptive_v11_execute_round.py"),

    "score_source":
        Path("scripts/adaptive_v11_score_round.py"),

    "advance_source":
        Path("scripts/kboot_advance_role_only.py"),
}


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def die(msg):
    raise RuntimeError(
        "RUNTIME_GUARD_FAIL: " + msg
    )


def check_locked_assets(
    root: Path,
):
    contract = json.loads(
        (
            MODEL_ROOT
            / "prospective_contract.json"
        ).read_text()
    )

    for name, path in SOURCE_FILES.items():

        got = sha256(path)

        want = contract[
            "source_hashes"
        ][name]

        if got != want:
            die(
                f"source drift: {name}"
            )

    for seed, want in contract[
        "checkpoint_hashes"
    ].items():

        p = (
            MODEL_ROOT
            / f"selector_seed_{seed}.pt"
        )

        if sha256(p) != want:
            die(
                f"selector checkpoint drift seed={seed}"
            )

    if (
        sha256(
            root
            / "states_dev_round_00.jsonl"
        )
        !=
        contract[
            "source_initial_state_hash"
        ]
    ):
        die(
            "R0 state hash mismatch"
        )

    if (
        sha256(
            root
            / "manifest_privileged.json"
        )
        !=
        contract[
            "source_manifest_hash"
        ]
    ):
        die(
            "manifest hash mismatch"
        )

    if (
        sha256(
            root
            / "frozen_belief.joblib"
        )
        !=
        contract[
            "frozen_belief_hash"
        ]
    ):
        die(
            "frozen belief hash mismatch"
        )

    if (
        sha256(
            GENERATOR
            / "adapter_config.json"
        )
        !=
        contract[
            "generator_adapter_config_hash"
        ]
    ):
        die(
            "generator adapter drift"
        )


def state_ids(
    root,
    rnd,
):
    p = (
        root
        / f"states_dev_round_{rnd:02d}.jsonl"
    )

    if not p.exists():
        die(
            f"missing state file {p}"
        )

    rows = read_jsonl(p)

    if len(rows) != 5:
        die(
            f"round {rnd}: "
            f"expected 5 states, got {len(rows)}"
        )

    ids = []

    for raw in rows:

        st = PublicBeliefState.from_public_dict(
            raw
        )

        if st.round_index != rnd:
            die(
                f"{st.session_id}: "
                f"round_index={st.round_index}, "
                f"expected={rnd}"
            )

        if len(st.history) != rnd:
            die(
                f"{st.session_id}: "
                f"history_len={len(st.history)}, "
                f"expected={rnd}"
            )

        if set(
            st.role_probs
        ) != set(
            TRAIN_ROLES
        ):
            die(
                f"{st.session_id}: role universe mismatch"
            )

        kvals = list(
            st.k_probs.values()
        )

        if (
            len(kvals) != 4
            or max(kvals)
            - min(kvals)
            > 1e-12
        ):
            die(
                f"{st.session_id}: K not uniform"
            )

        ids.append(
            str(
                st.session_id
            )
        )

    if len(set(ids)) != 5:
        die(
            "duplicate session IDs"
        )

    return set(ids)


def check_pre(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd - 1,
    )

    print(
        f"PRE_ROUND_{rnd}: PASS "
        f"systems={len(ids)} "
        f"history_len={rnd-1}"
    )


def check_post_gen(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd - 1,
    )

    p = (
        root
        / f"candidates_dev_round_{rnd:02d}.jsonl"
    )

    rows = read_jsonl(p)

    if len(rows) != 40:
        die(
            f"generator rows={len(rows)}, expected 40"
        )

    c = Counter(
        str(x["session_id"])
        for x in rows
    )

    if (
        set(c) != ids
        or set(c.values()) != {8}
    ):
        die(
            f"generator group counts bad: {dict(c)}"
        )

    for sid in ids:

        idx = sorted(
            int(x["candidate_index"])
            for x in rows
            if str(x["session_id"]) == sid
        )

        if idx != list(
            range(8)
        ):
            die(
                f"{sid}: candidate indices={idx}"
            )

    print(
        f"POST_GEN_R{rnd}: PASS rows=40 groups=5x8"
    )


def check_post_select(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd - 1,
    )

    pool = read_jsonl(
        root
        / f"candidate_pool_dev_round_{rnd:02d}.jsonl"
    )

    selected = read_jsonl(
        root
        / f"candidates_dev_round_{rnd:02d}.jsonl"
    )

    diag = read_jsonl(
        root
        / f"selector_diag_dev_round_{rnd:02d}.jsonl"
    )

    if len(pool) != 40:
        die(
            f"pool rows={len(pool)}, expected 40"
        )

    if len(selected) != 5:
        die(
            f"selected rows={len(selected)}, expected 5"
        )

    scount = Counter(
        str(x["session_id"])
        for x in selected
    )

    if (
        set(scount) != ids
        or set(scount.values()) != {1}
    ):
        die(
            f"selected cardinality bad: {dict(scount)}"
        )

    for x in selected:

        if not x.get(
            "chosen_for_continuation"
        ):
            die(
                "selected row missing chosen flag"
            )

        if not x.get(
            "selector_selected"
        ):
            die(
                "selected row missing selector flag"
            )

        if not x.get(
            "valid",
            True,
        ):
            die(
                "selector chose invalid query"
            )

        # No hindsight/truth fields may enter selector output.
        forbidden = [
            k
            for k in x
            if (
                "truth" in k.lower()
                or
                "selected_roles_training_only"
                in k.lower()
                or
                "auc_gain"
                in k.lower()
            )
        ]

        if forbidden:
            die(
                f"forbidden selector fields: {forbidden}"
            )

    dcount = Counter(
        str(x["session_id"])
        for x in diag
    )

    if set(dcount) != ids:
        die(
            "selector diagnostic system mismatch"
        )

    if any(
        n < 2
        or n > 8
        for n in dcount.values()
    ):
        die(
            f"selector viable counts outside [2,8]: {dict(dcount)}"
        )

    print(
        f"POST_SELECT_R{rnd}: PASS "
        f"pool=40 selected=5 viable={dict(dcount)}"
    )


def check_post_exec(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd - 1,
    )

    rows = read_jsonl(
        root
        / f"responses_dev_round_{rnd:02d}.jsonl"
    )

    if len(rows) != 5:
        die(
            f"responses rows={len(rows)}, expected 5"
        )

    got = {
        str(
            x.get(
                "session_id",
                x.get(
                    "system_id",
                    "",
                ),
            )
        )
        for x in rows
    }

    if got != ids:
        die(
            "response system IDs mismatch"
        )

    bad = [
        x
        for x in rows
        if x.get(
            "execution_status"
        ) != "ok"
    ]

    if bad:
        die(
            f"{len(bad)} executions failed"
        )

    print(
        f"POST_EXEC_R{rnd}: PASS responses=5 all_ok"
    )


def check_post_score(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd - 1,
    )

    rows = read_jsonl(
        root
        / f"scored_dev_round_{rnd:02d}.jsonl"
    )

    if len(rows) != 5:
        die(
            f"scored rows={len(rows)}, expected 5"
        )

    got = set()

    for x in rows:

        sid = str(
            x.get(
                "system_id",
                x.get(
                    "session_id",
                    "",
                ),
            )
        )

        got.add(sid)

        e135 = dict(
            x.get(
                "evidence",
                {},
            )
        )

        e134 = dict(
            x.get(
                "metadata",
                {},
            ).get(
                "evidence_v134",
                {},
            )
        )

        if set(e135) != set(
            TRAIN_ROLES
        ):
            die(
                f"{sid}: V13.5 role universe mismatch"
            )

        if set(e134) != set(
            TRAIN_ROLES
        ):
            die(
                f"{sid}: V13.4 role universe mismatch"
            )

    if got != ids:
        die(
            "scored system IDs mismatch"
        )

    print(
        f"POST_SCORE_R{rnd}: PASS "
        f"rows=5 V135=20 V134=20"
    )


def check_post_advance(
    root,
    rnd,
):
    ids = state_ids(
        root,
        rnd,
    )

    metrics = read_jsonl(
        root
        / f"kboot_metrics_dev_round_{rnd:02d}.jsonl"
    )

    if len(metrics) != 5:
        die(
            f"metrics rows={len(metrics)}, expected 5"
        )

    mp = {
        str(x["session_id"])
        for x in metrics
    }

    if mp != ids:
        die(
            "metrics IDs mismatch"
        )

    summary_path = (
        root
        / f"kboot_advance_dev_round_{rnd:02d}_summary.json"
    )

    if not summary_path.exists():
        die(
            f"missing advance summary {summary_path}"
        )

    s = json.loads(
        summary_path.read_text()
    )

    if int(
        s.get(
            "sessions",
            -1,
        )
    ) != 5:
        die(
            "advance summary sessions != 5"
        )

    if int(
        s.get(
            "observed",
            -1,
        )
    ) != 5:
        die(
            "advance summary observed != 5"
        )

    print(
        f"POST_ADVANCE_R{rnd}: PASS "
        f"states=5 history_len={rnd} observed=5"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--round",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "pre",
            "post_gen",
            "post_select",
            "post_exec",
            "post_score",
            "post_advance",
        ],
    )

    a = ap.parse_args()

    root = Path(
        a.root
    )

    check_locked_assets(
        root
    )

    {
        "pre":
            check_pre,

        "post_gen":
            check_post_gen,

        "post_select":
            check_post_select,

        "post_exec":
            check_post_exec,

        "post_score":
            check_post_score,

        "post_advance":
            check_post_advance,
    }[
        a.phase
    ](
        root,
        a.round,
    )

    print(
        "RUNTIME_GUARD_PASS"
    )


if __name__ == "__main__":
    main()
