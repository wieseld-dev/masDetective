#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

from adaptive_v11_active.io import (
    read_jsonl,
    stable_int,
    write_jsonl,
)
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import (
    PublicBeliefState,
    assert_policy_safe,
)
from v11.role_universe import TRAIN_ROLES
from v14.adaptive_belief_policy import V14AdaptiveBeliefPolicy


TARGET_RE = re.compile(
    r"TARGET CANDIDATE CAPABILITY:\s*([A-Za-z0-9_]+)"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def extract_target_role(prompt: str) -> str:
    m = TARGET_RE.search(
        str(prompt)
    )

    if m is None:
        raise RuntimeError(
            "Generator prompt does not expose "
            "TARGET CANDIDATE CAPABILITY"
        )

    return str(
        m.group(1)
    ).strip()


def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "ROLE DIRECT APPROACH V1: "
            "build a shared panel of targeted benign probes "
            "using the existing frozen R7 generator"
        )
    )

    ap.add_argument(
        "--root",
        default="artifacts/role_direct_approach_v1",
    )

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--state-source",
        required=True,
    )

    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/v14_adaptive_role_v1/"
            "checkpoints/round_07/policy"
        ),
    )

    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    ap.add_argument(
        "--split",
        default="dev",
        choices=[
            "train",
            "dev",
        ],
    )

    ap.add_argument(
        "--probes-per-role",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )

    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=3072,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20261008,
    )

    ap.add_argument(
        "--max-panels",
        type=int,
        default=24,
    )

    ap.add_argument(
        "--no-4bit",
        action="store_true",
    )

    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required"
        )

    root = Path(
        args.root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter = Path(
        args.adapter
    )

    if not adapter.is_dir():
        raise RuntimeError(
            f"Missing generator adapter: {adapter}"
        )

    state_rows = read_jsonl(
        args.state_source
    )

    if not state_rows:
        raise RuntimeError(
            "Empty state source"
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    # One canonical public R0 state is used ONLY to make
    # the generator produce candidate questions.
    #
    # Queries are generated ONCE, then the exact same shared
    # panel is executed against every test MAS.
    #
    # Therefore no system-specific truth, output, routing trace,
    # or sensor evidence can affect acquisition.
    # ---------------------------------------------------------

    raw_state = dict(
        state_rows[0]
    )

    assert_policy_safe(
        raw_state
    )

    state = PublicBeliefState.from_public_dict(
        raw_state
    )

    if state.history:
        raise RuntimeError(
            "ROLE DIRECT V1 requires clean R0 history"
        )

    roles = list(
        map(
            str,
            TRAIN_ROLES,
        )
    )

    if set(
        state.role_probs
    ) != set(
        roles
    ):
        raise RuntimeError(
            "Canonical R0 role universe mismatch"
        )

    n_needed = int(
        args.probes_per_role
    )

    if n_needed < 1:
        raise RuntimeError(
            "--probes-per-role must be >= 1"
        )

    manifest = json.loads(
        Path(
            args.manifest
        ).read_text(
            encoding="utf-8"
        )
    )

    # Only IDs are used during acquisition.
    # roles/mode/truth are NOT read for generation.
    system_ids = sorted(
        str(x["session_id"])
        for x in manifest["systems"]
        if str(
            x["split"]
        ) == args.split
    )

    if not system_ids:
        raise RuntimeError(
            f"No systems for split={args.split}"
        )

    print(
        "ROLE DIRECT APPROACH V1",
        flush=True,
    )

    print(
        "SYSTEMS =",
        len(system_ids),
        flush=True,
    )

    print(
        "ROLES =",
        len(roles),
        flush=True,
    )

    print(
        "PROBES / ROLE =",
        n_needed,
        flush=True,
    )

    print(
        "GENERATOR ADAPTER =",
        adapter,
        flush=True,
    )

    model = V14AdaptiveBeliefPolicy(
        args.model_name,
        adapter_path=str(
            adapter
        ),
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=int(
            args.max_seq_length
        ),
    )

    by_role = {
        role: []
        for role in roles
    }

    global_queries = set()

    focus_offset = (
        stable_int(
            "v14a_focus",
            state.session_id,
        )
        % len(
            state.role_probs
        )
    )

    # ---------------------------------------------------------
    # Each sweep contains 20 focus slots.
    #
    # We do NOT assume which slot maps to which role.
    # Instead we read TARGET CANDIDATE CAPABILITY from the
    # generator's own prompt and verify it.
    # ---------------------------------------------------------

    for panel_index in range(
        int(
            args.max_panels
        )
    ):

        if all(
            len(
                by_role[r]
            )
            >= n_needed
            for r in roles
        ):
            break

        for j in range(
            len(
                roles
            )
        ):

            candidate_index = (
                panel_index
                * len(
                    roles
                )
                + j
            )

            generator_focus = (
                candidate_index
                + focus_offset
            )

            seed = stable_int(
                args.seed,
                "ROLE_DIRECT_APPROACH",
                panel_index,
                j,
            )

            torch.manual_seed(
                seed
            )

            torch.cuda.manual_seed_all(
                seed
            )

            obs = {
                "belief_state":
                    state.to_public_dict(),

                "candidate_index":
                    int(
                        generator_focus
                    ),

                "max_history_turns":
                    2,

                "max_answer_chars":
                    0,
            }

            sample = model.sample_probe(
                obs,
                temperature=float(
                    args.temperature
                ),
                top_p=float(
                    args.top_p
                ),
                max_new_tokens=min(
                    int(
                        args.max_new_tokens
                    ),
                    64,
                ),
            )

            prompt = str(
                sample.prompt
            )

            role = extract_target_role(
                prompt
            )

            if role not in by_role:
                raise RuntimeError(
                    f"Unknown generated target role: {role}"
                )

            if len(
                by_role[
                    role
                ]
            ) >= n_needed:
                continue

            q = str(
                sample.query
                or ""
            ).strip()

            ok, why = surface_valid(
                q
            )

            valid = (
                bool(
                    sample.valid
                )
                and ok
                and bool(
                    q
                )
                and q not in global_queries
            )

            if not valid:

                print(
                    "[REJECT]",
                    f"panel={panel_index}",
                    f"slot={j}",
                    f"target={role}",
                    f"why={why!r}",
                    f"q={q[:120]!r}",
                    flush=True,
                )

                continue

            global_queries.add(
                q
            )

            by_role[
                role
            ].append(
                {
                    "target_role":
                        role,

                    "query":
                        q,

                    "generator_prompt":
                        prompt,

                    "generator_completion":
                        str(
                            sample.completion
                        ),

                    "raw_completion":
                        str(
                            sample.raw_completion
                        ),

                    "generator_seed":
                        int(
                            seed
                        ),

                    "generator_candidate_index":
                        int(
                            generator_focus
                        ),

                    "generator_panel":
                        int(
                            panel_index
                        ),
                }
            )

            print(
                "[ACCEPT]",
                f"target={role}",
                f"{len(by_role[role])}/{n_needed}",
                f"q={q[:140]!r}",
                flush=True,
            )

        counts = {
            r:
                len(
                    by_role[r]
                )
            for r in roles
        }

        print(
            f"[PANEL {panel_index}]",
            counts,
            flush=True,
        )

    missing = {
        r:
            len(
                by_role[r]
            )
        for r in roles
        if len(
            by_role[r]
        ) < n_needed
    }

    if missing:
        raise RuntimeError(
            "Could not obtain enough valid targeted probes: "
            f"{missing}"
        )

    # ---------------------------------------------------------
    # Shared panel:
    # EXACT SAME QUERY for a given role/probe_index is executed
    # against every MAS.
    # ---------------------------------------------------------

    panel_rows = []

    for role in roles:

        for i, item in enumerate(
            by_role[
                role
            ][
                :n_needed
            ],
            start=1,
        ):

            row = dict(
                item
            )

            row[
                "probe_index"
            ] = int(
                i
            )

            panel_rows.append(
                row
            )

    write_jsonl(
        root
        / "role_direct_probe_panel.jsonl",
        panel_rows,
    )

    candidates = []

    for sid in system_ids:

        candidate_index = 0

        for item in panel_rows:

            candidates.append(
                {
                    "session_id":
                        sid,

                    "split":
                        args.split,

                    "round":
                        1,

                    "candidate_index":
                        int(
                            candidate_index
                        ),

                    "group_id":
                        (
                            f"{sid}::ROLE_DIRECT::"
                            f"{item['target_role']}"
                        ),

                    "query":
                        str(
                            item[
                                "query"
                            ]
                        ),

                    "valid":
                        True,

                    "invalid_reason":
                        "",

                    "target_role":
                        str(
                            item[
                                "target_role"
                            ]
                        ),

                    "probe_index":
                        int(
                            item[
                                "probe_index"
                            ]
                        ),

                    "generator_prompt":
                        str(
                            item[
                                "generator_prompt"
                            ]
                        ),

                    "generator_seed":
                        int(
                            item[
                                "generator_seed"
                            ]
                        ),

                    "generator_candidate_index":
                        int(
                            item[
                                "generator_candidate_index"
                            ]
                        ),

                    "generator_mode":
                        "R7_ROLE_DIRECT_SHARED_PANEL",

                    "phase":
                        "ROLE_DIRECT_APPROACH",

                    "chosen_for_continuation":
                        True,
                }
            )

            candidate_index += 1

    out = (
        root
        / f"candidates_{args.split}_round_01.jsonl"
    )

    write_jsonl(
        out,
        candidates,
    )

    design = {
        "experiment":
            "ROLE DIRECT APPROACH V1",

        "systems":
            len(
                system_ids
            ),

        "roles":
            len(
                roles
            ),

        "probes_per_role":
            n_needed,

        "queries_per_system":
            len(
                panel_rows
            ),

        "total_mas_queries":
            len(
                candidates
            ),

        "generator":
            "existing frozen V14 R7 generator",

        "generator_adapter":
            str(
                adapter
            ),

        "acquisition":
            (
                "fixed shared role-targeted panel; "
                "no selector; no belief updater"
            ),

        "decision_time_inputs":
            [
                "target role chosen by fixed role universe",
                "clean public R0 generator state",
            ],

        "decision_time_forbidden":
            [
                "per-system truth roles",
                "routing trace",
                "MAS response",
                "sensor evidence",
            ],

        "routing_trace":
            "evaluation only",

        "sensor":
            "existing frozen V13.4 + V13.5",

        "source_hashes":
            {
                "generator_wrapper":
                    sha256(
                        Path(
                            __file__
                        )
                    ),

                "executor":
                    sha256(
                        Path(
                            "scripts/"
                            "adaptive_v11_execute_round.py"
                        )
                    ),

                "score_bridge":
                    sha256(
                        Path(
                            "scripts/"
                            "adaptive_v11_score_round.py"
                        )
                    ),

                "dual_sensor":
                    sha256(
                        Path(
                            "adaptive_score_v135.py"
                        )
                    ),
            },
    }

    (
        root
        / "design.json"
    ).write_text(
        json.dumps(
            design,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            design,
            indent=2,
        ),
        flush=True,
    )

    print(
        f"[DONE] candidates={out}",
        flush=True,
    )

    del model

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
