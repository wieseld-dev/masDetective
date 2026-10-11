#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_active.io import read_jsonl, write_jsonl

import analysis.belief_targeted_selector_v1 as bt
import analysis.build_llm_selector_v3_auc_dataset as dsbuilder


TRAINING_SOURCE = Path(
    "analysis/belief_targeted_selector_v1.py"
)

BUILDER_SOURCE = Path(
    "analysis/build_llm_selector_v3_auc_dataset.py"
)

DEFAULT_MODEL_ROOT = Path(
    "artifacts/belief_targeted_selector_v1"
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


class SelectorRuntime:

    def __init__(
        self,
        model_root: Path,
        *,
        require_contract: bool = True,
    ):
        self.model_root = Path(
            model_root
        )

        self.contract_path = (
            self.model_root
            / "prospective_contract.json"
        )

        if require_contract:
            if not self.contract_path.exists():
                raise RuntimeError(
                    "Missing prospective contract: "
                    f"{self.contract_path}. "
                    "Run the preflight first."
                )

            self._validate_contract()

        self.encoder = SentenceTransformer(
            bt.ENCODER,
            device=bt.DEVICE,
        )

        self.role_matrix = (
            self.encoder.encode(
                [
                    bt.ROLE_DESCRIPTIONS[r]
                    for r in bt.ROLES
                ],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            .astype(np.float32)
        )

        self.models = []

        for seed in bt.ENSEMBLE_SEEDS:

            p = (
                self.model_root
                / f"selector_seed_{seed}.pt"
            )

            if not p.exists():
                raise RuntimeError(
                    f"Missing checkpoint: {p}"
                )

            ckpt = torch.load(
                p,
                map_location="cpu",
                weights_only=False,
            )

            if list(
                ckpt["roles"]
            ) != list(
                bt.ROLES
            ):
                raise RuntimeError(
                    f"Role-order mismatch: {p}"
                )

            if (
                ckpt["encoder"]
                != bt.ENCODER
            ):
                raise RuntimeError(
                    f"Encoder mismatch: {p}"
                )

            if (
                int(
                    ckpt["seed"]
                )
                != int(
                    seed
                )
            ):
                raise RuntimeError(
                    f"Seed mismatch: {p}"
                )

            if (
                ckpt.get(
                    "role_descriptions"
                )
                != bt.ROLE_DESCRIPTIONS
            ):
                raise RuntimeError(
                    f"Role-description mismatch: {p}"
                )

            model = bt.SelectorNet(
                int(
                    ckpt[
                        "input_dim"
                    ]
                )
            ).to(
                bt.DEVICE
            )

            model.load_state_dict(
                ckpt[
                    "state_dict"
                ]
            )

            model.eval()

            self.models.append(
                (
                    seed,
                    model,
                    int(
                        ckpt[
                            "input_dim"
                        ]
                    ),
                )
            )

    def _validate_contract(
        self,
    ):
        c = json.loads(
            self.contract_path.read_text()
        )

        current = {
            "training_source":
                sha256(
                    TRAINING_SOURCE
                ),

            "builder_source":
                sha256(
                    BUILDER_SOURCE
                ),

            "online_source":
                sha256(
                    Path(__file__)
                ),
        }

        for k, v in current.items():
            if (
                c[
                    "source_hashes"
                ][k]
                != v
            ):
                raise RuntimeError(
                    f"CONTRACT HASH MISMATCH: {k}\n"
                    f"expected={c['source_hashes'][k]}\n"
                    f"current ={v}"
                )

        for seed in bt.ENSEMBLE_SEEDS:
            p = (
                self.model_root
                / f"selector_seed_{seed}.pt"
            )

            expected = (
                c[
                    "checkpoint_hashes"
                ][
                    str(seed)
                ]
            )

            got = sha256(
                p
            )

            if got != expected:
                raise RuntimeError(
                    f"CHECKPOINT HASH MISMATCH seed={seed}"
                )

    @torch.inference_mode()
    def score_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        xt = torch.tensor(
            X,
            dtype=torch.float32,
            device=bt.DEVICE,
        )

        all_scores = []

        for seed, model, input_dim in (
            self.models
        ):

            if (
                X.shape[1]
                != input_dim
            ):
                raise RuntimeError(
                    "Feature dimension mismatch: "
                    f"online={X.shape[1]} "
                    f"checkpoint={input_dim} "
                    f"seed={seed}"
                )

            all_scores.append(
                model(
                    xt
                )
                .float()
                .cpu()
                .numpy()
            )

        return np.mean(
            np.stack(
                all_scores,
                axis=0,
            ),
            axis=0,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--round",
        type=int,
        required=True,
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
        "--root",
        required=True,
    )

    ap.add_argument(
        "--model-root",
        default=str(
            DEFAULT_MODEL_ROOT
        ),
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    states_path = (
        root
        / (
            f"states_{args.split}_"
            f"round_{args.round-1:02d}.jsonl"
        )
    )

    pool_path = (
        root
        / (
            f"candidate_pool_{args.split}_"
            f"round_{args.round:02d}.jsonl"
        )
    )

    if not states_path.exists():
        raise RuntimeError(
            f"Missing state file: {states_path}"
        )

    if not pool_path.exists():
        raise RuntimeError(
            f"Missing candidate pool: {pool_path}"
        )

    state_rows = read_jsonl(
        states_path
    )

    pool_rows = read_jsonl(
        pool_path
    )

    states = {
        str(x["session_id"]):
            PublicBeliefState.from_public_dict(
                x
            )
        for x in state_rows
    }

    if len(states) != 5:
        raise RuntimeError(
            f"Expected exactly 5 Simple5 states, got {len(states)}"
        )

    for sid, st in states.items():

        if (
            int(
                st.round_index
            )
            != args.round - 1
        ):
            raise RuntimeError(
                f"{sid}: state round mismatch "
                f"{st.round_index} != {args.round-1}"
            )

        if set(
            st.role_probs.keys()
        ) != set(
            bt.ROLES
        ):
            raise RuntimeError(
                f"{sid}: role universe mismatch"
            )

    grouped = defaultdict(
        list
    )

    for x in pool_rows:

        sid = str(
            x["session_id"]
        )

        if sid not in states:
            raise RuntimeError(
                f"Candidate for unknown state: {sid}"
            )

        if int(
            x["round"]
        ) != args.round:
            raise RuntimeError(
                f"{sid}: candidate round mismatch"
            )

        grouped[
            sid
        ].append(
            dict(x)
        )

    if set(
        grouped
    ) != set(
        states
    ):
        raise RuntimeError(
            "Candidate/state ID mismatch"
        )

    histories = {}

    viable_by_sid = {}

    for sid in sorted(
        states
    ):

        st = states[
            sid
        ]

        # CRITICAL:
        # use EXACT DATASET-BUILDER history renderer.
        hist = dsbuilder.history_text(
            st
        )

        histories[
            sid
        ] = hist

        viable = [
            dict(x)
            for x in sorted(
                grouped[
                    sid
                ],
                key=lambda z:
                    int(
                        z[
                            "candidate_index"
                        ]
                    ),
            )
            if (
                bool(
                    x.get(
                        "valid",
                        True,
                    )
                )
                and
                str(
                    x.get(
                        "query",
                        ""
                    )
                ).strip()
            )
        ]

        # Generator requests G=8, but individual generations can
        # legitimately be invalid/empty.  The selector scores every
        # candidate independently and then performs argmax, so an
        # intermediate viable count such as 7 does not change the
        # feature contract or checkpoint input dimension.
        #
        # Reject only a degenerate candidate pool.
        if not (
            2
            <= len(viable)
            <= 8
        ):
            raise RuntimeError(
                f"{sid}: viable candidate count "
                f"{len(viable)} is outside safe range [2,8]"
            )

        viable_by_sid[
            sid
        ] = viable

    runtime = SelectorRuntime(
        Path(
            args.model_root
        ),
        require_contract=True,
    )

    all_queries = [
        str(
            x[
                "query"
            ]
        )
        for sid in sorted(
            viable_by_sid
        )
        for x in viable_by_sid[
            sid
        ]
    ]

    qmap = bt.encode_unique(
        runtime.encoder,
        all_queries,
    )

    hmap = bt.encode_unique(
        runtime.encoder,
        list(
            histories.values()
        ),
    )

    selected_rows = []
    diag_rows = []

    for sid in sorted(
        states
    ):

        st = states[
            sid
        ]

        viable = viable_by_sid[
            sid
        ]

        hist = histories[
            sid
        ]

        group = {
            "role_probs_before":
                {
                    r:
                        float(
                            st.role_probs[
                                r
                            ]
                        )
                    for r in bt.ROLES
                },

            "history_text":
                hist,

            "round":
                int(
                    args.round
                ),
        }

        X = np.stack([
            bt.build_feature(
                group,
                str(
                    x[
                        "query"
                    ]
                ),
                qmap,
                hmap,
                runtime.role_matrix,
            )
            for x in viable
        ]).astype(
            np.float32
        )

        scores = runtime.score_features(
            X
        )

        order = np.argsort(
            -scores
        )

        chosen_local = int(
            order[
                0
            ]
        )

        rank_by_local = {
            int(local):
                int(rank + 1)
            for rank, local
            in enumerate(
                order
            )
        }

        for local_i, row in enumerate(
            viable
        ):

            chosen = (
                local_i
                ==
                chosen_local
            )

            diag = dict(
                row
            )

            diag.update({
                "selector_model":
                    "belief_targeted_selector_v1_5seed",

                "selector_score":
                    float(
                        scores[
                            local_i
                        ]
                    ),

                "selector_rank":
                    rank_by_local[
                        local_i
                    ],

                "selector_selected":
                    bool(
                        chosen
                    ),

                "selector_model_samples":
                    len(
                        runtime.models
                    ),

                "selector_mode":
                    "belief_targeted_v1",

                "history_text_exact":
                    hist,
            })

            diag_rows.append(
                diag
            )

            if chosen:

                out = dict(
                    row
                )

                out.update({
                    "chosen_for_continuation":
                        True,

                    "selector_model":
                        "belief_targeted_selector_v1_5seed",

                    "selector_score":
                        float(
                            scores[
                                local_i
                            ]
                        ),

                    "selector_rank":
                        1,

                    "selector_selected":
                        True,

                    "selector_predicted_utility":
                        float(
                            scores[
                                local_i
                            ]
                        ),

                    "selector_model_samples":
                        len(
                            runtime.models
                        ),

                    "selector_mode":
                        "belief_targeted_v1",
                })

                selected_rows.append(
                    out
                )

        print(
            f"[SELECT] {sid} "
            f"round={args.round} "
            f"candidate={viable[chosen_local]['candidate_index']} "
            f"score={scores[chosen_local]:+.6f} "
            f"query={str(viable[chosen_local]['query'])[:160]!r}",
            flush=True,
        )

    if len(
        selected_rows
    ) != 5:
        raise RuntimeError(
            "Expected exactly one selected query "
            "for each of 5 systems"
        )

    by_sid = defaultdict(
        int
    )

    for x in selected_rows:
        by_sid[
            str(
                x[
                    "session_id"
                ]
            )
        ] += 1

    if (
        set(
            by_sid.values()
        )
        != {
            1
        }
    ):
        raise RuntimeError(
            f"Selection cardinality error: {dict(by_sid)}"
        )

    # IMPORTANT:
    # candidates_* contains ONLY the selected query.
    # Therefore downstream execution cannot accidentally
    # execute an unselected candidate.
    out_candidates = (
        root
        / (
            f"candidates_{args.split}_"
            f"round_{args.round:02d}.jsonl"
        )
    )

    out_diag = (
        root
        / (
            f"selector_diag_{args.split}_"
            f"round_{args.round:02d}.jsonl"
        )
    )

    write_jsonl(
        out_candidates,
        selected_rows,
    )

    write_jsonl(
        out_diag,
        diag_rows,
    )

    print(
        f"[DONE] selected={len(selected_rows)} "
        f"diagnostic_rows={len(diag_rows)}"
    )

    print(
        f"[DONE] candidates={out_candidates}"
    )

    print(
        f"[DONE] diag={out_diag}"
    )


if __name__ == "__main__":
    main()
