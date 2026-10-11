#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.io import (
    read_jsonl,
    write_jsonl,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--split",
        default="dev",
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    panel = read_jsonl(
        root
        / "role_direct_probe_panel.jsonl"
    )

    manifest = json.loads(
        Path(
            args.manifest
        ).read_text(
            encoding="utf-8"
        )
    )

    # Deliberately use ONLY:
    #   session_id
    #   split
    #
    # Never inspect roles / k / truth here.
    system_ids = sorted(
        str(
            x[
                "session_id"
            ]
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
    )

    if not system_ids:

        raise RuntimeError(
            "No systems found"
        )

    out = []

    for sid in system_ids:

        for j, p in enumerate(
            panel
        ):

            out.append(
                {
                    "session_id":
                        sid,

                    "split":
                        args.split,

                    "round":
                        1,

                    "candidate_index":
                        int(
                            j
                        ),

                    "group_id":
                        (
                            f"{sid}::ROLE_DIRECT::"
                            f"{p['target_role']}"
                        ),

                    "query":
                        str(
                            p[
                                "query"
                            ]
                        ),

                    "valid":
                        True,

                    "invalid_reason":
                        "",

                    "target_role":
                        str(
                            p[
                                "target_role"
                            ]
                        ),

                    "probe_index":
                        int(
                            p[
                                "probe_index"
                            ]
                        ),

                    "probe_strategy":
                        str(
                            p[
                                "strategy"
                            ]
                        ),

                    "generator_mode":
                        "DIRECT_PROMPT_NO_BELIEF",

                    "phase":
                        "ROLE_DIRECT_APPROACH",

                    "chosen_for_continuation":
                        True,
                }
            )

    expected = (
        len(
            panel
        )
        * len(
            system_ids
        )
    )

    if len(
        out
    ) != expected:

        raise RuntimeError(
            "Expansion count mismatch"
        )

    path = (
        root
        / f"candidates_{args.split}_round_01.jsonl"
    )

    write_jsonl(
        path,
        out,
    )

    print(
        "SHARED_PANEL_EXPANSION_PASS"
    )

    print(
        "systems =",
        len(
            system_ids
        )
    )

    print(
        "panel_queries =",
        len(
            panel
        )
    )

    print(
        "MAS executions =",
        len(
            out
        )
    )

    print(
        "output =",
        path
    )


if __name__ == "__main__":
    main()
