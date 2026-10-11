#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROOT = Path("artifacts/general_role_sensor_v1")
MANIFEST_PATH = ROOT / "manifest_privileged.json"
EXECUTIONS_PATH = ROOT / "executions_manifest.jsonl"

JOURNAL_PATH = ROOT / "collection_journal_training_eval_only.jsonl"
PUBLIC_PATH = ROOT / "target_public_responses.jsonl"
TRUTH_PATH = ROOT / "target_truth_training_eval_only.jsonl"
ERROR_PATH = ROOT / "collection_errors.jsonl"
RUN_CONFIG_PATH = ROOT / "collection_run_config.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256(
        "\0".join(map(str, parts)).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path):
    if not path.exists():
        return []

    rows = []
    bad = 0

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
                print(
                    f"[WARN] malformed JSONL line ignored: "
                    f"{path}:{line_no}",
                    flush=True,
                )

    if bad:
        print(f"[WARN] malformed lines ignored: {bad}", flush=True)

    return rows


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def rewrite_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)


def load_manifest():
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    executions = read_jsonl(EXECUTIONS_PATH)

    if len(executions) != 3888:
        raise RuntimeError(
            f"execution manifest has {len(executions)} rows, expected 3888"
        )

    systems = {
        str(x["system_id"]): x
        for x in manifest["systems"]
    }

    if len(systems) != 432:
        raise RuntimeError(
            f"system manifest has {len(systems)} rows, expected 432"
        )

    return manifest, executions, systems


def execution_id(manifest_sha: str, row: dict) -> str:
    return stable_hash(
        manifest_sha,
        row["system_id"],
        row["target_role"],
        row["probe_id"],
        row["query"],
    )


def blind_system_id(manifest_sha: str, system_id: str) -> str:
    return stable_hash(
        "PUBLIC_SYSTEM",
        manifest_sha,
        system_id,
    )[:20]


def select_pilot(rows):
    """
    Exactly one execution from every:
      split x pair configuration x branch

    = 2 splits x 3 pair configs x 2 branches
    = 12 executions.
    """
    chosen = {}

    for row in rows:
        key = (
            row["split"],
            int(row["pair_index"]),
            row["branch"],
        )
        if key not in chosen:
            chosen[key] = row

    expected = 2 * 3 * 2

    if len(chosen) != expected:
        raise RuntimeError(
            f"pilot selector found {len(chosen)}, expected {expected}"
        )

    return [
        chosen[k]
        for k in sorted(chosen)
    ]


def export_views(journal_rows, manifest_sha):
    # Deduplicate journal by execution_id; latest complete row wins.
    by_id = {}

    for row in journal_rows:
        if row.get("execution_status") == "ok":
            by_id[str(row["execution_id"])] = row

    ordered = sorted(
        by_id.values(),
        key=lambda x: int(x["manifest_position"]),
    )

    public_rows = []
    truth_rows = []

    for r in ordered:
        public_rows.append(
            {
                "execution_id": r["execution_id"],
                "split": r["split"],
                "candidate_role": r["target_role"],
                "candidate_family": r["target_family"],
                "probe_id": r["probe_id"],
                "probe_source": r["probe_source"],
                "query": r["query"],
                "public_system_id": blind_system_id(
                    manifest_sha,
                    r["system_id"],
                ),
                "final_output": r["final_output"],
            }
        )

        truth_rows.append(
            {
                "execution_id": r["execution_id"],
                "split": r["split"],
                "target_role": r["target_role"],
                "target_family": r["target_family"],
                "pair_index": r["pair_index"],
                "branch": r["branch"],
                "installed_label": r["installed_label"],
                "system_id_training_eval_only": r["system_id"],
                "installed_roles_training_eval_only":
                    r["installed_roles_training_eval_only"],
                "selected_roles_training_eval_only":
                    r["selected_roles_training_eval_only"],
                "target_activated_training_eval_only":
                    r["target_activated_training_eval_only"],
                "execution_mode_training_eval_only":
                    r["execution_mode"],
                "style_training_eval_only":
                    r["style"],
                "k_training_eval_only":
                    r["k"],
                "retries_used": r["retries_used"],
            }
        )

    rewrite_jsonl(PUBLIC_PATH, public_rows)
    rewrite_jsonl(TRUTH_PATH, truth_rows)

    return public_rows, truth_rows


def execute_one(
    *,
    manifest_position: int,
    row: dict,
    system: dict,
    manifest_sha: str,
    model: str,
    retries: int,
):
    # GENERAL experiment: register ALL 72 Role-Bank hidden prompts
    # before constructing the target.
    from targets.general_role_bank_v1 import (
        register_general_role_bank,
    )
    register_general_role_bank()

    from targets.single_router_fanout_mas import (
        SingleRouterFanoutMAS,
        SingleRouterSystemSpec,
    )

    eid = execution_id(
        manifest_sha,
        row,
    )

    spec = SingleRouterSystemSpec(
        system_id=str(system["system_id"]),
        split=f"general_sensor_{row['split']}",
        execution_mode=str(system["execution_mode"]),
        style=str(system["style"]),
        worker_roles=tuple(
            map(str, system["worker_roles"])
        ),
    )

    last_error = None

    for attempt in range(retries + 1):
        try:
            target = SingleRouterFanoutMAS(
                spec,
                model=model,
            )

            output = str(
                target.query(str(row["query"]))
            ).strip()

            if not output:
                raise RuntimeError("MAS returned empty output")

            tr = getattr(target, "_last_trace", None)

            if tr is None:
                raise RuntimeError(
                    "SingleRouterFanoutMAS did not expose _last_trace"
                )

            selected = list(
                map(
                    str,
                    getattr(tr, "selected_roles", ()),
                )
            )

            true_roles = list(
                map(
                    str,
                    getattr(
                        tr,
                        "true_roles",
                        tuple(system["worker_roles"]),
                    ),
                )
            )

            target_role = str(row["target_role"])

            return {
                "execution_id": eid,
                "manifest_position": manifest_position,
                "execution_status": "ok",

                "split": row["split"],
                "target_role": target_role,
                "target_family": row["target_family"],

                "pair_index": int(row["pair_index"]),
                "branch": row["branch"],
                "installed_label": int(row["installed_label"]),

                "probe_id": row["probe_id"],
                "probe_source": row["probe_source"],
                "selection_type": row.get("selection_type"),
                "canonical_anchor_index":
                    row.get("canonical_anchor_index"),

                "query": row["query"],
                "final_output": output,

                "system_id": row["system_id"],
                "execution_mode": row["execution_mode"],
                "style": row["style"],
                "k": int(row["k"]),

                "installed_roles_training_eval_only": true_roles,
                "selected_roles_training_eval_only": selected,
                "target_activated_training_eval_only":
                    target_role in selected,

                "retries_used": attempt,
                "model": model,
                "manifest_sha256": manifest_sha,
            }

        except Exception as e:
            last_error = e

            if attempt >= retries:
                break

            time.sleep(
                min(20.0, 2.0 ** attempt)
            )

    raise RuntimeError(
        f"{eid}: {type(last_error).__name__}: {last_error}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--model",
        default=None,
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
    )
    ap.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
    )

    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set"
        )

    model = (
        args.model
        or os.environ.get("MAS_MODEL")
        or "gpt-4o-mini"
    )

    manifest, executions, systems = load_manifest()

    manifest_sha = sha256_file(
        MANIFEST_PATH
    )
    executions_sha = sha256_file(
        EXECUTIONS_PATH
    )

    config = {
        "dataset_version": "general_role_sensor_v1",
        "manifest_sha256": manifest_sha,
        "executions_sha256": executions_sha,
        "model": model,
        "workers": int(args.workers),
        "retries": int(args.retries),
    }

    # Refuse to accidentally continue with changed experimental data/model.
    if RUN_CONFIG_PATH.exists():
        old = json.loads(
            RUN_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        for key in (
            "dataset_version",
            "manifest_sha256",
            "executions_sha256",
            "model",
        ):
            if old.get(key) != config.get(key):
                raise SystemExit(
                    f"Collection config mismatch for {key}: "
                    f"old={old.get(key)!r} new={config.get(key)!r}"
                )
    else:
        RUN_CONFIG_PATH.write_text(
            json.dumps(
                config,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # Number positions before filtering so IDs/order remain globally stable.
    indexed = list(
        enumerate(executions)
    )

    if args.split != "all":
        indexed = [
            (i, r)
            for i, r in indexed
            if r["split"] == args.split
        ]

    if args.pilot:
        selected_rows = select_pilot(
            [r for _, r in indexed]
        )
        selected_ids = {
            execution_id(
                manifest_sha,
                r,
            )
            for r in selected_rows
        }

        indexed = [
            (i, r)
            for i, r in indexed
            if execution_id(
                manifest_sha,
                r,
            )
            in selected_ids
        ]

    existing_rows = read_jsonl(
        JOURNAL_PATH
    )

    completed = {
        str(r["execution_id"])
        for r in existing_rows
        if r.get("execution_status") == "ok"
    }

    pending = []

    for pos, row in indexed:
        eid = execution_id(
            manifest_sha,
            row,
        )

        if eid in completed:
            continue

        sid = str(
            row["system_id"]
        )

        if sid not in systems:
            raise RuntimeError(
                f"missing system {sid}"
            )

        pending.append(
            (
                pos,
                row,
                systems[sid],
            )
        )

    print("=" * 88, flush=True)
    print("GENERAL SENSOR MAS COLLECTION V1", flush=True)
    print("=" * 88, flush=True)
    print("model:", model, flush=True)
    print("pilot:", args.pilot, flush=True)
    print("split:", args.split, flush=True)
    print("workers:", args.workers, flush=True)
    print("already complete:", len(completed), flush=True)
    print("selected this run:", len(indexed), flush=True)
    print("pending this run:", len(pending), flush=True)
    print("manifest sha:", manifest_sha, flush=True)
    print("=" * 88, flush=True)

    failures = []
    new_ok = 0
    activation = Counter()
    branch_counts = Counter()

    with ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as ex:
        futs = {}

        for pos, row, system in pending:
            fut = ex.submit(
                execute_one,
                manifest_position=pos,
                row=row,
                system=system,
                manifest_sha=manifest_sha,
                model=model,
                retries=args.retries,
            )
            futs[fut] = (pos, row)

        total_pending = len(futs)

        for done_i, fut in enumerate(
            as_completed(futs),
            1,
        ):
            pos, source = futs[fut]

            try:
                result = fut.result()

                append_jsonl(
                    JOURNAL_PATH,
                    result,
                )

                new_ok += 1

                activation[
                    bool(
                        result[
                            "target_activated_training_eval_only"
                        ]
                    )
                ] += 1

                branch_counts[
                    result["branch"]
                ] += 1

                if (
                    done_i <= 20
                    or done_i % 25 == 0
                    or done_i == total_pending
                ):
                    print(
                        f"[{done_i:04d}/{total_pending:04d}] "
                        f"OK "
                        f"{result['split']} "
                        f"{result['target_role']} "
                        f"{result['probe_id']} "
                        f"{result['branch']} "
                        f"act={result['target_activated_training_eval_only']} "
                        f"retry={result['retries_used']}",
                        flush=True,
                    )

            except Exception as e:
                err = {
                    "manifest_position": pos,
                    "target_role": source["target_role"],
                    "probe_id": source["probe_id"],
                    "system_id": source["system_id"],
                    "branch": source["branch"],
                    "error": repr(e),
                    "time": time.time(),
                }

                append_jsonl(
                    ERROR_PATH,
                    err,
                )

                failures.append(err)

                print(
                    f"[{done_i:04d}/{total_pending:04d}] "
                    f"FAILED "
                    f"{source['target_role']} "
                    f"{source['probe_id']} "
                    f"{source['branch']} | {e}",
                    flush=True,
                )

    journal_rows = read_jsonl(
        JOURNAL_PATH
    )

    public_rows, truth_rows = export_views(
        journal_rows,
        manifest_sha,
    )

    print()
    print("=" * 88, flush=True)
    print("RUN SUMMARY", flush=True)
    print("=" * 88, flush=True)
    print("new OK:", new_ok, flush=True)
    print("failures:", len(failures), flush=True)
    print("public rows total:", len(public_rows), flush=True)
    print("truth rows total:", len(truth_rows), flush=True)
    print("new branch counts:", dict(branch_counts), flush=True)
    print("new target activation:", dict(activation), flush=True)
    print("=" * 88, flush=True)

    if failures:
        raise SystemExit(
            f"{len(failures)} executions remain incomplete; rerun resumes them."
        )


if __name__ == "__main__":
    main()
