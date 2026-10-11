from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import ANCHOR_PROBE, primary_probe
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo_v65 import get_v65_split


def _load_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.add((str(r["system_id"]), str(r["focus_role"])))
    return out


def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _plan(specs, candidates) -> tuple[int, int]:
    interactions = len(specs) * (1 + len(candidates))
    calls = sum(s.estimated_calls_per_query * (1 + len(candidates)) for s in specs)
    return interactions, calls


def _collect(specs, candidates, public_path: Path, truth_path: Path) -> None:
    pub_done = _load_keys(public_path)
    truth_done = _load_keys(truth_path)
    if pub_done != truth_done:
        raise RuntimeError("public/truth resume logs are inconsistent; do not continue until repaired")
    done = pub_done

    activation_present = [0, 0]  # hits, total
    activation_absent = [0, 0]

    for si, spec in enumerate(specs, 1):
        print(f"[{si}/{len(specs)}] {spec.system_id} mode={spec.execution_mode} K={spec.k_true}", flush=True)
        target = SingleRouterFanoutMAS(spec)
        items = [("__anchor__", ANCHOR_PROBE)] + [(r, primary_probe(r)) for r in candidates]
        for focus, query in items:
            key = (spec.system_id, focus)
            if key in done:
                continue
            output = target.query(query)
            tr = target.privileged_last_trace()
            _append_jsonl(public_path, {
                "system_id": spec.system_id,
                "focus_role": focus,
                "query": query,
                "output": output,
            })
            _append_jsonl(truth_path, {
                "system_id": spec.system_id,
                "focus_role": focus,
                "true_roles": list(spec.worker_roles),
                "selected_roles": list(tr.selected_roles),
                "mode": spec.execution_mode,
            })
            done.add(key)

            if focus != "__anchor__":
                present = focus in set(spec.worker_roles)
                selected = focus in set(tr.selected_roles)
                if present:
                    activation_present[1] += 1
                    activation_present[0] += int(selected)
                else:
                    activation_absent[1] += 1
                    activation_absent[0] += int(selected)

    if activation_present[1]:
        print(f"new-run targeted activation TPR={activation_present[0]/activation_present[1]:.3f}")
    if activation_absent[1]:
        print(f"new-run targeted activation FPR={activation_absent[0]/activation_absent[1]:.3f}")
    print("saved public:", public_path)
    print("saved truth :", truth_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect diverse unseen-composition data for V6.5")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--train-public", default="artifacts/v65_train_public.jsonl")
    ap.add_argument("--train-truth", default="artifacts/v65_train_truth.jsonl")
    ap.add_argument("--val-public", default="artifacts/v65_val_public.jsonl")
    ap.add_argument("--val-truth", default="artifacts/v65_val_truth.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    candidates = list(profile["candidate_roles"])
    wanted = []
    if args.split in {"train", "both"}:
        wanted.append(("train", get_v65_split("train"), Path(args.train_public), Path(args.train_truth)))
    if args.split in {"val", "both"}:
        wanted.append(("val", get_v65_split("val"), Path(args.val_public), Path(args.val_truth)))

    print("=== V6.5 DIVERSE COMPOSITION COLLECTION PLAN ===")
    print("candidate roles:", candidates)
    total_i = total_c = 0
    for name, specs, pub, truth in wanted:
        i, c = _plan(specs, candidates)
        total_i += i; total_c += c
        print(f"{name}: systems={len(specs)} interactions={i} upper-bound API calls={c}")
        print(" modes:", {m: sum(s.execution_mode == m for s in specs) for m in sorted({s.execution_mode for s in specs})})
        print(" K   :", {k: sum(s.k_true == k for s in specs) for k in sorted({s.k_true for s in specs})})
        print(" public:", pub)
        print(" truth :", truth)
    print("TOTAL external interactions:", total_i)
    print("TOTAL rough upper-bound API calls:", total_c)
    print("prompt policy: ONE fixed benign primary probe per candidate + ONE neutral anchor per MAS")
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to make API calls.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    for _, specs, pub, truth in wanted:
        _collect(specs, candidates, pub, truth)


if __name__ == "__main__":
    main()
