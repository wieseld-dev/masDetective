from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from targets.single_router_zoo_v11 import KS, MODES, get_v11_split, validate_v11_design
from v11.role_universe import TRAIN_ROLES


def key(roles: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(map(str, roles)))


def canonical_hash(obj) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def manifest_role_sets(path: Path) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(obj, dict):
        return out
    for field in ("systems", "known_specs", "dev_specs", "train_specs", "specs"):
        rows = obj.get(field, [])
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            roles = r.get("roles", r.get("worker_roles", ()))
            if roles:
                out.add(key(roles))
    return out


def collect_forbidden(repo: Path, output: Path) -> tuple[set[tuple[str, ...]], list[str]]:
    forbidden: set[tuple[str, ...]] = set()
    sources: list[str] = []

    # Canonical V11 split(s).
    for split in ("train", "dev"):
        try:
            specs = get_v11_split(split)
        except Exception:
            continue
        n0 = len(forbidden)
        for s in specs:
            roles = getattr(s, "worker_roles", ())
            if roles:
                forbidden.add(key(roles))
        if len(forbidden) > n0:
            sources.append(f"get_v11_split({split!r})")

    # Every manifest-like JSON already under artifacts. This makes the new compositions
    # disjoint from the experiment manifests that currently exist in the repo.
    art = repo / "artifacts"
    if art.exists():
        for p in sorted(art.rglob("*.json")):
            name = p.name.lower()
            if "manifest" not in name:
                continue
            if p.resolve() == output.resolve():
                continue
            got = manifest_role_sets(p)
            if got:
                forbidden |= got
                sources.append(str(p.relative_to(repo)))
    return forbidden, sources


def choose_roles(
    rng: random.Random,
    *,
    k: int,
    forbidden: set[tuple[str, ...]],
    used: set[tuple[str, ...]],
    split_counts: Counter,
    mode_counts: Counter,
) -> tuple[str, ...]:
    best = None
    best_score = None
    roles = list(TRAIN_ROLES)
    for _ in range(4000):
        cand = tuple(rng.sample(roles, k))
        ck = key(cand)
        if ck in forbidden or ck in used:
            continue
        # Prefer role balance globally within the split and within the mode.
        score = (
            sum(split_counts[r] for r in cand),
            sum(mode_counts[r] for r in cand),
            max((split_counts[r] for r in cand), default=0),
            rng.random(),
        )
        if best_score is None or score < best_score:
            best = cand
            best_score = score
    if best is None:
        raise RuntimeError(f"could not find fresh composition for K={k}")
    return best


def build_split(
    *,
    split: str,
    per_cell: int,
    seed: int,
    forbidden: set[tuple[str, ...]],
    used: set[tuple[str, ...]],
) -> list[dict]:
    rng = random.Random(seed)
    split_counts = Counter({r: 0 for r in TRAIN_ROLES})
    mode_counts = {m: Counter({r: 0 for r in TRAIN_ROLES}) for m in MODES}
    rows: list[dict] = []
    styles = ("s1", "s2", "s3")

    cell_i = 0
    for mode in MODES:
        for k in KS:
            for j in range(per_cell):
                rr = choose_roles(
                    rng,
                    k=int(k),
                    forbidden=forbidden,
                    used=used,
                    split_counts=split_counts,
                    mode_counts=mode_counts[mode],
                )
                ck = key(rr)
                used.add(ck)
                split_counts.update(rr)
                mode_counts[mode].update(rr)
                sid_payload = {
                    "split": split, "mode": mode, "k": int(k),
                    "roles": list(rr), "j": j, "seed": seed,
                }
                sid = "av11kb_" + canonical_hash(sid_payload)[:16]
                rows.append({
                    "session_id": sid,
                    "split": split,
                    "mode": str(mode),
                    "style": styles[(cell_i + j) % len(styles)],
                    "k": int(k),
                    "roles": list(rr),
                })
            cell_i += 1

    return rows


def main():
    ap = argparse.ArgumentParser(description="Create fresh balanced K-bootstrap train/dev MAS manifests")
    ap.add_argument("--output", default="artifacts/adaptive_v11_k_bootstrap_v1/manifest_privileged.json")
    ap.add_argument("--seed", type=int, default=2026091701)
    ap.add_argument("--train-per-cell", type=int, default=6, help="4 modes x 4 K x this = train systems")
    ap.add_argument("--dev-per-cell", type=int, default=2, help="4 modes x 4 K x this = dev systems")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    validate_v11_design()
    repo = Path.cwd()
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; refusing to replace it without --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)

    forbidden, sources = collect_forbidden(repo, out)
    used: set[tuple[str, ...]] = set()

    train = build_split(
        split="train", per_cell=args.train_per_cell, seed=args.seed + 11,
        forbidden=forbidden, used=used,
    )
    dev = build_split(
        split="dev", per_cell=args.dev_per_cell, seed=args.seed + 29,
        forbidden=forbidden, used=used,
    )
    systems = train + dev

    assert len(train) == len(MODES) * len(KS) * args.train_per_cell
    assert len(dev) == len(MODES) * len(KS) * args.dev_per_cell
    assert not ({key(r["roles"]) for r in train} & {key(r["roles"]) for r in dev})
    assert not ({key(r["roles"]) for r in systems} & forbidden)

    payload_for_hash = {
        "version": "av11-k-bootstrap-manifest-v1",
        "seed": args.seed,
        "systems": systems,
    }
    obj = {
        **payload_for_hash,
        "sha256": canonical_hash(payload_for_hash),
        "notes": {
            "purpose": "K-signal bootstrap with frozen V11 generator and frozen V13.4/V13.5 judges",
            "train_systems": len(train),
            "dev_systems": len(dev),
            "balanced_cells": "mode x K",
            "forbidden_compositions": len(forbidden),
            "forbidden_sources": sources,
        },
    }
    out.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

    for split, rows in (("train", train), ("dev", dev)):
        print(f"[{split}] systems={len(rows)}")
        print("[mode]", dict(Counter(r["mode"] for r in rows)))
        print("[K]", dict(Counter(r["k"] for r in rows)))
        counts = Counter(x for r in rows for x in r["roles"])
        print("[role-minmax]", min(counts.values()), max(counts.values()))
    print(f"[forbidden] compositions={len(forbidden)} sources={len(sources)}")
    print(f"[done] {out} sha256={obj['sha256']}")


if __name__ == "__main__":
    main()
