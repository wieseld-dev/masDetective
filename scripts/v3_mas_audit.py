#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/eqs_v3_mas")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "audit_report.txt"

KEYWORDS = (
    "query", "prompt", "question", "completion", "response", "answer",
    "history", "reward", "utility", "advantage", "belief", "prob",
    "score", "margin", "auc", "brier", "logloss",
    "k", "num_agent", "agent_count", "role", "roles",
    "mas", "session", "round", "candidate", "task", "topology",
    "frame", "label", "target", "true"
)


class Logger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def p(self, *xs):
        s = " ".join(str(x) for x in xs)
        print(s, flush=True)
        print(s, file=self.f, flush=True)

    def close(self):
        self.f.close()


L = Logger(REPORT)


def short(obj, limit=5000):
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        s = repr(obj)

    if len(s) > limit:
        s = s[:limit] + "\n... [TRUNCATED]"

    return s


def flatten_paths(obj, prefix="", depth=0, max_depth=6):
    out = []

    if depth > max_depth:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)

            out.append((path, type(v).__name__))

            out.extend(
                flatten_paths(
                    v,
                    path,
                    depth + 1,
                    max_depth
                )
            )

    elif isinstance(obj, list) and obj:
        path = f"{prefix}[]"
        out.append((path, type(obj[0]).__name__))

        out.extend(
            flatten_paths(
                obj[0],
                path,
                depth + 1,
                max_depth
            )
        )

    return out


def interesting(path):
    x = path.lower()

    for k in KEYWORDS:
        if k in x:
            return True

    return False


def inspect_jsonl(path, max_rows=300):
    L.p()
    L.p("=" * 100)
    L.p("JSONL:", path)
    L.p("=" * 100)

    if not path.exists():
        L.p("MISSING")
        return

    L.p("SIZE_BYTES:", path.stat().st_size)

    rows = []
    bad = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                bad += 1
                if bad <= 3:
                    L.p("BAD JSON:", repr(e))

    L.p("PARSED_SAMPLE_ROWS:", len(rows))
    L.p("BAD_ROWS:", bad)

    if not rows:
        return

    top_keys = Counter()
    nested = Counter()
    type_map = defaultdict(Counter)

    for row in rows:
        if isinstance(row, dict):
            top_keys.update(row.keys())

        for p, typ in flatten_paths(row):
            nested[p] += 1
            type_map[p][typ] += 1

    L.p()
    L.p("TOP-LEVEL KEYS:")
    for k, n in top_keys.most_common():
        L.p(f"  {k:40s} {n}")

    L.p()
    L.p("INTERESTING NESTED PATHS:")
    for p, n in sorted(
        nested.items(),
        key=lambda kv: (-kv[1], kv[0])
    ):
        if interesting(p):
            types = ",".join(type_map[p].keys())
            L.p(f"  {p:70s} n={n:4d} type={types}")

    L.p()
    L.p("SAMPLE ROW 0:")
    L.p(short(rows[0]))

    if len(rows) > 1:
        L.p()
        L.p("SAMPLE ROW 1:")
        L.p(short(rows[1]))


def inspect_json(path):
    L.p()
    L.p("=" * 100)
    L.p("JSON:", path)
    L.p("=" * 100)

    if not path.exists():
        L.p("MISSING")
        return

    L.p("SIZE_BYTES:", path.stat().st_size)

    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        L.p("LOAD FAILED:", repr(e))
        return

    L.p("TOP TYPE:", type(obj).__name__)

    if isinstance(obj, dict):
        L.p("TOP KEYS:", sorted(obj.keys()))

        L.p()
        L.p("INTERESTING PATHS:")
        for p, typ in flatten_paths(obj):
            if interesting(p):
                L.p(f"  {p:70s} type={typ}")

        # Show one useful sample without dumping entire manifest.
        L.p()
        L.p("SAMPLE:")

        sample = {}
        for i, (k, v) in enumerate(obj.items()):
            sample[k] = v
            if i >= 1:
                break

        L.p(short(sample))

    elif isinstance(obj, list):
        L.p("LENGTH:", len(obj))

        if obj:
            L.p()
            L.p("ELEMENT 0 PATHS:")

            for p, typ in flatten_paths(obj[0]):
                if interesting(p):
                    L.p(f"  {p:70s} type={typ}")

            L.p()
            L.p("ELEMENT 0:")
            L.p(short(obj[0]))


def inventory():
    L.p()
    L.p("=" * 100)
    L.p("ARTIFACT INVENTORY")
    L.p("=" * 100)

    if not ROOT.exists():
        L.p("ROOT DOES NOT EXIST:", ROOT)
        return

    files = []

    for p in ROOT.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(ROOT)
                depth = len(rel.parts)

                if depth <= 4:
                    files.append(
                        (str(rel), p.stat().st_size)
                    )
            except Exception:
                pass

    for name, size in sorted(files):
        L.p(f"{size:12d}  {name}")


def scan_scripts():
    L.p()
    L.p("=" * 100)
    L.p("RELEVANT CODE CONTEXT")
    L.p("=" * 100)

    patterns = [
        "train_rows.append",
        "candidate_index",
        "completion",
        "new_k",
        "new_roles",
        "frozen_belief",
        "manifest_privileged",
        "states_train",
        "reward",
        "RewardBreakdown",
        "predict_proba",
    ]

    script_paths = [
        Path("scripts/v14_adaptive_generate_round_v1.py"),
        Path("scripts/v14_adaptive_grpo_update_v1.py"),
    ]

    # Also inspect nearby V14 scripts because execute/score may hold
    # the exact response + utility schema.
    for p in Path("scripts").glob("v14*.py"):
        if p not in script_paths:
            script_paths.append(p)

    shown = 0
    max_blocks = 80

    for path in sorted(set(script_paths)):

        if not path.exists():
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()
        except Exception:
            continue

        matches = []

        for i, line in enumerate(lines):
            if any(
                pat.lower() in line.lower()
                for pat in patterns
            ):
                matches.append(i)

        if not matches:
            continue

        L.p()
        L.p("-" * 100)
        L.p("FILE:", path)
        L.p("-" * 100)

        # Merge nearby matches into blocks.
        blocks = []

        for idx in matches:
            start = max(0, idx - 8)
            end = min(len(lines), idx + 13)

            if blocks and start <= blocks[-1][1]:
                blocks[-1] = (
                    blocks[-1][0],
                    max(blocks[-1][1], end)
                )
            else:
                blocks.append((start, end))

        for start, end in blocks:
            if shown >= max_blocks:
                break

            L.p(
                f"\n>>> {path}:{start + 1}-{end}"
            )

            for j in range(start, end):
                L.p(f"{j + 1:5d}: {lines[j]}")

            shown += 1

        if shown >= max_blocks:
            break


def discover_other_data():
    L.p()
    L.p("=" * 100)
    L.p("OTHER CANDIDATE DATA FILES")
    L.p("=" * 100)

    patterns = [
        "*.jsonl",
        "*.json",
        "*.csv",
        "*.joblib",
        "*.pkl",
        "*.pt",
    ]

    hits = []

    artifact_root = Path("artifacts")

    if artifact_root.exists():
        for pattern in patterns:
            for p in artifact_root.rglob(pattern):
                name = str(p).lower()

                if any(
                    word in name
                    for word in (
                        "v14",
                        "adaptive",
                        "candidate",
                        "execute",
                        "score",
                        "belief",
                        "state",
                        "eval"
                    )
                ):
                    try:
                        hits.append(
                            (
                                p.stat().st_mtime,
                                p.stat().st_size,
                                str(p)
                            )
                        )
                    except Exception:
                        pass

    # newest + most relevant
    hits = sorted(set(hits), reverse=True)

    for _, size, path in hits[:150]:
        L.p(f"{size:12d}  {path}")


def main():
    L.p("=" * 100)
    L.p("V3-MAS DATA AUDIT")
    L.p("=" * 100)
    L.p("PWD:", os.getcwd())
    L.p("ROOT:", ROOT.resolve())

    inventory()

    inspect_jsonl(
        ROOT / "states_train_round_06.jsonl"
    )

    inspect_jsonl(
        ROOT / "states_dev_round_00.jsonl"
    )

    inspect_json(
        ROOT / "manifest_privileged.json"
    )

    discover_other_data()
    scan_scripts()

    L.p()
    L.p("=" * 100)
    L.p("AUDIT COMPLETE")
    L.p("REPORT:", REPORT)
    L.p("=" * 100)

    L.close()


if __name__ == "__main__":
    main()
