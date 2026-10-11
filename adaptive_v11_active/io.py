from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_json(path: str | Path, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def sha256_json(obj: object) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_int(*parts: object, bits: int = 63) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:16], 16)
    return value & ((1 << bits) - 1)


def first_existing(paths: Sequence[str | Path]) -> Path | None:
    for x in paths:
        p = Path(x)
        if p.exists():
            return p
    return None
