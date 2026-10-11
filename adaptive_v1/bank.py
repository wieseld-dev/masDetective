from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .belief import BankSystem
from .schema import Action, EvidenceObservation


def read_jsonl(path: str | Path) -> list[dict]:
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(dict(row),ensure_ascii=False)+"\n")
    tmp.replace(p)


def load_bank(path: str | Path, roles: Sequence[str] | None = None) -> list[BankSystem]:
    by=defaultdict(list); truth={}; split={}; meta={}
    for row in read_jsonl(path):
        sid=str(row["system_id"]); tr=tuple(map(str,row.get("truth_roles",())))
        if sid in truth and set(tr)!=set(truth[sid]): raise RuntimeError(f"inconsistent truth for {sid}")
        truth[sid]=tr; split[sid]=str(row.get("split","train")); meta[sid]=dict(row.get("system_metadata",{}))
        ev={str(k):float(v) for k,v in dict(row["evidence"]).items()}
        if roles is not None:
            missing=set(roles)-set(ev)
            if missing: raise RuntimeError(f"{sid} observation missing roles: {sorted(missing)}")
        by[sid].append(EvidenceObservation(
            action=Action.from_dict(row["action"]),query=str(row.get("query","")),evidence=ev,
            system_id=sid,response_id=str(row.get("response_id","")),method=str(row.get("method","v11")),metadata=dict(row.get("metadata",{})),
        ))
    return [BankSystem(sid,truth[sid],by[sid],split[sid],meta[sid]) for sid in sorted(by)]


def split_bank(systems: Sequence[BankSystem], train_name: str="train", dev_name: str="dev") -> tuple[list[BankSystem],list[BankSystem]]:
    return [s for s in systems if s.split==train_name],[s for s in systems if s.split==dev_name]
