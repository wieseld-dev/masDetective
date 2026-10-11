from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class PublicInteraction:
    system_id: str
    probe_id: str
    target_role: str
    probe_text: str
    output: str


@dataclass(frozen=True)
class SystemTruth:
    """Training/evaluation-only truth. Never pass this object to a policy."""
    system_id: str
    comp_id: str
    style: str
    k_true: int
    roles: Tuple[str, ...]


class RealRolePresenceDataset:
    """Loader for the 300 already-collected real MAS interactions."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.probe_text: Dict[str, str] = {}
        self.probe_role: Dict[str, str] = {}
        self.outputs: Dict[Tuple[str, str], PublicInteraction] = {}
        self.truth: Dict[str, SystemTruth] = {}
        self.target_present: Dict[Tuple[str, str], int] = {}
        self._load_probe_bank()
        self._load_outputs()
        self._load_truth()
        self.validate_complete_grid()

    def _load_probe_bank(self) -> None:
        with (self.root / 'probe_bank.json').open('r', encoding='utf-8') as f:
            raw = json.load(f)
        for role, probes in raw.items():
            for probe in probes:
                pid = probe['probe_id']
                self.probe_text[pid] = probe['text']
                self.probe_role[pid] = role

    def _load_outputs(self) -> None:
        with (self.root / 'public_outputs.csv').open('r', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                pid = row['probe_id']
                sid = row['system_id']
                if pid not in self.probe_text:
                    raise ValueError(f'Unknown probe_id in outputs: {pid}')
                self.outputs[(sid, pid)] = PublicInteraction(
                    system_id=sid,
                    probe_id=pid,
                    target_role=row['target_role'],
                    probe_text=self.probe_text[pid],
                    output=row['output'],
                )

    def _load_truth(self) -> None:
        by_system: Dict[str, SystemTruth] = {}
        with (self.root / 'truth.csv').open('r', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                sid = row['system_id']
                truth = SystemTruth(
                    system_id=sid,
                    comp_id=row['comp_id'],
                    style=row['style'],
                    k_true=int(row['K']),
                    roles=tuple(row['roles'].split('|')),
                )
                old = by_system.get(sid)
                if old is not None and old != truth:
                    raise ValueError(f'Inconsistent system truth for {sid}')
                by_system[sid] = truth
                self.target_present[(sid, row['probe_id'])] = int(row['target_present'])
        self.truth = by_system

    @property
    def system_ids(self) -> List[str]:
        return sorted(self.truth)

    @property
    def probe_ids(self) -> List[str]:
        return sorted(self.probe_text)

    @property
    def roles(self) -> List[str]:
        return sorted(set(self.probe_role.values()))

    @property
    def comp_ids(self) -> List[str]:
        return sorted({t.comp_id for t in self.truth.values()})

    def systems_for_comps(self, comp_ids: Iterable[str]) -> List[str]:
        wanted = set(comp_ids)
        return [sid for sid in self.system_ids if self.truth[sid].comp_id in wanted]

    def get_output(self, system_id: str, probe_id: str) -> PublicInteraction:
        return self.outputs[(system_id, probe_id)]

    def get_truth(self, system_id: str) -> SystemTruth:
        return self.truth[system_id]

    def validate_complete_grid(self) -> None:
        missing = [(sid, pid) for sid in self.system_ids for pid in self.probe_ids if (sid, pid) not in self.outputs]
        if missing:
            raise ValueError(f'Replay grid incomplete: {len(missing)} missing; examples={missing[:10]}')
