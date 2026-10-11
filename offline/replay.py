from __future__ import annotations
from typing import List, Sequence, Tuple
from .real_dataset import RealRolePresenceDataset

History = List[Tuple[str, str]]

def replay_subset(dataset: RealRolePresenceDataset, system_id: str, probe_ids: Sequence[str]) -> History:
    """Replay already-collected outputs. Makes no API calls."""
    history: History = []
    seen = set()
    for probe_id in probe_ids:
        if probe_id in seen:
            raise ValueError(f'Duplicate probe in replay: {probe_id}')
        seen.add(probe_id)
        interaction = dataset.get_output(system_id, probe_id)
        history.append((probe_id, interaction.output))
    return history
