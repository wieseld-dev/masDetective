from __future__ import annotations

from typing import Mapping, Sequence

from adaptive_v11_active.schema import EvidenceFrame


def role_only_predict(bundle, history: Sequence[EvidenceFrame]) -> dict[str, float]:
    """Frozen role posterior without the unreliable K projection/head."""
    if not history:
        _, rp = bundle.initial()
        return {str(k): float(v) for k, v in rp.items()}
    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    calibrated = bundle._apply_platt(bundle.role_platt, raw)
    return {str(r): float(p) for r, p in zip(bundle.roles, calibrated)}


def dummy_k_probs(bundle) -> dict[int, float]:
    ks = tuple(int(k) for k in bundle.ks)
    return {k: 1.0 / len(ks) for k in ks}
