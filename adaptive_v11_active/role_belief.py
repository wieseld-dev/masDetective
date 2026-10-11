from __future__ import annotations

from typing import Sequence

from .belief import FrozenBeliefBundle
from .schema import EvidenceFrame


def uniform_k(bundle: FrozenBeliefBundle) -> dict[int, float]:
    return {int(k): 1.0 / len(bundle.ks) for k in bundle.ks}


def role_only_predict(
    bundle: FrozenBeliefBundle,
    history: Sequence[EvidenceFrame],
) -> tuple[dict[int, float], dict[str, float]]:
    """Return calibrated role marginals without any K/cardinality projection.

    Phase-1 Adaptive-V11 intentionally leaves K uninformative.  The v2 belief's
    role head is useful on held-out systems, while its K head is not.  Therefore
    this adapter uses only the calibrated role head and keeps P(K) uniform for
    schema compatibility and, critically, prevents a weak K estimate from
    shifting the role probabilities.
    """
    if not history:
        kp, rp = bundle.initial()
        return uniform_k(bundle), dict(rp)

    if not getattr(bundle, "_fitted", False):
        raise RuntimeError("frozen belief is not fitted")
    if getattr(bundle, "role_platt", None) is None:
        raise RuntimeError(
            "role-only adaptive probing requires frozen belief v2 with role_platt; "
            "refit artifacts/adaptive_v11_active_v1/frozen_belief.joblib first"
        )
    if not hasattr(bundle, "_raw_role_vector_with") or not hasattr(bundle, "_apply_platt"):
        raise RuntimeError("installed FrozenBeliefBundle is incompatible with role-only phase")

    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    pcal = bundle._apply_platt(bundle.role_platt, raw)
    rp = {str(r): float(pcal[i]) for i, r in enumerate(bundle.roles)}
    return uniform_k(bundle), rp
