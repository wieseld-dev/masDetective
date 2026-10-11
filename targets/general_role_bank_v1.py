from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_ROLE_BANK = Path(
    "artifacts/general_role_bank_v1/role_bank_v1.json"
)


def _sha(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _load_roles(path: str | Path = DEFAULT_ROLE_BANK):
    x = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if isinstance(x, dict):
        roles = (
            x.get("roles")
            or x.get("role_bank")
            or x.get("items")
        )
    else:
        roles = x

    if not isinstance(roles, list):
        raise RuntimeError(
            "Could not parse role bank"
        )

    return roles


def register_general_role_bank(
    path: str | Path = DEFAULT_ROLE_BANK,
):
    """
    Extend the controlled target with the new Role Bank roles.

    IMPORTANT:
    - Existing ROLE_PROMPTS are NEVER overwritten.
    - Missing Role Bank roles are registered from hidden_worker_prompt.
    - This function changes only the in-process ROLE_PROMPTS dict.
    """

    # Register historical V11 roles first.
    from targets.single_router_zoo_v11 import (
        validate_v11_design,
    )

    validate_v11_design()

    from targets.hierarchical_mas import (
        ROLE_PROMPTS,
    )

    roles = _load_roles(path)

    added = {}
    preserved = {}
    bank_roles = []

    for row in roles:
        role = str(
            _pick(
                row,
                "role",
                "name",
                "role_name",
                default="",
            )
        ).strip()

        prompt = str(
            _pick(
                row,
                "hidden_worker_prompt",
                default="",
            )
        ).strip()

        if not role:
            raise RuntimeError(
                f"Role entry missing role: {row}"
            )

        if not prompt:
            raise RuntimeError(
                f"{role}: missing hidden_worker_prompt"
            )

        bank_roles.append(role)

        if role in ROLE_PROMPTS:
            existing = str(
                ROLE_PROMPTS[role]
            ).strip()

            preserved[role] = {
                "existing_sha256": _sha(existing),
                "bank_sha256": _sha(prompt),
                "same_as_bank_prompt":
                    existing == prompt,
            }

        # GENERAL experiment is deliberately self-contained:
        # ALL 72 roles use the Role Bank hidden_worker_prompt.
        # This mutates only the current Python process.
        ROLE_PROMPTS[role] = prompt

        added[role] = {
            "sha256": _sha(prompt),
            "characters": len(prompt),
        }

    missing_after = sorted(
        set(bank_roles) - set(ROLE_PROMPTS)
    )

    if missing_after:
        raise RuntimeError(
            "Role registration incomplete: "
            + repr(missing_after)
        )

    return {
        "role_bank_roles": len(bank_roles),
        "registered_total_after": len(ROLE_PROMPTS),
        "historical_roles_preserved":
            len(preserved),
        "new_roles_added":
            len(added),
        "missing_after":
            missing_after,
        "preserved":
            preserved,
        "added":
            added,
    }
