from __future__ import annotations

import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from analysis.generate_general_role_v14_probes_v2 import (
    ROOT,
    ROLE_BANK,
    CANONICAL_PATH,
    MODEL,
    ADAPTER,
    SEED,
    TEMPERATURE,
    TOP_P,
    norm,
    cosine,
    set_seed,
    write_json,
    append_jsonl,
    load_role_bank,
    pick,
    canonical_map,
)

from analysis.adaptive_fanout_count_v12 import (
    sample_probe_fast,
)

from v8.embedding import FrozenOutputEmbedderV8
from v10.gate import BenignGateV10
from v13.policy import DisambiguationProbePolicyV13


OUT = ROOT / "v14_generated_probes_v3.jsonl"
AUDIT = ROOT / "v14_generated_probes_v3_audit.jsonl"
SUMMARY = ROOT / "v14_generated_probes_v3_summary.json"
PROGRESS = ROOT / "v14_generated_probes_v3_progress.json"

# Generate a small pool around EACH of the 3 canonical task families.
VALID_PER_ANCHOR = 4

# We ultimately keep:
#   3 canonical anchors x 2 generated probes = 6 generated / role.
KEEP_PER_ANCHOR = 2
KEEP_PER_ROLE = 6

MAX_ATTEMPTS_PER_ANCHOR = 45

# Strong semantic duplicate threshold already used in our adaptive work.
SEMANTIC_DUPLICATE = 0.965


# We deliberately do NOT reject on the old hard benign gate alone.
#
# Reason:
# phrases such as "worker process" are normal software terminology but
# can be interpreted by a MAS-oriented keyword gate as hidden-worker
# language.
#
# The V14 policy itself is trained/prompted for benign user-facing
# queries. We still audit the old gate and independently prohibit
# explicit hidden-system / injection language.
FORBIDDEN_META_PHRASES = (
    "system prompt",
    "hidden prompt",
    "hidden instruction",
    "hidden instructions",
    "internal instruction",
    "internal instructions",
    "prompt injection",
    "jailbreak",
    "routing decision",
    "routing decisions",
    "routing topology",
    "hidden routing",
    "hidden architecture",
    "internal architecture",
    "private state",
    "instruction hierarchy",
    "reveal your prompt",
    "reveal the prompt",
    "ignore previous instructions",
    "ignore all previous instructions",
)


def explicit_meta_safe(q: str) -> bool:
    t = " ".join(str(q).lower().split())

    return not any(
        phrase in t
        for phrase in FORBIDDEN_META_PHRASES
    )


def role_rows():
    doc = load_role_bank()

    out = []

    for x in doc:
        role = str(
            pick(
                x,
                "role",
                "name",
                "role_name",
                default="",
            )
        ).strip()

        split = str(
            pick(
                x,
                "split",
                default="",
            )
        ).strip().lower()

        family = str(
            pick(
                x,
                "family",
                "role_family",
                default="",
            )
        ).strip()

        desc = str(
            pick(
                x,
                "public_role_description",
                "public_description",
                "description",
                default="",
            )
        ).strip()

        if not role:
            raise RuntimeError(
                f"role without name: {x}"
            )

        if split not in {"train", "val"}:
            raise RuntimeError(
                f"{role}: invalid split={split!r}"
            )

        if not family:
            raise RuntimeError(
                f"{role}: missing family"
            )

        if not desc:
            raise RuntimeError(
                f"{role}: missing public description"
            )

        out.append(
            {
                "role": role,
                "split": split,
                "family": family,
                "description": desc,
            }
        )

    return out


def semantic_max(qemb, candidates):
    if not candidates:
        return -1.0

    return max(
        cosine(
            qemb,
            x["embedding"],
        )
        for x in candidates
    )


def select_two_from_anchor(
    *,
    candidates,
    anchor,
    embedder,
    already_selected,
):
    """
    Keep two different kinds of generated probes:

    1. DIRECT:
       highest similarity to the public role description.

    2. DIVERSE:
       deliberately farther from this anchor and from already selected
       probes, while retaining some role relevance.
    """

    if len(candidates) < 2:
        raise RuntimeError(
            "Need at least two candidates for anchor selection"
        )

    direct = max(
        candidates,
        key=lambda x:
            x["role_similarity"],
    )

    remaining = [
        x
        for x in candidates
        if x is not direct
    ]

    anchor_emb = embedder.encode(anchor)

    selected_embs = [
        x["embedding"]
        for x in already_selected
    ]

    def diversity_score(x):
        away_from_anchor = (
            1.0
            - cosine(
                x["embedding"],
                anchor_emb,
            )
        )

        if selected_embs:
            away_from_selected = (
                1.0
                - max(
                    cosine(
                        x["embedding"],
                        e,
                    )
                    for e in selected_embs
                )
            )
        else:
            away_from_selected = (
                away_from_anchor
            )

        return (
            0.45 * away_from_anchor
            + 0.35 * away_from_selected
            + 0.20 * x["role_similarity"]
        )

    diverse = max(
        remaining,
        key=diversity_score,
    )

    direct = {
        **direct,
        "selection_type":
            "direct_high_role_similarity",
    }

    diverse = {
        **diverse,
        "selection_type":
            "semantic_diverse",
    }

    return [direct, diverse]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    if not Path(ADAPTER).exists():
        raise SystemExit(
            f"missing adapter: {ADAPTER}"
        )

    roles = role_rows()
    canon = canonical_map()

    if len(roles) != 72:
        raise RuntimeError(
            f"expected 72 roles, got {len(roles)}"
        )

    for x in roles:
        r = x["role"]

        if len(canon.get(r, [])) < 3:
            raise RuntimeError(
                f"{r}: expected >=3 canonical probes, "
                f"got {len(canon.get(r, []))}"
            )

    by_family_train = defaultdict(list)

    for x in roles:
        if x["split"] == "train":
            by_family_train[
                x["family"]
            ].append(x)

    for family, rr in by_family_train.items():
        if len(rr) != 5:
            raise RuntimeError(
                f"{family}: TRAIN={len(rr)}, expected 5"
            )

    print("=" * 92, flush=True)
    print(
        "GENERAL ROLE V14 PROBES V3",
        flush=True,
    )
    print("=" * 92, flush=True)

    print(
        "roles:",
        len(roles),
        flush=True,
    )
    print(
        "TRAIN:",
        sum(x["split"] == "train" for x in roles),
        flush=True,
    )
    print(
        "VAL:",
        sum(x["split"] == "val" for x in roles),
        flush=True,
    )

    print(
        "protocol:",
        "3 canonical task families x "
        "4 generated candidates -> "
        "2 selected/family",
        flush=True,
    )

    print(
        "generated probes kept/role:",
        KEEP_PER_ROLE,
        flush=True,
    )

    print(
        "future total probes/role:",
        3 + KEEP_PER_ROLE,
        flush=True,
    )

    print(
        "IMPORTANT: no MAS outputs, reward, truth, "
        "activation, or routing is used.",
        flush=True,
    )

    print(
        "Old BenignGateV10 is AUDIT ONLY; "
        "explicit meta-language remains prohibited.",
        flush=True,
    )

    embedder = FrozenOutputEmbedderV8(
        device="cpu"
    )

    audit_gate = BenignGateV10(
        mode="hard",
        guard_model=None,
    )

    policy = DisambiguationProbePolicyV13(
        MODEL,
        adapter_path=ADAPTER,
        trainable=False,
        use_4bit=True,
        max_seq_length=1536,
    )

    policy.model.eval()

    # Resume only V3 artifacts.
    completed = set()

    if OUT.exists():
        counts = Counter()

        for line in OUT.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            row = json.loads(line)

            counts[
                row["role"]
            ] += 1

        completed = {
            role
            for role, n in counts.items()
            if n == KEEP_PER_ROLE
        }

        bad = {
            role: n
            for role, n in counts.items()
            if n != KEEP_PER_ROLE
        }

        if bad:
            raise RuntimeError(
                "Partial V3 role blocks detected; "
                f"do not resume automatically: {bad}"
            )

    total_selected = (
        len(completed)
        * KEEP_PER_ROLE
    )

    for role_idx, item in enumerate(
        roles,
        1,
    ):
        role = item["role"]

        if role in completed:
            print(
                f"[{role_idx:02d}/{len(roles)}] "
                f"{role}: RESUME SKIP",
                flush=True,
            )
            continue

        split = item["split"]
        family = item["family"]
        desc = item["description"]

        canonical = canon[role][:3]

        confusers = [
            x
            for x in by_family_train[family]
            if x["role"] != role
        ]

        desc_emb = embedder.encode(desc)

        # Canonical probes themselves are never generated again.
        seen_text = {
            norm(q)
            for q in canonical
        }

        all_role_candidates = []
        candidates_by_anchor = {}

        print(
            "\n"
            + "=" * 92,
            flush=True,
        )

        print(
            f"[{role_idx:02d}/{len(roles)}] "
            f"{split.upper():<5} "
            f"{family:<30} "
            f"{role}",
            flush=True,
        )

        for anchor_idx, anchor in enumerate(
            canonical,
            1,
        ):
            pool = []
            attempts = 0

            while (
                len(pool)
                < VALID_PER_ANCHOR
                and attempts
                < MAX_ATTEMPTS_PER_ANCHOR
            ):
                attempts += 1

                seed = (
                    SEED
                    + role_idx * 1_000_000
                    + anchor_idx * 10_000
                    + attempts
                )

                set_seed(seed)

                # Important change from V2:
                # every canonical probe becomes its OWN task-family anchor.
                obs = {
                    "candidate_role":
                        role,

                    "candidate_description":
                        desc,

                    "anchor_probe":
                        anchor,

                    "avoid_probes":
                        canonical
                        + [
                            x["query"]
                            for x in all_role_candidates
                        ][-12:],

                    "confusable_roles":
                        [
                            {
                                "role":
                                    x["role"],
                                "description":
                                    x["description"],
                            }
                            for x in confusers[:5]
                        ],

                    "public_history":
                        [],
                }

                # Important change from V2:
                # use the SAME fast sampling/validation wrapper used by
                # our adaptive inference code instead of raw sample_probe.
                (
                    q,
                    syntax_valid,
                    reason,
                    raw,
                ) = sample_probe_fast(
                    policy,
                    obs,
                    seed=seed,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

                q = str(q or "").strip()

                gate_allowed = None
                gate_stage = None
                gate_category = None

                status = "ok"

                if (
                    not syntax_valid
                    or not q
                ):
                    status = "invalid"

                elif norm(q) in seen_text:
                    status = "duplicate"

                elif not explicit_meta_safe(q):
                    status = "meta_blocked"

                else:
                    qemb = embedder.encode(q)

                    # Reject near duplicates across the entire role,
                    # not merely exact strings.
                    sim_previous = semantic_max(
                        qemb,
                        all_role_candidates,
                    )

                    if (
                        sim_previous
                        >= SEMANTIC_DUPLICATE
                    ):
                        status = (
                            "semantic_duplicate"
                        )

                    else:
                        gd = audit_gate.check(q)

                        gate_allowed = bool(
                            gd.allowed
                        )

                        gate_stage = getattr(
                            gd,
                            "stage",
                            None,
                        )

                        gate_category = getattr(
                            gd,
                            "category",
                            None,
                        )

                        # DO NOT reject merely because this old
                        # MAS-oriented gate says blocked.
                        #
                        # We preserve it as audit metadata.
                        status = (
                            "ok"
                            if gate_allowed
                            else "ok_gate_disagrees"
                        )

                append_jsonl(
                    AUDIT,
                    {
                        "role":
                            role,
                        "split":
                            split,
                        "family":
                            family,
                        "anchor_index":
                            anchor_idx,
                        "anchor":
                            anchor,
                        "attempt":
                            attempts,
                        "seed":
                            seed,
                        "query":
                            q,
                        "syntax_valid":
                            bool(
                                syntax_valid
                            ),
                        "syntax_reason":
                            reason,
                        "status":
                            status,
                        "old_gate_allowed":
                            gate_allowed,
                        "old_gate_stage":
                            gate_stage,
                        "old_gate_category":
                            gate_category,
                    },
                )

                if status not in {
                    "ok",
                    "ok_gate_disagrees",
                }:
                    continue

                qemb = embedder.encode(q)

                candidate = {
                    "query":
                        q,
                    "seed":
                        seed,
                    "embedding":
                        qemb,
                    "role_similarity":
                        cosine(
                            qemb,
                            desc_emb,
                        ),
                    "anchor_similarity":
                        cosine(
                            qemb,
                            embedder.encode(
                                anchor
                            ),
                        ),
                    "anchor_index":
                        anchor_idx,
                    "anchor":
                        anchor,
                    "old_gate_allowed":
                        gate_allowed,
                    "old_gate_stage":
                        gate_stage,
                    "old_gate_category":
                        gate_category,
                }

                pool.append(candidate)
                all_role_candidates.append(
                    candidate
                )

                seen_text.add(
                    norm(q)
                )

            if len(pool) < 2:
                raise RuntimeError(
                    f"{role} anchor={anchor_idx}: "
                    f"only {len(pool)} usable candidates "
                    f"after {attempts} attempts"
                )

            candidates_by_anchor[
                anchor_idx
            ] = pool

            print(
                f"  anchor {anchor_idx}: "
                f"{len(pool)} usable / "
                f"{attempts} attempts",
                flush=True,
            )

        selected = []

        # Guarantee task-family diversity:
        # exactly two probes around EACH canonical anchor.
        for anchor_idx, anchor in enumerate(
            canonical,
            1,
        ):
            picks = select_two_from_anchor(
                candidates=
                    candidates_by_anchor[
                        anchor_idx
                    ],
                anchor=anchor,
                embedder=embedder,
                already_selected=selected,
            )

            selected.extend(picks)

        if len(selected) != KEEP_PER_ROLE:
            raise RuntimeError(
                f"{role}: selected={len(selected)}, "
                f"expected={KEEP_PER_ROLE}"
            )

        for j, x in enumerate(
            selected,
            1,
        ):
            print(
                f"  G{j} "
                f"[A{x['anchor_index']}/"
                f"{x['selection_type']}] "
                f"roleSim="
                f"{x['role_similarity']:.3f} "
                f"anchorSim="
                f"{x['anchor_similarity']:.3f} "
                f"gate="
                f"{x['old_gate_allowed']} | "
                f"{x['query']}",
                flush=True,
            )

        # Write ONLY after all six for the role are available.
        for j, x in enumerate(
            selected,
            1,
        ):
            append_jsonl(
                OUT,
                {
                    "version":
                        "general_role_v14_probes_v3",

                    "role":
                        role,

                    "split":
                        split,

                    "family":
                        family,

                    "public_role_description":
                        desc,

                    "probe_source":
                        "v14_frozen",

                    "probe_index":
                        j,

                    "canonical_anchor_index":
                        x["anchor_index"],

                    "canonical_anchor":
                        x["anchor"],

                    "selection_type":
                        x["selection_type"],

                    "query":
                        x["query"],

                    "seed":
                        x["seed"],

                    "role_similarity":
                        float(
                            x[
                                "role_similarity"
                            ]
                        ),

                    "anchor_similarity":
                        float(
                            x[
                                "anchor_similarity"
                            ]
                        ),

                    "old_gate_allowed":
                        x[
                            "old_gate_allowed"
                        ],

                    "old_gate_stage":
                        x[
                            "old_gate_stage"
                        ],

                    "old_gate_category":
                        x[
                            "old_gate_category"
                        ],

                    "selection_uses_mas_output":
                        False,

                    "selection_uses_truth":
                        False,

                    "selection_uses_reward":
                        False,
                },
            )

        total_selected += KEEP_PER_ROLE

        write_json(
            PROGRESS,
            {
                "roles_completed":
                    role_idx,

                "roles_total":
                    len(roles),

                "last_role":
                    role,

                "selected_rows_so_far":
                    total_selected,

                "expected_total_rows":
                    len(roles)
                    * KEEP_PER_ROLE,
            },
        )

    summary = {
        "version":
            "general_role_v14_probes_v3",

        "roles_total":
            len(roles),

        "train_roles":
            sum(
                x["split"] == "train"
                for x in roles
            ),

        "val_roles":
            sum(
                x["split"] == "val"
                for x in roles
            ),

        "canonical_task_families_per_role":
            3,

        "candidate_target_per_family":
            VALID_PER_ANCHOR,

        "generated_kept_per_family":
            KEEP_PER_ANCHOR,

        "generated_kept_per_role":
            KEEP_PER_ROLE,

        "canonical_probes_retained_per_role":
            3,

        "future_total_probes_per_role":
            9,

        "selection_protocol":
            (
                "3 canonical task-family anchors; "
                "per anchor keep one direct/high-role-similarity "
                "probe and one semantic-diverse probe"
            ),

        "generation_wrapper":
            "sample_probe_fast",

        "old_benign_gate":
            "audit_only",

        "explicit_meta_filter":
            True,

        "selection_uses_mas_output":
            False,

        "selection_uses_truth":
            False,

        "selection_uses_reward":
            False,

        "adapter":
            ADAPTER,

        "seed":
            SEED,

        "expected_output_rows":
            len(roles)
            * KEEP_PER_ROLE,
    }

    write_json(
        SUMMARY,
        summary,
    )

    print(
        "\n" + "=" * 92,
        flush=True,
    )

    print("DONE", flush=True)

    print(
        "output:",
        OUT,
        flush=True,
    )

    print(
        "rows:",
        len(roles)
        * KEEP_PER_ROLE,
        flush=True,
    )

    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
