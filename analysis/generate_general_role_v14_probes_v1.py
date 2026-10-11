from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from v8.embedding import FrozenOutputEmbedderV8
from v10.gate import BenignGateV10
from v13.policy import DisambiguationProbePolicyV13


ROLE_BANK = Path(
    "artifacts/general_role_bank_v1/role_bank_v1.json"
)

OUTDIR = Path(
    "artifacts/general_role_bank_v1"
)

OUT = OUTDIR / "v14_generated_probes_v1.jsonl"
SUMMARY = OUTDIR / "v14_generated_probes_v1_summary.json"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "artifacts/v14_benign_policy"

SEED = 20260929

RAW_CANDIDATES_PER_ROLE = 6
KEEP_PER_ROLE = 2

TEMPERATURE = 0.90
TOP_P = 0.92


def norm(s: str) -> str:
    return " ".join(
        str(s).lower().replace("_", " ").split()
    )


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def get_canonical_queries(row: dict) -> list[str]:
    out = []

    for x in row.get("canonical_probes", []):
        if isinstance(x, dict):
            q = str(x.get("query", "")).strip()
        else:
            q = str(x).strip()

        if q:
            out.append(q)

    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    if not ROLE_BANK.exists():
        raise SystemExit(
            f"Missing role bank: {ROLE_BANK}"
        )

    if not Path(ADAPTER).exists():
        raise SystemExit(
            f"Missing V14 adapter: {ADAPTER}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    roles = json.loads(
        ROLE_BANK.read_text(encoding="utf-8")
    )

    if not isinstance(roles, list):
        raise RuntimeError(
            "role_bank_v1.json must be a list"
        )

    required = {
        "role",
        "split",
        "family",
        "public_role_description",
        "canonical_probes",
    }

    for row in roles:
        missing = required - set(row)
        if missing:
            raise RuntimeError(
                f"{row.get('role')} missing keys: "
                f"{sorted(missing)}"
            )

    by_family_train = {}

    for row in roles:
        if row["split"] != "train":
            continue

        by_family_train.setdefault(
            row["family"],
            [],
        ).append(row)

    # Every family should have 5 TRAIN roles.
    for family, rr in by_family_train.items():
        if len(rr) != 5:
            raise RuntimeError(
                f"family={family} has "
                f"{len(rr)} TRAIN roles, expected 5"
            )

    print("=" * 88)
    print("GENERAL ROLE V14 PROBE GENERATION")
    print("=" * 88)
    print("roles:", len(roles))
    print(
        "train:",
        sum(r["split"] == "train" for r in roles),
    )
    print(
        "val:",
        sum(r["split"] == "val" for r in roles),
    )
    print(
        "raw candidates/role:",
        RAW_CANDIDATES_PER_ROLE,
    )
    print("keep/role:", KEEP_PER_ROLE)
    print("adapter:", ADAPTER)
    print(
        "IMPORTANT: NO MAS outputs and NO reward "
        "are used for selection."
    )

    embedder = FrozenOutputEmbedderV8(
        device="cpu"
    )

    gate = BenignGateV10(
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

    selected_rows = []
    audit_rows = []

    for role_i, row in enumerate(roles):
        role = row["role"]
        family = row["family"]
        split = row["split"]

        desc = row["public_role_description"]
        canonical = get_canonical_queries(row)

        if len(canonical) < 3:
            raise RuntimeError(
                f"{role}: expected >=3 canonical probes, "
                f"got {len(canonical)}"
            )

        same_family_train = [
            x
            for x in by_family_train[family]
            if x["role"] != role
        ]

        # TRAIN target: 4 same-family TRAIN confusers.
        # VAL target: 5 same-family TRAIN confusers.
        confusers = [
            {
                "role": x["role"],
                "description":
                    x["public_role_description"],
            }
            for x in same_family_train
        ]

        obs = {
            "candidate_role": role,
            "candidate_description": desc,

            # Canonical query is only public guidance.
            "anchor_probe": canonical[0],

            # Explicitly encourage V14 to produce wording
            # different from the three canonical probes.
            "avoid_probes": canonical,

            "confusable_roles": confusers[:5],

            # No adaptive MAS history for dataset acquisition.
            "public_history": [],
        }

        raw = []
        seen = set(norm(x) for x in canonical)

        attempts = 0
        max_attempts = RAW_CANDIDATES_PER_ROLE * 5

        while (
            len(raw) < RAW_CANDIDATES_PER_ROLE
            and attempts < max_attempts
        ):
            attempts += 1

            seed = (
                SEED
                + role_i * 10000
                + attempts
            )

            set_seed(seed)

            sample = policy.sample_probe(
                obs,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            q = str(sample.query or "").strip()

            status = "ok"

            if not sample.valid or not q:
                status = "invalid"

            elif norm(q) in seen:
                status = "duplicate"

            else:
                gd = gate.check(q)

                if not gd.allowed:
                    status = "blocked"

            audit = {
                "role": role,
                "split": split,
                "family": family,
                "attempt": attempts,
                "seed": seed,
                "query": q,
                "status": status,
            }

            audit_rows.append(audit)

            if status != "ok":
                continue

            seen.add(norm(q))
            raw.append(
                {
                    "query": q,
                    "seed": seed,
                }
            )

        if len(raw) < KEEP_PER_ROLE:
            raise RuntimeError(
                f"{role}: only {len(raw)} valid unique "
                f"V14 probes after {attempts} attempts"
            )

        desc_e = embedder.encode(desc)

        canonical_e = [
            embedder.encode(q)
            for q in canonical
        ]

        for x in raw:
            qe = embedder.encode(x["query"])

            x["role_similarity"] = cosine(
                qe,
                desc_e,
            )

            x["max_canonical_similarity"] = max(
                cosine(qe, ce)
                for ce in canonical_e
            )

        # Greedy query-only selection.
        #
        # Prefer relevance to public role description,
        # but also reward novelty relative to canonical
        # probes and the first selected generated probe.
        chosen = []

        while len(chosen) < KEEP_PER_ROLE:
            best = None
            best_score = -1e9

            for x in raw:
                if x in chosen:
                    continue

                qe = embedder.encode(
                    x["query"]
                )

                refs = canonical_e[:]

                for y in chosen:
                    refs.append(
                        embedder.encode(
                            y["query"]
                        )
                    )

                max_sim = max(
                    cosine(qe, re)
                    for re in refs
                )

                diversity = 1.0 - max_sim

                # Query-only score.
                score = (
                    0.60
                    * x["role_similarity"]
                    +
                    0.40
                    * diversity
                )

                if score > best_score:
                    best_score = score
                    best = {
                        **x,
                        "selection_score":
                            float(score),
                        "max_reference_similarity":
                            float(max_sim),
                    }

            chosen.append(best)

        print(
            f"\n[{role_i+1:02d}/{len(roles)}] "
            f"{split:<5} "
            f"{family:<30} "
            f"{role}"
        )

        for j, x in enumerate(chosen, 1):
            print(
                f"  G{j}: "
                f"rel={x['role_similarity']:.3f} "
                f"maxRef={x['max_reference_similarity']:.3f} "
                f"| {x['query']}"
            )

            selected_rows.append(
                {
                    "version":
                        "general_role_v14_probes_v1",
                    "role": role,
                    "split": split,
                    "family": family,
                    "public_role_description": desc,
                    "probe_source": "v14_frozen",
                    "probe_family":
                        f"v14_generated_{j}",
                    "query": x["query"],
                    "seed": x["seed"],
                    "role_similarity":
                        x["role_similarity"],
                    "max_canonical_similarity":
                        x["max_canonical_similarity"],
                    "max_reference_similarity":
                        x["max_reference_similarity"],
                    "selection_score":
                        x["selection_score"],
                    "selection_uses_mas_output":
                        False,
                    "selection_uses_truth":
                        False,
                    "selection_uses_reward":
                        False,
                }
            )

    with OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in selected_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit_path = (
        OUTDIR
        / "v14_generated_probes_v1_audit.jsonl"
    )

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in audit_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "version":
            "general_role_v14_probes_v1",
        "roles_total":
            len(roles),
        "train_roles":
            sum(
                r["split"] == "train"
                for r in roles
            ),
        "val_roles":
            sum(
                r["split"] == "val"
                for r in roles
            ),
        "generated_probes_per_role":
            KEEP_PER_ROLE,
        "selected_rows":
            len(selected_rows),
        "expected_selected_rows":
            len(roles) * KEEP_PER_ROLE,
        "selection_uses_mas_output":
            False,
        "selection_uses_truth":
            False,
        "selection_uses_reward":
            False,
        "v14_adapter":
            ADAPTER,
        "seed":
            SEED,
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print(
        "selected probes:",
        OUT,
    )
    print(
        "audit:",
        audit_path,
    )
    print(
        "summary:",
        SUMMARY,
    )
    print(
        "rows:",
        len(selected_rows),
    )

    assert len(selected_rows) == 72 * 2

    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
