from __future__ import annotations

import gc
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v8.embedding import FrozenOutputEmbedderV8
from v10.gate import BenignGateV10
from v13.policy import DisambiguationProbePolicyV13


ROOT = Path("artifacts/general_role_bank_v1")
ROLE_BANK = ROOT / "role_bank_v1.json"
CANONICAL_PATH = ROOT / "canonical_probes_v1.jsonl"

OUT = ROOT / "v14_generated_probes_v2.jsonl"
AUDIT = ROOT / "v14_generated_probes_v2_audit.jsonl"
SUMMARY = ROOT / "v14_generated_probes_v2_summary.json"
PROGRESS = ROOT / "v14_generated_probes_v2_progress.json"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "artifacts/v14_benign_policy"

SEED = 20260929

RAW_VALID_TARGET = 18
KEEP_PER_ROLE = 6
MAX_ATTEMPTS_PER_ROLE = 90

TEMPERATURE = 0.95
TOP_P = 0.95


def norm(s):
    return " ".join(
        str(s).lower().replace("_", " ").split()
    )


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def load_role_bank():
    doc = json.loads(
        ROLE_BANK.read_text(encoding="utf-8")
    )

    # Support either:
    #   [ {...}, {...} ]
    # or:
    #   {"roles": [...]}
    if isinstance(doc, list):
        roles = doc

    elif isinstance(doc, dict):
        for key in (
            "roles",
            "role_bank",
            "items",
        ):
            if isinstance(doc.get(key), list):
                roles = doc[key]
                break
        else:
            raise RuntimeError(
                "Could not locate role list in role_bank_v1.json. "
                f"Top-level keys={list(doc)[:30]}"
            )

    else:
        raise RuntimeError(
            f"Unexpected role bank type: {type(doc)}"
        )

    return roles


def pick(row, *keys, default=None):
    for k in keys:
        if row.get(k) is not None:
            return row[k]
    return default


def canonical_map():
    out = defaultdict(list)

    for line in CANONICAL_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        x = json.loads(line)

        role = str(
            pick(
                x,
                "role",
                "target_role",
                "candidate_role",
                default="",
            )
        ).strip()

        query = str(
            pick(
                x,
                "query",
                "probe",
                "text",
                default="",
            )
        ).strip()

        if role and query:
            if norm(query) not in {
                norm(q) for q in out[role]
            }:
                out[role].append(query)

    return dict(out)


def greedy_diverse(candidates, n, embedder, refs):
    if not candidates:
        return []

    chosen = []

    ref_embs = [
        embedder.encode(x)
        for x in refs
        if str(x).strip()
    ]

    while (
        len(chosen) < n
        and len(chosen) < len(candidates)
    ):
        best = None
        best_score = -1e9

        for c in candidates:
            if c in chosen:
                continue

            qe = c["embedding"]

            existing = list(ref_embs)

            for x in chosen:
                existing.append(x["embedding"])

            if existing:
                max_sim = max(
                    cosine(qe, e)
                    for e in existing
                )
            else:
                max_sim = 0.0

            diversity = 1.0 - max_sim

            # Within a semantic stratum, prefer diversity.
            # Keep only a small relevance term.
            score = (
                0.75 * diversity
                + 0.25 * c["role_similarity"]
            )

            if score > best_score:
                best_score = score
                best = c

        chosen.append(best)

    return chosen


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    if not ROLE_BANK.exists():
        raise SystemExit(
            f"missing {ROLE_BANK}"
        )

    if not CANONICAL_PATH.exists():
        raise SystemExit(
            f"missing {CANONICAL_PATH}"
        )

    if not Path(ADAPTER).exists():
        raise SystemExit(
            f"missing {ADAPTER}"
        )

    roles = load_role_bank()
    canon = canonical_map()

    normalized = []

    for x in roles:
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
                f"Role without name: {x}"
            )

        if split not in {"train", "val"}:
            raise RuntimeError(
                f"{role}: bad split={split!r}"
            )

        if not family:
            raise RuntimeError(
                f"{role}: missing family"
            )

        if not desc:
            raise RuntimeError(
                f"{role}: missing public description"
            )

        if len(canon.get(role, [])) < 3:
            raise RuntimeError(
                f"{role}: only "
                f"{len(canon.get(role, []))} "
                "canonical probes"
            )

        normalized.append(
            {
                "role": role,
                "split": split,
                "family": family,
                "description": desc,
            }
        )

    print("=" * 90, flush=True)
    print(
        "GENERAL ROLE V14 PROBES V2",
        flush=True,
    )
    print("=" * 90, flush=True)
    print(
        "roles total:",
        len(normalized),
        flush=True,
    )
    print(
        "TRAIN:",
        sum(
            x["split"] == "train"
            for x in normalized
        ),
        flush=True,
    )
    print(
        "VAL:",
        sum(
            x["split"] == "val"
            for x in normalized
        ),
        flush=True,
    )
    print(
        "valid candidates target/role:",
        RAW_VALID_TARGET,
        flush=True,
    )
    print(
        "selected generated probes/role:",
        KEEP_PER_ROLE,
        flush=True,
    )
    print(
        "selection = 2 LOW + 2 MID + 2 HIGH "
        "semantic-similarity strata",
        flush=True,
    )
    print(
        "NO MAS OUTPUT / NO TRUTH / NO REWARD",
        flush=True,
    )

    by_family_train = defaultdict(list)

    for x in normalized:
        if x["split"] == "train":
            by_family_train[
                x["family"]
            ].append(x)

    for fam, rr in by_family_train.items():
        print(
            f"family {fam:<32} "
            f"TRAIN={len(rr)}",
            flush=True,
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

    # Resume safely.
    completed = set()

    if OUT.exists():
        for line in OUT.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            r = json.loads(line)

            if int(
                r.get(
                    "selected_count_for_role",
                    0,
                )
            ) == KEEP_PER_ROLE:
                completed.add(r["role"])

    total_selected = 0

    for role_idx, item in enumerate(
        normalized,
        1,
    ):
        role = item["role"]

        if role in completed:
            print(
                f"[{role_idx:02d}/{len(normalized)}] "
                f"{role}: RESUME SKIP",
                flush=True,
            )
            total_selected += KEEP_PER_ROLE
            continue

        family = item["family"]
        desc = item["description"]
        split = item["split"]

        canonical = canon[role][:3]

        confusers = [
            x
            for x in by_family_train[family]
            if x["role"] != role
        ]

        obs = {
            "candidate_role": role,
            "candidate_description": desc,
            "anchor_probe": canonical[0],
            "avoid_probes": canonical,
            "confusable_roles": [
                {
                    "role": x["role"],
                    "description":
                        x["description"],
                }
                for x in confusers[:5]
            ],
            "public_history": [],
        }

        desc_emb = embedder.encode(desc)

        seen = {
            norm(q)
            for q in canonical
        }

        candidates = []
        attempts = 0

        while (
            len(candidates)
            < RAW_VALID_TARGET
            and attempts
            < MAX_ATTEMPTS_PER_ROLE
        ):
            attempts += 1

            seed = (
                SEED
                + role_idx * 100000
                + attempts
            )

            set_seed(seed)

            sample = policy.sample_probe(
                obs,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            q = str(
                sample.query or ""
            ).strip()

            status = "ok"

            if not sample.valid or not q:
                status = "invalid"

            elif norm(q) in seen:
                status = "duplicate"

            else:
                gd = gate.check(q)

                if not gd.allowed:
                    status = "blocked"

            append_jsonl(
                AUDIT,
                {
                    "role": role,
                    "split": split,
                    "family": family,
                    "attempt": attempts,
                    "seed": seed,
                    "query": q,
                    "status": status,
                },
            )

            if status != "ok":
                continue

            seen.add(norm(q))

            emb = embedder.encode(q)

            candidates.append(
                {
                    "query": q,
                    "seed": seed,
                    "embedding": emb,
                    "role_similarity":
                        cosine(
                            emb,
                            desc_emb,
                        ),
                }
            )

        if len(candidates) < KEEP_PER_ROLE:
            raise RuntimeError(
                f"{role}: only "
                f"{len(candidates)} valid candidates "
                f"after {attempts} attempts"
            )

        # Sort from semantically less-direct to more-direct.
        candidates.sort(
            key=lambda x:
                x["role_similarity"]
        )

        n = len(candidates)

        # Relative strata, not arbitrary universal cosine thresholds.
        # This guarantees semantic spread for each role.
        cut1 = max(1, n // 3)
        cut2 = max(cut1 + 1, 2 * n // 3)

        low = candidates[:cut1]
        mid = candidates[cut1:cut2]
        high = candidates[cut2:]

        strata = [
            ("lower", low),
            ("medium", mid),
            ("high", high),
        ]

        selected = []

        for stratum_name, pool in strata:
            picks = greedy_diverse(
                pool,
                2,
                embedder,
                canonical
                + [
                    x["query"]
                    for x in selected
                ],
            )

            for x in picks:
                x["semantic_stratum"] = (
                    stratum_name
                )
                selected.append(x)

        if len(selected) != KEEP_PER_ROLE:
            raise RuntimeError(
                f"{role}: selected "
                f"{len(selected)}, expected "
                f"{KEEP_PER_ROLE}"
            )

        sims = [
            x["role_similarity"]
            for x in candidates
        ]

        print(
            f"\n[{role_idx:02d}/{len(normalized)}] "
            f"{split.upper():<5} "
            f"{family:<30} "
            f"{role}",
            flush=True,
        )

        print(
            "  candidate similarity range: "
            f"{min(sims):.3f} .. "
            f"{max(sims):.3f}",
            flush=True,
        )

        for j, x in enumerate(
            selected,
            1,
        ):
            print(
                f"  G{j} "
                f"[{x['semantic_stratum']:<6}] "
                f"sim={x['role_similarity']:.3f} | "
                f"{x['query']}",
                flush=True,
            )

        # One atomic logical block per role.
        for j, x in enumerate(
            selected,
            1,
        ):
            append_jsonl(
                OUT,
                {
                    "version":
                        "general_role_v14_probes_v2",
                    "role": role,
                    "split": split,
                    "family": family,
                    "public_role_description":
                        desc,
                    "probe_source":
                        "v14_frozen",
                    "probe_index": j,
                    "probe_family":
                        f"v14_{x['semantic_stratum']}_{j}",
                    "semantic_stratum":
                        x["semantic_stratum"],
                    "query": x["query"],
                    "seed": x["seed"],
                    "role_similarity":
                        float(
                            x[
                                "role_similarity"
                            ]
                        ),
                    "selection_uses_mas_output":
                        False,
                    "selection_uses_truth":
                        False,
                    "selection_uses_reward":
                        False,
                    "selected_count_for_role":
                        KEEP_PER_ROLE,
                },
            )

        total_selected += KEEP_PER_ROLE

        write_json(
            PROGRESS,
            {
                "roles_completed":
                    role_idx,
                "roles_total":
                    len(normalized),
                "last_role":
                    role,
                "selected_rows_so_far":
                    total_selected,
            },
        )

    summary = {
        "version":
            "general_role_v14_probes_v2",
        "roles_total":
            len(normalized),
        "train_roles":
            sum(
                x["split"] == "train"
                for x in normalized
            ),
        "val_roles":
            sum(
                x["split"] == "val"
                for x in normalized
            ),
        "raw_valid_candidates_target_per_role":
            RAW_VALID_TARGET,
        "generated_probes_per_role":
            KEEP_PER_ROLE,
        "semantic_strata": {
            "lower": 2,
            "medium": 2,
            "high": 2,
        },
        "canonical_probes_retained_per_role":
            3,
        "future_total_probes_per_role":
            9,
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
            len(normalized)
            * KEEP_PER_ROLE,
    }

    write_json(
        SUMMARY,
        summary,
    )

    print("\n" + "=" * 90, flush=True)
    print("DONE", flush=True)
    print("=" * 90, flush=True)
    print(
        "generated:",
        OUT,
        flush=True,
    )
    print(
        "summary:",
        SUMMARY,
        flush=True,
    )

    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
