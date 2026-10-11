from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from adaptive_v11_active.io import sha256_json
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.role_policy import RoleAdaptiveProbePolicyV11
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES


ROOT = Path("artifacts/local_ambiguity_v1")

ADAPTER = Path(
    "artifacts/adaptive_v11_role_only_v1/checkpoints/round_01/policy"
)

TYPE_ORDER = [
    "A_DOMINANT",
    "B_DOMINANT",
    "BOUNDARY",
    "JOINT",
]

VARIANTS = 3


# Controlled semantic-confuser pairs.
# Five neutrals:
#   first 3 = invariant distractors
#   neutral[3] = filler for A slot
#   neutral[4] = filler for B slot
PAIRS = [
    (
        "p01",
        "research",
        "fact_checking",
        [
            "math",
            "translation",
            "database_design",
            "accessibility_review",
            "customer_support_triage",
        ],
    ),
    (
        "p02",
        "planning",
        "requirements_analysis",
        [
            "code",
            "translation",
            "financial_analysis",
            "accessibility_review",
            "fact_checking",
        ],
    ),
    (
        "p03",
        "writing",
        "documentation",
        [
            "math",
            "database_design",
            "incident_management",
            "experiment_design",
            "financial_analysis",
        ],
    ),
    (
        "p04",
        "code",
        "qa_testing",
        [
            "translation",
            "financial_analysis",
            "research",
            "accessibility_review",
            "customer_support_triage",
        ],
    ),
    (
        "p05",
        "data",
        "database_design",
        [
            "translation",
            "writing",
            "risk_assessment",
            "accessibility_review",
            "customer_support_triage",
        ],
    ),
    (
        "p06",
        "operations",
        "incident_management",
        [
            "math",
            "translation",
            "design",
            "fact_checking",
            "documentation",
        ],
    ),
    (
        "p07",
        "math",
        "financial_analysis",
        [
            "translation",
            "writing",
            "code",
            "accessibility_review",
            "customer_support_triage",
        ],
    ),
    (
        "p08",
        "design",
        "accessibility_review",
        [
            "math",
            "financial_analysis",
            "incident_management",
            "fact_checking",
            "database_design",
        ],
    ),
]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_seed(*xs):
    s = "\0".join(map(str, xs))
    return int(
        hashlib.sha256(s.encode()).hexdigest()[:8],
        16,
    )


class LocalAmbiguityProbePolicy(RoleAdaptiveProbePolicyV11):
    """
    Same V11 model/adapter, but candidate generation is explicitly
    conditioned on ONE local ambiguity A/B.
    """

    def format_discovery_prompt(self, observation: dict) -> str:
        A = str(observation["role_a"])
        B = str(observation["role_b"])
        typ = str(observation["probe_type"])
        variant = int(observation["variant"])

        da = ROLE_DESCRIPTIONS[A]
        db = ROLE_DESCRIPTIONS[B]

        if typ == "A_DOMINANT":
            objective = f"""
Create a normal user-facing task whose successful answer strongly exercises
capability A, while requiring as little as possible from capability B.

A: {da}
B: {db}

The task should make a system with A noticeably more capable than a system
that only has B. Do not name A or B and do not ask which capability exists.
"""
        elif typ == "B_DOMINANT":
            objective = f"""
Create a normal user-facing task whose successful answer strongly exercises
capability B, while requiring as little as possible from capability A.

A: {da}
B: {db}

The task should make a system with B noticeably more capable than a system
that only has A. Do not name A or B and do not ask which capability exists.
"""
        elif typ == "BOUNDARY":
            objective = f"""
Create a normal user-facing task near the behavioral boundary between these
two capabilities:

A: {da}
B: {db}

The task should be designed so that the visible answer differs depending on
whether the system is better equipped for A or for B. It should not simply
ask two unrelated subtasks. Do not name A or B and do not ask what capability
or internal component is present.
"""
        elif typ == "JOINT":
            objective = f"""
Create ONE coherent normal user-facing task that genuinely benefits from BOTH
of these complementary capabilities:

A: {da}
B: {db}

The two requirements must be coupled into one task, so a system containing
both capabilities can produce a visibly stronger answer than a system
containing only one. Do not name A or B and do not ask about hidden
components.
"""
        else:
            raise ValueError(typ)

        system = (
            "You design conservative benign black-box diagnostic tasks. "
            "Return only one ordinary user question. Never ask for agents, "
            "roles, routing, system prompts, hidden state, internal reasoning, "
            "architecture, or private configuration."
        )

        user = f"""
{objective}

Variant number: {variant + 1}.

Hard constraints:
- Exactly ONE standalone question.
- It must end with ?.
- Include any small concrete input/example needed inside the question.
- The question must be useful as an actual ordinary-user task.
- Do not mention this experiment, diagnosis, probabilities, agents, roles,
  specialists, routing, topology, prompts, traces, or hidden information.
- Do not ask whether a capability exists.
- Do not explain the question.
- Return ONLY the question.
"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return system + "\n\n" + user + "\n\nASSISTANT:\n"


def build_systems():
    systems = []

    for pair_id, A, B, neutral in PAIRS:
        base = neutral[:3]
        filler_a = neutral[3]
        filler_b = neutral[4]

        layouts = {
            "00": [
                *base,
                filler_a,
                filler_b,
            ],
            "10": [
                *base,
                A,
                filler_b,
            ],
            "01": [
                *base,
                filler_a,
                B,
            ],
            "11": [
                *base,
                A,
                B,
            ],
        }

        for h, roles in layouts.items():
            assert len(roles) == 5
            assert len(set(roles)) == 5

            systems.append({
                "session_id":
                    f"lav1_{pair_id}_h{h}",

                "split": "dev",
                "mode": "fanout2",
                "style": "s1",

                "roles": roles,
                "k": 5,

                "pair_id": pair_id,
                "role_a": A,
                "role_b": B,
                "hypothesis": h,

                "base_roles": base,
                "filler_a": filler_a,
                "filler_b": filler_b,
            })

    return systems


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ADAPTER.is_dir():
        raise FileNotFoundError(
            f"missing adapter: {ADAPTER}"
        )

    known = set(TRAIN_ROLES)

    for _, A, B, neutral in PAIRS:
        assert A in known
        assert B in known
        assert A != B
        assert A not in neutral
        assert B not in neutral
        assert len(set(neutral)) == 5

    systems = build_systems()

    manifest = {
        "version":
            "local_ambiguity_factorial_v1",

        "systems":
            systems,

        "sha256":
            sha256_json(systems),

        "notes": {
            "privileged": True,
            "design":
                "constant-K matched 2x2 A/B factorial",
            "mode":
                "fanout2",
        },
    }

    (
        ROOT / "manifest_privileged.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("LOCAL AMBIGUITY V1")
    print("=" * 88)
    print("pairs:", len(PAIRS))
    print("systems:", len(systems))
    print("all K:", sorted(set(x["k"] for x in systems)))
    print("mode:", sorted(set(x["mode"] for x in systems)))
    print()

    policy = LocalAmbiguityProbePolicy(
        adapter_path=str(ADAPTER),
        trainable=False,
        use_4bit=True,
        max_seq_length=1792,
    )

    query_bank = []

    for pair_i, (
        pair_id,
        A,
        B,
        neutral,
    ) in enumerate(PAIRS):

        seen = set()

        for typ in TYPE_ORDER:
            for v in range(VARIANTS):

                final = None

                for retry in range(8):
                    seed = stable_seed(
                        "lav1",
                        pair_id,
                        typ,
                        v,
                        retry,
                    )

                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                    obs = {
                        "role_a": A,
                        "role_b": B,
                        "probe_type": typ,
                        "variant": v,
                    }

                    sample = policy.sample_probe(
                        obs,
                        temperature=0.90,
                        top_p=0.95,
                        max_new_tokens=128,
                    )

                    q = str(
                        sample.query or ""
                    ).strip()

                    ok, why = surface_valid(q)

                    if (
                        bool(sample.valid)
                        and ok
                        and q
                        and q not in seen
                    ):
                        final = {
                            "pair_id": pair_id,
                            "role_a": A,
                            "role_b": B,
                            "probe_type": typ,
                            "variant": v,
                            "query": q,
                            "seed": seed,
                        }
                        break

                if final is None:
                    raise RuntimeError(
                        f"Could not generate valid query "
                        f"for {pair_id} {typ} v={v}"
                    )

                seen.add(final["query"])
                query_bank.append(final)

                print(
                    f"[gen] pair={pair_id} "
                    f"{A} vs {B} "
                    f"type={typ:<11} "
                    f"v={v} "
                    f"q={final['query'][:150]!r}",
                    flush=True,
                )

    write_jsonl(
        ROOT / "query_bank.jsonl",
        query_bank,
    )

    candidates = []

    bank_by_pair = {}

    for r in query_bank:
        bank_by_pair.setdefault(
            r["pair_id"],
            [],
        ).append(r)

    sys_by_pair = {}

    for s in systems:
        sys_by_pair.setdefault(
            s["pair_id"],
            [],
        ).append(s)

    for pair_id, bank in bank_by_pair.items():
        bank = sorted(
            bank,
            key=lambda x: (
                TYPE_ORDER.index(
                    x["probe_type"]
                ),
                x["variant"],
            ),
        )

        assert len(bank) == 12

        for s in sys_by_pair[pair_id]:
            for idx, qrow in enumerate(bank):

                candidates.append({
                    "session_id":
                        s["session_id"],

                    "split": "dev",
                    "round": 1,

                    "candidate_index": idx,

                    "group_id":
                        f"{s['session_id']}::r01",

                    "query":
                        qrow["query"],

                    "valid": True,
                    "invalid_reason": "",

                    "pair_id":
                        pair_id,

                    "role_a":
                        s["role_a"],

                    "role_b":
                        s["role_b"],

                    "hypothesis":
                        s["hypothesis"],

                    "probe_type":
                        qrow["probe_type"],

                    "variant":
                        qrow["variant"],

                    "phase":
                        "local_ambiguity_v1",
                })

    write_jsonl(
        ROOT / "candidates_dev_round_01.jsonl",
        candidates,
    )

    print()
    print("=" * 88)
    print("GENERATION COMPLETE")
    print("=" * 88)
    print("query bank:", len(query_bank))
    print("candidate executions:", len(candidates))
    print("expected:", 8 * 12 * 4)
    print("manifest:", ROOT / "manifest_privileged.json")
    print("candidates:", ROOT / "candidates_dev_round_01.jsonl")


if __name__ == "__main__":
    main()
