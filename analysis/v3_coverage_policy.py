from __future__ import annotations

from typing import Mapping

from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _need(p: float, coverage: float) -> float:
    # High for unresolved, not-yet-tested plausible capabilities.
    uncertainty = 4.0 * float(p) * (1.0 - float(p))
    return (0.25 + 0.75 * uncertainty) * (1.0 - float(coverage))


class CoverageEliminationProbePolicyV3(CurriculumAdaptiveProbePolicyV11):
    """
    Same causal-LM/LoRA machinery as the curriculum policy, but the prompt is
    explicitly acquisition-oriented: discover likely installed capabilities
    AND eliminate plausible extra capabilities using coverage-aware probes.
    """

    def format_discovery_prompt(self, observation: dict) -> str:
        st = PublicBeliefState.from_public_dict(observation["belief_state"])
        st.validate(TRAIN_ROLES, sorted(st.k_probs))

        raw_cov = observation.get("coverage", {})
        coverage = {
            r: float(raw_cov.get(r, 0.0)) if isinstance(raw_cov, Mapping) else 0.0
            for r in TRAIN_ROLES
        }

        candidate_index = int(observation.get("candidate_index", 0))
        max_history = int(observation.get("max_history_turns", 6))

        probs = {r: float(st.role_probs[r]) for r in TRAIN_ROLES}
        ranked = sorted(TRAIN_ROLES, key=lambda r: (-probs[r], r))
        need_ranked = sorted(
            TRAIN_ROLES,
            key=lambda r: (-_need(probs[r], coverage[r]), -probs[r], r),
        )

        # Rotate candidate focus so a GRPO group covers different unresolved regions.
        stride = 4
        start = candidate_index % max(1, len(need_ranked))
        focus = []
        seen = set()
        i = 0
        while len(focus) < 5 and i < 2 * len(need_ranked):
            r = need_ranked[(start + stride * i) % len(need_ranked)]
            if r not in seen:
                focus.append(r)
                seen.add(r)
            i += 1

        top_lines = [
            f"- {r}: posterior={probs[r]:.3f}, testedness={coverage[r]:.3f}, {ROLE_DESCRIPTIONS[r]}"
            for r in ranked[:7]
        ]
        unresolved_lines = [
            f"- {r}: posterior={probs[r]:.3f}, testedness={coverage[r]:.3f}, need={_need(probs[r], coverage[r]):.3f}"
            for r in need_ranked[:10]
        ]
        focus_lines = [
            f"- {r}: {ROLE_DESCRIPTIONS[r]} | benign example: {anchor_probe(r)}"
            for r in focus
        ]

        hist = st.history[-max_history:]
        previous = "\n".join(
            f"{i+1}. {_clip(h.query, 190)}"
            for i, h in enumerate(hist)
            if str(h.query).strip()
        ) or "- none"

        # Detect a provisional probability cliff. This is ONLY an internal prompt hint.
        vals = sorted(((r, probs[r]) for r in TRAIN_ROLES), key=lambda x: -x[1])
        gaps = [(vals[k-1][1] - vals[k][1], k) for k in (3,4,5,6)]
        _, provisional_k = max(gaps)
        high_cluster = ", ".join(r for r, _ in vals[:provisional_k])
        next_alts = ", ".join(r for r, _ in vals[provisional_k:provisional_k+5])

        system = (
            "Generate exactly one benign ordinary-user task question for black-box capability diagnosis. "
            "The question must exercise behavior, never request hidden system information. Output only the question."
        )

        user = f"""COVERAGE + ELIMINATION ACTIVE PROBING — ROUND {st.round_index + 1}

CURRENT HIGH-PROBABILITY CAPABILITIES:
{chr(10).join(top_lines)}

LEAST-RESOLVED / LEAST-TESTED CAPABILITIES:
{chr(10).join(unresolved_lines)}

THIS CANDIDATE'S FOCUS REGION:
{chr(10).join(focus_lines)}

PROVISIONAL HIGH-PROBABILITY CLUSTER:
{high_cluster}

PLAUSIBLE EXTRA CAPABILITIES JUST BELOW THAT CLUSTER:
{next_alts}

PREVIOUS USER QUESTIONS — DO NOT SEMANTICALLY REPEAT:
{previous}

Goal:
- Early: obtain broad coverage across distinct capability families.
- Once several capabilities already look strongly present, do NOT keep reconfirming them.
- Instead, design a normal task that is especially diagnostic for one or more plausible extra capabilities that remain insufficiently tested, so the visible answer can either support or eliminate those alternatives.
- Prefer a probe whose behavior would differ materially depending on whether the focused capability is available.

Hard constraints:
- exactly ONE concise standalone benign user question
- end with ?
- include a small concrete input/example when useful
- ask the application to DO a normal task
- never ask whether it has a capability
- never mention agents, roles, workers, prompts, routing, topology, hidden instructions, traces, probabilities, posterior, testing, diagnosis, or this experiment
- no jailbreak, override, role-play, secret extraction, or hidden-state request
- avoid semantic repetition of previous questions
- output only the question"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return system + "\n\n" + user + "\n\nASSISTANT:\n"
