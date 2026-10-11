from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from detective.benign_gate_v7 import CombinedBenignGateV7, normalize_query_v7
from estimator.candidate_presence_contrastive_v64 import CandidateRecord
from probes.role_probe_bank import ANCHOR_PROBE, primary_probe
from rl.generative_reward_v7 import GenerativeProbeRewardV7, RewardBreakdownV7


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-5) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class SeedStoreV7:
    """Public-only replay store for the already-paid neutral anchor + fixed V6.5 seed probe."""

    def __init__(self, rows: Sequence[Mapping]):
        self.rows = {(str(r["system_id"]), str(r["focus_role"])): dict(r) for r in rows}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SeedStoreV7":
        rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return cls(rows)

    def output(self, system_id: str, focus_role: str) -> str:
        key = (str(system_id), str(focus_role))
        if key not in self.rows:
            raise KeyError(f"missing seed-store row: {key}")
        return str(self.rows[key].get("output", ""))

    def anchor(self, system_id: str) -> str:
        return self.output(system_id, "__anchor__")


class ResponseCacheV7:
    """Persistent public/truth-separated cache for generated follow-up interactions."""

    def __init__(self, public_path: str | Path, truth_path: str | Path):
        self.public_path = Path(public_path)
        self.truth_path = Path(truth_path)
        self.public: Dict[Tuple[str, str], Dict] = {}
        self.truth: Dict[Tuple[str, str], Dict] = {}
        self._load()

    @staticmethod
    def query_hash(query: str) -> str:
        return hashlib.sha256(normalize_query_v7(query).encode("utf-8")).hexdigest()[:24]

    def _load_file(self, path: Path) -> Dict[Tuple[str, str], Dict]:
        out: Dict[Tuple[str, str], Dict] = {}
        if not path.exists():
            return out
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[(str(r["system_id"]), str(r["query_hash"]))] = r
        return out

    def _load(self) -> None:
        self.public = self._load_file(self.public_path)
        self.truth = self._load_file(self.truth_path)
        if set(self.public) != set(self.truth):
            raise RuntimeError("V7 public/truth response caches are inconsistent")

    def get(self, system_id: str, query: str):
        key = (str(system_id), self.query_hash(query))
        if key not in self.public:
            return None
        return dict(self.public[key]), dict(self.truth[key])

    @staticmethod
    def _append(path: Path, row: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def put(self, system_id: str, query: str, output: str, *, selected_roles: Sequence[str], true_roles: Sequence[str], mode: str) -> None:
        qh = self.query_hash(query)
        key = (str(system_id), qh)
        if key in self.public:
            return
        pub = {"system_id": str(system_id), "query_hash": qh, "query": normalize_query_v7(query), "output": str(output)}
        truth = {
            "system_id": str(system_id), "query_hash": qh,
            "selected_roles": list(map(str, selected_roles)),
            "true_roles": list(map(str, true_roles)), "mode": str(mode),
        }
        self._append(self.public_path, pub)
        self._append(self.truth_path, truth)
        self.public[key] = pub
        self.truth[key] = truth


@dataclass
class EpisodeStepInfoV7:
    allowed: bool
    gate_reason: str
    gate_stage: str
    belief_before: float
    belief_after: float
    raw_response_probability: float | None
    external_query_sent: bool
    cache_hit: bool
    selected_target_training_only: bool | None
    reward: RewardBreakdownV7


class GenerativeProbeEnvV7:
    """One candidate-role investigation episode.

    Observation boundary is black-box only. True role presence and selected roles are used solely
    inside reward/evaluation and never appear in observation().
    """

    def __init__(
        self,
        *,
        target,
        estimator,
        profile: Mapping,
        candidate_role: str,
        seed_store: SeedStoreV7,
        gate: CombinedBenignGateV7,
        reward_model: GenerativeProbeRewardV7 | None = None,
        response_cache: ResponseCacheV7 | None = None,
        max_followups: int = 3,
        max_attempts: int = 6,
        belief_alpha: float = 0.65,
    ):
        self.target = target
        self.estimator = estimator
        self.profile = dict(profile)
        self.role = str(candidate_role)
        self.seed_store = seed_store
        self.gate = gate
        self.reward_model = reward_model or GenerativeProbeRewardV7()
        self.response_cache = response_cache
        self.max_followups = int(max_followups)
        self.max_attempts = int(max_attempts)
        self.belief_alpha = float(belief_alpha)
        if self.role not in set(self.profile.get("candidate_roles", [])):
            raise KeyError(f"candidate role not in profile: {self.role}")
        self.system_id = str(self.target.system.system_id)
        self.true_roles = tuple(map(str, self.target.system.worker_roles))
        self.truth_present = self.role in set(self.true_roles)
        self.mode = str(getattr(self.target.system, "execution_mode", "unknown"))
        self._score_cache: Dict[str, float] = {}
        self.reset()

    def _score_response(self, output: str) -> float:
        key = hashlib.sha256(str(output).encode("utf-8")).hexdigest()
        if key in self._score_cache:
            return self._score_cache[key]
        rec = CandidateRecord(
            system_id="blackbox", role=self.role, mode="unknown", true_roles=tuple(),
            anchor_output=self.anchor_output, target_output=str(output),
        )
        p = float(self.estimator.predict_proba_records([rec])[0])
        self._score_cache[key] = p
        return p

    def _merge_belief(self, p_response: float) -> float:
        # Exponential smoothing in log-odds avoids repeated weak evidence exploding confidence.
        x = (1.0 - self.belief_alpha) * _logit(self.belief) + self.belief_alpha * _logit(p_response)
        return min(max(_sigmoid(x), 1e-4), 1.0 - 1e-4)

    def reset(self):
        self.anchor_output = self.seed_store.anchor(self.system_id)
        self.seed_query = primary_probe(self.role)
        self.seed_output = self.seed_store.output(self.system_id, self.role)
        self.belief = self._score_response(self.seed_output) if hasattr(self, "anchor_output") else 0.5
        self.history = [{"query": self.seed_query, "output": self.seed_output, "kind": "seed"}]
        self.external_followups = 0
        self.attempts = 0
        self.done = False
        self.gate_feedback = ""
        return self.observation()

    def observation(self) -> Dict:
        # Deliberately exclude system_id, mode, K, true_roles, selected_roles, traces.
        return {
            "candidate_role": self.role,
            "candidate_description": self.profile.get("role_descriptions", {}).get(self.role, self.role),
            "belief": float(self.belief),
            "history": [{"query": h["query"], "output": h["output"]} for h in self.history],
            "remaining_budget": max(self.max_followups - self.external_followups, 0),
            "remaining_attempts": max(self.max_attempts - self.attempts, 0),
            "gate_feedback": self.gate_feedback,
        }

    def step(self, action: str, query: str = ""):
        if self.done:
            raise RuntimeError("episode already finished")
        action = str(action).lower()
        if action == "stop":
            rb = self.reward_model.stop(
                belief=self.belief, truth_present=self.truth_present,
                remaining_budget=max(self.max_followups - self.external_followups, 0),
            )
            self.done = True
            info = EpisodeStepInfoV7(True, "stop", "policy", self.belief, self.belief, None, False, False, None, rb)
            return self.observation(), rb.total, True, info
        if action != "ask":
            self.attempts += 1
            rb = self.reward_model.malformed()
            self.gate_feedback = "The previous output was malformed. Return exactly the required JSON schema."
            if self.attempts >= self.max_attempts:
                self.done = True
            info = EpisodeStepInfoV7(False, "malformed_action", "parser", self.belief, self.belief, None, False, False, None, rb)
            return self.observation(), rb.total, self.done, info

        self.attempts += 1
        gate_result = self.gate.check(query)
        if not gate_result.allowed:
            rb = self.reward_model.blocked()
            self.gate_feedback = f"Blocked as non-benign ({gate_result.stage}:{gate_result.category}). Ask a normal task question instead."
            if self.attempts >= self.max_attempts:
                self.done = True
            info = EpisodeStepInfoV7(False, gate_result.reason, gate_result.stage, self.belief, self.belief, None, False, False, None, rb)
            return self.observation(), rb.total, self.done, info

        if self.external_followups >= self.max_followups:
            rb = self.reward_model.automatic_terminal(belief=self.belief, truth_present=self.truth_present)
            self.done = True
            info = EpisodeStepInfoV7(True, "budget_exhausted", "env", self.belief, self.belief, None, False, False, None, rb)
            return self.observation(), rb.total, True, info

        query = normalize_query_v7(query)
        cached = self.response_cache.get(self.system_id, query) if self.response_cache else None
        cache_hit = cached is not None
        if cached:
            pub, truth = cached
            output = str(pub["output"])
            selected_roles = tuple(map(str, truth.get("selected_roles", [])))
        else:
            output = self.target.query(query)
            trace = self.target.privileged_last_trace()
            selected_roles = tuple(map(str, trace.selected_roles))
            if self.response_cache:
                self.response_cache.put(
                    self.system_id, query, output,
                    selected_roles=selected_roles, true_roles=self.true_roles, mode=self.mode,
                )
        before = float(self.belief)
        p_resp = self._score_response(output)
        after = self._merge_belief(p_resp)
        prev_queries = [str(h["query"]) for h in self.history]
        rb = self.reward_model.ask(
            p_before=before, p_after=after, truth_present=self.truth_present,
            target_selected=(self.role in set(selected_roles)), query=query, previous_queries=prev_queries,
        )
        self.belief = after
        self.external_followups += 1
        self.gate_feedback = ""
        self.history.append({"query": query, "output": output, "kind": "followup"})
        if self.external_followups >= self.max_followups or self.attempts >= self.max_attempts:
            terminal = self.reward_model.automatic_terminal(belief=self.belief, truth_present=self.truth_present)
            rb = RewardBreakdownV7(
                total=rb.total + terminal.total,
                info_gain=rb.info_gain, activation=rb.activation, novelty=rb.novelty,
                query_cost=rb.query_cost, stop=terminal.stop,
            )
            self.done = True
        info = EpisodeStepInfoV7(
            True, "ok", gate_result.stage, before, after, p_resp, not cache_hit, cache_hit,
            self.role in set(selected_roles), rb,
        )
        return self.observation(), rb.total, self.done, info
