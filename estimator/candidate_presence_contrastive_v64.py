from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class CandidateRecord:
    system_id: str
    role: str
    mode: str
    true_roles: Tuple[str, ...]
    anchor_output: str
    target_output: str

    @property
    def label(self) -> int:
        return int(self.role in set(self.true_roles))

    @property
    def k_true(self) -> int:
        return len(self.true_roles)

    @property
    def composition_key(self) -> str:
        return "|".join(sorted(self.true_roles))


def load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_profile(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_candidate_records(public_rows: Sequence[Mapping], truth_rows: Sequence[Mapping]) -> List[CandidateRecord]:
    """Build attacker-view candidate records.

    Features later use only: candidate role identity + anchor external output + targeted external output.
    Privileged truth is attached only as an evaluation/training label here.
    """
    pub: Dict[Tuple[str, str], Mapping] = {
        (str(r["system_id"]), str(r["focus_role"])): r for r in public_rows
    }
    truth: Dict[Tuple[str, str], Mapping] = {
        (str(r["system_id"]), str(r["focus_role"])): r for r in truth_rows
    }
    systems = sorted({sid for sid, _ in pub})
    records: List[CandidateRecord] = []
    for sid in systems:
        anchor = pub.get((sid, "__anchor__"))
        anchor_truth = truth.get((sid, "__anchor__"))
        if anchor is None or anchor_truth is None:
            raise ValueError(f"missing anchor row for system {sid}")
        true_roles = tuple(map(str, anchor_truth.get("true_roles", [])))
        mode = str(anchor_truth.get("mode", "unknown"))
        for (psid, focus), prow in sorted(pub.items()):
            if psid != sid or focus == "__anchor__":
                continue
            trow = truth.get((sid, focus))
            if trow is None:
                raise ValueError(f"missing truth row for {(sid, focus)}")
            # Never use selected_roles as an estimator feature. It is deliberately ignored.
            records.append(
                CandidateRecord(
                    system_id=sid,
                    role=str(focus),
                    mode=mode,
                    true_roles=true_roles,
                    anchor_output=str(anchor.get("output", "")),
                    target_output=str(prow.get("output", "")),
                )
            )
    if not records:
        raise ValueError("no candidate records were built")
    return records


def _role_conditioned(role: str, text: str) -> str:
    """Create role-conditioned lexical features without using any hidden state."""
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return f"role_{role}__empty"
    return " ".join(f"role_{role}__{t}" for t in toks)


def _safe_ratio(a: float, b: float) -> float:
    return a / max(b, 1.0)


class ContrastiveCandidateEstimatorV64:
    """Binary candidate-presence estimator for fan-out MAS outputs.

    Each sample asks one question only:
        Does the *chosen candidate role* exist?

    Observable inputs only:
      - candidate role identity (chosen by attacker),
      - neutral anchor output,
      - targeted probe output.

    No router choice, selected worker, true role set, topology trace, or internal message is used.
    """

    def __init__(self, candidate_roles: Sequence[str], role_descriptions: Mapping[str, str] | None = None,
                 random_state: int = 73):
        roles = tuple(dict.fromkeys(map(str, candidate_roles)))
        if len(roles) < 2:
            raise ValueError("candidate_roles must contain at least two roles")
        self.candidate_roles = roles
        self.role_descriptions = {str(k): str(v) for k, v in (role_descriptions or {}).items()}
        self.random_state = int(random_state)
        self.word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=24000,
        )
        self.char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
            max_features=24000,
        )
        self.role_word = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"[^ ]+",
            ngram_range=(1, 1),
            sublinear_tf=True,
            min_df=1,
            max_features=26000,
        )
        self.clf = LogisticRegression(
            C=0.35,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=self.random_state,
        )
        self._fitted = False

    def _fit_vectorizers(self, records: Sequence[CandidateRecord]) -> None:
        corpus: List[str] = []
        role_corpus: List[str] = []
        for r in records:
            corpus.extend([r.anchor_output, r.target_output])
            role_corpus.extend([
                _role_conditioned(r.role, r.anchor_output),
                _role_conditioned(r.role, r.target_output),
            ])
        self.word.fit(corpus)
        self.char.fit(corpus)
        self.role_word.fit(role_corpus)

    def _semantic_alignment(self, records: Sequence[CandidateRecord]) -> np.ndarray:
        """Cheap role-description alignment features using the fitted word TF-IDF space."""
        out = np.zeros((len(records), 7), dtype=np.float64)
        for i, r in enumerate(records):
            desc = self.role_descriptions.get(r.role, r.role)
            q = self.word.transform([desc])
            t = self.word.transform([r.target_output])
            a = self.word.transform([r.anchor_output])
            st = float(t.multiply(q).sum())
            sa = float(a.multiply(q).sum())
            lt = len(r.target_output)
            la = len(r.anchor_output)
            out[i] = [
                st,
                sa,
                st - sa,
                math.log1p(lt),
                math.log1p(la),
                math.log1p(abs(lt - la)),
                _safe_ratio(lt + 1.0, la + 1.0),
            ]
        return out

    def _transform(self, records: Sequence[CandidateRecord]):
        target = [r.target_output for r in records]
        anchor = [r.anchor_output for r in records]
        role_target = [_role_conditioned(r.role, r.target_output) for r in records]
        role_anchor = [_role_conditioned(r.role, r.anchor_output) for r in records]

        wt = self.word.transform(target)
        wa = self.word.transform(anchor)
        ct = self.char.transform(target)
        ca = self.char.transform(anchor)
        rt = self.role_word.transform(role_target)
        ra = self.role_word.transform(role_anchor)
        numeric = csr_matrix(self._semantic_alignment(records))

        # Target, anchor, and explicit deltas are all attacker-observable.
        return hstack([
            wt, wa, wt - wa,
            ct, ca, ct - ca,
            rt, ra, rt - ra,
            numeric,
        ], format="csr")

    def fit(self, records: Sequence[CandidateRecord]) -> "ContrastiveCandidateEstimatorV64":
        if not records:
            raise ValueError("records must be non-empty")
        y = np.asarray([r.label for r in records], dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain both present and absent candidates")
        self._fit_vectorizers(records)
        X = self._transform(records)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba_records(self, records: Sequence[CandidateRecord]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("estimator is not fitted")
        X = self._transform(records)
        return self.clf.predict_proba(X)[:, 1]

    def predict_candidate_proba(self, *, role: str, anchor_output: str, target_output: str,
                                system_id: str = "blackbox", mode: str = "unknown") -> float:
        """Inference API. The dummy true_roles field is not consumed by feature extraction."""
        rec = CandidateRecord(
            system_id=system_id,
            role=str(role),
            mode=mode,
            true_roles=tuple(),
            anchor_output=str(anchor_output),
            target_output=str(target_output),
        )
        return float(self.predict_proba_records([rec])[0])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ContrastiveCandidateEstimatorV64":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"unexpected estimator type: {type(obj)!r}")
        return obj
