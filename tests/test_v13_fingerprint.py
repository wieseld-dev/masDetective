import numpy as np

from v13.fingerprint import BehavioralFingerprintDetectorV13


class TinyEmbedder:
    def __init__(self):
        self.vocab = [
            "requirements", "acceptance", "stakeholder", "scope", "constraint",
            "database", "schema", "key", "normalization", "index",
            "generic", "question", "answer", "task", "check",
        ]

    def encode(self, text):
        t = str(text).lower()
        v = np.asarray([t.count(w) for w in self.vocab], dtype=np.float32) + 0.01
        return v / np.linalg.norm(v)


def test_role_fingerprint_crossfit_separates_behavior():
    emb = TinyEmbedder()
    desc = {"requirements_analysis": "requirements acceptance stakeholder scope constraint"}
    rows = []
    for i in range(8):
        q = f"generic question task {i}"
        rows.append({"role":"requirements_analysis","query":q,"output":"requirements acceptance stakeholder scope constraint","y":1,"weight":1.0,"group":f"g{i}"})
        rows.append({"role":"requirements_analysis","query":q,"output":"database schema key normalization index","y":0,"weight":1.0,"group":f"g{i}"})
    d = BehavioralFingerprintDetectorV13(emb, desc, c=0.1)
    rep = d.fit(rows, ["requirements_analysis"], folds=4)
    assert rep["macro_oof_auc"] > 0.9
    s_pos, p_pos = d.score("requirements_analysis", "generic question task", "requirements acceptance stakeholder scope constraint")
    s_neg, p_neg = d.score("requirements_analysis", "generic question task", "database schema key normalization index")
    assert s_pos > s_neg
    assert p_pos > p_neg
