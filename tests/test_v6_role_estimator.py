from estimator.role_presence_estimator import RolePresenceEstimator


def test_role_estimator_returns_valid_probabilities():
    roles = ["code", "writing", "math"]
    histories = [
        [{"query": "software", "output": "code error stack trace"}],
        [{"query": "draft", "output": "tone paragraph audience"}],
        [{"query": "equation", "output": "calculation units formula"}],
        [{"query": "software", "output": "code implementation tests"}],
    ]
    truths = [{"code"}, {"writing"}, {"math"}, {"code", "math"}]
    est = RolePresenceEstimator(roles).fit(histories, truths)
    probs = est.predict_proba(histories[0])
    assert set(probs) == set(roles)
    assert all(0.0 < p < 1.0 for p in probs.values())
    assert 0.0 <= est.expected_k(histories[0]) <= len(roles)
