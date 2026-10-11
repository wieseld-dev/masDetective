from estimator.k_estimator import KBeliefEstimator


def test_estimator_probability_distribution():
    h3 = [[{"query": "q", "output": "short code response"}]]
    h5 = [[{"query": "q", "output": "long research planning writing data response"}]]
    est = KBeliefEstimator(min_k=3, max_k=5).fit(h3 + h5, [3, 5])
    p = est.predict_proba(h3[0])
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert set(p) == {3, 4, 5}
