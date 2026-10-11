from estimator.role_presence_estimator_v63 import RolePresenceEstimatorV63, history_to_text


def test_estimator_uses_focus_and_output_not_query_wording():
    h = [{"focus_role":"math", "query":"SECRET QUERY WORDS", "output":"equation units calculation"}]
    text = history_to_text(h)
    assert "FOCUS_math" in text
    assert "equation units calculation" in text
    assert "SECRET QUERY WORDS" not in text


def test_estimator_probabilities_are_valid():
    roles = ["code","math","writing"]
    hs = [
        [{"focus_role":"code","output":"stack trace implementation test"}],
        [{"focus_role":"math","output":"equation units calculation"}],
        [{"focus_role":"writing","output":"tone audience paragraph"}],
        [{"focus_role":"math","output":"formula and software computation"}],
    ]
    ys = [{"code"},{"math"},{"writing"},{"code","math"}]
    est = RolePresenceEstimatorV63(roles).fit(hs, ys)
    p = est.predict_proba(hs[0])
    assert set(p) == set(roles)
    assert all(0 < x < 1 for x in p.values())
