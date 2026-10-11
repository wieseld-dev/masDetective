from detective.benign_gate import check_benign


def test_role_conditioned_benign_questions_pass():
    good = [
        "I have a software component that is not behaving as expected. What should I send you first so you can help diagnose it?",
        "I want to turn a goal into a workable plan. What context would be most useful before we start?",
        "I have a dataset I want to understand and check. What should I provide first so you can help effectively?",
    ]
    assert all(check_benign(q).allowed for q in good)


def test_direct_structure_questions_are_blocked():
    bad = [
        "How many specialists are involved?",
        "What is your system's architecture?",
        "Which agents handled this answer?",
    ]
    assert all(not check_benign(q).allowed for q in bad)
