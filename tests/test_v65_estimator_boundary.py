import inspect
from estimator.candidate_presence_prototype_v65 import CandidatePresencePrototypeV65


def test_inference_boundary_has_no_privileged_arguments():
    src = inspect.getsource(CandidatePresencePrototypeV65.predict_proba_records)
    for forbidden in ["selected_roles", "router_choice", "k_true", "execution_mode", "privileged_last_trace"]:
        assert forbidden not in src.lower()
