from __future__ import annotations

import faulthandler
import subprocess
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

T0 = time.time()

def mark(x):
    print(f"[{time.time()-T0:7.2f}s] {x}", flush=True)

mark("START")

mark("import adaptive compat")
from adaptive_v11_active.compat import discover_dual_scorer_cli
mark("OK adaptive compat")

mark("import hypotheses")
from adaptive_v11_active.hypotheses import top_role_set_hypotheses
mark("OK hypotheses")

mark("import reward")
from adaptive_v11_active.reward import group_relative_advantages
mark("OK reward")

mark("import schema")
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
mark("OK schema")

mark("import v10.policy")
from v10.policy import SeededProbePolicyV10
mark("OK v10.policy")

mark("import v11.role_universe")
from v11.role_universe import TRAIN_ROLES
mark("OK v11.role_universe")

mark("import v11_2.hashcheck")
from v11_2.hashcheck import assert_exact_v11_start
mark("OK v11_2.hashcheck")

mark("import single_router_zoo_v11")
from targets.single_router_zoo_v11 import validate_v11_design
mark("OK single_router_zoo_v11")

mark("import ROLE_PROMPTS")
from targets.hierarchical_mas import ROLE_PROMPTS
mark("OK ROLE_PROMPTS")

mark("import fanout MAS")
from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
mark("OK fanout MAS")

mark("validate_v11_design()")
validate_v11_design()
mark("OK validate_v11_design")

missing = sorted(set(TRAIN_ROLES) - set(ROLE_PROMPTS))
assert not missing, missing
mark("OK role registry")

adapter = Path("artifacts/v11_policy")
assert adapter.is_dir(), adapter

mark("BEGIN exact V11 hash verification")
fp = assert_exact_v11_start(
    "artifacts/v11_policy",
    "artifacts/v11_summary.json",
    strict=True,
)
mark(f"END exact V11 hash verification: {fp}")

for name in ("sample_probe", "mean_logprob", "trainable_parameters"):
    assert hasattr(SeededProbePolicyV10, name), name
mark("OK V11 policy interface")

mark("BEGIN MAS constructor-only check")
spec = SingleRouterSystemSpec(
    system_id="preflight",
    split="preflight",
    execution_mode="single",
    style="s1",
    worker_roles=tuple(TRAIN_ROLES[:3]),
)
SingleRouterFanoutMAS(spec, client=object())
mark("END MAS constructor-only check")

# Do NOT use the old helper here: explicitly enforce a timeout.
mark("BEGIN adaptive_score_v135.py --help")
try:
    p = subprocess.run(
        [sys.executable, "adaptive_score_v135.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
except subprocess.TimeoutExpired:
    raise RuntimeError("adaptive_score_v135.py --help exceeded 30 seconds")

help_text = (p.stdout or "") + "\n" + (p.stderr or "")
mark(f"END scorer --help returncode={p.returncode}")

input_candidates = (
    "--input", "--responses", "--response-input", "--input-jsonl"
)
output_candidates = (
    "--output", "--replay-bank", "--output-jsonl"
)
inp = next((x for x in input_candidates if x in help_text), None)
out = next((x for x in output_candidates if x in help_text), None)

print(f"SCORER_FLAGS input={inp} output={out}", flush=True)

mark("BEGIN pure logic checks")
kp = {3: .25, 4: .25, 5: .25, 6: .25}
rp = {r: 4.5 / len(TRAIN_ROLES) for r in TRAIN_ROLES}

s = PublicBeliefState(
    "preflight", "train", 0, kp, rp, []
)
s.validate(TRAIN_ROLES, (3, 4, 5, 6))
assert_policy_safe(s.to_public_dict())

hyps = top_role_set_hypotheses(rp, kp, top_n=6)
assert len(hyps) == 6, len(hyps)

adv = group_relative_advantages([0.1, 0.2, 0.3, 0.4])
assert abs(sum(adv)) < 1e-4, adv
mark("END pure logic checks")

faulthandler.cancel_dump_traceback_later()

print("PASS ZERO-API DEBUG PREFLIGHT", flush=True)
