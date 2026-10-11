import json
from pathlib import Path

import torch

from v10.checkpoint import atomic_write_json, commit_round_snapshot, load_committed_state


class TinyPolicy:
    def save_adapter(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text('{"ok": true}', encoding="utf-8")
        (p / "adapter_model.safetensors").write_bytes(b"tiny")


def test_atomic_json_and_committed_snapshot(tmp_path):
    state_path = tmp_path / "state.json"
    atomic_write_json(state_path, {"committed": True, "phase": "train", "adapter_dir": None, "optimizer_state": None})
    assert json.loads(state_path.read_text())["committed"] is True

    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = commit_round_snapshot(
        policy=TinyPolicy(),
        optimizer=opt,
        round_i=1,
        total_rounds=3,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_state=state_path,
        public_output_dir=tmp_path / "public_policy",
        log_path=tmp_path / "run.jsonl",
        torch_module=torch,
    )
    assert state["committed"] is True
    assert state["next_round"] == 2
    loaded = load_committed_state(state_path)
    assert Path(loaded["adapter_dir"]).is_dir()
    assert Path(loaded["optimizer_state"]).is_file()
    assert (tmp_path / "public_policy" / "adapter_model.safetensors").is_file()
