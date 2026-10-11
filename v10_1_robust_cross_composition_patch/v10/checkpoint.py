from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def atomic_replace_dir(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Replace dst with a complete directory tree copied from src.

    The copy is built beside dst and only renamed into place after it is complete.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}"
    old = dst.parent / f".{dst.name}.old-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    try:
        if dst.exists():
            if old.exists():
                shutil.rmtree(old)
            os.replace(dst, old)
        os.replace(tmp, dst)
        if old.exists():
            shutil.rmtree(old)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if old.exists() and not dst.exists():
            os.replace(old, dst)
        raise


def load_committed_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    state = json.loads(p.read_text(encoding="utf-8"))
    if not state.get("committed", False):
        raise RuntimeError(f"checkpoint state is not committed: {p}")
    phase = state.get("phase")
    if phase not in {"train", "holdout", "complete"}:
        raise RuntimeError(f"invalid checkpoint phase: {phase!r}")
    adapter = state.get("adapter_dir")
    opt = state.get("optimizer_state")
    if phase in {"train", "holdout", "complete"} and adapter and not Path(adapter).is_dir():
        raise RuntimeError(f"committed adapter directory missing: {adapter}")
    if phase in {"train", "holdout", "complete"} and opt and not Path(opt).is_file():
        raise RuntimeError(f"committed optimizer state missing: {opt}")
    return state


def commit_round_snapshot(
    *,
    policy,
    optimizer,
    round_i: int,
    total_rounds: int,
    checkpoint_root: str | Path,
    checkpoint_state: str | Path,
    public_output_dir: str | Path,
    log_path: str | Path,
    torch_module,
) -> dict[str, Any]:
    """Transactionally commit one fully-finished training round.

    A round is considered resumable only after adapter + optimizer + round metadata
    are all written and the root checkpoint JSON is atomically advanced.
    """
    root = Path(checkpoint_root)
    final_round = root / f"round_{round_i:02d}"
    staging = root / f".round_{round_i:02d}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    policy_dir = staging / "policy"
    optimizer_path = staging / "optimizer.pt"
    policy.save_adapter(str(policy_dir))
    torch_module.save(optimizer.state_dict(), optimizer_path)
    atomic_write_json(
        staging / "round_state.json",
        {
            "round": round_i,
            "total_rounds": total_rounds,
            "next_round": round_i + 1,
            "committed": True,
            "log": str(log_path),
        },
    )
    (staging / "COMMITTED").write_text("ok\n", encoding="utf-8")

    if final_round.exists():
        shutil.rmtree(final_round)
    os.replace(staging, final_round)

    # Convenience mirror for final downstream use. Resume never trusts this mirror;
    # it trusts only the immutable committed round snapshot referenced below.
    atomic_replace_dir(final_round / "policy", public_output_dir)

    state = {
        "version": "V10.1",
        "phase": "train" if round_i < total_rounds else "holdout",
        "completed_round": round_i,
        "next_round": round_i + 1,
        "total_rounds": total_rounds,
        "adapter_dir": str(final_round / "policy"),
        "optimizer_state": str(final_round / "optimizer.pt"),
        "round_snapshot": str(final_round),
        "log": str(log_path),
        "committed": True,
    }
    atomic_write_json(checkpoint_state, state)
    return state
