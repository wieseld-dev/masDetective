from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_exact_v11_start(
    adapter_dir: str,
    summary_path: str,
    *,
    strict: bool = True,
) -> dict:
    """Prove V11.2 begins from the already-trained V11 adapter, not fresh LoRA."""
    ad = Path(adapter_dir)
    model = ad / "adapter_model.safetensors"
    cfg = ad / "adapter_config.json"
    sp = Path(summary_path)

    problems: list[str] = []
    if not model.is_file():
        problems.append(f"missing adapter weights: {model}")
    if not cfg.is_file():
        problems.append(f"missing adapter config: {cfg}")
    if not sp.is_file():
        problems.append(f"missing V11 summary: {sp}")

    summary = {}
    if sp.is_file():
        summary = json.loads(sp.read_text(encoding="utf-8"))
        if str(summary.get("version")) != "V11":
            problems.append(f"summary version is {summary.get('version')!r}, expected 'V11'")
        policy_dir = summary.get("policy_dir")
        if policy_dir:
            try:
                if Path(policy_dir).resolve() != ad.resolve():
                    problems.append(
                        f"summary policy_dir={policy_dir!r} does not match adapter={adapter_dir!r}"
                    )
            except Exception:
                pass

    if problems and strict:
        raise RuntimeError("V11.2 start verification failed: " + "; ".join(problems))

    sha = _sha256(model) if model.is_file() else ""
    return {
        "policy_sha256": sha,
        "policy_match": not problems,
        "summary_match": not problems,
        "starting_version": summary.get("version"),
        "adapter_dir": str(ad),
        "summary": str(sp),
        "problems": problems,
    }
