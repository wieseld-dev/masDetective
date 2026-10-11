from __future__ import annotations

import hashlib
import os
from pathlib import Path

EXPECTED_V10_POLICY_SHA256 = "9abbb8cdfa04df70ab8f1dadba23fad15a82e098002df8c6186cf2befaf9a7f9"
EXPECTED_V10_SUMMARY_SHA256 = "b965cb9041d2fe323538f6c43591e6d9d82a5ddec95d642f4423d2430087973c"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        h.update(b"\0")
    return h.hexdigest()


def assert_exact_v10_start(policy_dir: str, summary_path: str, *, strict: bool = True) -> dict:
    p = Path(policy_dir)
    s = Path(summary_path)
    if not p.is_dir():
        raise RuntimeError(f"missing trained V10.1 adapter: {p}")
    if not s.is_file():
        raise RuntimeError(f"missing V10.1 summary: {s}")
    ph = sha256_tree(p)
    sh = sha256_file(s)
    result = {
        "policy_dir": str(p),
        "policy_sha256": ph,
        "summary_path": str(s),
        "summary_sha256": sh,
        "expected_policy_sha256": EXPECTED_V10_POLICY_SHA256,
        "expected_summary_sha256": EXPECTED_V10_SUMMARY_SHA256,
        "policy_match": ph == EXPECTED_V10_POLICY_SHA256,
        "summary_match": sh == EXPECTED_V10_SUMMARY_SHA256,
    }
    if strict and not result["policy_match"]:
        raise RuntimeError(
            "V11 REFUSES TO START: artifacts/v10_policy is not the exact trained V10.1 adapter "
            f"used in the consumed final test. got={ph} expected={EXPECTED_V10_POLICY_SHA256}"
        )
    if strict and not result["summary_match"]:
        raise RuntimeError(
            "V11 REFUSES TO START: artifacts/v10_summary.json does not match the frozen V10.1 summary. "
            f"got={sh} expected={EXPECTED_V10_SUMMARY_SHA256}"
        )
    return result
