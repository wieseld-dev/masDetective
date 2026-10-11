from __future__ import annotations

from pathlib import Path

from v11.hashcheck import sha256_file, sha256_tree


def test_sha256_tree_is_content_and_relative_path_stable(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        (root / "nested").mkdir(parents=True)
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "nested" / "weights.bin").write_bytes(b"trained-v10")
    assert sha256_tree(a) == sha256_tree(b)
    (b / "nested" / "weights.bin").write_bytes(b"different")
    assert sha256_tree(a) != sha256_tree(b)


def test_sha256_file_changes_with_content(tmp_path: Path):
    p = tmp_path / "summary.json"
    p.write_text('{"v":10}', encoding="utf-8")
    x = sha256_file(p)
    p.write_text('{"v":11}', encoding="utf-8")
    assert sha256_file(p) != x
