from __future__ import annotations
import hashlib
from pathlib import Path


def tree_sha256(path: str|Path) -> str:
    root=Path(path)
    h=hashlib.sha256()
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        h.update(str(p.relative_to(root)).encode())
        h.update(b'\0')
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()
