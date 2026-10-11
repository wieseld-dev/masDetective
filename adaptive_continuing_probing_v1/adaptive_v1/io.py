from __future__ import annotations
import json
from pathlib import Path

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8");t.replace(p)
