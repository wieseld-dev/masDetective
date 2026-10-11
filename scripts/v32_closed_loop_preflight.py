from __future__ import annotations
import os, re, json, argparse
from pathlib import Path

ROOT = Path(".").resolve()
OUT = Path("artifacts/eqs_v3_2_closed_loop/preflight_report.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

TOKENS = [
    "target_public_cache",
    "target_truth_cache_training_only",
    "execution_status",
    "final_output",
    "responses_",
    "score_input_",
    "scored_",
    "FrozenBeliefBundle",
    "role_only_predict",
    "adaptive_role_reward",
    "EvidenceFrame",
    "PublicBeliefState",
]

NAME_HINTS = [
    "v14a", "v14", "adaptive", "execute", "exec", "score", "advance",
    "target", "candidate", "response", "belief"
]

def safe_read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def context(text: str, needle: str, before=12, after=24):
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if needle.lower() in line.lower():
            lo=max(0,i-before); hi=min(len(lines),i+after+1)
            out.append((i+1, "\n".join(f"{j+1:5d}: {lines[j]}" for j in range(lo,hi))))
    return out

def argparse_context(text: str):
    lines=text.splitlines()
    hits=[]
    pats=("ArgumentParser(", "add_argument(", "if __name__")
    idxs=[i for i,l in enumerate(lines) if any(p in l for p in pats)]
    if not idxs: return ""
    lo=max(0,min(idxs)-5); hi=min(len(lines),max(idxs)+8)
    return "\n".join(f"{j+1:5d}: {lines[j]}" for j in range(lo,hi))

scripts = []
for rootname in ("scripts","jobs"):
    root=ROOT/rootname
    if not root.exists(): continue
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".py",".sh",".sbatch",""):
            txt=safe_read(p)
            low=txt.lower()
            score=sum(tok.lower() in low for tok in TOKENS)
            score += sum(h in p.name.lower() for h in NAME_HINTS)
            if score:
                scripts.append((score,p,txt))

scripts.sort(key=lambda x:(-x[0], str(x[1])))

lines=[]
def emit(s=""):
    lines.append(str(s))

emit("="*110)
emit("V3.2 CLOSED-LOOP PREFLIGHT AUDIT")
emit("="*110)
emit(f"PWD: {ROOT}")
emit()

emit("TOP RELEVANT SCRIPTS/JOBS")
emit("-"*110)
for score,p,txt in scripts[:80]:
    matched=[t for t in TOKENS if t.lower() in txt.lower()]
    emit(f"{score:3d}  {p.relative_to(ROOT)}")
    if matched:
        emit("     tokens: " + ", ".join(matched))
emit()

# Identify strongest likely executor / scorer / updater candidates
buckets = {
    "EXECUTOR": ("execution_status","final_output","target_public_cache"),
    "SCORER": ("score_input_","scored_"),
    "UPDATER": ("role_only_predict","EvidenceFrame","PublicBeliefState"),
}
for title, pats in buckets.items():
    emit("="*110)
    emit(title + " CANDIDATES")
    emit("="*110)
    ranked=[]
    for score,p,txt in scripts:
        c=sum(x.lower() in txt.lower() for x in pats)
        if c:
            ranked.append((c,score,p,txt))
    ranked.sort(key=lambda x:(-x[0],-x[1],str(x[2])))
    for c,score,p,txt in ranked[:12]:
        emit(f"\nFILE: {p.relative_to(ROOT)}  bucket_hits={c} total_score={score}")
        emit("-"*110)
        ac=argparse_context(txt)
        if ac:
            emit("ARGPARSE/ENTRYPOINT CONTEXT:")
            emit(ac)
        for pat in pats:
            cs=context(txt,pat,before=8,after=18)
            for lineno,block in cs[:2]:
                emit(f"\n>>> around '{pat}' @ line {lineno}")
                emit(block)
    emit()

emit("="*110)
emit("SBATCH COMMANDS THAT CALL RELEVANT PYTHON SCRIPTS")
emit("="*110)
for score,p,txt in scripts:
    if p.suffix not in (".sbatch",".sh"):
        continue
    rel=[]
    for line in txt.splitlines():
        if ("python" in line.lower() or "$PYTHON" in line or ".py" in line) and not line.strip().startswith("#"):
            rel.append(line)
    if rel and any(k in txt.lower() for k in ("v14","adaptive","response","score","exec")):
        emit(f"\nFILE: {p.relative_to(ROOT)}")
        for line in rel[:30]:
            emit("  "+line)

emit()
emit("="*110)
emit("SAMPLE JSONL SCHEMAS")
emit("="*110)
patterns = [
    "artifacts/**/candidates_*round_*.jsonl",
    "artifacts/**/responses_*round_*.jsonl",
    "artifacts/**/score_input_*round_*.jsonl",
    "artifacts/**/scored_*round_*.jsonl",
    "artifacts/**/states_*round_00.jsonl",
    "artifacts/**/target_public_cache.jsonl",
]
seen=set()
for pat in patterns:
    matches=sorted(ROOT.glob(pat))
    if not matches:
        continue
    # prefer v14 adaptive role path if present
    matches.sort(key=lambda p: (0 if "v14_adaptive_role_v1" in str(p) else 1, len(str(p))))
    p=matches[0]
    if p in seen: continue
    seen.add(p)
    emit(f"\nFILE: {p.relative_to(ROOT)}")
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            row=None
            for line in f:
                if line.strip():
                    row=json.loads(line); break
        if isinstance(row,dict):
            emit("TOP_KEYS: "+", ".join(sorted(row.keys())))
            # print compact selected fields/types
            for k,v in row.items():
                if k in {"prompt","completion","raw_completion","final_output","history"}:
                    if isinstance(v,str):
                        emit(f"  {k}: str len={len(v)} sample={v[:240]!r}")
                    elif isinstance(v,list):
                        emit(f"  {k}: list len={len(v)} sample={str(v[:1])[:600]}")
                    else:
                        emit(f"  {k}: {type(v).__name__}")
                elif k in {"session_id","system_id","candidate_index","round","split","execution_status","action","query","metadata"}:
                    emit(f"  {k}: {str(v)[:700]}")
        else:
            emit(f"ROW_TYPE: {type(row).__name__}")
    except Exception as e:
        emit(f"ERROR_READING_SAMPLE: {e}")

emit()
emit("="*110)
emit("PREFLIGHT COMPLETE")
emit("="*110)
emit("REPORT: "+str(OUT))

OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
