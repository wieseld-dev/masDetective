from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BULK = Path("artifacts/v4_bulk_mining")
FINAL = Path("artifacts/v4_final_dpo")
GOLD = Path("artifacts/v4_userlike_audit/gold_strict.jsonl")
OUT = Path("artifacts/v4_final_dpo/topic_match_rankings.jsonl")


def read(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write(p, xs):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


gold = read(GOLD)
dirs = read(BULK / "direction_negatives.jsonl")
metas = read(BULK / "meta_candidates.jsonl")

djud = {
    x["audit_id"]: x
    for x in read(FINAL / "direction_adjudications.jsonl")
}

mjud = {
    x["audit_id"]: x
    for x in read(FINAL / "meta_adjudications.jsonl")
}

gold_by = {int(x["gold_index"]): x for x in gold}


def did(x):
    return f"direction:{int(x['gold_index'])}:{int(x['sample_index'])}"


def mid(x):
    return f"meta:{int(x['gold_index'])}:{int(x['sample_index'])}"


# Only candidates that passed final high-precision adjudication.
dir_good = []

for x in dirs:
    j = djud[did(x)]

    if (
        j["direction"] == "ASSISTANT_TO_USER"
        and j["confidence"] == "HIGH"
        and j["strength"] in {
            "EXPLICIT_CLARIFICATION",
            "STRONG",
        }
    ):
        z = dict(x)
        z["secondary"] = j
        dir_good.append(z)


meta_good = []

for x in metas:
    j = mjud[mid(x)]

    if (
        j["target"] == "SYSTEM_SELF"
        and j["confidence"] == "HIGH"
        and j["strength"] in {"EXPLICIT", "STRONG"}
    ):
        z = dict(x)
        z["secondary"] = j
        meta_good.append(z)


print("clear direction candidates:", len(dir_good))
print("clear META candidates     :", len(meta_good))


# Encode every unique query once.
texts = []

for g in gold:
    texts.append(str(g["query"]))

for x in dir_good + meta_good:
    texts.append(str(x["query"]))

texts = list(dict.fromkeys(texts))

print("unique texts to embed:", len(texts))

model = SentenceTransformer("all-MiniLM-L6-v2")

E = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

emb = {
    t: E[i]
    for i, t in enumerate(texts)
}


def cosine(a, b):
    return float(np.dot(emb[a], emb[b]))


def word_jaccard(a, b):
    A = {
        w.strip(".,?!:;()[]{}'\"*").lower()
        for w in a.split()
        if len(w.strip(".,?!:;()[]{}'\"*")) >= 4
    }

    B = {
        w.strip(".,?!:;()[]{}'\"*").lower()
        for w in b.split()
        if len(w.strip(".,?!:;()[]{}'\"*")) >= 4
    }

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


by_dir = defaultdict(list)
by_meta = defaultdict(list)

for x in dir_good:
    by_dir[int(x["gold_index"])].append(x)

for x in meta_good:
    by_meta[int(x["gold_index"])].append(x)


ranked_rows = []


def rank_group(kind, gi, xs):
    chosen = str(gold_by[gi]["query"])

    out = []

    for x in xs:
        rejected = str(x["query"])

        sem = cosine(chosen, rejected)
        jac = word_jaccard(chosen, rejected)

        # Mostly semantic similarity; lexical overlap is only a small tie-breaker.
        score = 0.85 * sem + 0.15 * jac

        out.append({
            "kind": kind,
            "gold_index": gi,
            "sample_index": int(x["sample_index"]),
            "score": score,
            "semantic_similarity": sem,
            "word_jaccard": jac,
            "chosen": chosen,
            "rejected": rejected,
            "secondary": x["secondary"],
        })

    out.sort(
        key=lambda z: (
            z["score"],
            z["semantic_similarity"],
        ),
        reverse=True,
    )

    return out


for gi, xs in by_dir.items():
    ranked_rows.extend(rank_group("DIRECTION", gi, xs))

for gi, xs in by_meta.items():
    ranked_rows.extend(rank_group("META", gi, xs))


write(OUT, ranked_rows)


print()
print("========================================")
print(" TOPIC-MATCHED DIRECTION CANDIDATES")
print("========================================")

for gi in sorted(by_dir):
    xs = rank_group("DIRECTION", gi, by_dir[gi])

    print()
    print(
        f"===== DIRECTION G{gi} "
        f"| candidates={len(xs)} ====="
    )
    print("CHOSEN:", xs[0]["chosen"])

    for rank, x in enumerate(xs[:3], 1):
        print(
            f"  {rank}. "
            f"score={x['score']:.3f} "
            f"sem={x['semantic_similarity']:.3f} "
            f"jac={x['word_jaccard']:.3f}"
        )
        print("     -", x["rejected"])


print()
print("========================================")
print(" TOPIC-MATCHED META CANDIDATES")
print("========================================")

for gi in sorted(by_meta):
    xs = rank_group("META", gi, by_meta[gi])

    print()
    print(
        f"===== META G{gi} "
        f"| candidates={len(xs)} ====="
    )
    print("CHOSEN:", xs[0]["chosen"])

    for rank, x in enumerate(xs[:3], 1):
        print(
            f"  {rank}. "
            f"score={x['score']:.3f} "
            f"sem={x['semantic_similarity']:.3f} "
            f"jac={x['word_jaccard']:.3f}"
        )
        print("     -", x["rejected"])


print()
print("wrote:", OUT)
