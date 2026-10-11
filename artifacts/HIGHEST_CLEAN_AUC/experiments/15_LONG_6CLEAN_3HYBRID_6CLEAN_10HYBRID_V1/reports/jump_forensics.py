import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

JUMPS = [10, 16, 21, 24]


def read_jsonl(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sid(x):
    return str(
        x.get(
            "session_id",
            x.get("system_id", "")
        )
    )


def get_probs(x):
    candidates = [
        x.get("role_probs"),
        x.get("role_probabilities"),
        x.get("roles"),
    ]

    for container_key in [
        "belief",
        "public_belief",
        "state",
    ]:
        c = x.get(container_key)

        if isinstance(c, dict):
            candidates.extend([
                c.get("role_probs"),
                c.get("role_probabilities"),
            ])

    for c in candidates:
        if isinstance(c, dict):
            try:
                return {
                    str(k): float(v)
                    for k, v in c.items()
                }
            except Exception:
                pass

    raise RuntimeError(
        "Could not find role probabilities. "
        f"Available keys: {sorted(x.keys())}"
    )


def pair_value(tp, fp):
    if tp > fp:
        return 1.0
    if tp < fp:
        return 0.0
    return 0.5


def auc_from_probs(probs, truth):
    truth = set(truth)

    tr = [
        r for r in probs
        if r in truth
    ]

    fa = [
        r for r in probs
        if r not in truth
    ]

    vals = [
        pair_value(
            probs[t],
            probs[f],
        )
        for t in tr
        for f in fa
    ]

    return sum(vals) / len(vals)


def rank_text(probs, truth, n=8):
    truth = set(truth)

    xs = sorted(
        probs.items(),
        key=lambda z: z[1],
        reverse=True,
    )[:n]

    return " | ".join(
        f"{i+1}:{r}"
        f"{'*' if r in truth else ''}"
        f"={p:.4f}"
        for i, (r, p) in enumerate(xs)
    )


def uncertainty_text(probs, truth, n=6):
    truth = set(truth)

    # High p(1-p) = locally uncertain.
    xs = sorted(
        probs.items(),
        key=lambda z: z[1] * (1.0 - z[1]),
        reverse=True,
    )[:n]

    return " | ".join(
        f"{r}"
        f"{'*' if r in truth else ''}"
        f"={p:.4f}"
        for r, p in xs
    )


def target_hint(x):
    places = [x]

    for k in [
        "metadata",
        "meta",
        "generation_metadata",
        "action",
    ]:
        if isinstance(x.get(k), dict):
            places.append(x[k])

    wanted = [
        "target_role",
        "role",
        "focus_role",
        "hypothesis_role",
        "target",
        "focus",
    ]

    found = []

    for d in places:
        for k in wanted:
            if k in d and d[k] is not None:
                found.append(
                    f"{k}={d[k]}"
                )

    return ", ".join(found)


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): tuple(
        map(str, x["roles"])
    )
    for x in manifest["systems"]
    if x.get("split") == "dev"
}


print("=" * 110)
print("ADAPTIVE JUMP FORENSICS")
print("=" * 110)

for rnd in JUMPS:

    before_path = (
        ROOT /
        f"states_dev_round_{rnd-1:02d}.jsonl"
    )

    after_path = (
        ROOT /
        f"states_dev_round_{rnd:02d}.jsonl"
    )

    pool_path = (
        ROOT /
        f"candidate_pool_dev_round_{rnd:02d}.jsonl"
    )

    selected_path = (
        ROOT /
        f"candidates_dev_round_{rnd:02d}.jsonl"
    )

    response_path = (
        ROOT /
        f"responses_dev_round_{rnd:02d}.jsonl"
    )

    score_path = (
        ROOT /
        f"scored_dev_round_{rnd:02d}.jsonl"
    )

    before = {
        sid(x): x
        for x in read_jsonl(before_path)
    }

    after = {
        sid(x): x
        for x in read_jsonl(after_path)
    }

    selected = {
        sid(x): x
        for x in read_jsonl(selected_path)
    }

    responses = {
        sid(x): x
        for x in read_jsonl(response_path)
    }

    scored = {
        sid(x): x
        for x in read_jsonl(score_path)
    }

    pool = defaultdict(list)

    for x in read_jsonl(pool_path):
        pool[sid(x)].append(x)

    systems = sorted(
        set(before) &
        set(after) &
        set(truth)
    )

    print()
    print()
    print("#" * 110)
    print(f"ROUND {rnd}")
    print("#" * 110)

    total_before = []
    total_after = []

    contributions = []

    for system in systems:

        pb = get_probs(before[system])
        pa = get_probs(after[system])

        ab = auc_from_probs(
            pb,
            truth[system],
        )

        aa = auc_from_probs(
            pa,
            truth[system],
        )

        total_before.append(ab)
        total_after.append(aa)

        contributions.append(
            (
                aa - ab,
                system,
                ab,
                aa,
            )
        )

    mean_before = (
        sum(total_before) /
        len(total_before)
    )

    mean_after = (
        sum(total_after) /
        len(total_after)
    )

    delta = (
        mean_after -
        mean_before
    )

    print(
        f"GLOBAL: "
        f"{mean_before:.6f} -> "
        f"{mean_after:.6f} "
        f"delta={delta:+.6f} "
        f"≈ {delta*255:+.1f}/255 pair-ordering points"
    )

    print()
    print("PER-SYSTEM CONTRIBUTION:")

    for d, system, ab, aa in sorted(
        contributions,
        reverse=True,
    ):
        print(
            f"  {system:40s} "
            f"{ab:.6f} -> {aa:.6f} "
            f"delta={d:+.6f} "
            f"pairs≈{d*51:+.1f}"
        )

    for d, system, ab, aa in sorted(
        contributions,
        reverse=True,
    ):

        # Only deeply inspect systems that changed.
        if abs(d) < 1e-12:
            continue

        pb = get_probs(before[system])
        pa = get_probs(after[system])

        tr = set(truth[system])

        sel = selected.get(
            system,
            {}
        )

        resp = responses.get(
            system,
            {}
        )

        sc = scored.get(
            system,
            {}
        )

        chosen_idx = sel.get(
            "candidate_index",
            "?"
        )

        query = str(
            sel.get(
                "query",
                ""
            )
        )

        answer = str(
            resp.get(
                "final_output",
                ""
            )
        )

        evidence = sc.get(
            "evidence",
            {}
        )

        print()
        print("-" * 110)
        print(
            f"SYSTEM: {system}"
        )

        print(
            "TRUTH:",
            ", ".join(
                sorted(tr)
            )
        )

        print(
            f"AUC: {ab:.6f} -> {aa:.6f} "
            f"delta={aa-ab:+.6f} "
            f"≈ {(aa-ab)*51:+.1f} pair points"
        )

        print()
        print(
            "PRE TOP:",
            rank_text(
                pb,
                tr,
                10,
            )
        )

        print(
            "POST TOP:",
            rank_text(
                pa,
                tr,
                10,
            )
        )

        print(
            "PRE UNCERTAINTY:",
            uncertainty_text(
                pb,
                tr,
                7,
            )
        )

        print()
        print(
            f"SELECTED CANDIDATE = {chosen_idx}"
        )

        hint = target_hint(sel)

        if hint:
            print(
                "SELECTED TARGET META:",
                hint,
            )

        print(
            "QUERY:",
            query,
        )

        print()
        print("ALL 4 GENERATED CANDIDATES:")

        candidates = sorted(
            pool.get(system, []),
            key=lambda x: int(
                x.get(
                    "candidate_index",
                    0,
                )
            ),
        )

        for c in candidates:

            ci = c.get(
                "candidate_index",
                "?"
            )

            mark = (
                " <== SELECTED"
                if str(ci) == str(chosen_idx)
                else ""
            )

            hint = target_hint(c)

            print(
                f"  C{ci}{mark}"
            )

            if hint:
                print(
                    f"     META: {hint}"
                )

            print(
                "     Q:",
                str(
                    c.get(
                        "query",
                        "",
                    )
                ),
            )

        print()
        print(
            "MAS RESPONSE:",
            answer[:1400].replace(
                "\n",
                " ",
            )
        )

        if evidence:
            ev = {
                str(k): float(v)
                for k, v in evidence.items()
            }

            pos = sorted(
                ev.items(),
                key=lambda z: z[1],
                reverse=True,
            )[:7]

            neg = sorted(
                ev.items(),
                key=lambda z: z[1],
            )[:5]

            print()
            print(
                "SENSOR TOP POSITIVE:",
                " | ".join(
                    f"{r}"
                    f"{'*' if r in tr else ''}"
                    f"={v:+.3f}"
                    for r, v in pos
                )
            )

            print(
                "SENSOR MOST NEGATIVE:",
                " | ".join(
                    f"{r}"
                    f"{'*' if r in tr else ''}"
                    f"={v:+.3f}"
                    for r, v in neg
                )
            )

        flips_pos = []
        flips_neg = []

        for t in sorted(tr):

            if t not in pb or t not in pa:
                continue

            for f in sorted(
                set(pb) - tr
            ):

                if f not in pa:
                    continue

                vb = pair_value(
                    pb[t],
                    pb[f],
                )

                va = pair_value(
                    pa[t],
                    pa[f],
                )

                change = va - vb

                if change > 0:
                    flips_pos.append(
                        (
                            change,
                            t,
                            f,
                            pb[t],
                            pb[f],
                            pa[t],
                            pa[f],
                        )
                    )

                elif change < 0:
                    flips_neg.append(
                        (
                            change,
                            t,
                            f,
                            pb[t],
                            pb[f],
                            pa[t],
                            pa[f],
                        )
                    )

        print()
        print(
            f"GOOD TRUE-vs-FALSE FLIPS: "
            f"{len(flips_pos)}"
        )

        for (
            ch,
            t,
            f,
            bt,
            bf,
            at,
            af,
        ) in flips_pos[:15]:

            print(
                f"  + {t} > {f}: "
                f"before {bt:.4f} vs {bf:.4f} "
                f"-> after {at:.4f} vs {af:.4f}"
            )

        print(
            f"BAD TRUE-vs-FALSE FLIPS: "
            f"{len(flips_neg)}"
        )

        for (
            ch,
            t,
            f,
            bt,
            bf,
            at,
            af,
        ) in flips_neg[:10]:

            print(
                f"  - {t} > {f}: "
                f"before {bt:.4f} vs {bf:.4f} "
                f"-> after {at:.4f} vs {af:.4f}"
            )


print()
print()
print("=" * 110)
print("END")
print("=" * 110)
