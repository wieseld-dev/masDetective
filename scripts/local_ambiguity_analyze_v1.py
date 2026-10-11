from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import EvidenceFrame
from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/local_ambiguity_v1"
)

BELIEF_PATH = Path(
    "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
)

OUT = ROOT / "analysis"

HORDER = ["00", "10", "01", "11"]

TYPE_ORDER = [
    "A_DOMINANT",
    "B_DOMINANT",
    "BOUNDARY",
    "JOINT",
]

EPS = 1e-9


SCENARIOS = {
    # Is A present? B is believed mostly absent.
    "A_TEST": {
        "00": .45,
        "10": .45,
        "01": .05,
        "11": .05,
    },

    # Is B present? A is believed mostly absent.
    "B_TEST": {
        "00": .45,
        "10": .05,
        "01": .45,
        "11": .05,
    },

    # One is probably present; which one?
    "A_VS_B": {
        "00": .05,
        "10": .45,
        "01": .45,
        "11": .05,
    },

    # Strong possibility that both are present vs only one.
    "BOTH_TEST": {
        "00": .05,
        "10": .225,
        "01": .225,
        "11": .50,
    },
}


EXPECTED_TYPE = {
    "A_TEST": "A_DOMINANT",
    "B_TEST": "B_DOMINANT",
    "A_VS_B": "BOUNDARY",
    "BOTH_TEST": "JOINT",
}


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def clip(x):
    return float(
        np.clip(
            float(x),
            1e-6,
            1.0 - 1e-6,
        )
    )


def unprojected_role_probs(
    bundle,
    history,
):
    raw = bundle._raw_role_vector_with(
        bundle.role_model,
        history,
    )

    p = bundle._apply_platt(
        bundle.role_platt,
        raw,
    )

    return {
        r: clip(p[i])
        for i, r in enumerate(TRAIN_ROLES)
    }


def sid_of(row):
    return str(
        row.get("system_id")
        or row.get("session_id")
        or ""
    )


def entropy(p):
    x = np.asarray(p, float)
    x = np.clip(x, EPS, 1.0)
    return float(
        -np.sum(x * np.log(x))
    )


def state_likelihood(pA, pB):
    return np.asarray([
        (1-pA) * (1-pB),   # 00
        pA * (1-pB),       # 10
        (1-pA) * pB,       # 01
        pA * pB,           # 11
    ], dtype=float)


def prior_vec(scenario):
    d = SCENARIOS[scenario]

    w = np.asarray(
        [d[h] for h in HORDER],
        float,
    )

    return w / w.sum()


def expected_information_gain(
    rows_by_h,
    scenario,
):
    """
    Empirical counterfactual EIG.

    For every possible true local state h, we already observed
    what this exact query produces in the matched MAS H_h.

    Use pA,pB as a local pseudo-likelihood and compute expected
    posterior entropy reduction under the current scenario prior.
    """
    prior = prior_vec(scenario)
    H0 = entropy(prior)

    expected_H = 0.0
    expected_log_gain = 0.0

    for hi, h in enumerate(HORDER):
        r = rows_by_h[h]

        lik = state_likelihood(
            r["pA"],
            r["pB"],
        )

        post = prior * lik

        if post.sum() <= EPS:
            post = prior.copy()
        else:
            post /= post.sum()

        expected_H += (
            prior[hi]
            * entropy(post)
        )

        expected_log_gain += (
            prior[hi]
            * (
                math.log(
                    max(post[hi], EPS)
                )
                -
                math.log(
                    max(prior[hi], EPS)
                )
            )
        )

    return {
        "entropy_reduction":
            float(H0 - expected_H),

        "true_state_log_gain":
            float(expected_log_gain),
    }


def bootstrap_pair_ci(values, B=30000):
    vals = np.asarray(values, float)

    rng = np.random.default_rng(90173)

    boot = np.empty(B)

    for b in range(B):
        boot[b] = np.mean(
            rng.choice(
                vals,
                size=len(vals),
                replace=True,
            )
        )

    return [
        float(np.quantile(boot, .025)),
        float(np.quantile(boot, .975)),
    ]


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = json.loads(
        (
            ROOT / "manifest_privileged.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    candidates = {
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        ): x
        for x in read_jsonl(
            ROOT / "candidates_dev_round_01.jsonl"
        )
    }

    responses = {
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        ): x
        for x in read_jsonl(
            ROOT / "responses_dev_round_01.jsonl"
        )
    }

    scored = {}

    for x in read_jsonl(
        ROOT / "scored_dev_round_01.jsonl"
    ):
        sid = sid_of(x)

        idx = int(
            dict(
                x.get("action", {})
            ).get(
                "variant",
                x.get(
                    "candidate_index",
                    0,
                ),
            )
        )

        scored[(sid, idx)] = x

    truth = {
        str(x["session_id"]): x
        for x in manifest["systems"]
    }

    bundle = FrozenBeliefBundle.load(
        BELIEF_PATH
    )

    rows = []

    for key, c in candidates.items():
        if key not in responses:
            raise RuntimeError(
                f"missing response {key}"
            )

        if key not in scored:
            raise RuntimeError(
                f"missing score {key}"
            )

        resp = responses[key]
        sc = scored[key]

        if (
            not c.get("valid")
            or resp.get(
                "execution_status"
            ) != "ok"
        ):
            continue

        sid, idx = key

        m = truth[sid]

        A = str(m["role_a"])
        B = str(m["role_b"])

        v135 = {
            str(k): float(v)
            for k, v in dict(
                sc.get(
                    "evidence",
                    {},
                )
            ).items()
        }

        v134 = {
            str(k): float(v)
            for k, v in dict(
                sc.get(
                    "metadata",
                    {},
                ).get(
                    "evidence_v134",
                    {},
                )
            ).items()
        }

        if A not in v135 or B not in v135:
            raise RuntimeError(
                f"missing judge role evidence {A}/{B}"
            )

        answer = str(
            resp.get("final_output")
            or resp.get("output")
            or ""
        )

        frame = EvidenceFrame(
            query=str(c["query"]),
            answer=answer,
            v135=v135,
            v134=v134,
        )

        rp = unprojected_role_probs(
            bundle,
            [frame],
        )

        h = str(m["hypothesis"])

        rows.append({
            "sid": sid,
            "pair_id": m["pair_id"],
            "role_a": A,
            "role_b": B,
            "hypothesis": h,
            "candidate_index": idx,
            "probe_type":
                c["probe_type"],
            "variant":
                int(c["variant"]),
            "query":
                c["query"],
            "pA":
                float(rp[A]),
            "pB":
                float(rp[B]),
            "A_present":
                int(h in ("10", "11")),
            "B_present":
                int(h in ("01", "11")),
        })

    print(
        f"[data] rows={len(rows)} "
        f"pairs={len(set(r['pair_id'] for r in rows))}"
    )

    if len(rows) != 384:
        raise RuntimeError(
            f"Expected 384 scored observations, got {len(rows)}"
        )

    # ----------------------------------------------------------
    # SENSOR IDENTIFIABILITY
    # ----------------------------------------------------------

    auc_A = roc_auc_score(
        [r["A_present"] for r in rows],
        [r["pA"] for r in rows],
    )

    auc_B = roc_auc_score(
        [r["B_present"] for r in rows],
        [r["pB"] for r in rows],
    )

    state_correct = []

    for r in rows:
        pred = HORDER[
            int(
                np.argmax(
                    state_likelihood(
                        r["pA"],
                        r["pB"],
                    )
                )
            )
        ]

        state_correct.append(
            int(
                pred
                == r["hypothesis"]
            )
        )

    four_state_acc = float(
        np.mean(state_correct)
    )

    # Matched factorial causal-style presence effects.
    # For each pair/query:
    #   A effect on pA holding B slot both ways
    #   B effect on pB holding A slot both ways
    by_query = defaultdict(dict)

    for r in rows:
        key = (
            r["pair_id"],
            r["candidate_index"],
        )

        by_query[key][
            r["hypothesis"]
        ] = r

    a_effects = []
    b_effects = []

    for key, hh in by_query.items():
        if set(hh) != set(HORDER):
            continue

        aeff = 0.5 * (
            (
                hh["10"]["pA"]
                - hh["00"]["pA"]
            )
            +
            (
                hh["11"]["pA"]
                - hh["01"]["pA"]
            )
        )

        beff = 0.5 * (
            (
                hh["01"]["pB"]
                - hh["00"]["pB"]
            )
            +
            (
                hh["11"]["pB"]
                - hh["10"]["pB"]
            )
        )

        a_effects.append(aeff)
        b_effects.append(beff)

    # ----------------------------------------------------------
    # LOCAL ACTIVE-PROBING VALUE
    # ----------------------------------------------------------

    query_eig = {}

    for (
        pair_id,
        cand_idx,
    ), hh in by_query.items():

        if set(hh) != set(HORDER):
            continue

        anyrow = hh["00"]

        for scenario in SCENARIOS:
            v = expected_information_gain(
                hh,
                scenario,
            )

            query_eig[
                (
                    pair_id,
                    cand_idx,
                    scenario,
                )
            ] = {
                **v,
                "probe_type":
                    anyrow[
                        "probe_type"
                    ],
                "variant":
                    anyrow[
                        "variant"
                    ],
                "query":
                    anyrow[
                        "query"
                    ],
            }

    pair_ids = sorted(
        set(r["pair_id"] for r in rows)
    )

    per_pair = []

    aligned_hits = []
    switches = []
    type_adaptive_gains = []
    query_adaptive_gains = []

    for pair_id in pair_ids:

        best_types = {}
        best_queries = {}

        type_values = {}

        # ------------------------------------------------------
        # Value under each current-belief scenario
        # ------------------------------------------------------

        for scenario in SCENARIOS:

            qrows = [
                (
                    idx,
                    query_eig[
                        (
                            pair_id,
                            idx,
                            scenario,
                        )
                    ],
                )
                for idx in range(12)
            ]

            # query-level oracle
            qbest_idx, qbest = max(
                qrows,
                key=lambda z:
                    z[1][
                        "true_state_log_gain"
                    ],
            )

            best_queries[
                scenario
            ] = {
                "candidate_index":
                    qbest_idx,
                **qbest,
            }

            # type-level value = mean of three independent variants
            tvals = {}

            for typ in TYPE_ORDER:
                vals = [
                    x[
                        "true_state_log_gain"
                    ]
                    for _, x in qrows
                    if x["probe_type"]
                    == typ
                ]

                assert len(vals) == 3

                tvals[typ] = float(
                    np.mean(vals)
                )

            type_values[
                scenario
            ] = tvals

            bt = max(
                tvals,
                key=tvals.get,
            )

            best_types[
                scenario
            ] = bt

            aligned_hits.append(
                int(
                    bt
                    == EXPECTED_TYPE[
                        scenario
                    ]
                )
            )

        # ------------------------------------------------------
        # Does the optimal TYPE change with belief?
        # ------------------------------------------------------

        switches.append(
            int(
                len(
                    set(
                        best_types.values()
                    )
                )
                >= 2
            )
        )

        # Adaptive best type:
        adapt_type = float(
            np.mean([
                max(
                    type_values[s].values()
                )
                for s in SCENARIOS
            ])
        )

        # Best single fixed type used for ALL belief states.
        fixed_type_scores = {}

        for typ in TYPE_ORDER:
            fixed_type_scores[typ] = float(
                np.mean([
                    type_values[s][typ]
                    for s in SCENARIOS
                ])
            )

        best_fixed_type = max(
            fixed_type_scores,
            key=fixed_type_scores.get,
        )

        fixed_type = float(
            fixed_type_scores[
                best_fixed_type
            ]
        )

        type_gain = (
            adapt_type
            - fixed_type
        )

        type_adaptive_gains.append(
            type_gain
        )

        # Same calculation at exact-query level.
        adaptive_query = []

        for scenario in SCENARIOS:
            adaptive_query.append(
                max(
                    query_eig[
                        (
                            pair_id,
                            idx,
                            scenario,
                        )
                    ][
                        "true_state_log_gain"
                    ]
                    for idx in range(12)
                )
            )

        adaptive_query = float(
            np.mean(
                adaptive_query
            )
        )

        fixed_query_scores = []

        for idx in range(12):
            fixed_query_scores.append(
                float(
                    np.mean([
                        query_eig[
                            (
                                pair_id,
                                idx,
                                s,
                            )
                        ][
                            "true_state_log_gain"
                        ]
                        for s in SCENARIOS
                    ])
                )
            )

        fixed_query = max(
            fixed_query_scores
        )

        query_gain = (
            adaptive_query
            - fixed_query
        )

        query_adaptive_gains.append(
            query_gain
        )

        per_pair.append({
            "pair_id":
                pair_id,

            "best_types":
                best_types,

            "expected_types":
                EXPECTED_TYPE,

            "type_values":
                type_values,

            "type_switch":
                bool(
                    switches[-1]
                ),

            "best_fixed_type":
                best_fixed_type,

            "adaptive_type_value":
                adapt_type,

            "fixed_type_value":
                fixed_type,

            "type_adaptive_gain":
                type_gain,

            "query_adaptive_gain":
                query_gain,

            "best_queries":
                best_queries,
        })

    mean_type_gain = float(
        np.mean(
            type_adaptive_gains
        )
    )

    mean_query_gain = float(
        np.mean(
            query_adaptive_gains
        )
    )

    type_gain_ci = bootstrap_pair_ci(
        type_adaptive_gains
    )

    query_gain_ci = bootstrap_pair_ci(
        query_adaptive_gains
    )

    positive_type_pairs = sum(
        g > 1e-8
        for g in type_adaptive_gains
    )

    switch_rate = float(
        np.mean(switches)
    )

    aligned_rate = float(
        np.mean(aligned_hits)
    )

    # ----------------------------------------------------------
    # DECISION
    # ----------------------------------------------------------

    sensor_ok = (
        min(auc_A, auc_B) >= 0.65
    )

    adaptive_ok = (
        mean_type_gain > 0
        and positive_type_pairs >= 6
        and switch_rate >= 0.50
    )

    if sensor_ok and adaptive_ok:
        verdict = "GO_LOCAL_ADAPTIVE"
    elif sensor_ok:
        verdict = "STATIC_SIGNAL_ONLY"
    else:
        verdict = "SENSOR_NO_GO"

    summary = {
        "n_pairs":
            len(pair_ids),

        "n_systems":
            len(
                set(
                    r["sid"]
                    for r in rows
                )
            ),

        "n_queries_per_pair":
            12,

        "n_mas_observations":
            len(rows),

        "sensor": {
            "A_presence_auc":
                float(auc_A),

            "B_presence_auc":
                float(auc_B),

            "four_state_accuracy":
                four_state_acc,

            "mean_matched_A_effect_on_pA":
                float(
                    np.mean(
                        a_effects
                    )
                ),

            "mean_matched_B_effect_on_pB":
                float(
                    np.mean(
                        b_effects
                    )
                ),
        },

        "adaptivity": {
            "type_switch_pair_rate":
                switch_rate,

            "aligned_best_type_rate":
                aligned_rate,

            "mean_type_adaptive_gain":
                mean_type_gain,

            "type_adaptive_gain_ci95":
                type_gain_ci,

            "positive_type_gain_pairs":
                positive_type_pairs,

            "mean_query_adaptive_gain":
                mean_query_gain,

            "query_adaptive_gain_ci95":
                query_gain_ci,
        },

        "verdict":
            verdict,

        "per_pair":
            per_pair,
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (
        OUT / "observations.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    L = []

    L.append("=" * 104)
    L.append(
        "LOCAL AMBIGUITY ACTIVE PROBING V1"
    )
    L.append("=" * 104)

    L.append(
        f"pairs={len(pair_ids)} | "
        f"systems={len(set(r['sid'] for r in rows))} | "
        f"observations={len(rows)}"
    )

    L.append(
        "Controlled design: K=5 for every system; "
        "only A/B slots change; mode=fanout2."
    )

    L.append("")
    L.append("SENSOR IDENTIFIABILITY")
    L.append(
        f"  A-presence AUC             = {auc_A:.4f}"
    )
    L.append(
        f"  B-presence AUC             = {auc_B:.4f}"
    )
    L.append(
        f"  4-state local accuracy     = {four_state_acc:.4f} "
        f"(chance=.2500)"
    )
    L.append(
        f"  matched A->pA effect       = {np.mean(a_effects):+.6f}"
    )
    L.append(
        f"  matched B->pB effect       = {np.mean(b_effects):+.6f}"
    )

    L.append("")
    L.append("DOES PROBE VALUE DEPEND ON CURRENT BELIEF?")
    L.append(
        f"  pairs where best probe TYPE switches "
        f"across belief states = {sum(switches)}/{len(switches)} "
        f"({switch_rate:.3f})"
    )
    L.append(
        f"  best type matches intended local experiment "
        f"= {aligned_rate:.3f} (chance≈.250)"
    )
    L.append(
        f"  mean adaptive TYPE gain    = {mean_type_gain:+.6f}"
    )
    L.append(
        f"  type gain 95% pair CI      = "
        f"[{type_gain_ci[0]:+.6f},{type_gain_ci[1]:+.6f}]"
    )
    L.append(
        f"  positive type-gain pairs   = "
        f"{positive_type_pairs}/{len(pair_ids)}"
    )
    L.append(
        f"  mean adaptive QUERY gain   = {mean_query_gain:+.6f}"
    )
    L.append(
        f"  query gain 95% pair CI     = "
        f"[{query_gain_ci[0]:+.6f},{query_gain_ci[1]:+.6f}]"
    )

    L.append("")
    L.append("BEST TYPE BY PAIR / BELIEF")

    for p in per_pair:
        L.append(
            f"  {p['pair_id']} | "
            + " | ".join(
                f"{s}:{p['best_types'][s]}"
                for s in SCENARIOS
            )
            + f" | gain={p['type_adaptive_gain']:+.5f}"
        )

    L.append("")
    L.append("=" * 104)
    L.append(
        f"VERDICT: {verdict}"
    )

    if verdict == "GO_LOCAL_ADAPTIVE":
        L.append(
            "Local probe value changes with the current ambiguity. "
            "Next step: learn a selector that maps current pair-belief "
            "to A/B/BOUNDARY/JOINT probe acquisition."
        )

    elif verdict == "STATIC_SIGNAL_ONLY":
        L.append(
            "The A/B sensor works, but probe specialization does not "
            "show enough state-dependent value yet. Improve probe design "
            "before building an adaptive controller."
        )

    else:
        L.append(
            "The current Judge/belief sensor cannot reliably distinguish "
            "the matched A/B presence states. Do not train a selector yet."
        )

    L.append("=" * 104)

    report = "\n".join(L)

    (
        OUT / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)


if __name__ == "__main__":
    main()
