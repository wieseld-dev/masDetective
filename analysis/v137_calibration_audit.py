from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis.train_v13_6b_activation_judge as tr


ADAPTER = Path("artifacts/v13_7_fp3b_overnight/best_adapter")
OUT = Path("artifacts/v13_7_calibration_audit_v1")

V13 = Path("artifacts/v13_probe_disambiguation.jsonl")
V11 = Path("artifacts/v11_probe_discovery.jsonl")

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SEED = 1331
FOLD = 0
N_SPLITS = 5
V11_WEIGHT = 0.70

TARGET_FPR = 0.10
CAL_FRACTION = 0.60

MAX_LENGTH = 1024
BATCH = 8

ROLES = list(tr.ROLE_UNIVERSE)


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def objdict(x):
    if isinstance(x, dict):
        return x
    try:
        return vars(x)
    except TypeError:
        return {
            k: getattr(x, k)
            for k in dir(x)
            if not k.startswith("_")
            and not callable(getattr(x, k))
        }


def component_value(x):
    d = objdict(x)

    preferred = [
        "component_id",
        "component",
        "component_key",
        "dsu_component",
        "split_component",
    ]

    for k in preferred:
        if k in d:
            return str(d[k])

    for k, v in d.items():
        if "component" in k.lower():
            return str(v)

    raise RuntimeError(
        "Could not find component field. "
        f"Available keys: {sorted(d.keys())}"
    )


def stable_bucket(x, salt):
    h = hashlib.sha256(
        f"{salt}|{x}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") % 100


def field_value(row, key):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key)


def infer_fields(rows):
    first = objdict(rows[0])
    keys = list(first)

    text_field = "text" if "text" in keys else None
    label_field = "label" if "label" in keys else None

    if text_field is None or label_field is None:
        raise RuntimeError(
            f"Expected text/label fields, got {keys}"
        )

    role_field = None

    candidates = [
        "target_role",
        "role",
        "candidate_role",
        "target",
    ]

    for k in candidates:
        if k not in keys:
            continue

        vals = {
            str(field_value(x, k))
            for x in rows[: min(len(rows), 1000)]
        }

        if vals and vals <= set(ROLES):
            role_field = k
            break

    if role_field is None:
        for k in keys:
            vals = {
                str(field_value(x, k))
                for x in rows[: min(len(rows), 1000)]
                if field_value(x, k) is not None
            }
            if vals and vals <= set(ROLES):
                role_field = k
                break

    if role_field is None:
        raise RuntimeError(
            f"Could not infer role field from: {keys}"
        )

    self_field = None
    hard_field = None

    for k in keys:
        lk = k.lower()
        if "self_target_absent" in lk:
            self_field = k
        if "hard_neighbor_absent" in lk:
            hard_field = k

    print("FIELDS")
    print(" text :", text_field)
    print(" label:", label_field)
    print(" role :", role_field)
    print(" self :", self_field)
    print(" hard :", hard_field)

    return (
        text_field,
        label_field,
        role_field,
        self_field,
        hard_field,
    )


def counts_ok(rows, role_field, label_field):
    for role in ROLES:
        ys = [
            int(field_value(x, label_field))
            for x in rows
            if str(field_value(x, role_field)) == role
        ]

        pos = sum(ys)
        neg = len(ys) - pos

        if pos < 5 or neg < 20:
            return False

    return True


@torch.no_grad()
def infer_logits(model, tok, texts, device):
    out = []

    for s in range(0, len(texts), BATCH):
        batch = texts[s:s+BATCH]

        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        z = model(**enc).logits
        z = z.reshape(-1).float().cpu().numpy()

        out.append(z)

        if (s // BATCH) % 50 == 0:
            print(
                f"[infer] {min(s+BATCH,len(texts))}/"
                f"{len(texts)}",
                flush=True,
            )

    return np.concatenate(out)


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def choose_threshold(
    p,
    y,
    self_abs,
    target_fpr=0.10,
):
    # Prefer the exact confound subset when enough examples exist.
    neg_mask = (y == 0)

    if self_abs is not None:
        conf_mask = neg_mask & self_abs
    else:
        conf_mask = np.zeros_like(neg_mask)

    if conf_mask.sum() >= 10:
        constraint_mask = conf_mask
        source = "self_target_absent"
    else:
        constraint_mask = neg_mask
        source = "all_negatives"

    # Include an always-negative operating point.
    thresholds = sorted(
        set(
            [0.0, 1.0]
            + [float(x) for x in np.linspace(0.01, 0.99, 99)]
            + [float(x) for x in np.quantile(
                p,
                np.linspace(0.02, 0.98, 49)
            )]
        )
    )

    best = None

    for t in thresholds:
        pred = p >= t

        fpr = (
            float(pred[constraint_mask].mean())
            if constraint_mask.sum()
            else 0.0
        )

        pos = y == 1
        tpr = (
            float(pred[pos].mean())
            if pos.sum()
            else 0.0
        )

        overall_fpr = (
            float(pred[neg_mask].mean())
            if neg_mask.sum()
            else 0.0
        )

        feasible = fpr <= target_fpr + 1e-12

        if not feasible:
            continue

        cand = (
            tpr,
            -overall_fpr,
            -t,
            t,
            fpr,
            overall_fpr,
        )

        if best is None or cand[:3] > best[:3]:
            best = cand

    if best is None:
        return 1.0, source

    return float(best[3]), source


def metric_block(
    probs,
    labels,
    role_ids,
    thresholds,
    self_abs=None,
    hard_abs=None,
):
    pred = np.zeros_like(labels, dtype=np.int32)

    for j, role in enumerate(role_ids):
        pred[j] = int(
            probs[j] >= thresholds[ROLES.index(role)]
        )

    pos = labels == 1
    neg = labels == 0

    recall = float(pred[pos].mean()) if pos.sum() else None
    fpr = float(pred[neg].mean()) if neg.sum() else None

    bal = None
    if recall is not None and fpr is not None:
        bal = 0.5 * (recall + (1.0 - fpr))

    self_fpr = None
    if self_abs is not None and self_abs.sum():
        m = (labels == 0) & self_abs
        if m.sum():
            self_fpr = float(pred[m].mean())

    hard_fpr = None
    if hard_abs is not None and hard_abs.sum():
        m = (labels == 0) & hard_abs
        if m.sum():
            hard_fpr = float(pred[m].mean())

    per_auc = {}

    for role in ROLES:
        m = role_ids == role
        per_auc[role] = safe_auc(
            labels[m],
            probs[m],
        )

    aucs = [
        x for x in per_auc.values()
        if x is not None
    ]

    return {
        "n": int(len(labels)),
        "positive_n": int(pos.sum()),
        "negative_n": int(neg.sum()),
        "positive_recall": recall,
        "negative_fpr": fpr,
        "negative_specificity":
            None if fpr is None else 1.0-fpr,
        "balanced_accuracy": bal,
        "self_target_absent_fpr": self_fpr,
        "hard_neighbor_absent_fpr": hard_fpr,
        "brier": float(np.mean(
            (probs-labels) ** 2
        )),
        "macro_auc": float(np.mean(aucs)),
        "per_role_auc": per_auc,
        "predicted_positive_rate":
            float(pred.mean()),
    }


def main():
    if OUT.exists() and (OUT/"summary.json").exists():
        raise RuntimeError(
            f"Refusing overwrite: {OUT}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("V13.7 COMPONENT-SAFE CALIBRATION AUDIT")
    print("="*100)

    args = SimpleNamespace(
        seed=SEED,
        n_splits=N_SPLITS,
        fold=FOLD,
    )

    rd = dict(tr.ROLE_DESCRIPTIONS)

    r13 = tr.read_jsonl(V13)
    r11 = tr.read_jsonl(V11)

    c13, _ = tr.normalize_rows(
        r13, "v13", rd, 1.0
    )

    c11, _ = tr.normalize_rows(
        r11, "v11", rd, V11_WEIGHT
    )

    cases = tr.attach_components(c13 + c11)

    _, val_cases, split_report = tr.split_cases(
        cases,
        args,
    )

    print("validation cases:", len(val_cases))

    # ------------------------------------------------------
    # Split existing validation fold again BY COMPONENT.
    # 60% calibration, 40% untouched calibration-test.
    # Find deterministic salt that preserves every role.
    # ------------------------------------------------------
    chosen = None

    for salt in range(1000):
        cal_cases = []
        test_cases = []

        for c in val_cases:
            comp = component_value(c)

            if stable_bucket(comp, salt) < int(
                CAL_FRACTION*100
            ):
                cal_cases.append(c)
            else:
                test_cases.append(c)

        if not cal_cases or not test_cases:
            continue

        cal_rows, _ = tr.build_presence(
            cal_cases, rd
        )
        test_rows, _ = tr.build_presence(
            test_cases, rd
        )

        fields = infer_fields(cal_rows)
        tf, lf, rf, sf, hf = fields

        if (
            counts_ok(cal_rows, rf, lf)
            and counts_ok(test_rows, rf, lf)
        ):
            chosen = (
                salt,
                cal_cases,
                test_cases,
                cal_rows,
                test_rows,
                fields,
            )
            break

    if chosen is None:
        raise RuntimeError(
            "Could not make balanced component-safe "
            "calibration/test split."
        )

    (
        salt,
        cal_cases,
        test_cases,
        cal_rows,
        test_rows,
        fields,
    ) = chosen

    text_field, label_field, role_field, self_field, hard_field = fields

    print("split salt:", salt)
    print("cal cases :", len(cal_cases))
    print("test cases:", len(test_cases))
    print("cal rows  :", len(cal_rows))
    print("test rows :", len(test_rows))

    # Verify component separation.
    cc = {
        component_value(x)
        for x in cal_cases
    }
    tc = {
        component_value(x)
        for x in test_cases
    }

    assert not (cc & tc)
    print("component overlap: 0")

    # ------------------------------------------------------
    # Load epoch-1 best adapter.
    # ------------------------------------------------------
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)

    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(
        ADAPTER
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        torch_dtype=torch.float16,
    )

    model.config.pad_token_id = tok.pad_token_id

    model = PeftModel.from_pretrained(
        model,
        ADAPTER,
    )

    model.to(device)
    model.eval()

    print("model loaded:", BASE_MODEL)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    cal_text = [
        str(field_value(x, text_field))
        for x in cal_rows
    ]

    test_text = [
        str(field_value(x, text_field))
        for x in test_rows
    ]

    cal_logits = infer_logits(
        model, tok, cal_text, device
    )

    test_logits = infer_logits(
        model, tok, test_text, device
    )

    cal_y = np.asarray([
        int(field_value(x, label_field))
        for x in cal_rows
    ])

    test_y = np.asarray([
        int(field_value(x, label_field))
        for x in test_rows
    ])

    cal_role = np.asarray([
        str(field_value(x, role_field))
        for x in cal_rows
    ])

    test_role = np.asarray([
        str(field_value(x, role_field))
        for x in test_rows
    ])

    def flag_array(rows, field):
        if field is None:
            return None

        return np.asarray([
            bool(field_value(x, field))
            for x in rows
        ])

    cal_self = flag_array(
        cal_rows, self_field
    )
    test_self = flag_array(
        test_rows, self_field
    )

    cal_hard = flag_array(
        cal_rows, hard_field
    )
    test_hard = flag_array(
        test_rows, hard_field
    )

    raw_cal = sigmoid(cal_logits)
    raw_test = sigmoid(test_logits)

    calibrated_cal = np.zeros_like(
        raw_cal,
        dtype=np.float64,
    )

    calibrated_test = np.zeros_like(
        raw_test,
        dtype=np.float64,
    )

    thresholds = np.zeros(
        len(ROLES),
        dtype=np.float64,
    )

    calibration = {}

    # ------------------------------------------------------
    # Per-role Platt calibration + FPR-constrained threshold.
    # BOTH learned on calibration partition only.
    # ------------------------------------------------------
    for ri, role in enumerate(ROLES):
        mc = cal_role == role
        mt = test_role == role

        yc = cal_y[mc]
        zc = cal_logits[mc].reshape(-1, 1)

        if len(np.unique(yc)) != 2:
            raise RuntimeError(
                f"{role}: calibration has one class"
            )

        platt = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )

        platt.fit(zc, yc)

        pc = platt.predict_proba(zc)[:,1]
        pt = platt.predict_proba(
            test_logits[mt].reshape(-1,1)
        )[:,1]

        calibrated_cal[mc] = pc
        calibrated_test[mt] = pt

        selfc = (
            cal_self[mc]
            if cal_self is not None
            else None
        )

        threshold, source = choose_threshold(
            pc,
            yc,
            selfc,
            TARGET_FPR,
        )

        thresholds[ri] = threshold

        calibration[role] = {
            "coef":
                float(platt.coef_[0,0]),
            "intercept":
                float(platt.intercept_[0]),
            "threshold":
                float(threshold),
            "threshold_constraint_source":
                source,
            "cal_positive_n":
                int((yc==1).sum()),
            "cal_negative_n":
                int((yc==0).sum()),
        }

    raw_thresholds = np.full(
        len(ROLES),
        0.5,
    )

    cal_half_thresholds = np.full(
        len(ROLES),
        0.5,
    )

    raw_metrics = metric_block(
        raw_test,
        test_y,
        test_role,
        raw_thresholds,
        test_self,
        test_hard,
    )

    platt05_metrics = metric_block(
        calibrated_test,
        test_y,
        test_role,
        cal_half_thresholds,
        test_self,
        test_hard,
    )

    calibrated_metrics = metric_block(
        calibrated_test,
        test_y,
        test_role,
        thresholds,
        test_self,
        test_hard,
    )

    # ------------------------------------------------------
    # Counterfactual test-half metric.
    # Calibration is monotonic per role, so pair ordering
    # should not improve magically; this is our reality check.
    # ------------------------------------------------------
    test_pairs, pair_counts = tr.build_pairs(
        test_cases,
        rd,
    )

    eval_args = SimpleNamespace(
        max_length=MAX_LENGTH,
        eval_pair_batch_size=4,
    )

    pair_metrics = tr.eval_pairs(
        model,
        tok,
        test_pairs,
        eval_args,
        device,
        torch.float16,
    )

    summary = {
        "protocol":
            "v13.7_component_safe_posthoc_calibration",
        "adapter": str(ADAPTER),
        "base_model": BASE_MODEL,
        "validation_fold": FOLD,
        "calibration_fraction": CAL_FRACTION,
        "split_salt": salt,
        "component_overlap": 0,
        "cal_cases": len(cal_cases),
        "test_cases": len(test_cases),
        "cal_rows": len(cal_rows),
        "test_rows": len(test_rows),
        "target_fpr": TARGET_FPR,
        "raw_sigmoid_at_0_5": raw_metrics,
        "platt_at_0_5": platt05_metrics,
        "platt_fpr_constrained":
            calibrated_metrics,
        "counterfactual_test":
            pair_metrics,
        "test_pair_counts":
            dict(pair_counts),
    }

    dump(
        OUT / "calibration.json",
        calibration,
    )

    dump(
        OUT / "summary.json",
        summary,
    )

    np.savez_compressed(
        OUT / "predictions.npz",
        cal_logits=cal_logits,
        cal_y=cal_y,
        cal_role=cal_role,
        test_logits=test_logits,
        test_y=test_y,
        test_role=test_role,
        calibrated_test=calibrated_test,
        thresholds=thresholds,
    )

    print()
    print("="*100)
    print("FINAL CALIBRATION AUDIT")
    print("="*100)

    for name, m in [
        ("RAW@0.5", raw_metrics),
        ("PLATT@0.5", platt05_metrics),
        ("PLATT+FPR10", calibrated_metrics),
    ]:
        print()
        print(name)
        print(
            " macro_auc               =",
            m["macro_auc"],
        )
        print(
            " balanced_accuracy       =",
            m["balanced_accuracy"],
        )
        print(
            " positive_recall         =",
            m["positive_recall"],
        )
        print(
            " negative_fpr            =",
            m["negative_fpr"],
        )
        print(
            " self_target_absent_fpr  =",
            m["self_target_absent_fpr"],
        )
        print(
            " hard_neighbor_absent_fpr=",
            m["hard_neighbor_absent_fpr"],
        )
        print(
            " brier                   =",
            m["brier"],
        )

    print()
    print("COUNTERFACTUAL TEST:")
    print(json.dumps(
        pair_metrics,
        indent=2,
    ))

    print()
    print("wrote:", OUT/"summary.json")
    print("wrote:", OUT/"calibration.json")
    print("wrote:", OUT/"predictions.npz")


if __name__ == "__main__":
    main()
