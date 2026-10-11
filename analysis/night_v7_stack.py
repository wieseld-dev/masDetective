#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import train_v7_benign_output_sensor_pilot_v1 as bos
import v7_structured_reranker_v1 as sr
import v7_targeted_toprank_v1 as tr

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


ROLES = list(TRAIN_ROLES)
N_ROLES = len(ROLES)

ADAPTER = ROOT / "artifacts/v7_benign_output_sensor_pilot_v1/best_adapter"
RERANKER = ROOT / "artifacts/v7_structured_reranker_v1/reranker.pt"
RERANKER_SUMMARY = ROOT / "artifacts/v7_structured_reranker_v1/summary.json"
TARGETED = ROOT / "artifacts/v7_targeted_toprank_v1/targeted_ranker.pt"


def _to_np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _state_dict_from_checkpoint(obj):
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint is not dict: {type(obj)}")

    for key in (
        "model_state",
        "state_dict",
        "reranker_state",
        "best_state",
        "model",
    ):
        v = obj.get(key)
        if isinstance(v, dict):
            return v

    tensorish = {
        k: v for k, v in obj.items()
        if torch.is_tensor(v) or isinstance(v, np.ndarray)
    }

    if tensorish:
        return tensorish

    raise KeyError(
        f"could not find state dict; checkpoint keys={list(obj.keys())}"
    )


def _suffix_value(state, wanted):
    if wanted in state:
        return state[wanted]

    matches = [
        v for k, v in state.items()
        if k.endswith("." + wanted)
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def _parse_calibration(obj):
    """
    Find a 20-role scale+bias calibration in several formats.
    Returns (scale, bias) or None.
    """
    if isinstance(obj, dict):
        if "scale" in obj and "bias" in obj:
            s = obj["scale"]
            b = obj["bias"]

            if isinstance(s, dict) and isinstance(b, dict):
                if all(r in s and r in b for r in ROLES):
                    return (
                        np.asarray([float(s[r]) for r in ROLES], np.float32),
                        np.asarray([float(b[r]) for r in ROLES], np.float32),
                    )

            try:
                sa = _to_np(s).reshape(-1)
                ba = _to_np(b).reshape(-1)
                if len(sa) == N_ROLES and len(ba) == N_ROLES:
                    return sa, ba
            except Exception:
                pass

        if all(
            r in obj
            and isinstance(obj[r], dict)
            and "scale" in obj[r]
            and "bias" in obj[r]
            for r in ROLES
        ):
            return (
                np.asarray(
                    [float(obj[r]["scale"]) for r in ROLES],
                    dtype=np.float32,
                ),
                np.asarray(
                    [float(obj[r]["bias"]) for r in ROLES],
                    dtype=np.float32,
                ),
            )

        if "roles" in obj and isinstance(obj["roles"], dict):
            hit = _parse_calibration(obj["roles"])
            if hit is not None:
                return hit

        for v in obj.values():
            if isinstance(v, dict):
                hit = _parse_calibration(v)
                if hit is not None:
                    return hit

    return None


class V7NightStack:
    def __init__(
        self,
        device: str = "cuda",
        sensor_batch_size: int = 8,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. "
                "Run this on a Slurm GPU compute node, not the login node."
            )

        self.device = torch.device(device)
        self.sensor_batch_size = int(sensor_batch_size)

        if len(ROLES) != 20:
            raise RuntimeError(f"expected 20 roles, got {len(ROLES)}")

        self.max_length = int(
            getattr(sr, "MAX_LENGTH", 1024)
        )

        self._load_sensor()
        self._load_structured()
        self._load_structured_calibration()
        self._load_targeted()

    def _load_sensor(self):
        print("[stack] loading V7 sensor:", ADAPTER, flush=True)

        cfg = PeftConfig.from_pretrained(str(ADAPTER))
        base_name = cfg.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        base = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )

        base.config.pad_token_id = self.tokenizer.pad_token_id
        base.config.use_cache = False

        self.sensor = PeftModel.from_pretrained(
            base,
            str(ADAPTER),
            is_trainable=False,
        )

        self.sensor.eval()
        self.sensor.to(self.device)

        print(
            "[stack] sensor loaded on",
            self.device,
            "max_length=",
            self.max_length,
            flush=True,
        )

    def _load_structured(self):
        print("[stack] loading structured reranker:", RERANKER, flush=True)

        ck = torch.load(
            RERANKER,
            map_location="cpu",
            weights_only=False,
        )
        state = _state_dict_from_checkpoint(ck)

        mean = _suffix_value(state, "mean")
        std = _suffix_value(state, "std")

        if mean is None and isinstance(ck, dict):
            for key in ("mean", "score_mean", "train_mean"):
                if key in ck:
                    mean = ck[key]
                    break

        if std is None and isinstance(ck, dict):
            for key in ("std", "score_std", "train_std"):
                if key in ck:
                    std = ck[key]
                    break

        if mean is None or std is None:
            raise RuntimeError(
                "reranker checkpoint does not expose mean/std. "
                f"checkpoint keys={list(ck.keys()) if isinstance(ck, dict) else type(ck)}; "
                f"state keys={list(state.keys())[:50]}"
            )

        mean = _to_np(mean).reshape(-1)
        std = _to_np(std).reshape(-1)

        if len(mean) != N_ROLES or len(std) != N_ROLES:
            raise RuntimeError(
                f"structured mean/std mismatch: {mean.shape}, {std.shape}"
            )

        model = sr.StructuredReranker(
            N_ROLES,
            mean,
            std,
        )

        wanted = {}
        for name in ("delta", "bias", "mean", "std"):
            v = _suffix_value(state, name)
            if v is not None:
                wanted[name] = v

        missing, unexpected = model.load_state_dict(
            wanted,
            strict=False,
        )

        critical_missing = [
            x for x in missing
            if x in ("delta", "bias")
        ]

        if critical_missing:
            raise RuntimeError(
                f"structured reranker critical state missing: {critical_missing}; "
                f"available={list(state.keys())[:100]}"
            )

        model.eval()
        self.structured = model

        print(
            "[stack] structured reranker loaded",
            "missing=",
            missing,
            "unexpected=",
            unexpected,
            flush=True,
        )

    def _load_structured_calibration(self):
        summary = json.loads(
            RERANKER_SUMMARY.read_text()
        )

        if "role_calibration" not in summary:
            raise RuntimeError(
                "structured summary has no role_calibration. "
                "Refusing to silently use the old raw-sensor calibration."
            )

        parsed = _parse_calibration(
            summary["role_calibration"]
        )

        if parsed is None:
            raise RuntimeError(
                "could not parse structured role_calibration. "
                f"type={type(summary['role_calibration'])}; "
                f"value={repr(summary['role_calibration'])[:4000]}"
            )

        self.cal_scale, self.cal_bias = parsed

        print(
            "[stack] structured calibration loaded",
            flush=True,
        )

    def _load_targeted(self):
        print("[stack] loading targeted ranker:", TARGETED, flush=True)

        ck = torch.load(
            TARGETED,
            map_location="cpu",
            weights_only=False,
        )

        required = {
            "model_state",
            "feature_mean",
            "feature_std",
            "blend_alpha",
        }

        miss = required - set(ck.keys())
        if miss:
            raise RuntimeError(
                f"targeted checkpoint missing {miss}; keys={list(ck.keys())}"
            )

        self.feature_mean = _to_np(
            ck["feature_mean"]
        ).reshape(1, -1)

        self.feature_std = np.maximum(
            _to_np(ck["feature_std"]).reshape(1, -1),
            1e-6,
        )

        self.blend_alpha = float(
            ck["blend_alpha"]
        )

        self.targeted = tr.TargetedRanker(
            self.feature_mean.shape[1],
            N_ROLES,
        )

        self.targeted.load_state_dict(
            ck["model_state"],
            strict=True,
        )
        self.targeted.eval()

        print(
            "[stack] targeted loaded; blend_alpha=",
            self.blend_alpha,
            flush=True,
        )

    @torch.inference_mode()
    def score_outputs(self, outputs):
        outputs = [
            str(x)
            for x in outputs
        ]

        texts = []

        for output in outputs:
            for role in ROLES:
                texts.append(
                    bos.sensor_text(
                        role,
                        ROLE_DESCRIPTIONS[role],
                        output,
                    )
                )

        raw = bos.score_texts(
            self.sensor,
            self.tokenizer,
            texts,
            self.device,
            self.max_length,
            self.sensor_batch_size,
        ).reshape(
            len(outputs),
            N_ROLES,
        ).astype(np.float32)

        structured_t = self.structured(
            torch.tensor(
                raw,
                dtype=torch.float32,
            )
        )

        structured = (
            structured_t
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        calibrated = (
            structured
            * self.cal_scale.reshape(1, -1)
            + self.cal_bias.reshape(1, -1)
        ).astype(np.float32)

        features = tr.build_features(
            calibrated
        )

        features = (
            features - self.feature_mean
        ) / self.feature_std

        targeted_raw = tr.predict(
            self.targeted,
            features,
            calibrated,
        ).astype(np.float32)

        # This checkpoint's alpha was selected by blending the stronger
        # calibrated baseline with the learned residual targeted ranker.
        # alpha=0.74 therefore keeps 74% of calibrated geometry.
        final = (
            self.blend_alpha * calibrated
            + (1.0 - self.blend_alpha) * targeted_raw
        ).astype(np.float32)

        return {
            "raw": raw,
            "structured": structured,
            "calibrated": calibrated,
            "targeted_raw": targeted_raw,
            "final": final,
        }

    def score_output(self, output):
        x = self.score_outputs([output])
        return {
            k: v[0]
            for k, v in x.items()
        }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--text",
        default=(
            "The service is timing out after deployment. "
            "First inspect logs, compare before and after, "
            "identify the failing component, and prepare a rollback plan."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    stack = V7NightStack(
        sensor_batch_size=args.batch_size
    )

    s = stack.score_output(args.text)

    print("\n===== STACK SMOKE =====")

    for name in (
        "raw",
        "structured",
        "calibrated",
        "targeted_raw",
        "final",
    ):
        print("\n", name.upper())
        vals = s[name]
        order = np.argsort(-vals)

        for i in order[:8]:
            print(
                f"{ROLES[i]:<28} {float(vals[i]): .5f}"
            )


if __name__ == "__main__":
    main()
