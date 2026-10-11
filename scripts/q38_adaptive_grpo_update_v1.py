from __future__ import annotations

import argparse
import os
import gc
from pathlib import Path

from adaptive_v11_active.grpo import GRPOSample, save_policy_adapter, train_grpo_epoch
from adaptive_v11_active.io import read_jsonl, write_json
from q38.adaptive_belief_policy import Q38AdaptiveBeliefPolicy


def adapter_for_round(round_index: int, root: Path, v11: Path) -> Path:
    return v11 if round_index == 1 else root / "checkpoints" / f"round_{round_index-1:02d}" / "policy"


def main():
    ap = argparse.ArgumentParser(description="GRPO update for role-adaptive V11")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--root", default="artifacts/adaptive_v11_role_only_v1")
    ap.add_argument("--v11-adapter", default="artifacts/q38_utility_lora_v1/policy")
    ap.add_argument("--model-name", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    root = Path(args.root)
    rows = read_jsonl(root / f"grpo_train_round_{args.round:02d}.jsonl")
    samples = [
        GRPOSample(
            prompt=str(r["prompt"]), completion=str(r["completion"]),
            old_logp=float(r["old_logp"]), ref_logp=float(r["ref_logp"]),
            advantage=float(r["advantage"]), reward=float(r["reward"]), group_id=str(r["group_id"]),
        )
        for r in rows if abs(float(r.get("advantage", 0.0))) > 1e-10
    ]
    if not samples:
        raise SystemExit("no non-zero-advantage GRPO samples")

    env_start = os.environ.get("V14A_START_ADAPTER", "").strip()
    start = Path(env_start) if env_start else adapter_for_round(args.round, root, Path(args.v11_adapter))
    policy = Q38AdaptiveBeliefPolicy(
        args.model_name, adapter_path=str(start), trainable=True,
        use_4bit=not args.no_4bit, max_seq_length=args.max_seq_length,
    )
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    stats = []
    for epoch in range(1, args.epochs + 1):
        st = train_grpo_epoch(
            policy=policy, optimizer=opt, samples=samples,
            clip_eps=args.clip, kl_beta=args.kl_beta, grad_clip=1.0,
        )
        st["epoch"] = epoch
        stats.append(st)
        print(
            f"[role-grpo] r={args.round} epoch={epoch} loss={st['loss']:+.5f} "
            f"pg={st['pg']:+.5f} kl={st['kl']:.5f} n={st['n']}", flush=True,
        )

    outdir = root / "checkpoints" / f"round_{args.round:02d}"
    save_policy_adapter(policy, outdir / "policy")
    torch.save(opt.state_dict(), outdir / "optimizer.pt")
    write_json(outdir / "update_summary.json", {
        "phase": "role_only",
        "round": args.round,
        "start_adapter": str(start),
        "v11_reference": args.v11_adapter,
        "samples": len(samples),
        "epochs": args.epochs,
        "lr": args.lr,
        "clip": args.clip,
        "kl_beta": args.kl_beta,
        "stats": stats,
    })
    print(f"[done] checkpoint={outdir / 'policy'}", flush=True)
    del policy
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
