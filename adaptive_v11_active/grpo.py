from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class GRPOSample:
    prompt: str
    completion: str
    old_logp: float
    ref_logp: float
    advantage: float
    reward: float
    group_id: str


def kl_k3(new_lp, ref_lp, torch):
    delta = ref_lp - new_lp
    return torch.exp(torch.clamp(delta, -6.0, 6.0)) - delta - 1.0


def save_policy_adapter(policy, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(policy, "save_adapter"):
        policy.save_adapter(str(out))
        return

    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out))
        tok = getattr(policy, "tokenizer", None)
        if tok is not None and hasattr(tok, "save_pretrained"):
            tok.save_pretrained(str(out))
        return

    raise RuntimeError(
        "Cannot save policy adapter: no save_adapter() or model.save_pretrained()"
    )


def train_grpo_epoch(
    *,
    policy,
    optimizer,
    samples: Sequence[GRPOSample],
    clip_eps: float = 0.20,
    kl_beta: float = 0.02,
    grad_clip: float = 1.0,
):
    """
    Memory-safe GRPO epoch.

    Important:
    backward() is performed per sample with 1/N scaling.
    This is mathematically equivalent to backpropagating the mean loss,
    but does NOT retain all sequence graphs in GPU memory simultaneously.
    """
    import torch

    if not samples:
        return {
            "loss": 0.0,
            "pg": 0.0,
            "kl": 0.0,
            "n": 0,
        }

    optimizer.zero_grad(set_to_none=True)

    n = len(samples)

    loss_sum = 0.0
    pg_sum = 0.0
    kl_sum = 0.0

    for idx, tr in enumerate(samples, 1):
        new_lp = policy.mean_logprob(
            tr.prompt,
            tr.completion,
        )

        old_lp = torch.tensor(
            float(tr.old_logp),
            dtype=new_lp.dtype,
            device=new_lp.device,
        )

        ref_lp = torch.tensor(
            float(tr.ref_logp),
            dtype=new_lp.dtype,
            device=new_lp.device,
        )

        adv = torch.tensor(
            float(tr.advantage),
            dtype=new_lp.dtype,
            device=new_lp.device,
        )

        ratio = torch.exp(
            torch.clamp(
                new_lp - old_lp,
                -4.0,
                4.0,
            )
        )

        unclipped = ratio * adv

        clipped = (
            torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            )
            * adv
        )

        pg = -torch.minimum(
            unclipped,
            clipped,
        )

        kl = kl_k3(
            new_lp,
            ref_lp,
            torch,
        )

        loss = pg + float(kl_beta) * kl

        # Critical memory fix:
        # accumulate gradients, not computation graphs.
        (loss / n).backward()

        loss_sum += float(loss.detach().cpu())
        pg_sum += float(pg.detach().cpu())
        kl_sum += float(kl.detach().cpu())

        if idx == 1 or idx % 8 == 0 or idx == n:
            allocated = (
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            reserved = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )

            print(
                f"[grpo-step] {idx}/{n} "
                f"loss={float(loss.detach().cpu()):+.5f} "
                f"gpu_alloc={allocated:.2f}GiB "
                f"gpu_reserved={reserved:.2f}GiB",
                flush=True,
            )

        del (
            new_lp,
            old_lp,
            ref_lp,
            adv,
            ratio,
            unclipped,
            clipped,
            pg,
            kl,
            loss,
        )

    torch.nn.utils.clip_grad_norm_(
        policy.trainable_parameters(),
        float(grad_clip),
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss_sum / n,
        "pg": pg_sum / n,
        "kl": kl_sum / n,
        "n": n,
    }
