from __future__ import annotations

import argparse


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("Loading 4-bit model:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg, device_map={"": 0})
    prompt = "Return exactly this word: READY"
    batch = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=8)
    print("generated:", tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True))
    print("4-bit load/generation: OK")


if __name__ == "__main__":
    main()
