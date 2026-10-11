from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import PeftConfig, PeftModel

sys.path.insert(0, str(Path("analysis").resolve()))
import q38_vs_r7_generator_ab_v1 as ab


ADAPTER = Path("artifacts/q38_utility_lora_v1/policy")
OUT = Path("artifacts/q38_utility_lora_probes_v1")
OUT.mkdir(parents=True, exist_ok=True)

ARM = "QWEN38_27B_UTILITY_LORA"


def main():
    assert (ADAPTER / "adapter_config.json").is_file(), ADAPTER

    cfg = PeftConfig.from_pretrained(str(ADAPTER))
    base_name = cfg.base_model_name_or_path

    print("=" * 100)
    print("QWEN3.8-27B UTILITY-LoRA PROBE GENERATION")
    print("=" * 100)
    print("BASE        =", base_name)
    print("ADAPTER     =", ADAPTER)
    print("ROLES       =", len(ab.TRAIN_ROLES))
    print("PER ROLE    =", ab.N_PER_ROLE)
    print("TEMPERATURE =", ab.TEMPERATURE)
    print("TOP_P       =", ab.TOP_P)
    print("SEED        =", ab.SEED)

    assert len(ab.TRAIN_ROLES) == 20

    processor = AutoProcessor.from_pretrained(base_name)

    base = AutoModelForMultimodalLM.from_pretrained(
        base_name,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )
    model.eval()

    print(
        "VRAM after load =",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB",
        flush=True,
    )

    rows = []

    for ri, role in enumerate(ab.TRAIN_ROLES):
        user = ab.build_user_prompt(role)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ab.SYSTEM}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user}],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda")

        # IDENTICAL seed schedule to original Q38/R7 A/B.
        torch.manual_seed(ab.SEED + ri)

        t0 = time.time()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=ab.TEMPERATURE,
                top_p=ab.TOP_P,
                max_new_tokens=ab.MAX_NEW,
                num_return_sequences=ab.N_PER_ROLE,
            )

        torch.cuda.synchronize()
        dt = time.time() - t0

        prefix = inputs["input_ids"].shape[1]

        for j in range(ab.N_PER_ROLE):
            text = processor.decode(
                out[j, prefix:],
                skip_special_tokens=True,
            ).strip()

            rec = {
                "arm": ARM,
                "role": role,
                "candidate_index": j + 1,
                "seed_group": ab.SEED + ri,
                "query": text,
                "generation_group_seconds": dt,
            }
            rows.append(rec)

            print(
                f"LORA {role:28s} [{j+1}/{ab.N_PER_ROLE}] | {text}",
                flush=True,
            )

    assert len(rows) == 100, len(rows)

    probe_path = OUT / "probes.jsonl"
    with probe_path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {
        "experiment": "q38_utility_lora_probe_generation_v1",
        "arm": ARM,
        "base_model": base_name,
        "adapter": str(ADAPTER),
        "same_prompt_content_as_original_ab": True,
        "roles": list(ab.TRAIN_ROLES),
        "n_per_role": ab.N_PER_ROLE,
        "temperature": ab.TEMPERATURE,
        "top_p": ab.TOP_P,
        "max_new_tokens": ab.MAX_NEW,
        "seed": ab.SEED,
        **ab.basic_stats(rows),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("SAVED =", probe_path)
    print("Q38_UTILITY_LORA_PROBES_OK")


if __name__ == "__main__":
    main()
