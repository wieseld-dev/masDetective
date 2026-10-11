from __future__ import annotations


def main() -> None:
    import torch
    import transformers
    import peft

    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("compute capability:", torch.cuda.get_device_capability(0))
        print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    print("transformers:", transformers.__version__)
    print("peft:", peft.__version__)
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", bnb.__version__)
    except Exception as exc:
        print("bitsandbytes import FAILED:", repr(exc))
        raise


if __name__ == "__main__":
    main()
