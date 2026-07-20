"""Environment smoke test for the LLM serving platform.

Confirms the RTX 5080 baseline environment is wired up before any serving work:
- torch sees CUDA
- the GPU is the expected device with ~16 GB VRAM
- transformers imports cleanly (baseline backend dependency)

Run:
    uv run python -m src.env_check
Exit code is 0 only if CUDA is available (so it doubles as a CI/pre-flight gate).
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"torch          {torch.__version__}")
    print(f"cuda available {cuda_ok}")

    if not cuda_ok:
        print("!! CUDA not visible — check WSL GPU driver / torch cu130 wheel", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"gpu            {torch.cuda.get_device_name(0)}")
    print(f"vram           {props.total_memory / 1e9:.1f} GB")
    print(f"compute cap    {props.major}.{props.minor}")

    try:
        import transformers

        print(f"transformers   {transformers.__version__}")
    except ImportError:
        print("transformers   NOT INSTALLED (needed for baseline backend from D2)")

    # Tiny allocation to prove the runtime actually works, not just links.
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul smoke   ok ({y:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
