"""Smoke-check the venv on the H100 server (imports, CUDA, bf16 matmul, nvidia-smi per-process accounting).

Usage: <venv>/bin/python scripts/h100/env_check.py            (exit 0 == OK)
"""
import importlib
import os
import subprocess
import sys

MODULES = [
    "torch", "torchvision", "decord", "av", "cv2", "hydra", "pytorch_lightning", "numpy", "pandas",
    "scipy", "sklearn", "navsim", "nuplan", "timm", "einops", "shapely", "geopandas", "yaml", "omegaconf",
]


def main() -> int:
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)[:200]))
    import cv2  # noqa: E402
    import decord  # noqa: E402
    import hydra  # noqa: E402
    import numpy  # noqa: E402
    import pytorch_lightning  # noqa: E402
    import torch  # noqa: E402

    print("python", sys.version.split()[0])
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        return 2
    cap = torch.cuda.get_device_capability(0)
    print("device", torch.cuda.get_device_name(0), "capability", cap)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = a @ a
    torch.cuda.synchronize()
    print("bf16 matmul ok, mean|b| =", float(b.float().abs().mean()))
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout
    print("compute-apps:", out.strip().replace("\n", " | ") or "<empty>")
    pids = {ln.split(",")[0].strip() for ln in out.splitlines() if ln.strip()}
    visible = str(os.getpid()) in pids
    print("our pid visible to nvidia-smi:", visible)
    if not visible:
        print("per-process GPU accounting not visible (queue smoke-peak measurement would fail)")
        return 3
    print(
        "versions numpy", numpy.__version__, "hydra", hydra.__version__, "PL", pytorch_lightning.__version__,
        "decord", decord.__version__, "cv2", cv2.__version__,
    )
    if failures:
        print("IMPORT FAILURES:", failures)
        return 1
    print("ENV CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
