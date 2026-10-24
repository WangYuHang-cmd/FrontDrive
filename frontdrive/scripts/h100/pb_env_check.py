"""Smoke-check venv-pb: torch cu121 on sm_90, mmcv ms_deform_attn CUDA kernel vs the PyTorch reference, mmdet, navsim.

Exit codes: 0 ok, 2 no CUDA, 3 CUDA kernel unavailable/mismatch (prebuilt wheel without sm_90), 4 import failure.
"""
import sys

import torch

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE")
    sys.exit(2)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
try:
    import mmcv  # noqa: E402
    import mmdet  # noqa: E402
    from mmcv.ops.multi_scale_deform_attn import (  # noqa: E402
        MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch,
    )
    import navsim  # noqa: E402,F401
    import hydra  # noqa: E402
    import numpy  # noqa: E402
    import pytorch_lightning  # noqa: E402
except Exception as exc:  # noqa: BLE001
    print("IMPORT FAILURE:", repr(exc)[:300])
    sys.exit(4)
print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "hydra", hydra.__version__,
      "numpy", numpy.__version__, "PL", pytorch_lightning.__version__)

# CUDA kernel vs PyTorch reference (shapes follow mmcv's unit test)
torch.manual_seed(0)
bs, n_heads, n_levels, n_points, dim = 2, 2, 2, 2, 4
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long, device="cuda")
level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
s = int(sum((h * w).item() for h, w in shapes))
value = torch.rand(bs, s, n_heads, dim, device="cuda")
loc = torch.rand(bs, 7, n_heads, n_levels, n_points, 2, device="cuda")
attn = torch.rand(bs, 7, n_heads, n_levels, n_points, device="cuda")
attn = attn / attn.sum(-1, keepdim=True).sum(-2, keepdim=True)
ref = multi_scale_deformable_attn_pytorch(value, shapes, loc, attn)
try:
    out = MultiScaleDeformableAttnFunction.apply(value, shapes, level_start, loc, attn, 2)
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    print("ms_deform_attn CUDA kernel raised:", repr(exc)[:300])
    sys.exit(3)
err = float((out - ref).abs().max())
print("ms_deform_attn CUDA vs pytorch reference: max abs diff", err, "| out finite:", bool(torch.isfinite(out).all()))
if not torch.isfinite(out).all() or err > 1e-4 or float(out.abs().sum()) == 0.0:
    print("CUDA KERNEL MISMATCH -> no sm_90 kernel in this mmcv build (rebuild from source with TORCH_CUDA_ARCH_LIST=9.0)")
    sys.exit(3)
print("PB ENV CHECK OK")
