#!/usr/bin/env python3
"""Export the V-JEPA 2.1 ViT-L/16 (384, distilled from ViT-G) *encoder only* as bf16 safetensors.

Source checkpoint (already on disk): SDD/weights/vjepa2_1_vitl_dist_vitG_384.pt (4.8 GB, key `ema_encoder`).
We follow the SDD offline-loading pattern (sdd/video/backbones.py): build
`vjepa2_1_vit_large_384(pretrained=False)` from the cached torch.hub repo and load `ema_encoder`
with strict=False (only `pos_embed` is expected to be unexpected). The hub `pretrained=True` path is
broken (localhost URL), so we never use it.

Outputs:
  <out>.safetensors   bf16 encoder state dict (≈0.6 GB)
  <out>.json          export report: counts, shapes summary, missing/unexpected keys, sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


def sha256(p: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _strip(k: str) -> str:
    for prefix in ("module.", "backbone."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/henry/Desktop/Drive/ADAS_TO_model/SDD/weights/vjepa2_1_vitl_dist_vitG_384.pt")
    ap.add_argument("--hub", default="/home/henry/.cache/torch/hub/facebookresearch_vjepa2_main")
    ap.add_argument("--out", default="/home/henry/Desktop/Qwen_Drive/frontdrive/weights/vjepa21_vitl384_ema_encoder.bf16.safetensors")
    ap.add_argument("--num-frames", type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    sys.path.insert(0, a.hub)
    from src.hub.backbones import vjepa2_1_vit_large_384  # noqa: E402

    # Build encoder (+ predictor, discarded) without downloading anything.
    enc, _pred = vjepa2_1_vit_large_384(pretrained=False, num_frames=a.num_frames)
    enc = enc.eval()
    n_params = sum(p.numel() for p in enc.parameters())

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    keys = list(ck.keys()) if isinstance(ck, dict) else []
    sd = ck["ema_encoder"] if "ema_encoder" in ck else ck.get("encoder", ck)
    # strip DDP "module." and wrapper "backbone." prefixes (same as SDD backbones._clean).
    # Attempt 1 (2026-10-04 19:40) forgot "backbone." -> 302 missing keys -> random weights exported.
    sd = {_strip(k): v for k, v in sd.items()}
    res = enc.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)

    # export bf16 encoder
    from safetensors.torch import save_file  # noqa: E402

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bf16 = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in enc.state_dict().items()}
    meta = {"source": str(a.ckpt), "arch": "vjepa2_1_vit_large_384", "num_frames": str(a.num_frames),
            "dtype": "bfloat16", "note": "encoder only; ema_encoder weights; loaded strict=False"}
    save_file(bf16, str(out), metadata=meta)

    # smoke forward on CPU (tiny) to make sure the module runs: B=1, C=3, T=num_frames, 384x384 → tokens
    with torch.no_grad():
        x = torch.zeros(1, 3, a.num_frames, 384, 384)
        try:
            y = enc(x)
            tok_shape = list(y.shape)
        except Exception as ex:  # report but do not fail the export
            tok_shape = f"forward failed: {ex}"

    report = {
        "ckpt": a.ckpt, "ckpt_top_level_keys": keys[:20],
        "encoder_params": n_params, "encoder_params_M": round(n_params / 1e6, 1),
        "state_dict_tensors": len(bf16), "missing_keys": missing, "unexpected_keys": unexpected,
        "export_path": str(out), "export_bytes": out.stat().st_size, "export_sha256": sha256(out),
        "cpu_smoke_forward_output_shape": tok_shape, "elapsed_s": round(time.time() - t0, 1),
        "torch": torch.__version__,
    }
    Path(str(out).replace(".safetensors", ".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "ckpt_top_level_keys"}, indent=1))
    ok = (not unexpected or unexpected == ["pos_embed"]) and not missing
    print("EXPORT_OK" if ok else "EXPORT_CHECK_KEYS")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
