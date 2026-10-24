#!/usr/bin/env bash
# Print the training metadata stored in a Lightning checkpoint (epoch, global_step, hyper-parameters, callback state)
# without loading it onto the GPU. Used to pin what the released Drive-JEPA checkpoint is (which epoch / how many optimizer
# steps = effective batch), see memory "reproduce recipe from checkpoint metadata".
# Usage (server): bash scripts/h100/ckpt_meta.sh <checkpoint.ckpt> [more.ckpt ...]
set -u
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_pb_env.sh
for c in "$@"; do
  python - "$c" <<'PY'
import json, sys
import torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
out = {"file": p, "keys": sorted(ck.keys()) if isinstance(ck, dict) else type(ck).__name__}
if isinstance(ck, dict):
    for k in ("epoch", "global_step", "pytorch-lightning_version"):
        if k in ck:
            out[k] = ck[k]
    hp = ck.get("hyper_parameters") or ck.get("hparams")
    if hp is not None:
        out["hyper_parameters"] = {k: (str(v)[:200]) for k, v in (hp.items() if hasattr(hp, "items") else {})}
    cb = ck.get("callbacks")
    if isinstance(cb, dict):
        out["callbacks"] = {k: {kk: (str(vv)[:120]) for kk, vv in (v.items() if hasattr(v, "items") else {})} for k, v in cb.items()}
    sd = ck.get("state_dict") or {}
    out["state_dict_tensors"] = len(sd)
    out["state_dict_prefixes"] = sorted({k.split(".")[0] for k in sd})[:10]
    if "optimizer_states" in ck:
        try:
            pg = ck["optimizer_states"][0]["param_groups"]
            out["optimizer_param_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in pg]
        except Exception as e:  # noqa: BLE001
            out["optimizer_states"] = f"unreadable: {e}"
    if "lr_schedulers" in ck:
        out["lr_schedulers"] = str(ck["lr_schedulers"])[:300]
print(json.dumps(out, indent=1, default=str))
PY
done
