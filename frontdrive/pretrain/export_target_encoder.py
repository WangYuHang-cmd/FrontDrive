"""Export the EMA target encoder of a `train_motion.py` checkpoint in the layout the Drive-JEPA agent expects.

The e50 file (`vitl_merge_3dataset_e50.pt`) is a dict whose `target_encoder` entry holds 292 keys named
`module.backbone.*`; our `latest.pt` / `eN.pt` checkpoints use the identical layout, so the export is the
`target_encoder` state dict plus a few provenance fields. `agent.pretrain_pt_path=<export>` then initialises the
perception-free agent's ViT-L from it (vit_encoder.init_module reads only `target_encoder`).

Usage: python pretrain/export_target_encoder.py <ckpt.pt> <out.pt>   (prints key count, sha256, size)
"""
import hashlib, os, sys, torch

def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    te = ck["target_encoder"]
    assert len(te) == 292 and all(k.startswith("module.backbone.") for k in te), (len(te), list(te)[:3])
    out = {"target_encoder": te, "epoch": ck.get("epoch"), "source_ckpt": os.path.abspath(src),
           "config_hash": ck.get("config_hash"), "loss": ck.get("loss"), "batch_size": ck.get("batch_size"),
           "note": "EMA target encoder only (fp32); keys as in vitl_merge_3dataset_e50.pt"}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(f"{dst}: {len(te)} keys, epoch {out['epoch']}, {os.path.getsize(dst)/1e9:.2f} GB, sha256 {h[:16]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
