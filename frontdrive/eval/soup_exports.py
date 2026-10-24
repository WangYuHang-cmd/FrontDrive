#!/usr/bin/env python3
"""Weight-space average ("soup") of fp16 agent exports written by the frontdrive export callback
({"state_dict": {"agent.*": tensor}, "epoch": N, "global_step": S}) — the late-window exports of one run (same seed, same
optimisation trajectory) averaged uniformly. Floating tensors are averaged in fp32 and stored in fp16; non-float tensors
(buffers such as anchors / indices) are taken from the first export and checked for equality.
usage: soup_exports.py --out <soup.ckpt> <export1.ckpt> <export2.ckpt> ...
"""
import argparse
import json
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("exports", nargs="+")
    a = ap.parse_args()
    acc, counts, meta, first = {}, 0, [], None
    for p in a.exports:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        meta.append({"file": p, "epoch": ck.get("epoch"), "global_step": ck.get("global_step"), "n_tensors": len(sd)})
        if first is None:
            first = {k: v.clone() for k, v in sd.items()}
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] = v.float().clone()
        else:
            if set(sd) != set(first):
                sys.exit(f"key mismatch between {a.exports[0]} and {p}")
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.float()
                elif not torch.equal(v, first[k]):
                    sys.exit(f"non-float tensor {k} differs between exports (not soup-able)")
        counts += 1
    out_sd = {}
    for k, v in first.items():
        out_sd[k] = (acc[k] / counts).to(torch.float16) if v.is_floating_point() else v
    torch.save({"state_dict": out_sd, "epoch": f"soup({','.join(str(m['epoch']) for m in meta)})",
                "global_step": meta[-1]["global_step"], "soup_of": meta}, a.out)
    print(json.dumps({"out": a.out, "n_exports": counts, "n_tensors": len(out_sd), "members": meta}, indent=1, default=str))


if __name__ == "__main__":
    main()
