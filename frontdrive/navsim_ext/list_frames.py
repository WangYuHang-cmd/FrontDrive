#!/usr/bin/env python
"""List the camera files a NAVSIM split actually needs for a given sensor config (default: CAM_F0 history [2,3],
the Drive-JEPA two-frame protocol), using the navsim v1.1 SceneLoader so the token set equals the devkit's.

Output CSV columns: token, log_name, hist_idx, cam, rel_path, frame_idx, timestamp. Also writes a plain file list
(rel paths, unique) for `rsync --files-from` or a host-side crop job, and a JSON summary.

    source frontdrive/env/drivejepa_env.sh
    python -m navsim_ext.list_frames --split navtest --out /home/henry/Desktop/Qwen_Drive/benchmarks/navsim/lists/navtest_camf0_23
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "/home/henry/Desktop/Qwen_Drive/frontdrive/third_party/Drive-JEPA/navsim_v1"))
sys.path.insert(0, str(DEVKIT))
from navsim.common.dataloader import filter_scenes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest", help="train_test_split config name (navtest, navtrain, ...)")
    ap.add_argument("--data-root", default=os.environ.get("OPENSCENE_DATA_ROOT"))
    ap.add_argument("--logs-subdir", default=None, help="override: dir holding <split>/*.pkl (default <data-root>/<test|trainval>_navsim_logs)")
    ap.add_argument("--hist", default="2,3", help="history indices to keep (Drive-JEPA cam_f0=[2,3])")
    ap.add_argument("--cams", default="CAM_F0")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()
    cfg_dir = DEVKIT / "navsim/planning/script/config/common/train_test_split"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=a.split)
    scene_filter = instantiate(cfg.scene_filter)
    data_split = cfg.data_split  # 'test' or 'trainval'
    logs_dir = Path(a.logs_subdir) if a.logs_subdir else Path(a.data_root) / f"{data_split}_navsim_logs" / data_split
    if not logs_dir.exists():  # devkit default layout
        logs_dir = Path(a.data_root) / "navsim_logs" / data_split
    print(f"[list_frames] split={a.split} data_split={data_split} logs_dir={logs_dir}", flush=True)
    scenes = filter_scenes(logs_dir, scene_filter)
    hist = [int(x) for x in a.hist.split(",")]; cams = a.cams.split(",")
    rows, files = [], set()
    for token, frames in scenes.items():
        for h in hist:
            f = frames[h]
            for cam in cams:
                rel = f["cams"][cam]["data_path"]
                rows.append({"token": token, "log_name": f["log_name"], "hist_idx": h, "cam": cam, "rel_path": rel, "frame_idx": f["frame_idx"], "timestamp": f["timestamp"]})
                files.add(rel)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    Path(str(out) + ".files.txt").write_text("\n".join(sorted(files)) + "\n")
    summary = {"split": a.split, "data_split": data_split, "n_tokens": len(scenes), "n_rows": len(rows), "n_unique_files": len(files),
               "n_logs": len({r["log_name"] for r in rows}), "hist": hist, "cams": cams, "logs_dir": str(logs_dir)}
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
