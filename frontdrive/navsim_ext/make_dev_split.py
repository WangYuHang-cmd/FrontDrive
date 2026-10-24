#!/usr/bin/env python
"""Create a log-disjoint navtrain fit/dev split for model selection (plan v4 §6: never tune on navtest).

Reads the navtrain scene-filter log list from the vendored devkit, samples `--dev-frac` of the logs with a fixed
seed, and writes four hydra configs into the vendored config tree (tracked by our patch set):
  train_test_split/navtrain_fit.yaml, navtrain_dev.yaml and scene_filter/navtrain_fit.yaml, navtrain_dev.yaml
plus a JSON record of the split (log names + sha256 of the list) under frontdrive/navsim_ext/configs/.
    source env/drivejepa_env.sh && python -m navsim_ext.make_dev_split --dev-frac 0.10 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "/home/henry/Desktop/Qwen_Drive/frontdrive/third_party/Drive-JEPA/navsim_v1"))
CFG = DEVKIT / "navsim/planning/script/config/common/train_test_split"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-dir", default="/home/henry/Desktop/Qwen_Drive/frontdrive/navsim_ext/configs")
    ap.add_argument("--official", action="store_true", help="instead of a random split, restrict navtrain to the devkit's default train_logs / val_logs "
                    "(config/training/default_train_val_test_log_split.yaml, what run_training.py uses) -> navtrain_train_official / navtrain_val_official")
    a = ap.parse_args()
    base = yaml.safe_load((CFG / "scene_filter/navtrain.yaml").read_text())
    logs = sorted(set(base.get("log_names") or []))
    tokens = base.get("tokens") or []
    if not logs:  # navtrain filter without an explicit list -> take the trainval log files on disk
        root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs/trainval"
        logs = sorted(p.stem for p in root.glob("*.pkl"))
    if a.official:
        split = yaml.safe_load((DEVKIT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml").read_text())
        tr, va = set(split["train_logs"]), set(split["val_logs"])
        fit, dev = sorted(l for l in logs if l in tr), sorted(l for l in logs if l in va)
        assert len(fit) + len(dev) == len(logs), f"{len(logs) - len(fit) - len(dev)} navtrain logs are in neither train_logs nor val_logs"
        names = ("navtrain_train_official", "navtrain_val_official"); record = "navtrain_official_split.json"
    else:
        import random
        rng = random.Random(a.seed); shuffled = logs[:]; rng.shuffle(shuffled)
        n_dev = max(1, int(round(len(logs) * a.dev_frac)))
        dev, fit = sorted(shuffled[:n_dev]), sorted(shuffled[n_dev:])
        names = ("navtrain_fit", "navtrain_dev"); record = "navtrain_fit_dev_split.json"
    # navtrain selects explicit tokens: split them by the log they belong to (token -> log map from the trainval pickles)
    tok_fit, tok_dev = None, None
    if tokens:
        import pickle
        root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs/trainval"
        tok2log = {}
        for log in logs:
            fp = root / f"{log}.pkl"
            if not fp.exists():
                continue
            for fr in pickle.load(open(fp, "rb")):
                tok2log[fr["token"]] = log
        devset = set(dev)
        tok_fit = sorted(t for t in tokens if tok2log.get(t) and tok2log[t] not in devset)
        tok_dev = sorted(t for t in tokens if tok2log.get(t) in devset)
        unmapped = sum(1 for t in tokens if t not in tok2log)
        print(f"[make_dev_split] tokens total {len(tokens)}, fit {len(tok_fit)}, dev {len(tok_dev)}, unmapped {unmapped}")
    for name, lst, toks in ((names[0], fit, tok_fit), (names[1], dev, tok_dev)):
        sf = dict(base); sf["log_names"] = lst
        if toks is not None:
            sf["tokens"] = toks
        (CFG / f"scene_filter/{name}.yaml").write_text(yaml.safe_dump(sf, sort_keys=False))
        (CFG / f"{name}.yaml").write_text(f"defaults:\n  - scene_filter: {name}\n\ndata_split: trainval\n")
    rec = {"mode": "official" if a.official else "random", "seed": a.seed, "dev_frac": a.dev_frac, "n_logs_total": len(logs), "n_fit": len(fit), "n_dev": len(dev), "n_tokens_fit": len(tok_fit) if tok_fit is not None else None, "n_tokens_dev": len(tok_dev) if tok_dev is not None else None,
           "sha256_dev_list": hashlib.sha256("\n".join(dev).encode()).hexdigest(), "dev_logs": dev}
    Path(a.record_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.record_dir) / record).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "dev_logs"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
