#!/usr/bin/env python
"""Paired PDMS comparison with scene-level and log-clustered bootstrap (plan v4 §6).

    python eval/pdms_bootstrap.py --a runA/score.csv --b runB/score.csv --tokens lists/navtest_camf0_23.csv --out cmp.json

Inputs are navsim per-token score CSVs (columns token, score, sub-metrics...). The token->log map comes from the
list_frames CSV (token, log_name). Reports mean A, mean B, paired mean difference (B−A), 95 % CI from (i) scene
bootstrap and (ii) cluster bootstrap resampling logs (adjacent scenes within a log are correlated), both with
B=10,000 draws, plus the same for each sub-metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]


def paired_ci(d: np.ndarray, groups: np.ndarray | None, n_boot: int, rng: np.random.Generator):
    if groups is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        g_ids, inv = np.unique(groups, return_inverse=True)
        sums = np.bincount(inv, weights=d); counts = np.bincount(inv)
        pick = rng.integers(0, len(g_ids), size=(n_boot, len(g_ids)))
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.std())


def compare(a: pd.DataFrame, b: pd.DataFrame, tok2log: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    m = a.merge(b, on="token", suffixes=("_a", "_b"))
    if "valid_a" in m.columns:
        m = m[m.valid_a.astype(bool) & m.valid_b.astype(bool)]
    m["log"] = m.token.map(tok2log)
    rng = np.random.default_rng(seed)
    out = {"n_tokens": int(len(m)), "n_logs": int(m.log.nunique()), "n_boot": n_boot}
    for k in METRICS:
        if f"{k}_a" not in m.columns:
            continue
        d = (m[f"{k}_b"] - m[f"{k}_a"]).to_numpy(dtype=float) * 100
        lo_s, hi_s, se_s = paired_ci(d, None, n_boot, rng)
        lo_c, hi_c, se_c = paired_ci(d, m.log.to_numpy(), n_boot, rng)
        out[k] = {"mean_a": round(float(m[f"{k}_a"].mean() * 100), 3), "mean_b": round(float(m[f"{k}_b"].mean() * 100), 3), "diff_b_minus_a": round(float(d.mean()), 3),
                  "ci95_scene": [round(lo_s, 3), round(hi_s, 3)], "se_scene": round(se_s, 3), "ci95_log_cluster": [round(lo_c, 3), round(hi_c, 3)], "se_log_cluster": round(se_c, 3),
                  "significant_cluster": bool(lo_c > 0 or hi_c < 0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = pd.read_csv(a.tokens).drop_duplicates("token"); tok2log = dict(zip(tok.token, tok.log_name))
    res = compare(pd.read_csv(a.a), pd.read_csv(a.b), tok2log, a.n_boot, a.seed)
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res["score"] if "score" in res else res, indent=1))
    return 0


def _selftest():
    """Synthetic check: +0.5 PDMS true effect with per-log variation; no clipping (clipping at 1.0 biases a paired test)."""
    rng = np.random.default_rng(1)
    logs = np.repeat(np.arange(150), 80); n = len(logs)
    log_eff = rng.normal(0, 0.03, 150)[logs]
    base = 0.85 + log_eff + rng.normal(0, 0.08, n)
    delta = 0.005 + rng.normal(0, 0.003, 150)[logs]
    bb = base + delta + rng.normal(0, 0.03, n)
    tokens = [f"t{i}" for i in range(n)]
    a = pd.DataFrame({"token": tokens, "score": base}); b = pd.DataFrame({"token": tokens, "score": bb})
    res = compare(a, b, {t: int(l) for t, l in zip(tokens, logs)}, n_boot=3000)
    print("selftest:", json.dumps(res["score"]))
    assert 0.35 < res["score"]["diff_b_minus_a"] < 0.65 and res["score"]["ci95_log_cluster"][0] < 0.5 < res["score"]["ci95_log_cluster"][1]
    assert res["score"]["se_log_cluster"] > res["score"]["se_scene"]  # per-log effect variation inflates the clustered SE
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
