#!/usr/bin/env python3
"""Late-window summary of a pb run on dev (plan v4.3 ②/③ acceptance): per-export PDMS, the late mean / SD, and a paired
log-clustered bootstrap of the per-scene mean over the late exports vs a reference eval (official pb ckpt on dev, or the e50
baseline's late mean for the ③ encoder swaps).
usage: pb_late_window.py --run pb_e50_s0 --epochs 16 17 18 19 20 [--ref pb_official_v1_dev | --ref-run pb_e50_s0]
       [--threshold 92.96] [--n-boot 10000] [--out docs/evidence/.../late_window_<run>.json]
Reads benchmarks/navsim/exp/eval_pb/<tag>/<stamp>.csv (newest per tag) and the D5 attrs file for the log clustering.
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys

EXP = "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp"
ATTRS = "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/diagnostics/d5/attrs_navtrain_val_official.csv"
COLS = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
        "driving_direction_compliance"]


def newest_csv(tag):
    files = sorted(glob.glob(os.path.join(EXP, "eval_pb", tag, "*.csv")), key=os.path.getmtime)
    return files[-1] if files else None


def load_scores(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            tok = row.get("token")
            if not tok or tok == "average":
                continue
            out[tok] = {c: float(row[c]) for c in COLS if c in row and row[c] != ""}
    return out


def token_logs():
    logs = {}
    if not os.path.exists(ATTRS):
        return logs
    with open(ATTRS, newline="") as fh:
        r = csv.DictReader(fh)
        tok_col = "token" if "token" in r.fieldnames else r.fieldnames[0]
        log_col = next((c for c in ("log_name", "log") if c in r.fieldnames), None)
        for row in r:
            if log_col:
                logs[row[tok_col]] = row[log_col]
    return logs


def paired_cluster_bootstrap(a, b, logs, n_boot, seed=0):
    """mean(b - a) over common tokens with a log-clustered percentile bootstrap (tokens without a log form their own cluster)."""
    common = sorted(set(a) & set(b))
    diffs = {t: b[t]["score"] - a[t]["score"] for t in common}
    clusters = {}
    for t in common:
        clusters.setdefault(logs.get(t, t), []).append(diffs[t])
    keys = list(clusters)
    sums = [sum(clusters[k]) for k in keys]
    counts = [len(clusters[k]) for k in keys]
    point = sum(sums) / sum(counts)
    rng = random.Random(seed)
    boots = []
    n = len(keys)
    for _ in range(n_boot):
        s = c = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += sums[i]
            c += counts[i]
        boots.append(s / c)
    boots.sort()
    return point * 100, boots[int(0.025 * n_boot)] * 100, boots[int(0.975 * n_boot)] * 100, len(common), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", nargs="+", type=int, required=True)
    ap.add_argument("--ref", default="pb_official_v1_dev", help="reference eval tag (per-scene CSV)")
    ap.add_argument("--ref-run", default=None, help="instead of --ref: a run whose late window (same epochs) is the reference")
    ap.add_argument("--threshold", type=float, default=None, help="late-mean threshold for the verdict (e.g. 92.96)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logs = token_logs()
    per_export, scenes = [], []
    for ep in a.epochs:
        tag = f"{a.run}_ep{ep}"
        p = newest_csv(tag)
        if not p:
            print(f"missing per-scene CSV for {tag}", file=sys.stderr)
            continue
        s = load_scores(p)
        scenes.append(s)
        per_export.append({"epoch": ep, "tag": tag, "n": len(s), **{c: round(100 * statistics.mean(v[c] for v in s.values()), 2) for c in COLS}})
    if not scenes:
        sys.exit("no late-window evaluations found")
    pdms = [e["score"] for e in per_export]
    late_mean = statistics.mean(pdms)
    late_sd = statistics.stdev(pdms) if len(pdms) > 1 else 0.0
    # per-scene mean over the late exports
    common = set.intersection(*[set(s) for s in scenes])
    late_avg = {t: {c: statistics.mean(s[t][c] for s in scenes) for c in COLS if all(c in s[t] for s in scenes)} for t in common}
    # reference
    if a.ref_run:
        ref_scenes = []
        for ep in a.epochs:
            p = newest_csv(f"{a.ref_run}_ep{ep}")
            if p:
                ref_scenes.append(load_scores(p))
        if not ref_scenes:
            sys.exit(f"no reference evaluations for {a.ref_run}")
        rc = set.intersection(*[set(s) for s in ref_scenes])
        ref = {t: {c: statistics.mean(s[t][c] for s in ref_scenes) for c in COLS if all(c in s[t] for s in ref_scenes)} for t in rc}
        ref_label = f"{a.ref_run} late mean of epochs {sorted(set(e['epoch'] for e in per_export))} ({len(ref_scenes)} exports)"
    else:
        p = newest_csv(a.ref)
        if not p:
            sys.exit(f"no reference CSV for {a.ref}")
        ref = load_scores(p)
        ref_label = a.ref
    ref_pdms = 100 * statistics.mean(v["score"] for v in ref.values())
    d, lo, hi, n_common, n_clusters = paired_cluster_bootstrap(ref, late_avg, logs, a.n_boot)
    sub_delta = {c: round(100 * (statistics.mean(late_avg[t][c] for t in late_avg) - statistics.mean(ref[t][c] for t in ref)), 2) for c in COLS[1:]}
    verdict = None
    if a.threshold is not None:
        verdict = {"threshold": a.threshold, "pass": late_mean >= a.threshold, "margin": round(late_mean - a.threshold, 2)}
    res = {"run": a.run, "epochs": [e["epoch"] for e in per_export], "per_export": per_export,
           "late_mean_pdms": round(late_mean, 2), "late_sd_pdms": round(late_sd, 2), "late_min": min(pdms), "late_max": max(pdms),
           "reference": ref_label, "reference_pdms": round(ref_pdms, 2),
           "paired_delta_late_avg_vs_ref": {"delta": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)], "n_scenes": n_common,
                                            "n_clusters": n_clusters, "n_boot": a.n_boot, "clustered_by_log": bool(logs)},
           "subscore_delta_vs_ref": sub_delta, "verdict": verdict}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
