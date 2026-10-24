#!/usr/bin/env python3
"""Paired encoder comparison of frozen-feature probes (plan v4.1 §3.1 evidence chain, step 2: "does the synchronised motion
supervision improve the measurable dynamic representation, and does it hold on unseen devices?").

Inputs are ``diagnostics.d2_info_gap extract`` feature files and the matching ``probe --save-predictions`` results, one set
per encoder, over the SAME clip list(s) (the windows must be identical: clip_id and anchor arrays are checked).  Because the
probes are 5-fold GroupKFold by clip with a fixed seed, every window carries one held-out prediction per encoder and feature
variant, so encoder differences can be bootstrapped *paired over clips*: resample clips with replacement, recompute the pooled
metric of each encoder on the resampled windows, take the difference; 95 % percentile CIs, identical draws for all targets,
variants and pairs.  Continuous targets: pooled R^2 and MAE; binary targets: AUROC and AUPRC (held-out scores).

Several (features, results) pairs per encoder (e.g. one per held-out device) are concatenated -> pooled comparison; run the
script per device as well to see whether the sign is consistent across devices.

Usage:
  python -m diagnostics.probe_compare_encoders --run e50 F.npz R.json [F2.npz R2.json ...] --run Cp ... --run Dp ... \
      --pairs Cp-e50 Dp-Cp Dp-e50 --n-boot 2000 --seed 0 --out <dir>
Writes <out>/compare.json and <out>/compare.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURE_VARIANTS = ["f2_mean", "f8_mean", "f8_last"]


def r2_pooled(y: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), ties get half credit."""
    pos = s[y > 0.5]; neg = s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    if (y > 0.5).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = (y[order] > 0.5).astype(np.float64)
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    prec = tp / k
    return float((prec * yy).sum() / yy.sum())


METRICS = {"continuous": {"r2": (r2_pooled, +1), "mae": (mae, -1)}, "binary": {"auroc": (auroc, +1), "auprc": (auprc, +1)}}


def load_run(paths: Sequence[str]) -> Dict:
    """paths = [npz, json, npz, json, ...] -> dict with concatenated windows: clip_id, anchor, device, targets, target_names,
    and per-target per-variant held-out predictions (aligned to the finite-target rows, concatenated in file order)."""
    if len(paths) % 2 != 0:
        raise ValueError("each --run needs <features.npz> <results.json> pairs")
    parts = []
    for npz, js in zip(paths[0::2], paths[1::2]):
        with np.load(npz, allow_pickle=False) as z:
            arr = {k: z[k] for k in ("clip_id", "anchor", "targets", "target_names") if k in z.files}
            arr["device"] = z["device"] if "device" in z.files else np.array([""] * len(arr["clip_id"]))
        res = json.load(open(js))
        parts.append((npz, js, arr, res))
    names = [str(x) for x in parts[0][2]["target_names"]]
    for npz, js, arr, res in parts[1:]:
        if [str(x) for x in arr["target_names"]] != names:
            raise ValueError(f"{npz}: target_names differ from {parts[0][0]}")
    out = {"files": [(p[0], p[1]) for p in parts], "target_names": names,
           "clip_id": np.concatenate([p[2]["clip_id"].astype(str) for p in parts]),
           "anchor": np.concatenate([p[2]["anchor"] for p in parts]),
           "device": np.concatenate([p[2]["device"].astype(str) for p in parts]),
           "targets": np.concatenate([p[2]["targets"] for p in parts], axis=0), "pred": {}}
    for ti, t in enumerate(names):
        kind = None; per_fv: Dict[str, List[np.ndarray]] = {fv: [] for fv in FEATURE_VARIANTS}
        for npz, js, arr, res in parts:
            e = res["targets"].get(t, {})
            if "error" in e or not e:
                per_fv = None; break
            kind = e["kind"]
            key = "held_out_scores" if kind == "binary" else "held_out_predictions"
            if not e.get(key):
                raise ValueError(f"{js}: target {t} has no saved {key} (run probe with --save-predictions)")
            n_ok = int(np.isfinite(arr["targets"][:, ti]).sum())
            for fv in FEATURE_VARIANTS:
                v = np.asarray(e[key][fv], dtype=np.float64)
                if len(v) != n_ok:
                    raise ValueError(f"{js}: target {t} / {fv}: {len(v)} predictions for {n_ok} valid windows in {npz}")
                per_fv[fv].append(v)
        if per_fv is not None:
            out["pred"][t] = {"kind": kind, **{fv: np.concatenate(per_fv[fv]) for fv in FEATURE_VARIANTS}}
    return out


def check_aligned(a: Dict, b: Dict, la: str, lb: str) -> None:
    if len(a["clip_id"]) != len(b["clip_id"]) or not np.array_equal(a["clip_id"], b["clip_id"]) or not np.array_equal(a["anchor"], b["anchor"]):
        raise ValueError(f"windows of {la} and {lb} differ (clip_id / anchor); the encoders must be extracted on the same lists")
    if not np.allclose(np.nan_to_num(a["targets"], nan=-9e9), np.nan_to_num(b["targets"], nan=-9e9)):
        raise ValueError(f"targets of {la} and {lb} differ")


def clip_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> List[np.ndarray]:
    """Index arrays of windows for n_boot resamples of clips with replacement (same draws for every metric)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        draws.append(np.concatenate([members[i] for i in pick]))
    return draws


def compare(runs: Dict[str, Dict], pairs: List[Tuple[str, str]], n_boot: int, seed: int) -> Dict:
    labels = list(runs)
    ref = runs[labels[0]]
    for l in labels[1:]:
        check_aligned(ref, runs[l], labels[0], l)
    names = ref["target_names"]
    out = {"labels": labels, "pairs": [f"{b}-{a}" for b, a in pairs], "n_boot": n_boot, "seed": seed, "n_windows_total": int(len(ref["clip_id"])),
           "n_clips_total": int(len(np.unique(ref["clip_id"]))), "devices": {d: int(c) for d, c in zip(*np.unique(ref["device"], return_counts=True))},
           "targets": {}}
    for ti, t in enumerate(names):
        if any(t not in runs[l]["pred"] for l in labels):
            out["targets"][t] = {"error": "missing in at least one run"}; continue
        y_all = ref["targets"][:, ti]; ok = np.isfinite(y_all); y = y_all[ok]; g = ref["clip_id"][ok]
        kind = ref["pred"][t]["kind"]
        draws = clip_bootstrap_indices(g, n_boot, seed)
        entry = {"kind": kind, "n_windows": int(ok.sum()), "n_clips": int(len(np.unique(g))), "variants": {}}
        for fv in FEATURE_VARIANTS:
            preds = {l: runs[l]["pred"][t][fv] for l in labels}
            vrow = {"point": {}, "pairs": {}}
            for m, (fn, better) in METRICS[kind].items():
                vrow["point"][m] = {l: fn(y, preds[l]) for l in labels}
                for b, a in pairs:
                    diffs = np.array([fn(y[ix], preds[b][ix]) - fn(y[ix], preds[a][ix]) for ix in draws])
                    diffs = diffs[np.isfinite(diffs)]
                    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (float("nan"), float("nan")))
                    d = vrow["point"][m][b] - vrow["point"][m][a]
                    verdict = "better" if (better * lo > 0 if better > 0 else better * hi > 0) else ("worse" if (better * hi < 0 if better > 0 else better * lo < 0) else "n.s.")
                    vrow["pairs"].setdefault(f"{b}-{a}", {})[m] = {"delta": d, "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict}
            entry["variants"][fv] = vrow
        out["targets"][t] = entry
    return out


def to_markdown(res: Dict, title: str) -> str:
    L = [f"# {title}", "", f"Windows {res['n_windows_total']}, clips {res['n_clips_total']}, devices {res['devices']}; paired clip-clustered bootstrap "
         f"{res['n_boot']} draws (seed {res['seed']}), 95 % percentile CIs. 'better' / 'worse' = CI excludes 0 in the direction of the metric.", ""]
    labels = res["labels"]
    for t, e in res["targets"].items():
        if "error" in e:
            L.append(f"## {t}: {e['error']}"); L.append(""); continue
        main = "r2" if e["kind"] == "continuous" else "auprc"
        L.append(f"## {t} ({e['kind']}, {e['n_windows']} windows / {e['n_clips']} clips) -- {main}")
        L.append("")
        L.append("| variant | " + " | ".join(labels) + " | " + " | ".join(f"Δ {p}" for p in res["pairs"]) + " |")
        L.append("|---|" + "---:|" * (len(labels) + len(res["pairs"])))
        for fv, v in e["variants"].items():
            cells = [f"{v['point'][main][l]:.3f}" for l in labels]
            for p in res["pairs"]:
                d = v["pairs"][p][main]
                cells.append(f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {d['verdict']}")
            L.append(f"| {fv} | " + " | ".join(cells) + " |")
        L.append("")
    # summary counts over targets x variants for the main metric
    L.append("## Summary (main metric, counts over targets × variants)")
    L.append("")
    L.append("| pair | better | n.s. | worse |")
    L.append("|---|---:|---:|---:|")
    for p in res["pairs"]:
        c = {"better": 0, "n.s.": 0, "worse": 0}
        for t, e in res["targets"].items():
            if "error" in e: continue
            main = "r2" if e["kind"] == "continuous" else "auprc"
            for fv, v in e["variants"].items():
                c[v["pairs"][p][main]["verdict"]] += 1
        L.append(f"| {p} | {c['better']} | {c['n.s.']} | {c['worse']} |")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar="ARG", help="LABEL features.npz results.json [features.npz results.json ...]")
    ap.add_argument("--pairs", nargs="*", default=None, help="B-A pairs (default: every later run vs the first)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Encoder comparison of frozen-feature probes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = {}
    for r in a.run:
        if len(r) < 3:
            ap.error("--run needs LABEL and at least one features/results pair")
        runs[r[0]] = load_run(r[1:])
    labels = list(runs)
    pairs = [(l, labels[0]) for l in labels[1:]] if not a.pairs else [tuple(p.split("-", 1)) for p in a.pairs]
    for b, aa in pairs:
        if b not in runs or aa not in runs:
            ap.error(f"unknown label in pair {b}-{aa}")
    res = compare(runs, pairs, a.n_boot, a.seed)
    res["runs"] = {l: runs[l]["files"] for l in labels}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "compare.json"), "w"), indent=1)
    md = to_markdown(res, a.title)
    open(os.path.join(a.out, "compare.md"), "w").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
