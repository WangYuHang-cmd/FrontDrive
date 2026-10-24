"""Cross-device comparison of D2 info-gap probe results.

Reads several ``results.json`` files written by ``diagnostics.d2_info_gap probe`` (one per device subset) and writes a
markdown table plus a JSON summary of the 8f-vs-2f gaps per target and device, with the operational verdict used in the
D2 README (large: 95 % CI of the paired gap entirely above +0.10 R^2 / +0.05 AUROC or AUPRC; small: CI within +-0.05;
2f-better: CI entirely below 0 but not within +-0.05; moderate otherwise).  Usage:

    python -m diagnostics.d2_compare_devices --run mici=<dir>/results.json --run tici=... --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONT_TARGETS = ["v_now", "yaw_rate_now", "a_now", "dv_2s", "dyaw_2s", "dleft_2s", "steer_now"]
BIN_TARGETS = ["decel_event", "standstill_now"]
LARGE = {"r2": 0.10, "auroc": 0.05, "auprc": 0.05}
SMALL = 0.05


def verdict(metric: str, ci: list[float]) -> str:
    """large / moderate = 8f reads out more; small = no gap either way; 2f-better = CI entirely below 0 (beyond +-0.05)."""
    lo, hi = ci
    if lo > LARGE[metric]:
        return "large"
    if -SMALL <= lo and hi <= SMALL:
        return "small"
    if hi < 0:
        return "2f-better"
    return "moderate"


def fmt(v: float, ci: list[float] | None = None, nd: int = 3) -> str:
    s = f"{v:.{nd}f}"
    if ci is not None:
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def summarise(runs: dict[str, dict]) -> dict:
    out: dict = {"devices": {}, "targets": {}}
    for dev, r in runs.items():
        out["devices"][dev] = {
            "n_windows": r.get("n_windows_total"),
            "n_clips": r.get("n_clips_total"),
            "created": r.get("created"),
            "devices_in_windows": r.get("extract_meta", {}).get("devices"),
        }
    for tgt in CONT_TARGETS + BIN_TARGETS:
        metrics = ["r2"] if tgt in CONT_TARGETS else ["auroc", "auprc"]
        out["targets"][tgt] = {}
        for metric in metrics:
            row = {}
            for dev, r in runs.items():
                t = r["targets"].get(tgt)
                if t is None or metric not in t["bootstrap"]:
                    continue
                b = t["bootstrap"][metric]
                est, dif = b["estimates"], b["differences"]
                row[dev] = {
                    "n_windows": t["n_windows"],
                    "n_clips": t["n_clips"],
                    "f2_mean": est["f2_mean"]["value"],
                    "f8_mean": est["f8_mean"]["value"],
                    "f8_last": est["f8_last"]["value"],
                    "gap_f8_mean": dif["f8_mean-f2_mean"]["value"],
                    "gap_f8_mean_ci": dif["f8_mean-f2_mean"]["ci95"],
                    "gap_f8_last": dif["f8_last-f2_mean"]["value"],
                    "gap_f8_last_ci": dif["f8_last-f2_mean"]["ci95"],
                    "verdict_f8_last": verdict(metric, dif["f8_last-f2_mean"]["ci95"]),
                    "verdict_f8_mean": verdict(metric, dif["f8_mean-f2_mean"]["ci95"]),
                }
                if tgt in BIN_TARGETS:
                    row[dev]["prevalence"] = t["distribution"].get("prevalence", t["distribution"].get("mean"))
            out["targets"][tgt][metric] = row
    return out


def to_markdown(summary: dict, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("Each cell: held-out probe score (pooled over 5-fold GroupKFold by clip); gaps are paired clip-bootstrap "
                 "differences with 95 % CIs. Verdict rule (from the D2 README): large = CI entirely above +0.10 R^2 / "
                 "+0.05 AUROC or AUPRC; small = CI within +-0.05; 2f-better = CI entirely below 0 but not within +-0.05; moderate otherwise.")
    lines.append("")
    lines.append("| device | windows | clips | created |")
    lines.append("|---|---:|---:|---|")
    for dev, d in summary["devices"].items():
        lines.append(f"| {dev} | {d['n_windows']} | {d['n_clips']} | {d['created']} |")
    lines.append("")
    for tgt, per_metric in summary["targets"].items():
        for metric, row in per_metric.items():
            if not row:
                continue
            lines.append(f"## {tgt} ({metric.upper()})")
            lines.append("")
            lines.append("| device | f2_mean | f8_mean | f8_last | gap f8_last-f2 | verdict | gap f8_mean-f2 | verdict |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for dev, c in row.items():
                lines.append(
                    f"| {dev} | {fmt(c['f2_mean'])} | {fmt(c['f8_mean'])} | {fmt(c['f8_last'])} | "
                    f"{fmt(c['gap_f8_last'], c['gap_f8_last_ci'])} | {c['verdict_f8_last']} | "
                    f"{fmt(c['gap_f8_mean'], c['gap_f8_mean_ci'])} | {c['verdict_f8_mean']} |"
                )
            lines.append("")
    # agreement summary
    lines.append("## Verdict agreement across devices (f8_last - f2_mean)")
    lines.append("")
    lines.append("| target | metric | " + " | ".join(summary["devices"]) + " | all agree |")
    lines.append("|---|---|" + "---|" * len(summary["devices"]) + "---|")
    for tgt, per_metric in summary["targets"].items():
        for metric, row in per_metric.items():
            if not row:
                continue
            vs = [row[d]["verdict_f8_last"] if d in row else "-" for d in summary["devices"]]
            lines.append(f"| {tgt} | {metric} | " + " | ".join(vs) + f" | {'yes' if len(set(vs)) == 1 else 'no'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=path/to/results.json (repeatable, ordered)")
    ap.add_argument("--out", required=True, help="output directory (compare.md, compare.json)")
    ap.add_argument("--title", default="D2 info gap by device: 8-frame vs 2-frame frozen-feature probes")
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = load(Path(path))
    summary = summarise(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(summary, indent=1))
    (out / "compare.md").write_text(to_markdown(summary, args.title))
    print((out / "compare.md").read_text())


if __name__ == "__main__":
    main()
