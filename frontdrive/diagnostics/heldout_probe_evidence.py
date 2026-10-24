#!/usr/bin/env python3
"""Assemble the unseen-device probe evidence (card C-33, docs/evidence/2026-10-15_unseen_device_probe/).

Copies the per-group ``probe_compare_encoders`` outputs, extraction metadata and single-encoder probe tables from the
diagnostics directory into the evidence directory, writes ``summary.md`` (headline tables per pair x feature variant with one
column per group, the 9 x 3 verdict counts, and the pre-registered criterion check), and refreshes the ``## Results`` block of the
README between ``<!-- RESULTS:BEGIN -->`` / ``<!-- RESULTS:END -->``.  Idempotent; re-run as groups finish.

Usage: python -m diagnostics.heldout_probe_evidence [--diag DIR] [--ev DIR] [--groups tici tizi mici pooled neo]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

DIAG = "/home/henry/Desktop/Qwen_Drive/benchmarks/pretrain/diagnostics/heldout_probe"
EV = "docs/evidence/2026-10-15_unseen_device_probe"
GROUPS = ["tici", "tizi", "mici", "pooled", "neo"]
ENCS = ["e50", "Cp", "Dp"]
PAIRS = ["Dp-Cp", "Cp-e50", "Dp-e50"]
VARIANTS = ["f8_mean", "f2_mean", "f8_last"]
SUPERVISED = ["v_now", "yaw_rate_now", "a_now", "dv_2s", "dyaw_2s", "dleft_2s"]
EXPLORATORY = ["steer_now", "decel_event", "standstill_now"]
MAIN_METRIC = {"continuous": "r2", "binary": "auprc"}
PAIR_LABEL = {"Dp-Cp": "D' - C' (objective effect)", "Cp-e50": "C' - e50 (data effect)", "Dp-e50": "D' - e50 (both)"}


def load_group(diag: str, g: str) -> dict | None:
    p = os.path.join(diag, f"compare_{g}", "compare.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt_delta(d: dict) -> str:
    flag = {"better": "+", "worse": "-", "n.s.": ""}.get(d["verdict"], "?")
    mark = " **better**" if d["verdict"] == "better" else (" **worse**" if d["verdict"] == "worse" else " n.s.")
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{mark}"


def counts(cmp: dict, pair: str, variant: str, targets: List[str]) -> Dict[str, int]:
    c = {"better": 0, "n.s.": 0, "worse": 0, "missing": 0}
    for t in targets:
        tgt = cmp["targets"].get(t)
        if not tgt or variant not in tgt["variants"] or pair not in tgt["variants"][variant]["pairs"]:
            c["missing"] += 1; continue
        m = MAIN_METRIC[tgt["kind"]]
        c[tgt["variants"][variant]["pairs"][pair][m]["verdict"]] += 1
    return c


def copy_inputs(diag: str, ev: str, groups: List[str]) -> List[str]:
    copied = []
    for g in groups:
        src = os.path.join(diag, f"compare_{g}")
        if os.path.isdir(src):
            for ext in ("md", "json"):
                s = os.path.join(src, f"compare.{ext}")
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(ev, f"compare_{g}.{ext}")); copied.append(f"compare_{g}.{ext}")
        if g == "pooled":
            continue
        for enc in ENCS:
            meta = os.path.join(diag, f"features_{enc}_heldout_{g}_stride8_fp32.npz.meta.json")
            if os.path.exists(meta):
                shutil.copy2(meta, os.path.join(ev, f"features_{enc}_heldout_{g}.meta.json")); copied.append(f"features_{enc}_heldout_{g}.meta.json")
            pdir = os.path.join(diag, f"probe_{enc}_{g}")
            if os.path.isdir(pdir):
                dst = os.path.join(ev, f"probe_{enc}_{g}"); os.makedirs(dst, exist_ok=True)
                for f in ("table.md", "probe.log"):
                    if os.path.exists(os.path.join(pdir, f)):
                        shutil.copy2(os.path.join(pdir, f), os.path.join(dst, f)); copied.append(f"probe_{enc}_{g}/{f}")
    return copied


def build_summary(cmps: Dict[str, dict]) -> str:
    groups = [g for g in GROUPS if g in cmps]
    out = ["# Summary: unseen-device motion probes (e50 vs C' vs D')", ""]
    out.append("Groups available: " + ", ".join(f"{g} ({cmps[g]['n_clips_total']} clips / {cmps[g]['n_windows_total']} windows)" for g in groups) + ".")
    out.append("Cells: paired difference of the main metric (R^2 for continuous targets, AUPRC for binary) with the 95 % clip-clustered "
               "bootstrap CI; **better** / **worse** = CI excludes 0 in that direction. tici / tizi / mici = unseen dongles of device "
               "types present in comma v0; neo = unseen device type (EON); pooled = tici + tizi + mici concatenated.")
    out.append("")
    # criterion check
    out.append("## Pre-registered criterion check (six motion-supervised targets, f8_mean; pass = >= 4 better and 0 worse)")
    out.append("")
    out.append("| group | role | f8_mean D'-C' better / n.s. / worse | result | f2_mean D'-C' better / n.s. / worse (secondary) | C'-e50 f8_mean better / n.s. / worse |")
    out.append("|---|---|---|---|---|---|")
    for g in groups:
        c8 = counts(cmps[g], "Dp-Cp", "f8_mean", SUPERVISED); c2 = counts(cmps[g], "Dp-Cp", "f2_mean", SUPERVISED)
        ce = counts(cmps[g], "Cp-e50", "f8_mean", SUPERVISED)
        ok = c8["better"] >= 4 and c8["worse"] == 0
        role = "pre-registered test (unseen device type)" if g == "neo" else "descriptive (threshold fixed after tici/tizi seen)"
        res = ("**PASS**" if ok else "**FAIL**") if g == "neo" else ("meets" if ok else "does not meet")
        out.append(f"| {g} | {role} | {c8['better']} / {c8['n.s.']} / {c8['worse']} | {res} | {c2['better']} / {c2['n.s.']} / {c2['worse']} | {ce['better']} / {ce['n.s.']} / {ce['worse']} |")
    out.append("")
    # verdict counts over all 9 targets x 3 variants
    out.append("## Verdict counts over all 9 targets x 3 feature variants (main metric)")
    out.append("")
    out.append("| group | " + " | ".join(f"{p} better / n.s. / worse" for p in PAIRS) + " |")
    out.append("|---|" + "---|" * len(PAIRS))
    for g in groups:
        cells = []
        for p in PAIRS:
            tot = {"better": 0, "n.s.": 0, "worse": 0}
            for v in VARIANTS:
                c = counts(cmps[g], p, v, SUPERVISED + EXPLORATORY)
                for k in tot: tot[k] += c[k]
            cells.append(f"{tot['better']} / {tot['n.s.']} / {tot['worse']}")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    out.append("")
    # headline tables
    for pair in PAIRS:
        for variant in ["f8_mean", "f2_mean"]:
            out.append(f"## {PAIR_LABEL[pair]}, feature variant `{variant}`")
            out.append("")
            out.append("| target | metric | " + " | ".join(groups) + " |")
            out.append("|---|---|" + "---|" * len(groups))
            for t in SUPERVISED + EXPLORATORY:
                row = []
                metric = None
                for g in groups:
                    tgt = cmps[g]["targets"].get(t)
                    if not tgt or variant not in tgt["variants"]:
                        row.append("–"); continue
                    metric = MAIN_METRIC[tgt["kind"]]
                    row.append(fmt_delta(tgt["variants"][variant]["pairs"][pair][metric]))
                tag = " (supervised)" if t in SUPERVISED else " (exploratory)"
                out.append(f"| {t}{tag} | {metric or ''} | " + " | ".join(row) + " |")
            out.append("")
    # point estimates per group (f8_mean and f2_mean)
    out.append("## Point estimates (main metric) per group")
    out.append("")
    for g in groups:
        out.append(f"### {g}")
        out.append("")
        out.append("| target | metric | e50 f8 | C' f8 | D' f8 | e50 f2 | C' f2 | D' f2 |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for t in SUPERVISED + EXPLORATORY:
            tgt = cmps[g]["targets"].get(t)
            if not tgt: continue
            m = MAIN_METRIC[tgt["kind"]]
            p8 = tgt["variants"]["f8_mean"]["point"][m]; p2 = tgt["variants"]["f2_mean"]["point"][m]
            out.append(f"| {t} | {m} | " + " | ".join(f"{p8[e]:.3f}" for e in ENCS) + " | " + " | ".join(f"{p2[e]:.3f}" for e in ENCS) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def refresh_readme(ev: str, summary: str) -> bool:
    p = os.path.join(ev, "README.md")
    if not os.path.exists(p):
        return False
    s = open(p, encoding="utf-8").read()
    b, e = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    if b not in s or e not in s:
        return False
    # keep the headline + criterion + counts sections only (the full tables live in summary.md)
    cut = summary.split("## D' - C' (objective effect), feature variant `f8_mean`")[0]
    body = cut.split("\n", 2)[2] if cut.startswith("# ") else cut
    new = s[: s.index(b) + len(b)] + "\n" + body.strip() + "\n\nFull per-target tables: `summary.md`; per-group reports `compare_<group>.md`.\n" + s[s.index(e):]
    open(p, "w", encoding="utf-8").write(new)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default=DIAG); ap.add_argument("--ev", default=EV)
    ap.add_argument("--groups", nargs="*", default=GROUPS)
    a = ap.parse_args()
    os.makedirs(a.ev, exist_ok=True)
    copied = copy_inputs(a.diag, a.ev, a.groups)
    cmps = {g: c for g in a.groups if (c := load_group(a.diag, g)) is not None}
    if not cmps:
        print("no compare.json found"); return 1
    summary = build_summary(cmps)
    open(os.path.join(a.ev, "summary.md"), "w", encoding="utf-8").write(summary)
    ok = refresh_readme(a.ev, summary)
    print(f"groups: {list(cmps)}; copied {len(copied)} files; README results block {'refreshed' if ok else 'NOT refreshed (markers missing)'}")
    for g in cmps:
        c8 = counts(cmps[g], "Dp-Cp", "f8_mean", SUPERVISED)
        print(f"  {g}: D'-C' f8_mean supervised better/n.s./worse = {c8['better']}/{c8['n.s.']}/{c8['worse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
