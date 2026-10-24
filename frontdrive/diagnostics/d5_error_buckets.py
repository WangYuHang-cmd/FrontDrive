"""D5 (plan v4.1 §3.1): bucket planning errors of a NAVSIM agent on the dev split by scene situation.

Two sub-commands:

  attrs  : compute per-token situation attributes from the navsim log pickles (ego future, turn geometry, lead vehicle,
           cut-in, stop-and-go, VRUs, agent density, red lights) for every token of a scene filter -> CSV.
  report : join the attributes with one or two per-scene PDM-score CSVs (navsim `run_pdm_score*` output) and write
           markdown bucket tables (mean PDMS and sub-scores, failure rates, deficit share per bucket; paired per-bucket
           differences with log-clustered bootstrap CIs when two score files are given).

    python -m diagnostics.d5_error_buckets attrs --split navtrain_val_official --out <attrs.csv> [--limit-logs N]
    python -m diagnostics.d5_error_buckets report --attrs <attrs.csv> --scores <score.csv> [--scores-b <score.csv>] --out <dir>

Conventions (navsim / OpenScene log pickles): frames at 2 Hz; `ego2global_rotation` = quaternion (w, x, y, z);
`ego_dynamic_state` = (vx, vy, ax, ay) in the ego frame (x forward, y left); `anns.gt_boxes` = (x, y, z, l, w, h, yaw) in
the ego frame; `anns.gt_velocity_3d` = box velocity (checked by `attrs`: in-lane vehicles ahead move along +x).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FUT = 8          # 4 s at 2 Hz (PDMS horizon)
DT = 0.5
CORRIDOR = 1.75  # half-width of the ego lane corridor used for lead / cut-in tests [m]
EGO_HALF_LEN = 2.3
COMMANDS = ["left", "straight", "right", "unknown"]


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def scene_filter_tokens(split: str) -> tuple[set[str], list[str]]:
    root = os.environ["DRIVEJEPA_ROOT"]
    y = yaml.safe_load(open(f"{root}/navsim/planning/script/config/common/train_test_split/scene_filter/{split}.yaml"))
    return set(y["tokens"] or []), list(y["log_names"] or [])


def _vehicle_mask(names: np.ndarray) -> np.ndarray:
    return np.isin(names, ["vehicle", "truck", "bus", "trailer", "construction"])


def _vru_mask(names: np.ndarray) -> np.ndarray:
    return np.isin(names, ["pedestrian", "bicycle", "motorcycle", "czone_sign", "generic_object"]) & np.isin(names, ["pedestrian", "bicycle", "motorcycle"])


def _lead(boxes: np.ndarray, names: np.ndarray, vel: np.ndarray, tracks: list[str]):
    """Nearest vehicle ahead inside the ego corridor: (index, dRel, vx_abs, track) or None."""
    if len(boxes) == 0:
        return None
    m = _vehicle_mask(names) & (boxes[:, 0] > 0) & (np.abs(boxes[:, 1]) <= CORRIDOR + boxes[:, 4] / 2.0)
    if not m.any():
        return None
    idx = np.where(m)[0]
    i = idx[np.argmin(boxes[idx, 0])]
    d_rel = float(boxes[i, 0] - boxes[i, 3] / 2.0 - EGO_HALF_LEN)
    return i, d_rel, float(vel[i, 0]), tracks[i]


def frame_attrs(frames: list[dict], k: int, stats: dict) -> dict:
    f = frames[k]
    yaw0 = quat_yaw(f["ego2global_rotation"])
    p0 = np.asarray(f["ego2global_translation"][:2])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    R = np.array([[c, -s], [s, c]])
    vx, vy, ax, ay = f["ego_dynamic_state"]
    v0 = float(math.hypot(vx, vy))
    out = {"token": f["token"], "log_name": f["log_name"], "frame_idx": k,
           "command": COMMANDS[int(np.argmax(f["driving_command"]))], "v0": v0, "a0": float(ax)}
    fut = frames[k + 1:k + 1 + FUT]
    out["n_future"] = len(fut)
    if len(fut) >= 2:
        pos = np.stack([R @ (np.asarray(g["ego2global_translation"][:2]) - p0) for g in fut])
        yaws = np.array([wrap(quat_yaw(g["ego2global_rotation"]) - yaw0) for g in fut])
        speeds = np.array([math.hypot(*g["ego_dynamic_state"][:2]) for g in fut])
        allv = np.concatenate([[v0], speeds])
        out.update({
            "turn_deg_4s": float(math.degrees(yaws[-1])), "turn_deg_2s": float(math.degrees(yaws[min(3, len(yaws) - 1)])),
            "lateral_4s": float(pos[-1, 1]), "progress_4s": float(np.sum(np.linalg.norm(np.diff(np.vstack([[0, 0], pos]), axis=0), axis=1))),
            "v_min_4s": float(allv.min()), "v_max_4s": float(allv.max()), "v_end_4s": float(speeds[-1]),
            "max_decel_4s": float(np.min(np.diff(allv)) / DT), "max_accel_4s": float(np.max(np.diff(allv)) / DT),
        })
    a = f["anns"]
    boxes = np.asarray(a["gt_boxes"], dtype=float).reshape(-1, 7)
    names = np.asarray(a["gt_names"]).astype(str)
    vel = np.asarray(a["gt_velocity_3d"], dtype=float).reshape(-1, 3)
    tracks = list(a["track_tokens"])
    dist = np.hypot(boxes[:, 0], boxes[:, 1]) if len(boxes) else np.zeros(0)
    out["n_agents_30m"] = int((dist <= 30).sum())
    out["n_vehicles_30m"] = int(((dist <= 30) & _vehicle_mask(names)).sum()) if len(boxes) else 0
    vru = _vru_mask(names) if len(boxes) else np.zeros(0, bool)
    ahead_vru = vru & (boxes[:, 0] > 0) & (boxes[:, 0] < 20) & (np.abs(boxes[:, 1]) < 5) if len(boxes) else vru
    out["vru_ahead_20m"] = bool(ahead_vru.any())
    out["vru_min_dist"] = float(dist[vru].min()) if vru.any() else float("nan")
    lead = _lead(boxes, names, vel, tracks)
    out.update({"lead_d_rel": float("nan"), "lead_v_rel": float("nan"), "lead_ttc": float("nan"), "lead_dv_2s": float("nan")})
    if lead is not None:
        i, d_rel, lead_vx, trk = lead
        # sanity statistic for the velocity frame: in-lane vehicles ahead should move along +x (angle ~ 0)
        sp = math.hypot(vel[i, 0], vel[i, 1])
        if sp > 2.0:
            stats["lead_vel_angle_deg"].append(math.degrees(math.atan2(vel[i, 1], vel[i, 0])))
        v_rel = lead_vx - v0
        out["lead_d_rel"], out["lead_v_rel"] = d_rel, float(v_rel)
        out["lead_ttc"] = float(d_rel / -v_rel) if v_rel < -0.1 else float("inf")
        # lead speed change over the next 2 s (same track token)
        if len(fut) >= 4:
            g = fut[3]
            gt = list(g["anns"]["track_tokens"])
            if trk in gt:
                j = gt.index(trk)
                gv = np.asarray(g["anns"]["gt_velocity_3d"], dtype=float).reshape(-1, 3)
                out["lead_dv_2s"] = float(math.hypot(gv[j, 0], gv[j, 1]) - math.hypot(vel[i, 0], vel[i, 1]))
    # cut-in: a vehicle outside the corridor now that is inside it (ahead, < 40 m) 1-2 s later, while the ego is not turning
    cut_in = False
    if len(fut) >= 4 and abs(out.get("turn_deg_2s", 0.0)) < 10.0 and len(boxes):
        in_now = set(np.asarray(tracks)[(np.abs(boxes[:, 1]) <= CORRIDOR + boxes[:, 4] / 2.0) & (boxes[:, 0] > -5)])
        for g in (fut[1], fut[3]):
            gb = np.asarray(g["anns"]["gt_boxes"], dtype=float).reshape(-1, 7)
            gn = np.asarray(g["anns"]["gt_names"]).astype(str)
            gtk = np.asarray(list(g["anns"]["track_tokens"]))
            if len(gb) == 0:
                continue
            m = _vehicle_mask(gn) & (gb[:, 0] > 0) & (gb[:, 0] < 40) & (np.abs(gb[:, 1]) <= CORRIDOR)
            if any(t not in in_now for t in gtk[m]):
                cut_in = True
                break
    out["cut_in"] = cut_in
    tl = f.get("traffic_lights") or []
    out["n_red_lights"] = int(sum(1 for t in tl if len(t) > 1 and bool(t[1])))
    return out


def cmd_attrs(args: argparse.Namespace) -> None:
    tokens, log_names = scene_filter_tokens(args.split)
    root = os.environ["OPENSCENE_DATA_ROOT"]
    rows, stats = [], {"lead_vel_angle_deg": []}
    logs = log_names[: args.limit_logs] if args.limit_logs else log_names
    for n, ln in enumerate(logs):
        p = Path(root) / "navsim_logs" / args.log_dir / f"{ln}.pkl"
        if not p.exists():
            print(f"[attrs] missing log pickle {p}", file=sys.stderr)
            continue
        frames = pickle.load(open(p, "rb"))
        for k, f in enumerate(frames):
            if f["token"] in tokens:
                rows.append(frame_attrs(frames, k, stats))
        if (n + 1) % 20 == 0:
            print(f"[attrs] {n + 1}/{len(logs)} logs, {len(rows)} tokens", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ang = np.asarray(stats["lead_vel_angle_deg"])
    meta = {"split": args.split, "n_tokens_filter": len(tokens), "n_tokens_attrs": int(len(df)), "n_logs": len(logs),
            "lead_velocity_angle_deg": {"n": int(len(ang)), "median": float(np.median(ang)) if len(ang) else None,
                                        "p90_abs": float(np.percentile(np.abs(ang), 90)) if len(ang) else None,
                                        "note": "in-lane leads should move along +x if gt_velocity_3d is in the ego frame"}}
    json.dump(meta, open(str(args.out) + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


# ----------------------------------------------------------------------------------------------------------------- report
BUCKETS = {
    "all": lambda d: pd.Series(True, index=d.index),
    "cmd_left": lambda d: d.command == "left",
    "cmd_straight": lambda d: d.command == "straight",
    "cmd_right": lambda d: d.command == "right",
    "turn_left_ge20": lambda d: d.turn_deg_4s >= 20,
    "turn_right_ge20": lambda d: d.turn_deg_4s <= -20,
    "straight_lt10": lambda d: d.turn_deg_4s.abs() < 10,
    "standstill_4s": lambda d: (d.v0 < 0.5) & (d.v_max_4s < 0.5),
    "start_from_stop": lambda d: (d.v0 < 0.5) & (d.v_max_4s > 2.0),
    "stopping": lambda d: (d.v0 > 3.0) & (d.v_min_4s < 0.5),
    "hard_brake_le-2": lambda d: d.max_decel_4s <= -2.0,
    "low_speed_lt3": lambda d: (d.v0 >= 0.5) & (d.v0 < 3.0),
    "high_speed_ge12": lambda d: d.v0 >= 12.0,
    "lead_lt30": lambda d: d.lead_d_rel < 30,
    "lead_closing": lambda d: (d.lead_d_rel < 30) & (d.lead_v_rel < -1.0),
    "lead_braking": lambda d: (d.lead_d_rel < 40) & (d.lead_dv_2s < -1.0),
    "lead_ttc_lt4": lambda d: d.lead_ttc < 4.0,
    "cut_in": lambda d: d.cut_in.astype(bool),
    "vru_ahead_20m": lambda d: d.vru_ahead_20m.astype(bool),
    "dense_ge15": lambda d: d.n_agents_30m >= 15,
    "red_light": lambda d: d.n_red_lights >= 1,
}
SUB = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
       "comfort", "driving_direction_compliance"]
SHORT = {"score": "PDMS", "no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP",
         "time_to_collision_within_bound": "TTC", "comfort": "C", "driving_direction_compliance": "DDC"}


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["token"].astype(str) != "average"].copy()
    return df.set_index("token")


def bucket_table(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    total_def = (1.0 - d["score"]).sum()
    for name, fn in BUCKETS.items():
        m = fn(d).fillna(False).astype(bool)
        if m.sum() == 0:
            continue
        s = d[m]
        row = {"bucket": name, "n": int(m.sum()), "share_%": 100.0 * m.mean()}
        for c in SUB:
            if c in s:
                row[SHORT[c]] = 100.0 * s[c].mean()
        row["NC<1_%"] = 100.0 * (s["no_at_fault_collisions"] < 1).mean()
        row["DAC<1_%"] = 100.0 * (s["drivable_area_compliance"] < 1).mean()
        row["TTC<1_%"] = 100.0 * (s["time_to_collision_within_bound"] < 1).mean()
        row["EP<0.5_%"] = 100.0 * (s["ego_progress"] < 0.5).mean()
        row["deficit_share_%"] = 100.0 * (1.0 - s["score"]).sum() / total_def if total_def > 0 else float("nan")
        rows.append(row)
    t = pd.DataFrame(rows)
    t.insert(0, "run", tag)
    return t


def md_table(t: pd.DataFrame, nd: int = 2) -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> None:
    attrs = pd.read_csv(args.attrs).set_index("token")
    a = load_scores(args.scores)
    da = attrs.join(a, how="inner")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [f"# D5 planning-error buckets: `{args.tag_a}` on {len(da)} dev scenes", "",
             "Attributes from the navsim log pickles (`attrs` sub-command); scores from the per-scene PDM-score CSV. "
             "Sub-scores are means x100; failure rates are the share of scenes with the sub-score below 1 (EP below 0.5); "
             "deficit share = this bucket's share of the total (1 - PDMS) mass (buckets overlap).", ""]
    ta = bucket_table(da, args.tag_a)
    parts += [f"## {args.tag_a}", "", md_table(ta), ""]
    result = {"a": {"tag": args.tag_a, "scores": args.scores, "n": int(len(da)), "table": ta.to_dict(orient="records")}}
    if args.scores_b:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eval.pdms_bootstrap import paired_ci  # noqa: E402

        b = load_scores(args.scores_b)
        db = attrs.join(b, how="inner")
        common = da.index.intersection(db.index)
        da2, db2 = da.loc[common], db.loc[common]
        tb = bucket_table(db2, args.tag_b)
        parts += [f"## {args.tag_b}", "", md_table(tb), ""]
        rng = np.random.default_rng(args.seed)
        rows = []
        for name, fn in BUCKETS.items():
            m = fn(da2).fillna(False).astype(bool).values
            if m.sum() < 20:
                continue
            d = (db2["score"].values[m] - da2["score"].values[m]) * 100.0
            groups = da2["log_name"].values[m]
            lo, hi = paired_ci(d, groups, args.n_boot, rng)[:2]
            rows.append({"bucket": name, "n": int(m.sum()), "delta_PDMS": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                         "delta_EP": 100.0 * float((db2["ego_progress"].values[m] - da2["ego_progress"].values[m]).mean()),
                         "delta_DAC": 100.0 * float((db2["drivable_area_compliance"].values[m] - da2["drivable_area_compliance"].values[m]).mean()),
                         "delta_NC": 100.0 * float((db2["no_at_fault_collisions"].values[m] - da2["no_at_fault_collisions"].values[m]).mean()),
                         "delta_TTC": 100.0 * float((db2["time_to_collision_within_bound"].values[m] - da2["time_to_collision_within_bound"].values[m]).mean())})
        td = pd.DataFrame(rows)
        parts += [f"## Paired difference {args.tag_b} - {args.tag_a} per bucket (log-clustered bootstrap, {args.n_boot} draws)", "",
                  md_table(td), ""]
        result["b"] = {"tag": args.tag_b, "scores": args.scores_b, "n": int(len(db2)), "table": tb.to_dict(orient="records")}
        result["diff"] = td.to_dict(orient="records")
    (out / "report.md").write_text("\n".join(parts))
    json.dump(result, open(out / "report.json", "w"), indent=1)
    print("\n".join(parts[:6] + [md_table(ta)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("attrs")
    p.add_argument("--split", required=True)
    p.add_argument("--log-dir", default="trainval", help="navsim_logs/<dir>")
    p.add_argument("--out", required=True)
    p.add_argument("--limit-logs", type=int, default=0)
    p.set_defaults(fn=cmd_attrs)
    p = sp.add_parser("report")
    p.add_argument("--attrs", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--scores-b", default=None)
    p.add_argument("--tag-a", default="A")
    p.add_argument("--tag-b", default="B")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
