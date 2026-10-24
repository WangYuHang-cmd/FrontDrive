#!/usr/bin/env python3
"""Showcase renderings for the README (plan v4.3 ⑤): where the FrontDrive models do better than the released Drive-JEPA checkpoint
on navtest, drawn from data that is already on disk — per-scene PDMS CSVs, the navtest metric cache (map, agents, human trajectory),
the navsim log pickles (camera calibration, front image path) and the submission pickles (our / the released model's trajectories).

  summary  --out assets/            grouped sub-score bars (released vs 3 seeds) + histogram of per-scene PDMS differences
  scenes   --out assets/ --tokens … GIF per scene: front camera at t with the projected trajectories (ours green, released red,
                                    human white) + BEV animation over the 4 s horizon (drivable area, agents from the PDM
                                    observation, ego footprints along ours / released, score box)
Everything is in the ego frame at t (x forward, y left); trajectories in the pickles are already relative to it.
"""
import argparse
import glob
import lzma
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

B = "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim"
EVAL = f"{B}/exp/eval_pb"
MC = f"{B}/exp/Drive-JEPA-cache/metric_cache"
LOGS = f"{B}/dataset/navsim_logs/test"
BLOBS = f"{B}/dataset/test_sensor_blobs/test"
OFFICIAL_TAG = "pb_official_v1_navtest_sidecalib"
SEED_TAGS = [f"pb_off_ft_all_s{s}_soup1_5_navtest" for s in (0, 1, 2)]
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
       "driving_direction_compliance"]
SUB_SHORT = ["NC", "DAC", "EP", "TTC", "C", "DDC"]
GREEN, RED, WHITE, BLUE = (60, 200, 90), (230, 70, 60), (245, 245, 245), (70, 130, 230)
VEH_L, VEH_W, VEH_RA2C = 5.176, 2.297, 1.461   # nuPlan Pacifica (metric cache vehicle parameters)


def load_scores(tag: str) -> pd.DataFrame:
    df = pd.read_csv(sorted(glob.glob(f"{EVAL}/{tag}/*.csv"))[-1], index_col=0)
    return df[df["token"].astype(str) != "average"].set_index("token")


# ----------------------------------------------------------------------------------------------------------------- summary
def cmd_summary(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    off = load_scores(OFFICIAL_TAG); seeds = [load_scores(t) for t in SEED_TAGS]
    idx = off.index
    for s in seeds:
        idx = idx.intersection(s.index)
    # 1) grouped bars: PDMS + sub-scores
    cols = ["score"] + SUB; names = ["PDMS"] + SUB_SHORT
    vals = np.array([[df.loc[idx, c].mean() * 100 for c in cols] for df in [off] + seeds])   # [4, 7]
    fig, axes = plt.subplots(1, 7, figsize=(14, 3.6))
    labels = ["released\nDrive-JEPA", "FrontDrive\nseed 0", "FrontDrive\nseed 1", "FrontDrive\nseed 2"]
    colors = ["#c94a3f", "#3fae5a", "#3fae5a", "#3fae5a"]
    for j, (ax, name) in enumerate(zip(axes, names)):
        v = vals[:, j]; ax.bar(range(4), v, color=colors, alpha=[1.0, 0.95, 0.75, 0.55][0] if False else 1.0)
        for k in range(4):
            ax.text(k, v[k], f"{v[k]:.2f}", ha="center", va="bottom", fontsize=7)
        lo = max(0.0, v.min() - 1.2); ax.set_ylim(lo, min(100.0, v.max() + 0.8)); ax.set_title(name, fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels(["rel.", "s0", "s1", "s2"], fontsize=8); ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("NAVSIM navtest (12,146 scenes): released Drive-JEPA perception-based v1 checkpoint vs the three FrontDrive seeds", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{args.out}/summary_subscores.png", dpi=150); plt.close(fig)
    # 2) histogram of per-scene PDMS difference (mean of the three seeds − released)
    mean_ours = sum(s.loc[idx, "score"] for s in seeds) / 3
    d = (mean_ours - off.loc[idx, "score"]).values * 100
    better, worse, same = (d > 1e-6).sum(), (d < -1e-6).sum(), (np.abs(d) <= 1e-6).sum()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    nz = d[np.abs(d) > 1e-6]
    ax.hist(nz, bins=np.arange(-100, 101, 5), color="#5b8def", alpha=0.9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("per-scene PDMS difference, mean of the 3 FrontDrive seeds − released checkpoint (points)")
    ax.set_ylabel("scenes (log)")
    ax.set_title(f"{better:,} scenes better, {worse:,} worse, {same:,} identical (of {len(d):,}); mean difference {d.mean():+.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{args.out}/summary_scene_gains.png", dpi=150); plt.close(fig)
    print(f"summary: better {better} worse {worse} same {same}; mean {d.mean():+.3f}; wrote {args.out}/summary_subscores.png, summary_scene_gains.png")


# ------------------------------------------------------------------------------------------------------------------ scenes
def find_cache(token: str) -> str:
    hits = glob.glob(f"{MC}/*/unknown/{token}/metric_cache.pkl")
    if not hits:
        raise FileNotFoundError(f"no metric cache for {token}")
    return hits[0]


_LOG_CACHE: Dict[str, list] = {}


def load_log(log_name: str) -> list:
    if log_name not in _LOG_CACHE:
        _LOG_CACHE[log_name] = pickle.load(open(f"{LOGS}/{log_name}.pkl", "rb"))
    return _LOG_CACHE[log_name]


def to_ego(xy: np.ndarray, origin: Tuple[float, float, float]) -> np.ndarray:
    """global xy [N, 2] -> ego frame at (x0, y0, heading)."""
    x0, y0, th = origin
    c, s = np.cos(-th), np.sin(-th)
    d = xy - np.array([x0, y0])
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def footprint(x: float, y: float, h: float) -> np.ndarray:
    """vehicle rectangle (ego frame) for a rear-axle pose."""
    cx, cy = x + VEH_RA2C * np.cos(h), y + VEH_RA2C * np.sin(h)
    dx, dy = VEH_L / 2, VEH_W / 2
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return corners @ R.T + np.array([cx, cy])


def project_to_image(pts_ego: np.ndarray, cam: dict) -> List[Tuple[int, int]]:
    """ground points [N, 2] in the ego frame -> pixel coordinates (full-resolution image), points behind the camera dropped."""
    R = np.asarray(cam["sensor2lidar_rotation"], dtype=float); t = np.asarray(cam["sensor2lidar_translation"], dtype=float)
    K = np.asarray(cam["cam_intrinsic"], dtype=float)
    P = np.concatenate([pts_ego, np.zeros((len(pts_ego), 1))], axis=1)       # z = 0 (ground; lidar frame == ego frame in these logs)
    pc = (R.T @ (P - t).T).T                                                  # camera frame: z forward
    out = []
    for p in pc:
        if p[2] < 0.5:
            continue
        u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
        out.append((int(round(u)), int(round(v))))
    return out


def densify(poses: np.ndarray, n: int = 6) -> np.ndarray:
    """poses [T, 3] starting at the ego origin -> interpolated xy [(T)*n+1, 2] incl. the origin."""
    pts = np.concatenate([np.zeros((1, 2)), poses[:, :2]], axis=0)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def camera_panel(fr: dict, trajs: Dict[str, Optional[np.ndarray]], size=(640, 360)) -> np.ndarray:
    cam = fr["cams"]["CAM_F0"]
    img = cv2.imread(f"{BLOBS}/{cam['data_path']}")
    if img is None:
        img = np.full((1080, 1920, 3), 40, np.uint8)
    for name, color in (("human", WHITE), ("released", RED), ("ours", GREEN)):
        tr = trajs.get(name)
        if tr is None:
            continue
        px = project_to_image(densify(tr), cam)
        if len(px) >= 2:
            cv2.polylines(img, [np.array(px, np.int32)], False, color[::-1], 6 if name != "human" else 3, cv2.LINE_AA)
            cv2.circle(img, px[-1], 12, color[::-1], -1, cv2.LINE_AA)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bev_panel(cache, origin, trajs: Dict[str, Optional[np.ndarray]], step: int, scores: Dict[str, dict], seed_label: str,
              size_px: int = 480) -> np.ndarray:
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#1b1f2a")
    dm = cache.drivable_area_map
    for geom, mt in zip(dm._geometries, dm._map_types):
        name = str(mt)
        if geom.geom_type != "Polygon":
            continue
        xy = to_ego(np.asarray(geom.exterior.coords), origin)
        if "INTERSECTION" in name:
            ax.add_patch(MplPolygon(np.stack([-xy[:, 1], xy[:, 0]], 1), closed=True, fc="#3a4152", ec="none", zorder=1))
        elif "ROADBLOCK" in name or "CARPARK" in name:
            ax.add_patch(MplPolygon(np.stack([-xy[:, 1], xy[:, 0]], 1), closed=True, fc="#2f3542", ec="none", zorder=0))
        elif "LANE" in name:
            ax.add_patch(MplPolygon(np.stack([-xy[:, 1], xy[:, 0]], 1), closed=True, fc="none", ec="#59627a", lw=0.5, zorder=2))
    # agents at this step (10 Hz observation -> 0.5 s steps)
    om = cache.observation[min(step * 5, len(cache.observation._global_to_local_idcs) - 1)]
    for geom, tok in zip(om._geometries, om.tokens):
        if geom.geom_type != "Polygon":
            continue
        xy = to_ego(np.asarray(geom.exterior.coords), origin)
        col = "#d84d4d" if tok == cache.observation.red_light_token else "#9aa3b5"
        ax.add_patch(MplPolygon(np.stack([-xy[:, 1], xy[:, 0]], 1), closed=True, fc=col, ec="#e6e9f0", lw=0.4, alpha=0.9, zorder=3))
    # trajectories (full path faint) + footprint at this step
    for name, color, z in (("human", "#ffffff", 4), ("released", "#e64a3c", 5), ("ours", "#3cc85a", 6)):
        tr = trajs.get(name)
        if tr is None:
            continue
        pts = np.concatenate([np.zeros((1, 2)), tr[:, :2]], axis=0)
        ax.plot(-pts[:, 1], pts[:, 0], color=color, lw=1.6 if name != "human" else 1.0, ls="-" if name != "human" else "--", alpha=0.9, zorder=z)
        if name != "human":
            k = min(step, len(tr)) ; pose = (0.0, 0.0, 0.0) if k == 0 else (tr[k - 1, 0], tr[k - 1, 1], tr[k - 1, 2])
            fp = footprint(*pose)
            ax.add_patch(MplPolygon(np.stack([-fp[:, 1], fp[:, 0]], 1), closed=True, fc=color, ec="k", lw=0.6, alpha=0.85, zorder=z + 2))
    ax.set_xlim(-30, 30); ax.set_ylim(-12, 48); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    def fmt(sc):
        return f"PDMS {sc['score']*100:5.1f}  " + " ".join(f"{k}{sc[c]*100:.0f}" for k, c in zip(SUB_SHORT, SUB))
    lines = [f"t = {step * 0.5:.1f} s"]
    if "released" in scores:
        lines.append("released:   " + fmt(scores["released"]))
    if "ours" in scores:
        lines.append(f"{seed_label}: " + fmt(scores["ours"]))
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7.5, family="monospace", color="w",
            bbox=dict(boxstyle="round,pad=0.35", fc="#000000", ec="none", alpha=0.55), zorder=10)
    ax.text(0.98, 0.02, "green: FrontDrive   red: released   white: human", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7, color="#dfe3ea", zorder=10)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def human_traj_ego(cache, origin) -> np.ndarray:
    st = cache.trajectory.get_sampled_trajectory()      # 0.1 s states incl. t = 0
    poses = []
    for k in range(5, 45, 5):
        if k >= len(st):
            break
        s = st[k].rear_axle
        xy = to_ego(np.array([[s.x, s.y]]), origin)[0]
        poses.append([xy[0], xy[1], s.heading - origin[2]])
    return np.array(poses)


def cmd_scenes(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    ours_pk = pickle.load(open(args.ours, "rb"))["predictions"][0] if args.ours else None
    rel_pk = pickle.load(open(args.released, "rb"))["predictions"][0] if args.released else None
    off = load_scores(OFFICIAL_TAG); ours_sc = load_scores(SEED_TAGS[args.seed])
    for token in args.tokens:
        cache = pickle.loads(lzma.decompress(open(find_cache(token), "rb").read()))
        log_name = find_cache(token).split("/metric_cache/")[1].split("/")[0]
        frames = load_log(log_name); fr = [f for f in frames if f["token"] == token][0]
        ra = cache.ego_state.rear_axle; origin = (ra.x, ra.y, ra.heading)
        trajs = {"human": human_traj_ego(cache, origin),
                 "ours": np.asarray(ours_pk[token].poses) if ours_pk and token in ours_pk else None,
                 "released": np.asarray(rel_pk[token].poses) if rel_pk and token in rel_pk else None}
        scores = {"released": off.loc[token].to_dict(), "ours": ours_sc.loc[token].to_dict()}
        cam = camera_panel(fr, trajs)
        frames_out = []
        for step in range(0, 9):
            bev = bev_panel(cache, origin, trajs, step, scores, f"FrontDrive s{args.seed}")
            H = max(cam.shape[0], bev.shape[0]); canvas = np.full((H, cam.shape[1] + bev.shape[1], 3), 27, np.uint8)
            canvas[(H - cam.shape[0]) // 2:(H - cam.shape[0]) // 2 + cam.shape[0], :cam.shape[1]] = cam
            canvas[:bev.shape[0], cam.shape[1]:] = bev
            frames_out.append(Image.fromarray(canvas))
        gif = f"{args.out}/showcase_{token}.gif"
        frames_out[0].save(gif, save_all=True, append_images=frames_out[1:], duration=[600] * (len(frames_out) - 1) + [1800], loop=0, optimize=True)
        frames_out[-1].save(f"{args.out}/showcase_{token}.png")
        print(f"{token}: log {log_name}, released PDMS {scores['released']['score']*100:.1f}, ours {scores['ours']['score']*100:.1f} -> {gif} ({os.path.getsize(gif)//1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summary"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("scenes"); p.add_argument("--out", required=True); p.add_argument("--tokens", nargs="+", required=True)
    p.add_argument("--ours", default=None, help="submission.pkl of the FrontDrive seed to draw"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--released", default=None, help="submission.pkl of the released checkpoint (overlay)"); p.set_defaults(fn=cmd_scenes)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
