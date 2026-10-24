"""Derive the Drive-JEPA perception-based (pb) training cache from the perception-free compact JPEG cache + log pickles.

Why: the upstream pb cache (`run_dataset_caching.py` with `DriveJEPAFeatureBuilder`) needs the CAM_F0 frames [2,3] AND the
CAM_L0/R0/B0 frame-3 JPEGs of every navtrain token (~300 GB of float tensors; the side views only fill lidar2img rows the
network never reads, see docs/evidence/2026-10-17_pb_recipe). The pf compact cache (navsim_ext/compact_cache.py) already
holds the identical CAM_F0 crops (JPEG q95, 512x256, `[28:-28]` crop + cv2.resize) for all 103,288 navtrain tokens, and
everything else the pb builder emits is a function of the log pickle:

  img_shape  (4,3)   float32  padded image shape per camera row [b0,f0,l0,r0]: scale 0.4 then pad to /32 -> (448,768,3)
  lidar2img  (4,4,4) float32  scale(0.4) @ K4 @ lidar2cam^T per camera row from sensor2lidar_{rotation,translation}, cam_intrinsic
  ego_status (4,11)  float32  [pose(3) relative to the current frame, velocity(2), acceleration(2), driving_command(4)] x 4 frames
  trajectory (8,3)   float64  future poses (== the pf `trajectory_target`), plus the token (str)

Two derivation engines give bit-identical output (asserted on the first token of every log when both are importable):
  builder  the patched vendored code itself: devkit `AgentInput.from_scene_dict_list` with calibration-only cameras
           (`SensorConfig.keep_calibration`), `bev_feature_build.LoadMultiViewImageFromFiles(side_cams_from_calibration=True)`,
           the vendored `RandomScaleImageMultiViewImage` / `PadMultiViewImage` transforms and
           `drive_jepa_features._ego_status_feature` -> identical to the training-time feature builder by construction
           (needs mmcv, i.e. venv-pb);
  numpy    a torch/mmcv-free mirror of the same arithmetic (any venv; what venv-pf uses).

Layout = upstream (`<cache>/<log>/<token>/drive_jepa_feature.gz` + `drive_jepa_target.gz`, gzip-1 pickles) plus the
`_COMPACT_JPEG` marker directory, so the patched run_training.py uses `CompactCacheOnlyDataset` (decodes the JPEG bytes).
Arrays are stored as numpy (no torch objects -> readable by venv-pf torch 2.8 and venv-pb torch 2.1 alike); the
DataLoader's default_collate turns them into tensors of the upstream dtypes.

CLI (run from the frontdrive root, any venv with torch + navsim; ~103k tokens, no image I/O, ~12 GB, minutes):
    python -m navsim_ext.compact_cache_pb build --split navtrain_val_official --compact <pf cache> --logs <navsim_logs/trainval> \
        --out <pb cache> --workers 8 [--engine auto|builder|numpy]      (then --split navtrain_train_official; resumable)
    python -m navsim_ext.compact_cache_pb verify --compact ... --logs ... --out ... --n 50 [--mode auto|builder|pf] [--blobs <sensor_blobs/trainval>]
        builder: decodes N random tokens of logs with local CAM_F0 frames through CompactCacheOnlyDataset and compares them with
                 the patched upstream DriveJEPAFeatureBuilder/DriveJEPATargetBuilder (exact img_shape/lidar2img/ego_status, trajectory exact or within TRAJ_ATOL,
                 PSNR >= 35 dB camera features); pf: pf-cache bytes / independent projection / pickle fields (no blobs needed)
    python -m navsim_ext.compact_cache_pb coverage --out ... --metric-cache $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache [--check-anchors] [--move-aside DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "/home/henry/Desktop/Qwen_Drive/frontdrive/third_party/Drive-JEPA/navsim_v1"))
if str(DEVKIT) not in sys.path:
    sys.path.insert(0, str(DEVKIT))
SPLIT_DIR = DEVKIT / "navsim/planning/script/config/common/train_test_split/scene_filter"

MARKER = "_COMPACT_JPEG"  # empty directory marker read by the patched run_training.py (== navsim_ext.compact_cache.MARKER)
PF_FEATURE = "drive_jepa_feature"  # perception-free DriveJEPAFeatureDIBuilder.get_unique_name()
PF_TARGET = "trajectory_target"  # perception-free TrajectoryTargetBuilder.get_unique_name()
PB_FEATURE = "drive_jepa_feature"  # perception-based DriveJEPAFeatureBuilder.get_unique_name() (ViT / front-camera path)
PB_TARGET = "drive_jepa_target"  # perception-based DriveJEPATargetBuilder.get_unique_name()
JPEG_KEYS = ("camera_feature_1", "camera_feature_2")  # frame 3 (current), frame 2 (t-0.5 s); same in pf and pb builders
CAM_ORDER = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")  # row order of bevformer/bev_feature_build.LoadMultiViewImageFromFiles
F0_ROW = 1  # drive_jepa_model.py: features['lidar2img'][:, 1:2]
NUM_HISTORY_FRAMES = 4
SCALE = 0.4  # RandomScaleImageMultiViewImage(scales=[0.4])
PAD_DIVISOR = 32  # PadMultiViewImage(size_divisor=32)
SENSOR_HW = (1080, 1920)  # every nuPlan/OpenScene camera: 1920x1080 (checked on local CAM_F0 JPEGs; all 8 cams share the intrinsics)
FEATURE_KEYS = ("img_shape", "lidar2img", "camera_feature_1", "camera_feature_2", "ego_status")
META_KEYS = ("img_shape", "lidar2img", "ego_status")
TARGET_KEYS = ("trajectory", "token")
TRAJ_ATOL = 1e-9  # verify: trajectory tolerance vs a scene target recomputed on another CPU (last-ulp sin/cos differences); exact match still reported
ENGINES = ("auto", "builder", "numpy")


# ----------------------------------------------------------------------------------------------------------------------
# helpers
def gz_load(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def gz_dump(path: Path, obj) -> None:
    """gzip level 1 like navsim.planning.training.dataset.dump_feature_target_to_pickle, written atomically."""
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def split_members(split: Optional[str]) -> Tuple[Optional[List[str]], Optional[Set[str]]]:
    """(sorted log names, token set) of a train_test_split scene filter (`scene_filter/<split>.yaml`, e.g. navtrain,
    navtrain_train_official, navtrain_val_official, navtrain_fit, navtrain_dev); 'all' / None -> (None, None) = no restriction.
    Parsed with PyYAML (no omegaconf: the 100k-token yamls exceed omegaconf>=2.4's node limit)."""
    if not split or split == "all":
        return None, None
    path = SPLIT_DIR / f"{split}.yaml"
    assert path.exists(), f"unknown split {split!r}: no {path}"
    import yaml

    d = yaml.load(path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    logs = sorted(d["log_names"]) if d.get("log_names") else None
    tokens = set(d["tokens"]) if d.get("tokens") else None
    return logs, tokens


def list_cache_logs(cache: Path) -> List[str]:
    return sorted(d.name for d in Path(cache).iterdir() if d.is_dir() and d.name != MARKER)


def _history(frames: List[dict], index: Dict[str, int], token: str) -> List[dict]:
    pos = index[token]
    assert pos >= NUM_HISTORY_FRAMES - 1, f"{token}: frame index {pos} has no {NUM_HISTORY_FRAMES - 1} history frames"
    history = frames[pos - NUM_HISTORY_FRAMES + 1 : pos + 1]
    assert history[-1]["token"] == token
    return history


# ----------------------------------------------------------------------------------------------------------------------
# engine "numpy": torch/mmcv-free mirror of the upstream feature-builder arithmetic
def padded_img_shape(hw: Tuple[int, int] = SENSOR_HW, scale: float = SCALE, divisor: int = PAD_DIVISOR) -> Tuple[int, int, int]:
    """One `img_shape` row: mmcv.imresize to (int(w*scale), int(h*scale)), then impad_to_multiple(divisor) -> (448, 768, 3)."""
    h, w = int(hw[0] * scale), int(hw[1] * scale)
    return (int(np.ceil(h / divisor)) * divisor, int(np.ceil(w / divisor)) * divisor, 3)


def lidar2img_from_calibration(rotation: np.ndarray, translation: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unscaled 4x4 lidar2img, the operations of bev_feature_build.LoadMultiViewImageFromFiles in the same order."""
    lidar2cam_r = np.linalg.inv(rotation)
    lidar2cam_t = translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def scaled_lidar2img(lidar2img: np.ndarray, scale: float = SCALE) -> np.ndarray:
    """RandomScaleImageMultiViewImage: scale_factor @ lidar2img with scale_factor = diag(s, s, 1, 1)."""
    scale_factor = np.eye(4)
    scale_factor[0, 0] *= scale
    scale_factor[1, 1] *= scale
    return scale_factor @ lidar2img


def bev_meta_from_frame(frame: dict) -> Tuple[np.ndarray, np.ndarray]:
    """`img_shape` (4,3) float32 and `lidar2img` (4,4,4) float32 of one frame's cameras in the upstream row order."""
    rows = []
    for name in CAM_ORDER:
        cam = frame["cams"][name]
        raw = lidar2img_from_calibration(np.asarray(cam["sensor2lidar_rotation"]), np.asarray(cam["sensor2lidar_translation"]),
                                         np.asarray(cam["cam_intrinsic"]))
        rows.append(scaled_lidar2img(raw))
    lidar2img = np.stack(rows).astype(np.float32)  # torch.FloatTensor(np.stack(...)) upstream: same float64->float32 rounding
    img_shape = np.asarray([padded_img_shape()] * len(rows), dtype=np.float32)
    return img_shape, lidar2img


def ego_status_from_frames(history_frames: Sequence[dict]) -> np.ndarray:
    """(4,11) float32 as DriveJEPAFeatureBuilder.compute_features builds it from AgentInput.ego_statuses (devkit code)."""
    from navsim.common.dataclasses import AgentInput, SensorConfig  # heavy import (nuplan), kept local for the workers

    agent_input = AgentInput.from_scene_dict_list(list(history_frames), Path("/nonexistent"), num_history_frames=len(history_frames),
                                                  sensor_config=SensorConfig.build_no_sensors())
    rows = [np.concatenate([np.asarray(e.ego_pose, dtype=np.float32), np.asarray(e.ego_velocity, dtype=np.float32),
                            np.asarray(e.ego_acceleration, dtype=np.float32), np.asarray(e.driving_command, dtype=np.float32)])
            for e in agent_input.ego_statuses]
    return np.stack(rows).astype(np.float32)


def bev_meta_numpy(history: Sequence[dict]) -> Dict[str, np.ndarray]:
    img_shape, lidar2img = bev_meta_from_frame(history[-1])
    return {"img_shape": img_shape, "lidar2img": lidar2img, "ego_status": ego_status_from_frames(history)}


# ----------------------------------------------------------------------------------------------------------------------
# engine "builder": the patched vendored feature-builder code on calibration-only cameras (needs mmcv -> venv-pb)
def builder_available() -> bool:
    try:
        import navsim.agents.drive_jepa_perception_based.bevformer.bev_feature_build  # noqa: F401  (imports mmcv)
        import navsim.agents.drive_jepa_perception_based.drive_jepa_features  # noqa: F401
        return True
    except Exception:
        return False


def resolve_engine(engine: str) -> str:
    assert engine in ENGINES, engine
    if engine == "auto":
        return "builder" if builder_available() else "numpy"
    if engine == "builder" and not builder_available():
        raise RuntimeError("engine=builder needs the vendored pb feature builder (mmcv); use venv-pb or --engine numpy")
    return engine


def calibration_only_agent_input(history: Sequence[dict]):
    """Devkit AgentInput of the history frames with every camera calibration-only (no image I/O): the flagged pb agent's
    SensorConfig.keep_calibration path with no sensor requested."""
    from navsim.common.dataclasses import AgentInput, SensorConfig

    sensor_config = SensorConfig.build_no_sensors()
    sensor_config.keep_calibration = True
    return AgentInput.from_scene_dict_list(list(history), Path("/nonexistent"), num_history_frames=len(history), sensor_config=sensor_config)


_PADDED_SHAPE_BUILDER: Dict[Tuple[int, int], Tuple[int, int, int]] = {}


def padded_img_shape_builder(hw: Tuple[int, int] = SENSOR_HW) -> Tuple[int, int, int]:
    """The vendored scale + pad transforms applied once to a blank image of the sensor size (cached per process)."""
    if hw not in _PADDED_SHAPE_BUILDER:
        from navsim.agents.drive_jepa_perception_based.bevformer import bev_feature_build as bfb

        res = {"img": [np.zeros((hw[0], hw[1], 3), dtype=np.float32)], "lidar2img": []}
        res = bfb.RandomScaleImageMultiViewImage(res)
        res = bfb.PadMultiViewImage(res)
        _PADDED_SHAPE_BUILDER[hw] = tuple(int(x) for x in res["img_shape"][0])
    return _PADDED_SHAPE_BUILDER[hw]


def bev_meta_builder(history: Sequence[dict]) -> Dict[str, np.ndarray]:
    """img_shape / lidar2img / ego_status through the vendored (patched) builder code: LoadMultiViewImageFromFiles with
    side_cams_from_calibration=True on calibration-only cameras (4 rows [b0,f0,l0,r0], no image), the vendored
    RandomScaleImageMultiViewImage on those rows, torch.FloatTensor like _get_bev_feature, _ego_status_feature."""
    import torch
    from navsim.agents.drive_jepa_perception_based.bevformer import bev_feature_build as bfb
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import _ego_status_feature

    agent_input = calibration_only_agent_input(history)
    res = bfb.LoadMultiViewImageFromFiles(agent_input, side_cams_from_calibration=True)
    assert len(res["lidar2img"]) == len(CAM_ORDER) and not res["img"], (len(res["lidar2img"]), len(res["img"]))
    res = bfb.NormalizeMultiviewImage(res)  # image-only transforms: no-ops on the empty image list
    res = bfb.RandomScaleImageMultiViewImage(res)  # scales every lidar2img row exactly like the builder
    res = bfb.PadMultiViewImage(res)
    shape = padded_img_shape_builder()  # _get_bev_feature replicates the (single) padded F0 shape to all rows
    img_shape = torch.FloatTensor(np.stack([shape] * len(res["lidar2img"]))).numpy()
    lidar2img = torch.FloatTensor(np.stack(res["lidar2img"])).numpy()
    return {"img_shape": img_shape, "lidar2img": lidar2img, "ego_status": _ego_status_feature(agent_input).numpy()}


def bev_meta(history: Sequence[dict], engine: str) -> Dict[str, np.ndarray]:
    return bev_meta_builder(history) if engine == "builder" else bev_meta_numpy(history)


def engines_agree(history: Sequence[dict]) -> Dict[str, bool]:
    """Bitwise agreement of the two engines per key (both must be importable)."""
    a, b = bev_meta_builder(history), bev_meta_numpy(history)
    return {k: bool(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype) for k in META_KEYS}


# ----------------------------------------------------------------------------------------------------------------------
def derive_token(pf_token_dir: Path, token: str, frames: List[dict], index: Dict[str, int], engine: str = "numpy") -> Tuple[dict, dict]:
    """pb feature / target dicts of one token from its pf compact pickles and the log frames."""
    history = _history(frames, index, token)
    pf_feat = gz_load(pf_token_dir / f"{PF_FEATURE}.gz")
    pf_tgt = gz_load(pf_token_dir / f"{PF_TARGET}.gz")
    meta = bev_meta(history, engine)
    feats = {"img_shape": meta["img_shape"], "lidar2img": meta["lidar2img"]}
    for k in JPEG_KEYS:
        v = pf_feat[k]
        assert isinstance(v, (bytes, bytearray)), f"{token}: pf key {k} is {type(v).__name__}, expected JPEG bytes"
        feats[k] = bytes(v)
    feats["ego_status"] = meta["ego_status"]
    traj = pf_tgt["trajectory"]
    traj = traj.numpy() if hasattr(traj, "numpy") else np.asarray(traj)
    assert traj.shape == (8, 3), traj.shape
    tgts = {"trajectory": np.ascontiguousarray(traj), "token": token}
    return feats, tgts


# ----------------------------------------------------------------------------------------------------------------------
# build
def _build_log(args) -> Tuple[str, int, int, int, int, str]:
    """Worker: all (selected) tokens of ONE log. Returns (log, n_written, n_skipped, n_failed, n_not_in_split, first errors)."""
    compact, logs_dir, out, log, force, tokens, engine, selfcheck = args
    pkl = Path(logs_dir) / f"{log}.pkl"
    if not pkl.exists():
        return log, 0, 0, 0, 0, "missing log pickle"
    with open(pkl, "rb") as f:
        frames = pickle.load(f)
    index = {fr["token"]: i for i, fr in enumerate(frames)}
    n_ok = n_skip = n_fail = n_out = 0
    errors: List[str] = []
    checked = False
    for tdir in sorted(Path(compact, log).iterdir()):
        if not tdir.is_dir():
            continue
        token = tdir.name
        if tokens is not None and token not in tokens:
            n_out += 1
            continue
        odir = Path(out, log, token)
        if not force and (odir / f"{PB_FEATURE}.gz").exists() and (odir / f"{PB_TARGET}.gz").exists():
            n_skip += 1
            continue
        try:
            if selfcheck and engine == "builder" and not checked:  # both engines on the first derived token of the log
                agree = engines_agree(_history(frames, index, token))
                assert all(agree.values()), f"engine disagreement {agree}"
                checked = True
            feats, tgts = derive_token(tdir, token, frames, index, engine)
            odir.mkdir(parents=True, exist_ok=True)
            gz_dump(odir / f"{PB_FEATURE}.gz", feats)
            gz_dump(odir / f"{PB_TARGET}.gz", tgts)
            n_ok += 1
        except Exception as e:  # keep going; the parent reports counts and the first errors
            n_fail += 1
            errors.append(f"{token}: {type(e).__name__}: {str(e)[:120]}")
    return log, n_ok, n_skip, n_fail, n_out, "; ".join(errors[:3])


def build(compact: str, logs_dir: str, out: str, workers: int = 8, force: bool = False, log_names: Optional[Sequence[str]] = None,
          limit_logs: int = 0, split: Optional[str] = None, engine: str = "auto", selfcheck: bool = True) -> dict:
    compact_p, out_p = Path(compact), Path(out)
    assert (compact_p / MARKER).is_dir(), f"{compact} is not a compact JPEG cache (no {MARKER} marker)"
    engine = resolve_engine(engine)
    split_logs, split_tokens = split_members(split)
    logs = list_cache_logs(compact_p)
    n_cache_logs = len(logs)
    if split_logs is not None:
        keep = set(split_logs)
        logs = [l for l in logs if l in keep]
    if log_names:
        keep = set(log_names)
        logs = [l for l in logs if l in keep]
    if limit_logs:
        logs = logs[:limit_logs]
    # per-log token subsets (parent): tiny per job instead of shipping the 100k-token set to every worker
    tokens_by_log: Dict[str, Optional[frozenset]] = {}
    n_split_tokens_in_cache = 0
    for log in logs:
        if split_tokens is None:
            tokens_by_log[log] = None
        else:
            toks = frozenset(d.name for d in (compact_p / log).iterdir() if d.is_dir() and d.name in split_tokens)
            tokens_by_log[log] = toks
            n_split_tokens_in_cache += len(toks)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / MARKER).mkdir(exist_ok=True)  # must stay EMPTY (CacheOnlyDataset would treat its entries as tokens)
    info = Path(str(out_p).rstrip("/") + "_info.txt")
    with open(info, "a") as f:
        f.write("compact jpeg cache, perception-based layout; derived %s by navsim_ext/compact_cache_pb.py (engine %s, split %s) from %s "
                "(CAM_F0 [2,3] 512x256 JPEG) + log pickles %s: img_shape (4,3), lidar2img (4,4,4) scale %.1f pad /%d from calibration, "
                "ego_status (4,11), drive_jepa_target {trajectory (8,3), token}\n"
                % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), engine, split or "all", compact, logs_dir, SCALE, PAD_DIVISOR))
    print(f"[compact_cache_pb] split={split or 'all'} engine={engine} logs {len(logs)}/{n_cache_logs} "
          f"{'tokens ' + str(n_split_tokens_in_cache) + ' ' if split_tokens is not None else ''}from {compact} -> {out} "
          f"(workers={workers}, force={force}, selfcheck={selfcheck})", flush=True)
    t0 = time.time()
    tot = {"logs": 0, "written": 0, "skipped": 0, "failed": 0, "not_in_split": 0, "missing_pickle": 0}
    errors: List[str] = []
    jobs = [(str(compact_p), str(logs_dir), str(out_p), log, force, tokens_by_log[log], engine, selfcheck) for log in logs]

    def _account(res):
        log, n_ok, n_skip, n_fail, n_out, err = res
        tot["logs"] += 1
        tot["written"] += n_ok
        tot["skipped"] += n_skip
        tot["failed"] += n_fail
        tot["not_in_split"] += n_out
        if err == "missing log pickle":
            tot["missing_pickle"] += 1
            errors.append(f"{log}: missing log pickle")
        elif err:
            errors.append(f"{log}: {err}")
        if tot["logs"] % 50 == 0 or tot["logs"] == len(logs):
            el = time.time() - t0
            rate = tot["logs"] / max(el, 1e-6)
            print(f"[compact_cache_pb] logs {tot['logs']}/{len(logs)} written {tot['written']} skipped {tot['skipped']} failed "
                  f"{tot['failed']} | {el / 60:.1f} min, eta {(len(logs) - tot['logs']) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    if workers <= 1:
        for job in jobs:
            _account(_build_log(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:  # spawn: small workers
            for fut in as_completed([ex.submit(_build_log, job) for job in jobs]):
                _account(fut.result())
    summary = {**tot, "n_logs_requested": len(logs), "n_cache_logs": n_cache_logs, "split": split or "all", "engine": engine,
               "n_split_tokens_in_cache": n_split_tokens_in_cache if split_tokens is not None else None, "selfcheck": selfcheck,
               "elapsed_s": round(time.time() - t0, 1), "compact": str(compact_p), "logs_dir": str(logs_dir), "out": str(out_p),
               "errors_head": errors[:20]}
    with open(str(out_p).rstrip("/") + "_build_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("[compact_cache_pb] " + json.dumps({k: v for k, v in summary.items() if k != "errors_head"}), flush=True)
    for e in errors[:20]:
        print("[compact_cache_pb] error:", e, flush=True)
    return summary


def count_tokens(out: str) -> Tuple[int, int]:
    """(token dirs with both pickles, token dirs) under a derived cache."""
    out_p = Path(out)
    n_dirs = n_complete = 0
    for log in list_cache_logs(out_p):
        for d in (out_p / log).iterdir():
            if d.is_dir():
                n_dirs += 1
                if (d / f"{PB_FEATURE}.gz").exists() and (d / f"{PB_TARGET}.gz").exists():
                    n_complete += 1
    return n_complete, n_dirs


# ----------------------------------------------------------------------------------------------------------------------
# verify, mode "pf": pf-cache bytes, independent projection, pickle fields (no blobs / mmcv needed)
def independent_lidar2img(cam: dict, scale: float = SCALE) -> np.ndarray:
    """A second formulation for cross-checking: P = S @ K4 @ [R^T | -R^T t] with S = diag(s, s, 1, 1)."""
    R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
    K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    K4 = np.eye(4)
    K4[:3, :3] = K
    S = np.diag([scale, scale, 1.0, 1.0])
    return S @ K4 @ T


def check_token(out_dir: Path, pf_dir: Path, token: str, frames: List[dict], index: Dict[str, int]) -> List[str]:
    """Returns a list of problems (empty = ok) for one derived token."""
    import cv2

    problems: List[str] = []
    feats = gz_load(out_dir / f"{PB_FEATURE}.gz")
    tgts = gz_load(out_dir / f"{PB_TARGET}.gz")
    if set(feats) != set(FEATURE_KEYS):
        problems.append(f"feature keys {sorted(feats)}")
    if set(tgts) != set(TARGET_KEYS):
        problems.append(f"target keys {sorted(tgts)}")
    for k, shape in (("img_shape", (4, 3)), ("lidar2img", (4, 4, 4)), ("ego_status", (4, 11))):
        v = np.asarray(feats.get(k))
        if v.shape != shape or v.dtype != np.float32:
            problems.append(f"{k} {v.shape} {v.dtype}")
    traj = np.asarray(tgts.get("trajectory"))
    if traj.shape != (8, 3) or traj.dtype != np.float64:
        problems.append(f"trajectory {traj.shape} {traj.dtype}")
    if tgts.get("token") != token:
        problems.append(f"token {tgts.get('token')!r}")
    pf_feat = gz_load(pf_dir / f"{PF_FEATURE}.gz")
    pf_tgt = gz_load(pf_dir / f"{PF_TARGET}.gz")
    for k in JPEG_KEYS:
        if bytes(feats.get(k, b"")) != bytes(pf_feat[k]):
            problems.append(f"{k} bytes differ from the pf cache")
        img = cv2.imdecode(np.frombuffer(feats[k], np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape != (256, 512, 3):
            problems.append(f"{k} does not decode to 256x512x3")
    pf_traj = pf_tgt["trajectory"]
    pf_traj = pf_traj.numpy() if hasattr(pf_traj, "numpy") else np.asarray(pf_traj)
    if not np.array_equal(traj, pf_traj):
        problems.append("trajectory differs from the pf target")
    cur = frames[index[token]]
    if not np.array_equal(feats["img_shape"], np.asarray([padded_img_shape()] * 4, np.float32)):
        problems.append(f"img_shape rows {feats['img_shape'].tolist()}")
    for row, name in enumerate(CAM_ORDER):
        ref = independent_lidar2img(cur["cams"][name])
        if not np.allclose(feats["lidar2img"][row], ref, rtol=1e-5, atol=1e-2):  # float32 storage of values up to ~1e3
            problems.append(f"lidar2img row {row} ({name}) deviates from the independent projection by {np.abs(feats['lidar2img'][row] - ref).max():.3g}")
    if not np.array_equal(feats["lidar2img"], bev_meta_from_frame(cur)[1]):
        problems.append("lidar2img differs from the numpy engine")
    es = feats["ego_status"]
    if not np.array_equal(es[-1, :3], np.zeros(3, np.float32)):
        problems.append(f"current-frame ego pose not at the origin: {es[-1, :3]}")
    for i, fr in enumerate(frames[index[token] - NUM_HISTORY_FRAMES + 1 : index[token] + 1]):
        dyn = np.asarray(fr["ego_dynamic_state"], np.float32)
        if not (np.array_equal(es[i, 3:5], dyn[:2]) and np.array_equal(es[i, 5:7], dyn[2:4])):
            problems.append(f"ego velocity/acceleration row {i} differ from ego_dynamic_state")
        if not np.array_equal(es[i, 7:], np.asarray(fr["driving_command"], np.float32)):
            problems.append(f"driving_command row {i} differs")
    return problems


def _verify_pf(compact_p: Path, logs_dir: str, out_p: Path, n: int, seed: int, log_names: Optional[Sequence[str]]) -> dict:
    import random

    out_logs = list_cache_logs(out_p)
    if log_names:
        out_logs = [l for l in out_logs if l in set(log_names)]
    pairs = [(log, d.name) for log in out_logs for d in sorted((out_p / log).iterdir()) if d.is_dir()]
    n_total = len(pairs)
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n] if n else pairs
    pairs.sort()
    by_log: Dict[str, List[str]] = {}
    for log, tok in pairs:
        by_log.setdefault(log, []).append(tok)
    n_ok = n_fail = 0
    fails: List[str] = []
    for log, toks in by_log.items():
        with open(Path(logs_dir) / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        index = {fr["token"]: i for i, fr in enumerate(frames)}
        for tok in toks:
            probs = check_token(out_p / log / tok, compact_p / log / tok, tok, frames, index)
            if probs:
                n_fail += 1
                fails.append(f"{log}/{tok}: " + " | ".join(probs))
            else:
                n_ok += 1
    return {"mode": "pf", "tokens_in_out": n_total, "checked": n_ok + n_fail, "ok": n_ok, "fail": n_fail, "logs_sampled": len(by_log),
            "fails_head": fails[:10]}


# ----------------------------------------------------------------------------------------------------------------------
# verify, mode "builder": CompactCacheOnlyDataset output vs the patched upstream builders on logs with local CAM_F0 frames
def _flagged_sensor_config(cfg):
    """The pb agent's SensorConfig with side_cams_from_calibration (via DriveJEPAAgent.get_sensor_config when importable)."""
    import types

    try:
        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent

        return DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=cfg))  # reads only self._config
    except Exception:  # same object as the agent builds (tests/test_compact_cache_pb.py::test_agent_sensor_config_flag)
        from navsim.common.dataclasses import SensorConfig

        sc = SensorConfig(cam_f0=[2, 3], cam_l0=[], cam_l1=[], cam_l2=[], cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[], lidar_pc=[])
        sc.keep_calibration = True
        return sc


def _sample_builder_tokens(out_p: Path, logs_dir: Path, blobs: Path, n: int, seed: int, log_names: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    """Up to n random derived tokens whose CAM_F0 frames (p-1, p) exist under `blobs`, spread over ~n/10 logs."""
    import random

    rng = random.Random(seed)
    cand = [l for l in list_cache_logs(out_p) if (blobs / l).is_dir() and (logs_dir / f"{l}.pkl").exists()]
    if log_names:
        cand = [l for l in cand if l in set(log_names)]
    rng.shuffle(cand)
    per_log = max(1, math.ceil(n / max(1, min(10, len(cand)))))  # ~10 logs when available, fill from fewer otherwise
    chosen: Dict[str, List[str]] = {}
    total = 0
    for log in cand:
        if total >= n:
            break
        with open(logs_dir / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        index = {fr["token"]: i for i, fr in enumerate(frames)}
        toks = []
        for d in sorted((out_p / log).iterdir()):
            tok = d.name
            if not d.is_dir() or tok not in index or index[tok] < NUM_HISTORY_FRAMES - 1:
                continue
            p = index[tok]
            if all((blobs / frames[q]["cams"]["CAM_F0"]["data_path"]).exists() for q in (p - 1, p)):
                toks.append(tok)
        if not toks:
            continue
        rng.shuffle(toks)
        take = toks[: min(per_log, n - total)]
        chosen[log] = sorted(take)
        total += len(take)
    return chosen


def _verify_builder(compact_p: Path, logs_dir: str, out_p: Path, blobs: str, n: int, seed: int, log_names: Optional[Sequence[str]],
                    min_psnr: float) -> dict:
    import torch
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder, DriveJEPATargetBuilder
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim_ext.compact_cache import CompactCacheOnlyDataset

    logs_p, blobs_p = Path(logs_dir), Path(blobs)
    cfg = DriveJEPAConfig(side_cams_from_calibration=True)
    sensor_config = _flagged_sensor_config(cfg)
    fb, tb = DriveJEPAFeatureBuilder(cfg), DriveJEPATargetBuilder(cfg)
    chosen = _sample_builder_tokens(out_p, logs_p, blobs_p, n, seed, log_names)
    exact = {k: 0 for k in META_KEYS + ("trajectory",)}
    target_ref = {"scene": 0, "pf": 0}
    psnrs: List[float] = []
    traj_diffs: List[float] = []
    close_traj = 0  # trajectory equal to the reference within TRAJ_ATOL but not bit-identical (see below)
    n_ok = n_fail = 0
    fails: List[str] = []
    scene_error: Optional[str] = None
    for log, toks in chosen.items():
        scene_filter = SceneFilter(num_history_frames=NUM_HISTORY_FRAMES, num_future_frames=10, frame_interval=1, has_route=True,
                                   log_names=[log], tokens=list(toks))
        loader = SceneLoader(data_path=logs_p, sensor_blobs_path=blobs_p, scene_filter=scene_filter, sensor_config=sensor_config)
        ds = CompactCacheOnlyDataset(cache_path=str(out_p), feature_builders=[fb], target_builders=[tb], log_names=[log])
        for tok in toks:
            probs: List[str] = []
            try:
                if tok not in loader.tokens:
                    raise KeyError(f"{tok} not produced by the SceneLoader filter")
                feats, tgts = ds._load_scene_with_token(tok)  # what the training DataLoader sees (JPEGs decoded)
                agent_input = loader.get_agent_input_from_token(tok)  # no side JPEG is opened (flag on)
                assert agent_input.cameras[-1].cam_l0.image is None and agent_input.cameras[-1].cam_l0.intrinsics is not None
                ref = fb.compute_features(agent_input)
                if set(feats) != set(ref) or set(ref) != set(FEATURE_KEYS):
                    probs.append(f"feature keys {sorted(feats)} vs {sorted(ref)}")
                for k in META_KEYS:
                    v = np.asarray(feats[k])
                    if np.array_equal(v, ref[k].numpy()) and v.dtype == np.float32 and ref[k].dtype == torch.float32:
                        exact[k] += 1
                    else:
                        probs.append(f"{k} differs from the builder (max abs {np.abs(v.astype(np.float64) - ref[k].numpy()).max():.3g})")
                for k in JPEG_KEYS:
                    dec = feats[k]
                    if not (isinstance(dec, torch.Tensor) and dec.shape == (3, 256, 512) and dec.dtype == torch.float32):
                        probs.append(f"{k} not a [3,256,512] float tensor after decoding")
                        continue
                    mse = float(((dec - ref[k]) ** 2).mean())
                    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
                    psnrs.append(psnr)
                    if psnr < min_psnr:
                        probs.append(f"{k} PSNR {psnr:.2f} < {min_psnr}")
                try:
                    scene = loader.get_scene_from_token(tok)  # needs the nuPlan maps (NUPLAN_MAPS_ROOT)
                    rt = tb.compute_targets(scene)
                    ref_traj, ref_tok, src = rt["trajectory"].numpy(), rt["token"], "scene"
                except Exception as e:  # no maps -> the pf target (what the derived cache copies)
                    scene_error = scene_error or f"{type(e).__name__}: {str(e)[:100]}"
                    pf_traj = gz_load(compact_p / log / tok / f"{PF_TARGET}.gz")["trajectory"]
                    ref_traj, ref_tok, src = (pf_traj.numpy() if hasattr(pf_traj, "numpy") else np.asarray(pf_traj)), tok, "pf"
                target_ref[src] += 1
                traj = np.asarray(tgts["trajectory"])
                ref_arr = np.asarray(ref_traj)
                same_shape = traj.shape == ref_arr.shape
                diff = float(np.abs(traj.astype(np.float64) - ref_arr.astype(np.float64)).max()) if same_shape else float("inf")
                traj_diffs.append(diff)
                tok_ok = tgts["token"] == tok == ref_tok
                if same_shape and diff == 0.0 and traj.dtype == np.float64 and tok_ok:
                    exact["trajectory"] += 1
                elif same_shape and diff <= TRAJ_ATOL and traj.dtype == np.float64 and tok_ok:
                    # The derived trajectory is the pf target computed on the machine that built the compact cache; the scene target
                    # is recomputed HERE (SE(2) transform with sin/cos) -> last-ulp differences between CPUs / numpy SIMD paths are
                    # possible (H100 host 2026-10-17: 50/50 tokens non-identical while img_shape/lidar2img/ego_status were exact; the
                    # magnitude is reported as trajectory_max_abs_diff so the tolerance can be judged). Physically meaningless at 1e-9.
                    close_traj += 1
                else:
                    probs.append(f"trajectory/token differ from the {src} target (max abs {diff:.3g}, dtype {traj.dtype}, "
                                 f"shape {traj.shape} vs {ref_arr.shape}, token_ok {tok_ok})")
            except Exception as e:
                probs.append(f"{type(e).__name__}: {str(e)[:160]}")
            if probs:
                n_fail += 1
                fails.append(f"{log}/{tok}: " + " | ".join(probs))
            else:
                n_ok += 1
    return {"mode": "builder", "blobs": str(blobs_p), "checked": n_ok + n_fail, "ok": n_ok, "fail": n_fail, "logs_sampled": len(chosen),
            "exact": exact, "trajectory_close_within_atol": close_traj, "trajectory_atol": TRAJ_ATOL,
            "trajectory_max_abs_diff": (max(traj_diffs) if traj_diffs else None),
            "psnr_min": round(min(psnrs), 2) if psnrs else None, "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None,
            "min_psnr_required": min_psnr, "target_ref": target_ref, "scene_error": scene_error, "fails_head": fails[:10]}


def verify(compact: str, logs_dir: str, out: str, n: int = 50, seed: int = 0, log_names: Optional[Sequence[str]] = None, mode: str = "auto",
           blobs: Optional[str] = None, min_psnr: float = 35.0) -> dict:
    """mode builder (default when mmcv and dev CAM_F0 blobs are available): CompactCacheOnlyDataset vs the patched upstream
    builder; mode pf: pf-cache bytes / independent projection / pickle fields."""
    assert mode in ("auto", "builder", "pf"), mode
    compact_p, out_p = Path(compact), Path(out)
    blobs = blobs or os.path.join(os.environ.get("OPENSCENE_DATA_ROOT", "/nonexistent"), "sensor_blobs", "trainval")
    if mode == "auto":
        has_blobs = Path(blobs).is_dir() and any((Path(blobs) / l).is_dir() for l in list_cache_logs(out_p))
        mode = "builder" if (builder_available() and has_blobs) else "pf"
    if mode == "builder":
        rep = _verify_builder(compact_p, logs_dir, out_p, blobs, n, seed, log_names, min_psnr)
    else:
        rep = _verify_pf(compact_p, logs_dir, out_p, n, seed, log_names)
    rep.update({"out": str(out_p), "marker": (out_p / MARKER).is_dir() and not any((out_p / MARKER).iterdir())})
    print("[compact_cache_pb] verify " + json.dumps(rep), flush=True)
    return rep


# ----------------------------------------------------------------------------------------------------------------------
# coverage against train_metric_cache / anchors_scores_index (pb pitfall: compute_score does metric_cache_paths[token])
def metric_cache_tokens(metric_cache_dir: Path) -> Dict[str, str]:
    """Token -> metric_cache.pkl path from the FIRST metadata csv, exactly like navsim.common.dataloader.MetricCacheLoader."""
    meta = sorted(p for p in (metric_cache_dir / "metadata").iterdir() if ".csv" in p.name)
    assert meta, f"no metadata csv under {metric_cache_dir}/metadata"
    if len(meta) > 1:
        print(f"[compact_cache_pb] WARNING {len(meta)} metadata csv files; MetricCacheLoader reads only {meta[0].name} "
              "(scripts/fix_metric_cache_paths.sh merges them)", flush=True)
    with open(meta[0]) as f:
        rows = f.read().splitlines()[1:]
    return {r.split("/")[-2]: r for r in rows if r}


def coverage(out: str, metric_cache_dir: str, move_aside: Optional[str] = None, check_anchors: bool = False) -> dict:
    out_p, mc = Path(out), Path(metric_cache_dir)
    mtok = metric_cache_tokens(mc)
    cache_tokens = [(log, d.name) for log in list_cache_logs(out_p) for d in (out_p / log).iterdir() if d.is_dir()]
    uncovered = [(log, tok) for log, tok in cache_tokens if tok not in mtok]
    missing_anchor: List[Tuple[str, str]] = []
    if check_anchors:
        for log, tok in cache_tokens:
            p = mtok.get(tok)
            if p is not None and not os.path.exists(p.replace("train_metric_cache", "anchors_scores_index") + ".npy"):
                missing_anchor.append((log, tok))
    moved = 0
    if move_aside:
        for log, tok in uncovered + missing_anchor:
            dst = Path(move_aside) / log / tok
            if (out_p / log / tok).exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(out_p / log / tok), str(dst))
                moved += 1
        for log in list_cache_logs(out_p):  # drop log dirs emptied by the move (CacheOnlyDataset iterates them)
            if not any((out_p / log).iterdir()):
                (out_p / log).rmdir()
    rep = {"out": str(out_p), "metric_cache": str(mc), "n_cache_tokens": len(cache_tokens), "n_metric_tokens": len(mtok),
           "n_uncovered": len(uncovered), "n_missing_anchor": len(missing_anchor) if check_anchors else None, "moved_aside": moved,
           "uncovered_head": [f"{l}/{t}" for l, t in uncovered[:10]], "missing_anchor_head": [f"{l}/{t}" for l, t in missing_anchor[:10]]}
    print("[compact_cache_pb] coverage " + json.dumps(rep), flush=True)
    return rep


# ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="derive the pb cache (resumable, parallel, per split)")
    b.add_argument("--compact", required=True, help="pf compact JPEG cache (navsim_ext.compact_cache build output)")
    b.add_argument("--logs", required=True, help="navsim_logs/<split> directory with the <log>.pkl files")
    b.add_argument("--out", required=True, help="pb cache to write")
    b.add_argument("--split", default="all", help="train_test_split scene filter restricting logs+tokens: navtrain_val_official, "
                   "navtrain_train_official, navtrain, navtrain_fit, navtrain_dev; 'all' = every log of the compact cache")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--engine", default="auto", choices=ENGINES, help="builder = the patched vendored feature-builder code (needs mmcv); "
                   "numpy = torch/mmcv-free mirror; auto = builder when importable")
    b.add_argument("--no-selfcheck", action="store_true", help="skip the builder-vs-numpy agreement check on the first token of each log")
    b.add_argument("--force", action="store_true", help="rewrite tokens that already exist in --out")
    b.add_argument("--limit-logs", type=int, default=0, help="only the first N logs (smoke)")
    b.add_argument("--log-names", default="", help="comma-separated subset of logs")
    v = sub.add_parser("verify", help="re-check derived tokens (builder: through CompactCacheOnlyDataset vs the patched upstream builder)")
    v.add_argument("--compact", required=True)
    v.add_argument("--logs", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--n", "--limit", dest="n", type=int, default=50, help="random tokens to check (0 = all, pf mode only)")
    v.add_argument("--mode", default="auto", choices=("auto", "builder", "pf"))
    v.add_argument("--blobs", default="", help="sensor_blobs/<split> with CAM_F0 frames of (some) cache logs; default $OPENSCENE_DATA_ROOT/sensor_blobs/trainval")
    v.add_argument("--min-psnr", type=float, default=35.0)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--log-names", default="")
    c = sub.add_parser("coverage", help="cache tokens missing from train_metric_cache metadata (and anchors index)")
    c.add_argument("--out", required=True)
    c.add_argument("--metric-cache", required=True, help="$NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache")
    c.add_argument("--check-anchors", action="store_true")
    c.add_argument("--move-aside", default="", help="move uncovered token dirs into this directory (same <log>/<token> layout)")
    k = sub.add_parser("count", help="count complete token dirs of a derived cache")
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        s = build(a.compact, a.logs, a.out, a.workers, a.force, [x for x in a.log_names.split(",") if x] or None, a.limit_logs,
                  None if a.split == "all" else a.split, a.engine, not a.no_selfcheck)
        return 0 if (s["failed"] == 0 and s["missing_pickle"] == 0) else 1
    if a.cmd == "verify":
        r = verify(a.compact, a.logs, a.out, a.n, a.seed, [x for x in a.log_names.split(",") if x] or None, a.mode, a.blobs or None, a.min_psnr)
        return 0 if (r["fail"] == 0 and r["ok"] > 0 and r["marker"]) else 1
    if a.cmd == "coverage":
        r = coverage(a.out, a.metric_cache, a.move_aside or None, a.check_anchors)
        return 0 if (r["n_uncovered"] == 0 and not r["n_missing_anchor"]) or r["moved_aside"] else 1
    if a.cmd == "count":
        complete, dirs = count_tokens(a.out)
        print(json.dumps({"out": a.out, "tokens_complete": complete, "token_dirs": dirs}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
