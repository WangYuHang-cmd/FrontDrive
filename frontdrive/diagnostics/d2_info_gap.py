"""D2 information-gap diagnostic: 8-frame (3.5 s) pretraining input vs 2-frame (0.5 s) deployment input.

Plan v4.1 section 3.1 (D2) / section 5 (E3).  Question: how much more about the vehicle's dynamic state and near
future can be READ OUT (frozen e50 target-encoder features + simple probes) from the 8-frame input the encoder was
pretrained with than from the 2-frame input the deployed Drive-JEPA perception-free agent feeds it?  A small gap
means no long-sequence teacher is needed; a large gap makes "long history teacher -> two-frame student" distillation
a phase-2 candidate.

Sub-commands
------------
extract   clip list (``<mp4> 0`` per line) -> features npz (+ ``.meta.json``).  ``--exclude-list`` drops clips (ids or
          mp4 paths, e.g. the E2 continued-pretraining pool) so a later run is a held-out comparison; ``--shard i/n``
          processes the i-th of n contiguous chunks of the (filtered) list so a long run fits several short foreground
          calls (``merge-features`` joins them); ``--precision fp32`` (default, TF32 disabled: batch-shape independent
          and bit-reproducible) or ``bf16`` autocast (2-3x faster, ~1 % pooled deviation that depends on the batch
          shape).  Per clip, anchors a = history,
          history+stride, ... with a + horizon < n_frames (defaults: history 7, horizon 4, stride 8 kept frames = 4 s
          at 2 fps).  Inputs per anchor: 8f = frames a-7..a (4 tubelets x 512 tokens), 2f = frames a-1, a (one
          tubelet, exactly what the deployed agent sees).  Stored (float16): ``f2_mean`` (mean over the 512 tokens of
          the 2f pass), ``f8_mean`` (mean over all 2048 tokens of the 8f pass), ``f8_last`` (mean over the last slot's
          512 tokens of the 8f pass) and ``f8_slots`` (per-slot means of the 8f pass, [N, 4, D]).  Targets at the
          anchor come from the ``<clip>.can.npz`` sidecar (NaN = unavailable).
merge-features
          shard npz files from ``extract --shard`` (same list / checkpoint / stride / precision / batch sizes) -> one
          features npz + ``.meta.json`` (windows concatenated in shard order, GPU / wall time summed).
probe     features npz -> ``results.json`` + ``table.md``: ridge / logistic probes, 5-fold GroupKFold by clip,
          inner 4-fold CV for the regulariser, pooled held-out R^2 / MAE / AUROC / AUPRC, constant baselines,
          shuffled-label control with ``--n-perm`` label permutations per feature variant (reported as mean / sd /
          min / max: the held-out labels keep their clip clustering, so the null is as wide as ~1,000 clips allow, not
          as narrow as ~15,000 windows would suggest), paired clip bootstrap CIs for every metric and for the gaps
          f8_mean - f2_mean and f8_last - f2_mean.  The ridge Gram eigendecompositions are label-independent and are
          shared across targets, shuffled controls and slot probes (``RidgeFoldCache``); the logistic inner CV and the
          logistic permutation controls run in up to ``--n-jobs`` worker processes (1 BLAS thread each).
check-t2  numerically confirm that a 2-frame input through the max_num_frames=8 RoPE model built here equals the
          deployed agent's construction (``vit_large(num_frames=2)`` + target_encoder weights), quantify how much
          the 6 context frames change the last slot's tokens, and measure the bf16-vs-fp32 and batch-shape
          dependence of the pooled features.
retarget  features npz -> the same npz with ``targets`` (and ``dt_hist`` / ``dt_fut``) recomputed from the sidecars
          under ``--clips-root`` with the current ``window_targets`` (features untouched; written to ``.tmp`` then
          ``os.replace``; ``meta_json`` and the ``.meta.json`` gain ``retargeted`` / ``retarget``).  Every target not
          named in ``--changed`` must come back bit-identical (checked, max abs diff reported); ``--backup`` keeps
          the previous target arrays, ``--report`` writes the comparison as json.  Used 2026-10-12 to fix the
          yaw-sign targets of features extracted before the fix (see ``D_YAW_SIGN``).

Conventions: frames scaled to [0, 1] then ImageNet mean/std normalised (vendored ``app/vjepa/transforms.py``
VideoTransform default), full 256x512 frame, no crop, eval mode, no grad, fp32 with TF32 disabled (the evidence
protocol; ``--precision bf16`` autocast is available for quick runs).  Token order inside the ViT
is time-major (idx = slot*512 + row*32 + col), so slot k = tokens [k*512, (k+1)*512).  Sidecar conventions follow
``pretrain/motion_data.py``: ``d_*[i]`` = motion from kept frame i to i+1 in frame i's vehicle frame, with
``d_forward`` / ``d_left`` forward / LEFT positive (m) but ``d_yaw`` CLOCKWISE positive (rad: the difference of the
NED dead-reckoning heading), and ``wdev_z`` is the gyro yaw rate about the openpilot device z axis, which points DOWN,
so it is clockwise positive too (rad/s).  ``window_targets`` multiplies both by ``D_YAW_SIGN`` / ``YAW_RATE_SIGN``
= -1 before use, so the stored yaw targets (``yaw_rate_now``, ``dyaw_2s``) are CCW / left-turn positive and
``dleft_2s`` is composed from a consistent (forward, left, CCW) triple; see the constants for the measurement.
Features extracted before 2026-10-12 carry the unconverted targets and need ``retarget``.  Row 0 is often NaN;
rows missing from a short sidecar are NaN; clips without a sidecar are skipped.
"""
from __future__ import annotations

import os

# Shared CPU: cap BLAS threads before numpy is imported (setdefault: an explicit environment wins).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import hashlib
import json
import logging
import math
import pathlib
import sys
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("d2_info_gap")

# ----------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------
FPS = 2.0
HISTORY = 7  # frames before the anchor in the 8-frame input: a-7 .. a
HORIZON = 4  # future rows used by the targets: a+1 .. a+4 (2 s at 2 fps)
N_FRAMES_8F = 8
N_FRAMES_2F = 2
TUBELET = 2
PATCH = 16
IMG_HW = (256, 512)
TOKENS_PER_SLOT = (IMG_HW[0] // PATCH) * (IMG_HW[1] // PATCH)  # 512
N_SLOTS_8F = N_FRAMES_8F // TUBELET  # 4
EMBED_DIM = 1024  # vit_large
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DECEL_THRESHOLD = -2.0  # m/s^2, decel_event = any(a_ego[a+1..a+4] < threshold)
SIDECAR_SKIP_KEYS = ("heading_src",)  # object array, never loaded (allow_pickle=False)

# Sidecar yaw sign convention (same constants as pretrain/motion_data.py; confirmed by an independent geometry review
# 2026-10-12).  The comma sidecar's ``d_yaw`` is the difference of the dead-reckoning heading ``psi``, which
# comma/openpilot express as a NED angle (clockwise positive, z down), and ``wdev_z`` is the gyro rate about the
# openpilot device-frame z axis, which also points down; both are therefore right-turn (clockwise) positive and agree
# with each other (corr(wdev_z, d_yaw) = +0.83 .. +0.89 on the local clips), while ``d_forward`` / ``d_left`` are
# forward / left positive.  Measured on 600 local sidecars (17,117 turning rows with |d_yaw| > 0.01 rad; tici / tizi /
# mici): sign(d_left) == -sign(d_yaw) on 98.8 % of the rows, corr(d_left, d_yaw) = -0.82, and the unicycle chord
# relation holds as d_left = -d_forward * tan(d_yaw / 2) (mean residual 0.012 m, vs 0.237 m for the mirrored sign).
# ``window_targets`` therefore uses ``D_YAW_SIGN * d_yaw`` (CCW) in the dyaw_2s sum and in ``compose_se2``, and
# ``YAW_RATE_SIGN * wdev_z`` (CCW) for yaw_rate_now, so every yaw quantity stored in the features npz is
# counter-clockwise / left-turn positive and consistent with ``d_left``.  Before 2026-10-12 the raw values were used:
# yaw_rate_now / dyaw_2s carried the opposite sign of what the docs claimed (harmless for the sign-invariant linear
# probes) and dleft_2s was the lateral displacement under a MIRRORED rotation (wrong; fixed with ``retarget``).
D_YAW_SIGN = -1.0  # multiplies the sidecar d_yaw (clockwise +) -> CCW rad
YAW_RATE_SIGN = -1.0  # multiplies the sidecar wdev_z (clockwise +) -> CCW rad/s

# Optional plausibility mask (off by default; ``retarget --plausible`` turns it on while recomputing targets): sidecar
# values outside these physical ranges -- dead-reckoning heading resets / gyro spikes, 34 clips in subset v0 as of
# 2026-10-12 -- are treated as missing by ``_col``, so every window target that touches them becomes NaN and the window
# is dropped for that target by the probes.  Same ranges as ``pretrain.motion_data.PLAUSIBLE_RANGES`` (checked by a test).
PLAUSIBLE_RANGES = {"v_ego": (-2.0, 60.0), "a_ego": (-12.0, 12.0), "wdev_z": (-1.0, 1.0), "d_yaw": (-0.5, 0.5),
                    "d_forward": (-5.0, 30.0), "d_left": (-6.0, 6.0), "steer_deg": (-720.0, 720.0)}
PLAUSIBLE_MASK = False


def _set_plausible_mask(flag: bool) -> None:
    global PLAUSIBLE_MASK
    PLAUSIBLE_MASK = bool(flag)


CONTINUOUS_TARGETS = ["v_now", "yaw_rate_now", "a_now", "dv_2s", "dyaw_2s", "dleft_2s", "steer_now"]
BINARY_TARGETS = ["decel_event", "standstill_now"]
TARGET_NAMES = ["v_now", "yaw_rate_now", "a_now", "dv_2s", "dyaw_2s", "dleft_2s", "decel_event", "steer_now",
                "standstill_now"]
TARGET_UNITS = {"v_now": "m/s", "yaw_rate_now": "rad/s", "a_now": "m/s^2", "dv_2s": "m/s", "dyaw_2s": "rad",
                "dleft_2s": "m", "decel_event": "bool", "steer_now": "deg", "standstill_now": "bool"}
TARGET_DESCRIPTIONS = {
    "v_now": "v_ego[a]",
    "yaw_rate_now": "YAW_RATE_SIGN * wdev_z[a] = -wdev_z[a] (CCW +; the sidecar stores wdev_z clockwise +)",
    "a_now": "a_ego[a]", "dv_2s": "v_ego[a+4] - v_ego[a]",
    "dyaw_2s": "wrap(sum D_YAW_SIGN * d_yaw[a..a+3]) = wrap(-sum d_yaw) (CCW +; the sidecar stores d_yaw clockwise +)",
    "dleft_2s": "y of the SE(2)-composed pose at a+4 in frame a from (d_forward, d_left, D_YAW_SIGN * d_yaw) (left +)",
    "decel_event": f"any(a_ego[a+1..a+4] < {DECEL_THRESHOLD})", "steer_now": "steer_deg[a] (comma-only field)",
    "standstill_now": "standstill[a]",
}
# Targets whose values the 2026-10-12 sign fix changes (``retarget --changed`` default); the others must be bit-identical.
SIGN_FIX_CHANGED_TARGETS = ("yaw_rate_now", "dyaw_2s", "dleft_2s")
FEATURE_VARIANTS = ["f2_mean", "f8_mean", "f8_last"]
RIDGE_LAMBDAS = np.logspace(-3, 3, 13)  # lambda multiplies ||w||^2 in the per-sample MEAN squared loss (= sklearn alpha / n)
LOGREG_CS = np.logspace(-4, 1, 6)  # sklearn C grid (inverse regularisation on the summed log loss)


# ----------------------------------------------------------------------------------------------------------------
# Sidecar / window / target construction (numpy only; unit-tested on synthetic sidecars)
# ----------------------------------------------------------------------------------------------------------------
def sidecar_path(mp4_path: str) -> str:
    return os.path.splitext(mp4_path)[0] + ".can.npz"


def clip_id_of(mp4_path: str) -> str:
    return os.path.splitext(os.path.basename(mp4_path))[0]


def load_sidecar(path: Optional[str]) -> Optional[dict]:
    """Numeric keys of ``<clip>.can.npz`` (allow_pickle=False); ``device`` as str; None if missing/unreadable."""
    if path is None or not os.path.exists(path):
        return None
    out: dict = {}
    try:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key in SIDECAR_SKIP_KEYS:
                    continue
                try:
                    arr = z[key]
                except ValueError:  # another object array
                    continue
                if key == "device":
                    out[key] = str(arr.reshape(-1)[0]) if arr.size else ""
                elif arr.ndim == 0:
                    out[key] = arr.item()
                else:
                    out[key] = arr
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.warning("unreadable sidecar %s: %s", path, exc)
        return None
    return out


def sidecar_n_rows(side: dict) -> int:
    v = side.get("v_ego")
    return int(len(v)) if v is not None else 0


def _col(side: dict, key: str, rows) -> np.ndarray:
    """``side[key][rows]`` as float64 with NaN where the key is missing or ``rows`` fall outside the array."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.full(rows.shape, np.nan, dtype=np.float64)
    arr = side.get(key)
    if arr is None:
        return out
    arr = np.asarray(arr)
    ok = (rows >= 0) & (rows < arr.shape[0])
    if ok.any():
        out[ok] = arr[rows[ok]].astype(np.float64)
    if PLAUSIBLE_MASK and key in PLAUSIBLE_RANGES:
        lo, hi = PLAUSIBLE_RANGES[key]
        with np.errstate(invalid="ignore"):
            out[~((out >= lo) & (out <= hi))] = np.nan  # out-of-range (or already NaN) -> missing
    return out


def clip_anchors(n_frames: int, stride: int = 8, history: int = HISTORY, horizon: int = HORIZON) -> np.ndarray:
    """Anchors a = history, history+stride, ... with a + horizon < n_frames (int64 array, possibly empty)."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if n_frames <= history + horizon:
        return np.zeros(0, dtype=np.int64)
    return np.arange(history, n_frames - horizon, stride, dtype=np.int64)


def wrap_angle(x):
    """Wrap to (-pi, pi]; NaN propagates."""
    x = np.asarray(x, dtype=np.float64)
    return x - 2.0 * np.pi * np.ceil((x - np.pi) / (2.0 * np.pi))


def compose_se2(d_forward, d_left, d_yaw) -> Tuple[float, float, float]:
    """Chain per-step SE(2) deltas (step k: pose k -> k+1 in pose k's frame: forward, left, CCW yaw).

    Returns the final pose (x, y, yaw) relative to pose 0, in pose 0's frame (x forward, y left, yaw CCW,
    unwrapped).  Any NaN step makes the result NaN."""
    df, dl, dy = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (d_forward, d_left, d_yaw))
    if not (len(df) == len(dl) == len(dy)):
        raise ValueError("d_forward, d_left, d_yaw must have the same length")
    if len(df) == 0:
        return 0.0, 0.0, 0.0
    if not (np.isfinite(df).all() and np.isfinite(dl).all() and np.isfinite(dy).all()):
        return float("nan"), float("nan"), float("nan")
    yaw = np.concatenate([[0.0], np.cumsum(dy)])  # heading of pose k in frame 0
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    x = float(np.sum(c * df - s * dl))
    y = float(np.sum(s * df + c * dl))
    return x, y, float(yaw[-1])


def window_targets(side: dict, anchor: int, horizon: int = HORIZON, history: int = HISTORY) -> Dict[str, float]:
    """Targets (float, NaN = unavailable) at ``anchor`` from a loaded sidecar; also dt_hist / dt_fut from t_sof.

    The sidecar's ``wdev_z`` and ``d_yaw`` are clockwise positive (see ``D_YAW_SIGN``); they are converted to CCW
    here, so yaw_rate_now / dyaw_2s are left-turn positive and dleft_2s composes a consistent (forward, left, CCW)
    triple (left-turn -> positive lateral displacement)."""
    a = int(anchor)
    out: Dict[str, float] = {}
    v_now, v_fut = _col(side, "v_ego", [a, a + horizon])
    out["v_now"] = float(v_now)
    out["yaw_rate_now"] = float(YAW_RATE_SIGN * _col(side, "wdev_z", [a])[0])  # sidecar wdev_z is clockwise +
    out["a_now"] = float(_col(side, "a_ego", [a])[0])
    out["dv_2s"] = float(v_fut - v_now)
    steps = np.arange(a, a + horizon)
    d_yaw_ccw = D_YAW_SIGN * _col(side, "d_yaw", steps)  # sidecar d_yaw is clockwise +; CCW from here on
    out["dyaw_2s"] = float(wrap_angle(np.sum(d_yaw_ccw)))  # NaN if any step NaN
    _, y_final, _ = compose_se2(_col(side, "d_forward", steps), _col(side, "d_left", steps), d_yaw_ccw)
    out["dleft_2s"] = float(y_final)
    acc = _col(side, "a_ego", np.arange(a + 1, a + horizon + 1))
    out["decel_event"] = float(np.any(acc < DECEL_THRESHOLD)) if np.isfinite(acc).all() else float("nan")
    out["steer_now"] = float(_col(side, "steer_deg", [a])[0])
    out["standstill_now"] = float(_col(side, "standstill", [a])[0])
    t = _col(side, "t_sof", [a - history, a, a + horizon])
    out["dt_hist"] = float((t[1] - t[0]) * 1e-9)
    out["dt_fut"] = float((t[2] - t[1]) * 1e-9)
    return out


def targets_matrix(side: dict, anchors: Sequence[int], horizon: int = HORIZON, history: int = HISTORY):
    """(targets float32 [A, len(TARGET_NAMES)], dt_hist float32 [A], dt_fut float32 [A])."""
    rows = [window_targets(side, a, horizon=horizon, history=history) for a in anchors]
    tg = np.array([[r[k] for k in TARGET_NAMES] for r in rows], dtype=np.float32).reshape(len(rows), len(TARGET_NAMES))
    dth = np.array([r["dt_hist"] for r in rows], dtype=np.float32)
    dtf = np.array([r["dt_fut"] for r in rows], dtype=np.float32)
    return tg, dth, dtf


# ----------------------------------------------------------------------------------------------------------------
# Video decoding (DataLoader workers)
# ----------------------------------------------------------------------------------------------------------------
def decode_clip(path: str) -> Tuple[np.ndarray, str]:
    """All frames of an mp4 as uint8 [T, H, W, 3] RGB; decord (1 thread) with a PyAV fallback.  Returns (frames, decoder)."""
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return np.ascontiguousarray(frames), "decord"
    except Exception as exc:  # noqa: BLE001 - fall back to PyAV on any decord failure
        logger.debug("decord failed on %s (%s); trying PyAV", path, exc)
    import av  # type: ignore

    with av.open(path) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 1
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
    return np.ascontiguousarray(np.stack(frames)), "pyav"


def read_clip_list(path: str) -> List[str]:
    """``<mp4> [label]`` per line -> list of mp4 paths (blank lines ignored)."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                out.append(parts[0])
    return out


def read_exclude_list(path: str) -> set:
    """Clip ids to exclude: one clip id or mp4 path per line (first token; blank lines and ``#`` comments ignored)."""
    out = set()
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts and not parts[0].startswith("#"):
                out.add(clip_id_of(parts[0]))
    return out


def apply_exclusions(paths: Sequence[str], exclude_ids) -> Tuple[List[str], List[str]]:
    """Drop the clips whose id is in ``exclude_ids``; returns (kept paths, excluded clip ids in list order)."""
    exclude_ids = set(exclude_ids or ())
    kept, dropped = [], []
    for p in paths:
        cid = clip_id_of(p)
        if cid in exclude_ids:
            dropped.append(cid)
        else:
            kept.append(p)
    return kept, dropped


def parse_shard(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """``"i/n"`` -> (i, n) with 0 <= i < n; None or "" -> None."""
    if not spec:
        return None
    try:
        i_s, n_s = str(spec).split("/")
        i, n = int(i_s), int(n_s)
    except ValueError as exc:
        raise ValueError(f"shard must be 'i/n', got {spec!r}") from exc
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"shard must satisfy 0 <= i < n, got {spec!r}")
    return i, n


def shard_paths(paths: Sequence[str], spec: Optional[str]) -> List[str]:
    """The i-th of n contiguous, near-equal chunks of ``paths`` (``np.array_split`` boundaries), or all of them."""
    sh = parse_shard(spec)
    if sh is None:
        return list(paths)
    i, n = sh
    return [paths[int(j)] for j in np.array_split(np.arange(len(paths)), n)[i]]


class ClipWindowsDataset:
    """One item per clip: decoded frames + anchors + targets (numpy only, runs in DataLoader workers)."""

    def __init__(self, mp4_paths: Sequence[str], stride: int = 8, history: int = HISTORY, horizon: int = HORIZON,
                 expect_hw: Tuple[int, int] = IMG_HW):
        self.paths = list(mp4_paths)
        self.stride, self.history, self.horizon, self.expect_hw = int(stride), int(history), int(horizon), tuple(expect_hw)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        path = self.paths[i]
        item = {"clip_id": clip_id_of(path), "path": path, "frames": None, "anchors": np.zeros(0, np.int64),
                "targets": None, "dt_hist": None, "dt_fut": None, "device": "", "n_frames": 0, "sidecar_rows": 0,
                "decoder": "", "error": ""}
        side = load_sidecar(sidecar_path(path))
        if side is None:
            item["error"] = "no_sidecar"
            return item
        item["device"] = str(side.get("device", ""))
        item["sidecar_rows"] = sidecar_n_rows(side)
        try:
            frames, item["decoder"] = decode_clip(path)
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"decode_error: {type(exc).__name__}: {exc}"
            return item
        if frames.ndim != 4 or frames.shape[-1] != 3:
            item["error"] = f"bad_frames_shape: {frames.shape}"
            return item
        if tuple(frames.shape[1:3]) != self.expect_hw:
            item["error"] = f"unexpected_resolution: {frames.shape[1:3]} != {self.expect_hw}"
            return item
        n = int(frames.shape[0])
        item["n_frames"] = n
        anchors = clip_anchors(n, self.stride, self.history, self.horizon)
        item["anchors"] = anchors
        if len(anchors) == 0:
            item["error"] = "too_short"
            return item
        item["frames"] = frames
        item["targets"], item["dt_hist"], item["dt_fut"] = targets_matrix(side, anchors, self.horizon, self.history)
        return item


# ----------------------------------------------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------------------------------------------
def _quiet_root_logger():
    """The vendored init_video_model calls logging.basicConfig(level=INFO) on the root logger and logs the whole
    module repr; silence that (returns a restore callable)."""
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.WARNING)
    return lambda: root.setLevel(prev)


def strip_module_prefix(state_dict: dict) -> dict:
    """Strip ONLY the DDP ``module.`` prefix (the wrappers keep ``backbone.``)."""
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def build_encoder(ckpt_path: str, device, model_name: str = "vit_large"):
    """init_video_model(...) with the pretraining configuration, TARGET_ENCODER weights loaded into the encoder
    wrapper (predictor discarded), eval mode.  Returns (encoder_wrapper, load_report)."""
    import torch
    from app.vjepa.utils import init_video_model  # vendored (PYTHONPATH from env/drivejepa_env.sh)

    restore = _quiet_root_logger()
    try:
        encoder, predictor = init_video_model(
            model_name=model_name, crop_size=IMG_HW, patch_size=PATCH, max_num_frames=N_FRAMES_8F, tubelet_size=TUBELET,
            uniform_power=True, use_rope=True, use_sdpa=True, use_mask_tokens=True, num_mask_tokens=2,
            zero_init_mask_tokens=True, pred_depth=12, pred_embed_dim=384, pred_num_heads=12, device=device,
        )
    finally:
        restore()
    del predictor
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = strip_module_prefix(ck["target_encoder"])
    msg = encoder.load_state_dict(sd, strict=False)
    report = {
        "checkpoint": str(ckpt_path), "state_dict_key": "target_encoder", "prefix_stripped": "module.",
        "n_tensors": len(sd), "n_params": int(sum(int(v.numel()) for v in sd.values())),
        "missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys),
        "epoch": int(ck["epoch"]) if "epoch" in ck else None,
        "model_kwargs": {"model_name": model_name, "crop_size": list(IMG_HW), "patch_size": PATCH,
                         "max_num_frames": N_FRAMES_8F, "tubelet_size": TUBELET, "uniform_power": True, "use_rope": True,
                         "use_sdpa": True},
    }
    del ck
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(f"target_encoder load mismatch: missing={msg.missing_keys} unexpected={msg.unexpected_keys}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, report


def normalize_frames(frames_u8, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 tensor [B, T, H, W, 3] -> float32 [B, 3, T, H, W], /255 then ImageNet mean/std (vendored convention)."""
    import torch

    x = frames_u8.permute(0, 4, 1, 2, 3).to(torch.float32).div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1, 1)
    return (x - m) / s


def pool_slots(tokens, tokens_per_slot: int = TOKENS_PER_SLOT):
    """[B, N, D] tokens (time-major, slot = N // tokens_per_slot) -> per-slot means [B, n_slots, D] (float32)."""
    B, N, D = tokens.shape
    if N % tokens_per_slot:
        raise ValueError(f"{N} tokens is not a multiple of {tokens_per_slot}")
    return tokens.float().reshape(B, N // tokens_per_slot, tokens_per_slot, D).mean(dim=2)


def set_strict_fp32() -> None:
    """Disable TF32 in cuBLAS matmuls and in the cuDNN patch-embedding conv (cuDNN TF32 is on by default), so the fp32
    protocol is plain fp32: batch-shape independent and bit-reproducible on the same hardware / library versions."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def encode_tokens(backbone, frames_u8, use_bf16: bool = True):
    """Frozen forward: uint8 [B, T, H, W, 3] on the model device -> float32 tokens [B, N, D]."""
    import torch

    x = normalize_frames(frames_u8)
    with torch.no_grad():
        if use_bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = backbone(x)
        else:
            out = backbone(x)
    return out.float()


# ----------------------------------------------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------------------------------------------
def run_extract(args) -> dict:
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(int(args.threads))
    t_wall0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    paths = read_clip_list(args.list)
    n_listed = len(paths)
    excluded: List[str] = []
    exclude_list = getattr(args, "exclude_list", None)
    if exclude_list:
        paths, excluded = apply_exclusions(paths, read_exclude_list(exclude_list))
        logger.info("excluded %d of %d clips (ids listed in %s)", len(excluded), n_listed, exclude_list)
    shard = getattr(args, "shard", None) or None
    if shard:
        n_before = len(paths)
        paths = shard_paths(paths, shard)
        logger.info("shard %s: %d of %d clips", shard, len(paths), n_before)
    if args.max_clips and args.max_clips > 0:
        paths = paths[: args.max_clips]
    logger.info("%d clips from %s", len(paths), args.list)
    use_bf16 = device.type == "cuda" and args.precision == "bf16"
    if device.type == "cuda" and not use_bf16:
        set_strict_fp32()

    encoder, load_report = build_encoder(args.ckpt, device)
    backbone = encoder.backbone
    logger.info("encoder loaded: missing=%d unexpected=%d params=%d epoch=%s", len(load_report["missing"]),
                len(load_report["unexpected"]), load_report["n_params"], load_report["epoch"])

    ds = ClipWindowsDataset(paths, stride=args.stride, history=HISTORY, horizon=HORIZON)
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=int(args.num_workers), collate_fn=lambda x: x,
                    prefetch_factor=(2 if args.num_workers > 0 else None), persistent_workers=False)

    feats = {k: [] for k in ("f2_mean", "f8_mean", "f8_last", "f8_slots")}
    meta_cols = {k: [] for k in ("clip_id", "anchor", "n_frames", "device", "sidecar_rows", "targets", "dt_hist", "dt_fut")}
    skipped: Dict[str, List[str]] = {}
    decoders: Dict[str, int] = {}
    gpu_time = 0.0
    n_clips_done = 0
    idx8 = torch.arange(-HISTORY, 1, device=device)  # a-7 .. a
    idx2 = torch.arange(-(N_FRAMES_2F - 1), 1, device=device)  # a-1, a
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for n_item, item in enumerate(dl):
        if item["error"]:
            skipped.setdefault(item["error"].split(":")[0], []).append(item["clip_id"])
            continue
        decoders[item["decoder"]] = decoders.get(item["decoder"], 0) + 1
        frames = torch.from_numpy(item["frames"]).to(device, non_blocking=False)  # [T, H, W, 3] uint8
        anchors = torch.from_numpy(item["anchors"]).to(device)
        A = int(anchors.numel())
        slots_all, f2_all = [], []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        win8 = frames[(anchors[:, None] + idx8[None, :])]  # [A, 8, H, W, 3]
        for s in range(0, A, args.batch):
            tok = encode_tokens(backbone, win8[s: s + args.batch], use_bf16=use_bf16)
            slots_all.append(pool_slots(tok, TOKENS_PER_SLOT))  # [b, 4, D]
        del win8
        win2 = frames[(anchors[:, None] + idx2[None, :])]  # [A, 2, H, W, 3]
        b2 = max(1, args.batch * (N_FRAMES_8F // N_FRAMES_2F))
        for s in range(0, A, b2):
            tok = encode_tokens(backbone, win2[s: s + b2], use_bf16=use_bf16)
            f2_all.append(pool_slots(tok, TOKENS_PER_SLOT)[:, 0])  # [b, D]
        del win2, frames
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gpu_time += time.perf_counter() - t0
        slots = torch.cat(slots_all, 0)  # [A, 4, D] float32
        f2 = torch.cat(f2_all, 0)
        feats["f8_slots"].append(slots.half().cpu().numpy())
        feats["f8_mean"].append(slots.mean(dim=1).half().cpu().numpy())
        feats["f8_last"].append(slots[:, -1].half().cpu().numpy())
        feats["f2_mean"].append(f2.half().cpu().numpy())
        meta_cols["clip_id"].extend([item["clip_id"]] * A)
        meta_cols["anchor"].append(item["anchors"].astype(np.int32))
        meta_cols["n_frames"].append(np.full(A, item["n_frames"], np.int32))
        meta_cols["sidecar_rows"].append(np.full(A, item["sidecar_rows"], np.int32))
        meta_cols["device"].extend([item["device"]] * A)
        meta_cols["targets"].append(item["targets"])
        meta_cols["dt_hist"].append(item["dt_hist"])
        meta_cols["dt_fut"].append(item["dt_fut"])
        n_clips_done += 1
        if n_clips_done % 50 == 0 or n_item + 1 == len(ds):
            logger.info("[%d/%d] clips done, %d windows, gpu %.1fs, wall %.1fs", n_item + 1, len(ds),
                        sum(len(a) for a in meta_cols["anchor"]), gpu_time, time.time() - t_wall0)

    if n_clips_done == 0:
        raise RuntimeError(f"no clip produced windows; skipped={ {k: len(v) for k, v in skipped.items()} }")
    out = {
        "f2_mean": np.concatenate(feats["f2_mean"]), "f8_mean": np.concatenate(feats["f8_mean"]),
        "f8_last": np.concatenate(feats["f8_last"]), "f8_slots": np.concatenate(feats["f8_slots"]),
        "clip_id": np.array(meta_cols["clip_id"]), "anchor": np.concatenate(meta_cols["anchor"]),
        "n_frames": np.concatenate(meta_cols["n_frames"]), "sidecar_rows": np.concatenate(meta_cols["sidecar_rows"]),
        "device": np.array(meta_cols["device"]), "targets": np.concatenate(meta_cols["targets"]),
        "target_names": np.array(TARGET_NAMES), "dt_hist": np.concatenate(meta_cols["dt_hist"]),
        "dt_fut": np.concatenate(meta_cols["dt_fut"]),
    }
    n_windows = int(out["anchor"].shape[0])
    peak_mem = float(torch.cuda.max_memory_allocated(device)) / 2**30 if device.type == "cuda" else None
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "list": os.path.abspath(args.list), "ckpt": os.path.abspath(args.ckpt),
        "stride": args.stride, "history": HISTORY, "horizon": HORIZON, "batch_8f": args.batch,
        "batch_2f": max(1, args.batch * (N_FRAMES_8F // N_FRAMES_2F)), "num_workers": args.num_workers,
        "n_clips_listed": n_listed, "n_clips_excluded": len(excluded),
        "exclude_list": os.path.abspath(exclude_list) if exclude_list else None, "excluded_clip_ids": excluded,
        "shard": shard, "n_clips_processed": len(paths), "n_clips_with_windows": n_clips_done,
        "n_clips_skipped": {k: len(v) for k, v in skipped.items()}, "skipped_clip_ids": skipped,
        "n_windows": n_windows, "windows_per_clip_mean": n_windows / max(1, n_clips_done), "decoders": decoders,
        "devices": {d: int(c) for d, c in zip(*np.unique(out["device"], return_counts=True))},
        "gpu_forward_time_s": gpu_time, "wall_time_s": time.time() - t_wall0, "peak_gpu_mem_gib": peak_mem,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "precision": "bf16 autocast" if use_bf16 else "fp32",
        "tf32": (bool(torch.backends.cuda.matmul.allow_tf32) or bool(torch.backends.cudnn.allow_tf32)) if device.type == "cuda" else None,
        "feature_dtype": "float16", "embed_dim": int(out["f2_mean"].shape[1]), "encoder_load": load_report,
        "input_norm": {"scale": "1/255", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "dt_hist_mean_s": float(np.nanmean(out["dt_hist"])), "dt_fut_mean_s": float(np.nanmean(out["dt_fut"])),
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), meta_json=np.array(json.dumps(meta)), **out)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1))
    logger.info("wrote %s (%d windows from %d clips; gpu %.1fs; peak %.2f GiB) and %s", out_path, n_windows,
                n_clips_done, gpu_time, peak_mem or 0.0, meta_path)
    return meta


# ----------------------------------------------------------------------------------------------------------------
# merge-features
# ----------------------------------------------------------------------------------------------------------------
MERGE_CONSISTENT_META = ("list", "ckpt", "stride", "history", "horizon", "precision", "tf32", "batch_8f", "batch_2f",
                         "exclude_list", "n_clips_listed", "n_clips_excluded", "feature_dtype", "embed_dim", "input_norm")


def _sum_counts(dicts) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def merge_features(inputs: Sequence[str], out: str) -> dict:
    """Concatenate shard npz files written by ``extract --shard i/n`` into one features npz + ``.meta.json``.

    The shards must agree on list / checkpoint / stride / precision / batch sizes (``MERGE_CONSISTENT_META``) and on
    ``target_names``; no clip may occur in two shards.  Inputs that carry ``shard = "i/n"`` are ordered by i, so the
    merged window order equals that of an unsharded run.  GPU / wall times are summed, peak memory is the max, skipped
    clips are unioned, counts are recomputed from the merged arrays."""
    if not inputs:
        raise ValueError("merge_features needs at least one input")
    arrays, metas = [], []
    for p in inputs:
        with np.load(p, allow_pickle=False) as z:
            arrays.append({k: z[k] for k in z.files if k != "meta_json"})
            metas.append(json.loads(str(z["meta_json"])) if "meta_json" in z.files else {})
    shards = [parse_shard(m.get("shard")) for m in metas]
    if all(s is not None for s in shards) and len({s[1] for s in shards}) == 1 and len({s[0] for s in shards}) == len(shards):
        order = sorted(range(len(inputs)), key=lambda i: shards[i][0])
        inputs, arrays, metas = [inputs[i] for i in order], [arrays[i] for i in order], [metas[i] for i in order]
    keys = set(arrays[0])
    for a, p in zip(arrays[1:], inputs[1:]):
        if set(a) != keys:
            raise ValueError(f"{p}: keys {sorted(set(a) ^ keys)} differ from {inputs[0]}")
    m0 = metas[0]
    for m, p in zip(metas[1:], inputs[1:]):
        for k in MERGE_CONSISTENT_META:
            if m.get(k) != m0.get(k):
                raise ValueError(f"{p}: meta {k!r} = {m.get(k)!r} differs from {inputs[0]} ({m0.get(k)!r})")
    merged = {}
    for k in sorted(keys):
        if k == "target_names":
            if any(not np.array_equal(a[k], arrays[0][k]) for a in arrays[1:]):
                raise ValueError("target_names differ between shards")
            merged[k] = arrays[0][k]
        else:
            merged[k] = np.concatenate([a[k] for a in arrays], axis=0)
    ids = [set(a["clip_id"].astype(str).tolist()) for a in arrays]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dup = ids[i] & ids[j]
            if dup:
                raise ValueError(f"clips {sorted(dup)[:5]} occur in both {inputs[i]} and {inputs[j]}")
    clip_ids = merged["clip_id"].astype(str)
    skipped: Dict[str, List[str]] = {}
    for m in metas:
        for k, v in (m.get("skipped_clip_ids") or {}).items():
            skipped.setdefault(k, []).extend(v)
    n_windows = int(len(clip_ids))
    n_clips = int(len(np.unique(clip_ids)))
    meta = dict(m0)
    meta.update({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "merged_from": [os.path.abspath(p) for p in inputs],
        "shards": [m.get("shard") for m in metas], "shard": None,
        "n_clips_processed": int(sum(int(m.get("n_clips_processed", 0) or 0) for m in metas)), "n_clips_with_windows": n_clips,
        "n_clips_skipped": {k: len(v) for k, v in skipped.items()}, "skipped_clip_ids": skipped,
        "n_windows": n_windows, "windows_per_clip_mean": n_windows / max(1, n_clips),
        "decoders": _sum_counts([m.get("decoders", {}) for m in metas]),
        "gpu_forward_time_s": float(sum(float(m.get("gpu_forward_time_s", 0.0) or 0.0) for m in metas)),
        "wall_time_s": float(sum(float(m.get("wall_time_s", 0.0) or 0.0) for m in metas)),
        "peak_gpu_mem_gib": max(float(m.get("peak_gpu_mem_gib") or 0.0) for m in metas),
        "per_shard": [{k: m.get(k) for k in ("shard", "created", "n_clips_processed", "n_clips_with_windows", "n_windows",
                                              "gpu_forward_time_s", "wall_time_s", "peak_gpu_mem_gib")} for m in metas],
    })
    if "device" in merged:
        meta["devices"] = {d: int(c) for d, c in zip(*np.unique(merged["device"], return_counts=True))}
    if "dt_hist" in merged:
        meta["dt_hist_mean_s"] = float(np.nanmean(merged["dt_hist"]))
    if "dt_fut" in merged:
        meta["dt_fut_mean_s"] = float(np.nanmean(merged["dt_fut"]))
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), meta_json=np.array(json.dumps(meta)), **merged)
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(json.dumps(meta, indent=1))
    logger.info("merged %d shards -> %s (%d windows from %d clips; gpu %.1fs)", len(inputs), out_path, n_windows, n_clips,
                meta["gpu_forward_time_s"])
    return meta


def run_merge_features(args) -> dict:
    return merge_features(list(args.inputs), args.out)


# ----------------------------------------------------------------------------------------------------------------
# retarget: recompute the stored targets from the sidecars (features untouched)
# ----------------------------------------------------------------------------------------------------------------
DEFAULT_CLIPS_ROOT = "/home/henry/Desktop/Qwen_Drive/benchmarks/pretrain/clips/subset_v0"


def build_sidecar_index(clips_root: str) -> Dict[str, str]:
    """clip id -> ``.can.npz`` path for every sidecar below ``clips_root`` (one os.walk; fallback lookup)."""
    index: Dict[str, str] = {}
    for dirpath, _, files in os.walk(clips_root):
        for f in files:
            if f.endswith(".can.npz"):
                index.setdefault(f[: -len(".can.npz")], os.path.join(dirpath, f))
    return index


def find_sidecar(clip_id: str, clips_root: str, index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """``<root>/<id[:2]>/<id>.can.npz`` (the comma export layout), then ``<root>/<id>.can.npz``, then ``index``."""
    for cand in (os.path.join(clips_root, clip_id[:2], clip_id + ".can.npz"), os.path.join(clips_root, clip_id + ".can.npz")):
        if os.path.exists(cand):
            return cand
    return (index or {}).get(clip_id)


def compare_columns(old: np.ndarray, new: np.ndarray) -> dict:
    """NaN-aware comparison of two target columns: bit identity, max abs diff over the jointly finite rows, NaN pattern."""
    old = np.asarray(old, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)
    fin_o, fin_n = np.isfinite(old), np.isfinite(new)
    both = fin_o & fin_n
    same_nan = bool(np.array_equal(fin_o, fin_n))
    return {
        "bit_identical": bool(same_nan and np.array_equal(old[both].view(np.uint32), new[both].view(np.uint32))),
        "max_abs_diff": float(np.max(np.abs(old[both].astype(np.float64) - new[both]))) if both.any() else 0.0,
        "nan_pattern_identical": same_nan, "n_finite_old": int(fin_o.sum()), "n_finite_new": int(fin_n.sum()),
    }


def retarget_features(features: str, clips_root: str = DEFAULT_CLIPS_ROOT, changed: Sequence[str] = SIGN_FIX_CHANGED_TARGETS,
                      backup: Optional[str] = None, report_path: Optional[str] = None, plausible: bool = False) -> dict:
    """Recompute ``targets`` (and ``dt_hist`` / ``dt_fut`` when stored) of a features npz for every stored
    (clip_id, anchor) row from the sidecars under ``clips_root`` with the current ``window_targets`` / ``targets_matrix``,
    and overwrite the npz atomically (``.tmp`` then ``os.replace``; the ``.meta.json`` next to it is rewritten too).

    Every target NOT listed in ``changed`` and the dt arrays must come back bit-identical to the stored values (same
    NaN pattern, same float32 bits), otherwise ``AssertionError`` is raised before anything is written.  The npz
    ``meta_json`` gains ``retargeted: true`` and a ``retarget`` block (date, sign constants, target definitions,
    per-target comparison).  ``backup`` (optional npz) receives the previous target / dt arrays with their row keys;
    ``report_path`` (optional) receives the returned report as json.  Returns the report (max abs diffs etc.)."""
    path = pathlib.Path(features)
    with np.load(str(path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta_json"))) if "meta_json" in arrays else {}
    tnames = [str(s) for s in arrays["target_names"]]
    if tnames != TARGET_NAMES:
        raise ValueError(f"target_names {tnames} differ from TARGET_NAMES {TARGET_NAMES}; retarget needs the current target set")
    changed = [str(t) for t in changed]
    if changed == ["all"]:
        changed = list(TARGET_NAMES)
    unknown = [t for t in changed if t not in TARGET_NAMES]
    if unknown:
        raise ValueError(f"unknown targets in changed: {unknown}")
    clip_ids = np.asarray(arrays["clip_id"]).astype(str)
    anchors = np.asarray(arrays["anchor"], dtype=np.int64)
    old_targets = np.asarray(arrays["targets"], dtype=np.float32)
    if old_targets.shape != (len(anchors), len(TARGET_NAMES)):
        raise ValueError(f"targets shape {old_targets.shape} != ({len(anchors)}, {len(TARGET_NAMES)})")
    horizon, history = int(meta.get("horizon", HORIZON)), int(meta.get("history", HISTORY))
    old_dt = {k: np.asarray(arrays[k], dtype=np.float32) for k in ("dt_hist", "dt_fut") if k in arrays}

    new_targets = np.full_like(old_targets, np.nan)
    new_dt = {k: np.full(len(anchors), np.nan, dtype=np.float32) for k in old_dt}
    uniq, inv = np.unique(clip_ids, return_inverse=True)
    index: Optional[Dict[str, str]] = None
    missing: List[str] = []
    prev_mask = PLAUSIBLE_MASK
    _set_plausible_mask(plausible)
    try:
        for g, cid in enumerate(uniq):
            rows = np.flatnonzero(inv == g)
            sc = find_sidecar(cid, clips_root, index)
            if sc is None and index is None:
                index = build_sidecar_index(clips_root)
                sc = find_sidecar(cid, clips_root, index)
            side = load_sidecar(sc) if sc else None
            if side is None:
                missing.append(cid)
                continue
            tg, dth, dtf = targets_matrix(side, anchors[rows], horizon=horizon, history=history)
            new_targets[rows] = tg
            if "dt_hist" in new_dt:
                new_dt["dt_hist"][rows] = dth
            if "dt_fut" in new_dt:
                new_dt["dt_fut"][rows] = dtf
    finally:
        _set_plausible_mask(prev_mask)
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(uniq)} clips have no readable sidecar under {clips_root}: {missing[:5]}")

    report = {"features": str(path.resolve()), "clips_root": os.path.abspath(clips_root), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "n_windows": int(len(anchors)), "n_clips": int(len(uniq)), "horizon": horizon, "history": history,
              "sign_constants": {"D_YAW_SIGN": D_YAW_SIGN, "YAW_RATE_SIGN": YAW_RATE_SIGN},
              "plausible_mask": bool(plausible), "plausible_ranges": (dict(PLAUSIBLE_RANGES) if plausible else None),
              "changed_targets": changed, "unchanged": {}, "changed": {}, "dt": {}}
    problems = []
    for i, t in enumerate(TARGET_NAMES):
        c = compare_columns(old_targets[:, i], new_targets[:, i])
        if t in changed:
            o, n = old_targets[:, i].astype(np.float64), new_targets[:, i].astype(np.float64)
            both = np.isfinite(o) & np.isfinite(n)
            nz = both & (o != 0) & (n != 0)
            c.update({
                "corr_old_new": float(np.corrcoef(o[both], n[both])[0, 1]) if both.sum() > 2 else float("nan"),
                "frac_sign_flipped": float(np.mean(np.sign(o[nz]) == -np.sign(n[nz]))) if nz.any() else float("nan"),
                "exact_negation": bool(both.any() and np.array_equal(old_targets[:, i][both], -new_targets[:, i][both])),
                "old_mean": float(o[both].mean()), "old_std": float(o[both].std()), "new_mean": float(n[both].mean()),
                "new_std": float(n[both].std()),
                "n_newly_nan": int(np.sum(np.isfinite(o) & ~np.isfinite(n))),  # windows dropped by the plausibility mask
            })
            report["changed"][t] = c
        else:
            report["unchanged"][t] = c
            if not c["bit_identical"]:
                problems.append(f"{t}: max abs diff {c['max_abs_diff']:.3g}, nan pattern identical {c['nan_pattern_identical']}")
    for k, old in old_dt.items():
        c = compare_columns(old, new_dt[k])
        report["dt"][k] = c
        if not c["bit_identical"]:
            problems.append(f"{k}: max abs diff {c['max_abs_diff']:.3g}, nan pattern identical {c['nan_pattern_identical']}")
    if problems:
        raise AssertionError("retarget: targets that must not change differ from the stored ones -- " + "; ".join(problems))
    report["max_abs_diff_unchanged"] = float(max([c["max_abs_diff"] for c in report["unchanged"].values()]
                                                 + [c["max_abs_diff"] for c in report["dt"].values()] + [0.0]))

    if backup:
        bp = pathlib.Path(backup)
        bp.parent.mkdir(parents=True, exist_ok=True)
        with open(bp, "wb") as fh:
            np.savez(fh, targets=old_targets, target_names=np.array(TARGET_NAMES), clip_id=arrays["clip_id"], anchor=arrays["anchor"],
                     meta_json=np.array(json.dumps(meta)), **old_dt)
        report["backup"] = str(bp.resolve())
    arrays["targets"] = new_targets
    for k, v in new_dt.items():
        arrays[k] = v
    meta = dict(meta)
    history_block = list(meta.get("retarget_history", [])) + ([meta["retarget"]] if meta.get("retarget") else [])
    meta.update({
        "retargeted": True, "retargeted_at": report["date"],
        "retarget": {"date": report["date"], "clips_root": report["clips_root"], "D_YAW_SIGN": D_YAW_SIGN, "YAW_RATE_SIGN": YAW_RATE_SIGN,
                     "changed_targets": changed, "target_definitions": dict(TARGET_DESCRIPTIONS),
                     "plausible_mask": bool(plausible), "plausible_ranges": (dict(PLAUSIBLE_RANGES) if plausible else None),
                     "max_abs_diff_unchanged": report["max_abs_diff_unchanged"],
                     "changed": {t: {k: c[k] for k in ("max_abs_diff", "corr_old_new", "frac_sign_flipped", "exact_negation", "old_std", "new_std", "n_newly_nan")}
                                 for t, c in report["changed"].items()},
                     "note": "sidecar d_yaw / wdev_z are clockwise +; targets now use D_YAW_SIGN * d_yaw and YAW_RATE_SIGN * wdev_z (CCW)"},
    })
    if history_block:
        meta["retarget_history"] = history_block
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta_json=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=1))
    os.replace(meta_tmp, meta_path)
    report["meta_path"] = str(meta_path.resolve())
    if report_path:
        rp = pathlib.Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=1, default=_json_default))
        report["report_path"] = str(rp.resolve())
    logger.info("retargeted %s: %d windows / %d clips; unchanged targets max abs diff %.3g; changed: %s", path, len(anchors), len(uniq),
                report["max_abs_diff_unchanged"],
                {t: f"corr {c['corr_old_new']:+.3f}, max|d| {c['max_abs_diff']:.3g}" for t, c in report["changed"].items()})
    return report


def run_retarget(args) -> dict:
    changed = [t.strip() for t in str(args.changed).split(",") if t.strip()]
    return retarget_features(args.features, args.clips_root, changed=changed, backup=args.backup, report_path=args.report,
                             plausible=bool(getattr(args, "plausible", False)))


# ----------------------------------------------------------------------------------------------------------------
# Probes (numpy / sklearn)
# ----------------------------------------------------------------------------------------------------------------
def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _group_kfold(groups: np.ndarray, n_splits: int):
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    k = int(min(n_splits, n_groups))
    if k < 2:
        raise ValueError(f"need >= 2 groups for GroupKFold, got {n_groups}")
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def ridge_decompose(Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the Gram matrix of standardised features -> (evals >= 0, V).  It depends on the features
    and the fold only (not on the labels), so ``probe_regression`` can share it across targets and shuffled runs."""
    G = Xs.T @ Xs
    evals, V = np.linalg.eigh(G)
    return np.clip(evals, 0.0, None), V


def ridge_solve(evals: np.ndarray, V: np.ndarray, Xs: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ridge weights for every lambda from a Gram eigendecomposition: W [d, L] and intercept (= mean y)."""
    n = Xs.shape[0]
    yb = float(y.mean())
    Vb = V.T @ (Xs.T @ (y - yb))
    W = V @ (Vb[:, None] / (evals[:, None] + np.asarray(lambdas, dtype=np.float64)[None, :] * n))
    return W, yb


def ridge_path(Xs: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ridge on standardised X for every lambda at once: W [d, L] and intercept (= mean y).  The penalty is
    ``lambda * n * ||w||^2`` on the summed squared loss, i.e. ``lambda * ||w||^2`` on the per-sample mean loss."""
    evals, V = ridge_decompose(Xs)
    return ridge_solve(evals, V, Xs, y, lambdas)


class RidgeFoldCache:
    """Label-independent per-fold quantities of a ridge probe (training-fold mean/std, Gram eigendecompositions of
    the outer and inner training folds) shared by every probe that uses the same feature matrix and the same folds
    (same valid-window mask): all continuous targets, their shuffled controls and the slot probes.  The caller
    supplies a ``feature_key`` that identifies the feature matrix; the folds are recorded on first use and any later
    use with different folds raises."""

    def __init__(self):
        self.store: dict = {}

    def get(self, key, compute):
        if key not in self.store:
            self.store[key] = compute()
        return self.store[key]

    def check_folds(self, key, folds) -> None:
        sig = [np.asarray(te, dtype=np.int64) for _, te in folds]
        prev = self.store.get((key, "folds"))
        if prev is None:
            self.store[(key, "folds")] = sig
        elif len(prev) != len(sig) or any(a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(prev, sig)):
            raise ValueError(f"RidgeFoldCache key {key!r} reused with different folds")

    def __len__(self):
        return len(self.store)


def select_ridge_lambda(Xs: np.ndarray, y: np.ndarray, groups: np.ndarray, lambdas: np.ndarray, n_inner: int = 4,
                        cache: Optional[RidgeFoldCache] = None, feature_key=None) -> Tuple[int, np.ndarray]:
    """Inner GroupKFold CV on (already standardised) training data; returns (best index, pooled MSE per lambda).
    With ``cache``, the inner-fold Gram eigendecompositions are stored under ``feature_key``."""
    sse = np.zeros(len(lambdas))
    inner = _group_kfold(groups, n_inner)
    if cache is not None:
        cache.check_folds((feature_key, "inner"), inner)
    for j, (tr, va) in enumerate(inner):
        if cache is not None:
            evals, V = cache.get((feature_key, "inner", j), lambda: ridge_decompose(Xs[tr]))
        else:
            evals, V = ridge_decompose(Xs[tr])
        W, yb = ridge_solve(evals, V, Xs[tr], y[tr], lambdas)
        pred = Xs[va] @ W + yb  # [n_va, L]
        sse += ((pred - y[va][:, None]) ** 2).sum(axis=0)
    mse = sse / len(y)
    return int(np.argmin(mse)), mse


def probe_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lambdas: np.ndarray = RIDGE_LAMBDAS, n_outer: int = 5,
                     n_inner: int = 4, shuffle_train_labels: bool = False, rng: Optional[np.random.Generator] = None,
                     cache: Optional[RidgeFoldCache] = None, feature_key=None) -> dict:
    """Ridge probe with outer GroupKFold (groups intact) and inner CV for lambda.  Returns pooled held-out
    predictions (same order as X), constant baselines (train-fold mean / median), chosen lambdas and the folds.
    ``cache`` / ``feature_key`` share the label-independent fold decompositions between calls on the same X."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    pred = np.full(n, np.nan)
    base_mean = np.full(n, np.nan)
    base_median = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=np.int32)
    chosen, lam_idx = [], []
    folds = _group_kfold(groups, n_outer)
    if cache is not None:
        cache.check_folds(feature_key, folds)
    for k, (tr, te) in enumerate(folds):
        y_tr = y[tr]
        if shuffle_train_labels:
            y_tr = (rng or np.random.default_rng(0)).permutation(y_tr)
        if cache is not None:
            mu, sd = cache.get((feature_key, "std", k), lambda: standardize_fit(X[tr]))
        else:
            mu, sd = standardize_fit(X[tr])
        Xs_tr = (X[tr] - mu) / sd
        Xs_te = (X[te] - mu) / sd
        j, _ = select_ridge_lambda(Xs_tr, y_tr, groups[tr], lambdas, n_inner, cache=cache, feature_key=(feature_key, k))
        if cache is not None:
            evals, V = cache.get((feature_key, "outer", k), lambda: ridge_decompose(Xs_tr))
        else:
            evals, V = ridge_decompose(Xs_tr)
        W, yb = ridge_solve(evals, V, Xs_tr, y_tr, lambdas[j: j + 1])
        pred[te] = Xs_te @ W[:, 0] + yb
        base_mean[te] = y_tr.mean()
        base_median[te] = np.median(y_tr)
        fold_id[te] = k
        chosen.append(float(lambdas[j]))
        lam_idx.append(j)
    return {"pred": pred, "baseline_mean": base_mean, "baseline_median": base_median, "fold": fold_id,
            "lambdas": chosen, "lambda_at_grid_edge": [bool(j in (0, len(lambdas) - 1)) for j in lam_idx], "folds": folds}


def _logistic_fit(Xs: np.ndarray, yy: np.ndarray, C: float, max_iter: int, tol: float, clf=None):
    """L2 logistic regression (sklearn lbfgs, class_weight=None); passing ``clf`` warm-starts from its solution."""
    from sklearn.linear_model import LogisticRegression

    if clf is None:
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=max_iter, tol=tol, class_weight=None, warm_start=True)
    clf.set_params(C=float(C))
    return clf.fit(Xs, yy)


def _logistic_inner_fold(Xs_tr: np.ndarray, y_tr: np.ndarray, itr: np.ndarray, iva: np.ndarray, Cs, max_iter: int, tol: float):
    """One inner fold of the C search: warm-started path over increasing C -> (summed held-out log loss per C in the
    order of ``Cs``, fold size).  Pure function so it can run in a worker process."""
    from sklearn.metrics import log_loss

    Cs = np.asarray(Cs, dtype=np.float64)
    losses = np.zeros(len(Cs))
    clf = None
    for j in np.argsort(Cs):
        clf = _logistic_fit(Xs_tr[itr], y_tr[itr], Cs[j], max_iter, tol, clf)
        p = clf.predict_proba(Xs_tr[iva])[:, 1]
        losses[j] = log_loss(y_tr[iva], p, labels=[0, 1]) * len(iva)
    return losses, int(len(iva))


def _logistic_inner_cv(Xs_tr: np.ndarray, y_tr: np.ndarray, inner_folds, Cs, max_iter: int, tol: float, n_jobs: int = 1):
    """Pooled held-out log loss per C over the inner folds (folds with one class are skipped); ``n_jobs`` > 1 runs
    the folds in loky worker processes limited to one BLAS thread each (falls back to sequential on any failure)."""
    jobs = [(itr, iva) for itr, iva in inner_folds if len(np.unique(y_tr[itr])) >= 2]
    results = None
    if int(n_jobs) > 1 and len(jobs) > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=min(int(n_jobs), len(jobs)), backend="loky", inner_max_num_threads=1)(
                delayed(_logistic_inner_fold)(Xs_tr, y_tr, itr, iva, Cs, max_iter, tol) for itr, iva in jobs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parallel inner CV failed (%s: %s); running the inner folds sequentially", type(exc).__name__, exc)
            results = None
    if results is None:
        results = [_logistic_inner_fold(Xs_tr, y_tr, itr, iva, Cs, max_iter, tol) for itr, iva in jobs]
    losses = np.zeros(len(Cs))
    counts = 0
    for l, c in results:
        losses += l
        counts += c
    return losses, counts


def probe_logistic(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs: np.ndarray = LOGREG_CS, n_outer: int = 5, n_inner: int = 4,
                   shuffle_train_labels: bool = False, rng: Optional[np.random.Generator] = None, max_iter: int = 1000,
                   fixed_Cs: Optional[Sequence[float]] = None, n_jobs: int = 1, tol: float = 1e-4) -> dict:
    """L2 logistic regression (sklearn, class_weight=None) with outer GroupKFold and inner CV over C (pooled log
    loss).  Returns pooled held-out scores (decision function) and the training prevalence baseline.
    ``fixed_Cs`` (one per outer fold) skips the inner CV and uses that C -- the shuffled-label control reuses the C
    selected on the true labels of the same outer fold; ``n_jobs`` > 1 parallelises the inner folds over processes."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    Cs = np.asarray(Cs, dtype=np.float64)
    n = len(y)
    score = np.full(n, np.nan)
    base_prev = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=np.int32)
    chosen, c_idx = [], []
    folds = _group_kfold(groups, n_outer)
    if fixed_Cs is not None and len(fixed_Cs) != len(folds):
        raise ValueError(f"fixed_Cs must have one entry per outer fold ({len(folds)}), got {len(fixed_Cs)}")
    for k, (tr, te) in enumerate(folds):
        y_tr = y[tr]
        if shuffle_train_labels:
            y_tr = (rng or np.random.default_rng(0)).permutation(y_tr)
        mu, sd = standardize_fit(X[tr])
        Xs_tr = (X[tr] - mu) / sd
        Xs_te = (X[te] - mu) / sd
        base_prev[te] = y_tr.mean()
        fold_id[te] = k
        if len(np.unique(y_tr)) < 2:  # degenerate training fold: constant score
            score[te] = 0.0
            chosen.append(float("nan"))
            c_idx.append(-1)
            continue
        if fixed_Cs is not None and np.isfinite(fixed_Cs[k]):
            C = float(fixed_Cs[k])
            hits = np.flatnonzero(np.isclose(Cs, C, rtol=1e-9))
            j = int(hits[0]) if len(hits) else -1
        else:
            losses, counts = _logistic_inner_cv(Xs_tr, y_tr, _group_kfold(groups[tr], n_inner), Cs, max_iter, tol, n_jobs)
            j = int(np.argmin(losses / max(counts, 1))) if counts > 0 else len(Cs) // 2
            C = float(Cs[j])
        clf = _logistic_fit(Xs_tr, y_tr, C, max_iter, tol)
        score[te] = clf.decision_function(Xs_te)
        chosen.append(C)
        c_idx.append(j)
    return {"score": score, "baseline_prevalence": base_prev, "fold": fold_id, "Cs": chosen,
            "C_at_grid_edge": [bool(j in (0, len(Cs) - 1)) for j in c_idx], "folds": folds}


def _shuffled_logistic_job(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, C: float, seed: int, max_iter: int, tol: float):
    """One (permutation, outer fold) control fit: training labels permuted with ``seed``, features standardised on the
    training fold, C fixed -> held-out decision scores (zeros for a degenerate fold).  Pure function (worker-safe)."""
    y_tr = np.random.default_rng(seed).permutation(y[tr])
    if len(np.unique(y_tr)) < 2 or not np.isfinite(C):
        return np.zeros(len(te))
    mu, sd = standardize_fit(X[tr])
    clf = _logistic_fit((X[tr] - mu) / sd, y_tr, float(C), max_iter, tol)
    return clf.decision_function((X[te] - mu) / sd)


def shuffled_logistic_controls(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs_per_fold: Sequence[float], n_perm: int,
                               rng: np.random.Generator, n_outer: int = 5, max_iter: int = 1000, tol: float = 1e-4,
                               n_jobs: int = 1) -> np.ndarray:
    """``n_perm`` label-permutation controls of the logistic probe: for every outer fold the training labels are
    permuted across windows (fresh permutation per control), C is the value selected on the true labels of that fold
    and the model is scored on the true held-out windows.  Returns pooled held-out scores ``[n_perm, n]``.  The
    (permutation, fold) fits are independent and run in up to ``n_jobs`` loky workers (1 BLAS thread each); the
    permutation seeds are drawn from ``rng`` up front, so the result does not depend on ``n_jobs``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    folds = _group_kfold(groups, n_outer)
    if len(Cs_per_fold) != len(folds):
        raise ValueError(f"Cs_per_fold must have one entry per outer fold ({len(folds)}), got {len(Cs_per_fold)}")
    n_perm = int(n_perm)
    seeds = rng.integers(0, 2**31 - 1, size=(n_perm, len(folds)))
    jobs = [(p, k) for p in range(n_perm) for k in range(len(folds))]
    results = None
    if int(n_jobs) > 1 and len(jobs) > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=min(int(n_jobs), len(jobs)), backend="loky", inner_max_num_threads=1)(
                delayed(_shuffled_logistic_job)(X, y, folds[k][0], folds[k][1], Cs_per_fold[k], int(seeds[p, k]), max_iter, tol)
                for p, k in jobs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parallel permutation controls failed (%s: %s); running them sequentially", type(exc).__name__, exc)
            results = None
    if results is None:
        results = [_shuffled_logistic_job(X, y, folds[k][0], folds[k][1], Cs_per_fold[k], int(seeds[p, k]), max_iter, tol) for p, k in jobs]
    scores = np.full((n_perm, len(y)), np.nan)
    for (p, k), s in zip(jobs, results):
        scores[p, folds[k][1]] = s
    return scores


def summarize_null(metric: str, values: Sequence[float]) -> dict:
    """Permutation-null summary: ``{metric: mean, metric_sd, metric_range: [min, max], metric_per_perm}`` (NaN-safe)."""
    v = np.asarray(values, dtype=np.float64)
    ok = v[np.isfinite(v)]
    if len(ok) == 0:
        return {metric: float("nan"), f"{metric}_sd": float("nan"), f"{metric}_range": [float("nan"), float("nan")], f"{metric}_per_perm": v.tolist()}
    return {metric: float(ok.mean()), f"{metric}_sd": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0,
            f"{metric}_range": [float(ok.min()), float(ok.max())], f"{metric}_per_perm": v.tolist()}


# -- metrics --------------------------------------------------------------------------------------------------------
def r2_score_pooled(y: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


METRICS = {"r2": r2_score_pooled, "mae": mae, "auroc": auroc, "auprc": auprc}


def paired_clip_bootstrap(groups: np.ndarray, y: np.ndarray, preds: Dict[str, np.ndarray], metric: str, n_boot: int = 1000,
                          seed: int = 0, diffs: Sequence[Tuple[str, str]] = (("f8_mean", "f2_mean"), ("f8_last", "f2_mean"))) -> dict:
    """Resample clips with replacement (same draw for every prediction set), recompute the pooled metric from the
    stored held-out predictions.  Returns point estimates + 95 % percentile CIs for each set and for the paired
    differences ``a - b``."""
    fn = METRICS[metric]
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_lists = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    names = list(preds)
    point = {k: fn(y, preds[k]) for k in names}
    rng = np.random.default_rng(seed)
    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        sel = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_lists[i] for i in sel])
        yy = y[idx]
        for k in names:
            boot[k][b] = fn(yy, preds[k][idx])
    out = {"metric": metric, "n_boot": n_boot, "n_clips": int(len(uniq)), "estimates": {}, "differences": {}}
    for k in names:
        bk = boot[k]
        ok = np.isfinite(bk)
        out["estimates"][k] = {"value": point[k], "ci95": [float(np.percentile(bk[ok], 2.5)), float(np.percentile(bk[ok], 97.5))]
                               if ok.any() else [float("nan"), float("nan")]}
    for a, b_ in diffs:
        if a in boot and b_ in boot:
            d = boot[a] - boot[b_]
            ok = np.isfinite(d)
            out["differences"][f"{a}-{b_}"] = {
                "value": point[a] - point[b_],
                "ci95": [float(np.percentile(d[ok], 2.5)), float(np.percentile(d[ok], 97.5))] if ok.any() else [float("nan")] * 2,
                "p_boot_le0": float(np.mean(d[ok] <= 0)) if ok.any() else float("nan"),
            }
    return out


# ----------------------------------------------------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------------------------------------------------
def _fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _fmt_ci(est: dict, nd=3):
    lo, hi = est["ci95"]
    return f"{_fmt(est['value'], nd)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def run_probe(args) -> dict:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(int(args.threads))
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    z = np.load(args.features, allow_pickle=False)
    feats = {k: np.asarray(z[k], dtype=np.float32) for k in FEATURE_VARIANTS}
    slots = np.asarray(z["f8_slots"], dtype=np.float32) if (args.slots and "f8_slots" in z.files) else None
    targets = np.asarray(z["targets"], dtype=np.float64)
    tnames = [str(s) for s in z["target_names"]]
    clip_ids = np.asarray(z["clip_id"]).astype(str)
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    n_all = len(clip_ids)
    logger.info("%d windows, %d clips, features %s", n_all, len(np.unique(clip_ids)), {k: v.shape for k, v in feats.items()})
    for k, v in feats.items():
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite values in {k}")

    results = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "features": os.path.abspath(args.features), "extract_meta": meta,
        "protocol": {
            "outer_cv": f"{args.outer_folds}-fold GroupKFold, groups = clip_id", "inner_cv": f"{args.inner_folds}-fold GroupKFold on the training folds",
            "ridge": "closed form on standardised features (mean/std from the training folds); penalty lambda*||w||^2 on the "
                     "per-sample mean squared loss (sklearn alpha = lambda*n_train); lambda grid " + ", ".join(f"{v:.3g}" for v in RIDGE_LAMBDAS),
            "logistic": "sklearn LogisticRegression(l2, lbfgs, class_weight=None), C by inner-CV pooled log loss, C grid "
                        + ", ".join(f"{v:.3g}" for v in LOGREG_CS),
            "metrics": "pooled held-out predictions: R^2 and MAE (continuous), AUROC and AUPRC (binary)",
            "baseline": "constant: training-fold mean (R^2) / median (MAE); prevalence (AUROC = 0.5, AUPRC = prevalence)",
            "bootstrap": f"paired over clips, {args.n_boot} draws, seed {args.seed}, 95 % percentile CIs, same draws for all variants and differences",
            "shuffled_control": f"{int(getattr(args, 'n_perm', 8) or 1)} independent label permutations per feature variant: training labels permuted across "
                                "windows within each training fold, evaluated on the true held-out labels; ridge re-selects lambda by inner CV on the "
                                "permuted labels, logistic reuses the C selected on the true labels of the same outer fold; reported as mean / sd / "
                                "[min, max] over the permutations (the held-out labels keep their clip clustering, so the null width is set by the "
                                "number of clips, resp. positive clips, not by the number of windows)",
            "compute": "ridge Gram eigendecompositions are label-independent and shared across targets / controls / slots (RidgeFoldCache); "
                       f"logistic inner folds run in {getattr(args, 'n_jobs', 1)} worker process(es) with 1 BLAS thread each",
        },
        "n_windows_total": n_all, "n_clips_total": int(len(np.unique(clip_ids))), "targets": {},
    }
    if slots is not None:
        results["slot_analysis"] = {"description": "ridge R^2 (same protocol, no bootstrap) of each single slot mean of the 8f pass; "
                                                   "slot 3 = frames a-1, a (the 2f frames), slot 0 = frames a-7, a-6", "targets": {}}
    rng_shuffle = np.random.default_rng(args.seed + 1)
    ridge_cache = RidgeFoldCache()
    n_jobs = int(getattr(args, "n_jobs", 1) or 1)
    n_perm = max(1, int(getattr(args, "n_perm", 8) or 1))
    sel = getattr(args, "targets", None)
    wanted = [t.strip() for t in sel.split(",") if t.strip()] if sel else list(tnames)
    unknown = [t for t in wanted if t not in tnames]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; available: {tnames}")
    for ti, tname in enumerate(tnames):
        if tname not in wanted:
            continue
        t_target = time.time()
        y_all = targets[:, ti]
        ok = np.isfinite(y_all)
        y = y_all[ok]
        g = clip_ids[ok]
        n = int(ok.sum())
        ok_key = hashlib.sha1(ok.tobytes()).hexdigest()[:12]  # same valid rows -> same folds -> cache hit
        is_binary = tname in BINARY_TARGETS
        entry = {"n_windows": n, "n_clips": int(len(np.unique(g))), "unit": TARGET_UNITS.get(tname, ""),
                 "definition": TARGET_DESCRIPTIONS.get(tname, ""), "kind": "binary" if is_binary else "continuous"}
        if n < 20 or len(np.unique(g)) < 2:
            entry["error"] = "too few valid windows"
            results["targets"][tname] = entry
            continue
        if is_binary:
            prev = float(y.mean())
            entry["distribution"] = {"prevalence": prev, "n_pos": int(y.sum()), "n_pos_clips": int(len(np.unique(g[y > 0.5])))}
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                entry["error"] = "too few positives/negatives"
                results["targets"][tname] = entry
                continue
            scores, shuffled, chosen = {}, {}, {}
            for fv in FEATURE_VARIANTS:
                t1 = time.time()
                r = probe_logistic(feats[fv][ok], y, g, LOGREG_CS, args.outer_folds, args.inner_folds, n_jobs=n_jobs)
                scores[fv] = r["score"]
                chosen[fv] = {"C_per_fold": r["Cs"], "C_at_grid_edge": r["C_at_grid_edge"]}
                null_scores = shuffled_logistic_controls(feats[fv][ok], y, g, r["Cs"], n_perm, rng_shuffle, args.outer_folds, n_jobs=n_jobs)
                shuffled[fv] = {"n_perm": n_perm, "C_per_fold": list(r["Cs"]),
                                **summarize_null("auroc", [auroc(y, s) for s in null_scores]),
                                **summarize_null("auprc", [auprc(y, s) for s in null_scores])}
                logger.info("%s / %s: AUROC %.3f AUPRC %.3f (shuffled AUROC %.3f [%.3f, %.3f], AUPRC %.3f [%.3f, %.3f] over %d perms) C=%s  %.0fs",
                            tname, fv, auroc(y, scores[fv]), auprc(y, scores[fv]), shuffled[fv]["auroc"], *shuffled[fv]["auroc_range"],
                            shuffled[fv]["auprc"], *shuffled[fv]["auprc_range"], n_perm, r["Cs"], time.time() - t1)
            base = r["baseline_prevalence"]
            entry["baseline"] = {"auroc": 0.5, "auprc": prev, "note": "constant score; AUPRC of a constant = prevalence"}
            entry["bootstrap"] = {m: paired_clip_bootstrap(g, y, scores, m, args.n_boot, args.seed) for m in ("auroc", "auprc")}
            entry["shuffled_control"] = shuffled
            entry["regularisation"] = chosen
            entry["held_out_scores"] = {fv: scores[fv].astype(np.float32).tolist() for fv in FEATURE_VARIANTS} if args.save_predictions else None
        else:
            entry["distribution"] = {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max()),
                                     "p05": float(np.percentile(y, 5)), "p95": float(np.percentile(y, 95))}
            preds, shuffled, chosen = {}, {}, {}
            for fv in FEATURE_VARIANTS:
                t1 = time.time()
                fkey = f"{fv}@{ok_key}"
                r = probe_regression(feats[fv][ok], y, g, RIDGE_LAMBDAS, args.outer_folds, args.inner_folds, cache=ridge_cache, feature_key=fkey)
                preds[fv] = r["pred"]
                chosen[fv] = {"lambda_per_fold": r["lambdas"], "lambda_at_grid_edge": r["lambda_at_grid_edge"]}
                null_r2, null_mae, null_lams = [], [], []
                for _ in range(n_perm):
                    rs = probe_regression(feats[fv][ok], y, g, RIDGE_LAMBDAS, args.outer_folds, args.inner_folds, shuffle_train_labels=True,
                                          rng=rng_shuffle, cache=ridge_cache, feature_key=fkey)
                    null_r2.append(r2_score_pooled(y, rs["pred"]))
                    null_mae.append(mae(y, rs["pred"]))
                    null_lams.append(list(rs["lambdas"]))
                shuffled[fv] = {"n_perm": n_perm, **summarize_null("r2", null_r2), **summarize_null("mae", null_mae),
                                "lambda_per_fold_per_perm": null_lams}
                logger.info("%s / %s: R2 %.3f MAE %.3f (shuffled R2 %.3f [%.3f, %.3f] over %d perms) lambda=%s  %.0fs", tname, fv,
                            r2_score_pooled(y, preds[fv]), mae(y, preds[fv]), shuffled[fv]["r2"], *shuffled[fv]["r2_range"], n_perm,
                            r["lambdas"], time.time() - t1)
            preds_b = dict(preds)
            preds_b["baseline_mean"] = r["baseline_mean"]
            preds_b["baseline_median"] = r["baseline_median"]
            entry["baseline"] = {"r2_mean_predictor": r2_score_pooled(y, r["baseline_mean"]), "mae_mean_predictor": mae(y, r["baseline_mean"]),
                                 "mae_median_predictor": mae(y, r["baseline_median"])}
            entry["bootstrap"] = {m: paired_clip_bootstrap(g, y, preds_b, m, args.n_boot, args.seed) for m in ("r2", "mae")}
            entry["shuffled_control"] = shuffled
            entry["regularisation"] = chosen
            entry["held_out_predictions"] = {fv: preds[fv].astype(np.float32).tolist() for fv in FEATURE_VARIANTS} if args.save_predictions else None
            if slots is not None:
                sa = {}
                for s in range(slots.shape[1]):
                    r = probe_regression(slots[ok, s], y, g, RIDGE_LAMBDAS, args.outer_folds, args.inner_folds, cache=ridge_cache,
                                         feature_key=f"slot{s}@{ok_key}")
                    sa[f"slot{s}"] = {"r2": r2_score_pooled(y, r["pred"]), "mae": mae(y, r["pred"])}
                results["slot_analysis"]["targets"][tname] = sa
        entry["probe_time_s"] = time.time() - t_target
        results["targets"][tname] = entry
        logger.info("target %s done in %.0fs (cache entries %d)", tname, entry["probe_time_s"], len(ridge_cache))
    results["probe_wall_time_s"] = time.time() - t0
    results["probe_calls"] = [{"created": results["created"], "targets": wanted, "wall_time_s": results["probe_wall_time_s"],
                               "n_boot": args.n_boot, "seed": args.seed, "n_perm": n_perm}]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    if getattr(args, "merge", False) and res_path.exists():
        results = merge_results(json.loads(res_path.read_text()), results, tnames)
    res_path.write_text(json.dumps(results, indent=1, default=_json_default))
    (out_dir / "table.md").write_text(render_table(results))
    logger.info("wrote %s and %s in %.0fs", out_dir / "results.json", out_dir / "table.md", results["probe_wall_time_s"])
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def merge_results(prev: dict, new: dict, target_order: Sequence[str]) -> dict:
    """``probe --merge``: keep the targets of an earlier results.json that this call did not recompute (so the probe
    can be run in several shorter calls), in canonical target order; wall time and calls are accumulated."""
    out = dict(new)
    targets = dict(prev.get("targets", {}))
    targets.update(new.get("targets", {}))
    out["targets"] = {t: targets[t] for t in target_order if t in targets}
    if "slot_analysis" in prev or "slot_analysis" in new:
        sa = dict(new.get("slot_analysis") or prev.get("slot_analysis") or {})
        merged = dict((prev.get("slot_analysis") or {}).get("targets", {}))
        merged.update((new.get("slot_analysis") or {}).get("targets", {}))
        sa["targets"] = {t: merged[t] for t in target_order if t in merged}
        out["slot_analysis"] = sa
    out["probe_calls"] = list(prev.get("probe_calls", [])) + list(new.get("probe_calls", []))
    out["probe_wall_time_s"] = float(prev.get("probe_wall_time_s", 0.0)) + float(new.get("probe_wall_time_s", 0.0))
    return out


def render_table(results: dict) -> str:
    """Markdown tables: continuous targets (R^2, then MAE) and binary targets (AUROC, then AUPRC)."""
    lines = []
    meta = results.get("extract_meta", {})
    lines.append(f"# D2 info gap: 8-frame (3.5 s) vs 2-frame (0.5 s) frozen-feature probes\n")
    lines.append(f"Windows: {results['n_windows_total']} from {results['n_clips_total']} clips"
                 + (f" (stride {meta.get('stride')}, devices {meta.get('devices')})" if meta else "") + ". "
                 "Values are pooled held-out metrics over 5-fold GroupKFold by clip; brackets are 95 % paired clip-bootstrap CIs. "
                 "Gaps are f8_mean - f2_mean and f8_last - f2_mean (same bootstrap draws). "
                 "Shuffled = training labels permuted (control).\n")
    cont = [t for t in results["targets"] if results["targets"][t].get("kind") == "continuous" and "bootstrap" in results["targets"][t]]
    binr = [t for t in results["targets"] if results["targets"][t].get("kind") == "binary" and "bootstrap" in results["targets"][t]]
    if cont:
        lines.append("## Continuous targets: R^2 (higher is better)\n")
        lines.append("| target | n_windows | mean +- std | baseline R^2 | f2_mean | f8_mean | f8_last | gap f8_mean-f2 | gap f8_last-f2 | shuffled R^2, mean over perms (f2 / f8m / f8l) |")
        lines.append("|---|---:|---|---:|---|---|---|---|---|---|")
        for t in cont:
            e = results["targets"][t]
            b = e["bootstrap"]["r2"]
            d = e["distribution"]
            sh = e["shuffled_control"]
            lines.append(f"| {t} [{e['unit']}] | {e['n_windows']} | {_fmt(d['mean'])} +- {_fmt(d['std'])} | {_fmt(e['baseline']['r2_mean_predictor'])} | "
                         f"{_fmt_ci(b['estimates']['f2_mean'])} | {_fmt_ci(b['estimates']['f8_mean'])} | {_fmt_ci(b['estimates']['f8_last'])} | "
                         f"{_fmt_ci(b['differences']['f8_mean-f2_mean'])} | {_fmt_ci(b['differences']['f8_last-f2_mean'])} | "
                         f"{_fmt(sh['f2_mean']['r2'])} / {_fmt(sh['f8_mean']['r2'])} / {_fmt(sh['f8_last']['r2'])} |")
        lines.append("")
        lines.append("## Continuous targets: MAE (lower is better; gap = f8 - f2, negative = 8f better)\n")
        lines.append("| target | baseline MAE (median) | f2_mean | f8_mean | f8_last | gap f8_mean-f2 | gap f8_last-f2 |")
        lines.append("|---|---:|---|---|---|---|---|")
        for t in cont:
            e = results["targets"][t]
            b = e["bootstrap"]["mae"]
            lines.append(f"| {t} [{e['unit']}] | {_fmt(e['baseline']['mae_median_predictor'])} | {_fmt_ci(b['estimates']['f2_mean'])} | "
                         f"{_fmt_ci(b['estimates']['f8_mean'])} | {_fmt_ci(b['estimates']['f8_last'])} | "
                         f"{_fmt_ci(b['differences']['f8_mean-f2_mean'])} | {_fmt_ci(b['differences']['f8_last-f2_mean'])} |")
        lines.append("")
    if binr:
        for m, label in (("auroc", "AUROC (baseline 0.5)"), ("auprc", "AUPRC (baseline = prevalence)")):
            lines.append(f"## Binary targets: {label}\n")
            lines.append(f"| target | n_windows | prevalence | baseline | f2_mean | f8_mean | f8_last | gap f8_mean-f2 | gap f8_last-f2 | shuffled {m}, mean over perms (f2 / f8m / f8l) |")
            lines.append("|---|---:|---:|---:|---|---|---|---|---|---|")
            for t in binr:
                e = results["targets"][t]
                b = e["bootstrap"][m]
                sh = e["shuffled_control"]
                lines.append(f"| {t} | {e['n_windows']} | {_fmt(e['distribution']['prevalence'])} | {_fmt(e['baseline'][m])} | "
                             f"{_fmt_ci(b['estimates']['f2_mean'])} | {_fmt_ci(b['estimates']['f8_mean'])} | {_fmt_ci(b['estimates']['f8_last'])} | "
                             f"{_fmt_ci(b['differences']['f8_mean-f2_mean'])} | {_fmt_ci(b['differences']['f8_last-f2_mean'])} | "
                             f"{_fmt(sh['f2_mean'][m])} / {_fmt(sh['f8_mean'][m])} / {_fmt(sh['f8_last'][m])} |")
            lines.append("")
    skipped = [t for t in results["targets"] if "error" in results["targets"][t]]
    if skipped:
        lines.append("Targets without a probe: " + ", ".join(f"{t} ({results['targets'][t]['error']}, n={results['targets'][t]['n_windows']})" for t in skipped) + "\n")
    null_tables = {"continuous": (("r2", "R^2"), ("mae", "MAE")), "binary": (("auroc", "AUROC"), ("auprc", "AUPRC"))}
    for kind, metrics in null_tables.items():
        rows = []
        for t in results["targets"]:
            e = results["targets"][t]
            if e.get("kind") != kind or not e.get("shuffled_control"):
                continue
            for fv, s in e["shuffled_control"].items():
                cells = []
                for m, _ in metrics:
                    if f"{m}_range" in s:
                        cells.append(f"{_fmt(s[m])} +- {_fmt(s.get(m + '_sd'))} [{_fmt(s[m + '_range'][0])}, {_fmt(s[m + '_range'][1])}]")
                    else:
                        cells.append(_fmt(s.get(m)))
                extra = f" (prev. {_fmt(e['distribution']['prevalence'])}, {e['distribution'].get('n_pos_clips', '?')} pos. clips)" if kind == "binary" else ""
                rows.append(f"| {t}{extra} | {fv} | {s.get('n_perm', 1)} | " + " | ".join(cells) + " |")
        if rows:
            lines.append(f"## Shuffled-label control, {kind} targets: null over permutations (mean +- sd [min, max])\n")
            lines.append("Training labels permuted across windows within each training fold, scored on the true held-out labels. The held-out "
                         "labels keep their clip clustering, so the width of this null is governed by the number of clips (and of positive "
                         "clips), not by the number of windows; a real effect must clear the [min, max] range, not just 0 / 0.5 / prevalence.\n")
            lines.append("| target | features | n_perm | " + " | ".join(lbl for _, lbl in metrics) + " |")
            lines.append("|---|---|---:|---|---|")
            lines.extend(rows)
            lines.append("")
    sa = results.get("slot_analysis")
    if sa and sa.get("targets"):
        lines.append("## Per-slot analysis: ridge R^2 of single slot means of the 8f pass (slot 3 = the 2f frames with context)\n")
        slots = sorted({s for v in sa["targets"].values() for s in v})
        lines.append("| target | " + " | ".join(slots) + " |")
        lines.append("|---|" + "---:|" * len(slots))
        for t, v in sa["targets"].items():
            lines.append(f"| {t} | " + " | ".join(_fmt(v[s]["r2"]) for s in slots) + " |")
        lines.append("")
    reg_lines = []
    for t, e in results["targets"].items():
        if "regularisation" in e:
            for fv, r in e["regularisation"].items():
                key = "lambda_per_fold" if "lambda_per_fold" in r else "C_per_fold"
                edge = r.get("lambda_at_grid_edge", r.get("C_at_grid_edge", []))
                reg_lines.append(f"| {t} | {fv} | {', '.join(f'{v:.3g}' for v in r[key])} | {sum(bool(x) for x in edge)}/{len(edge)} |")
    if reg_lines:
        lines.append("## Chosen regularisation per outer fold (inner CV)\n")
        lines.append("| target | features | lambda or C per fold | folds at grid edge |")
        lines.append("|---|---|---|---:|")
        lines.extend(reg_lines)
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# check-t2: deployed construction vs this construction for 2-frame inputs
# ----------------------------------------------------------------------------------------------------------------
def build_deployed_style_encoder(ckpt_path: str, device):
    """The deployed agent's construction (navsim drive_jepa_model.py -> vjepa2 evals/.../vit_encoder.init_module):
    ``vit_large(input_size=(256, 512), num_frames=2, patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True)``
    with the target_encoder weights after replacing 'module.' and 'backbone.'.  (``input_size`` is not a
    VisionTransformer argument and is ignored there; img_size stays 224, which only affects the unused sincos
    pos_embed / grid_size fallbacks because RoPE positions are derived from the real H/W patches at forward time.)"""
    import torch
    import vjepa2.src.models.vision_transformer as vit

    model = vit.vit_large(input_size=IMG_HW, num_frames=2, patch_size=PATCH, tubelet_size=TUBELET, uniform_power=True, use_rope=True,
                          use_activation_checkpointing=False, img_temporal_dim_size=None, checkpoint_key="target_encoder", model_name="vit_large")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck["target_encoder"].items()}
    msg = model.load_state_dict(sd, strict=False)
    del ck
    model.to(device).eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def rope_positions(T: int, H_patches: int = IMG_HW[0] // PATCH, W_patches: int = IMG_HW[1] // PATCH) -> dict:
    """(frame, height, width) RoPE position ids the vendored RoPEAttention assigns for a [B, 3, T*tubelet, H, W] input."""
    import torch
    from vjepa2.src.models.utils.modules import RoPEAttention

    attn = RoPEAttention(dim=64, num_heads=1, grid_size=H_patches)
    ids = torch.arange(T * H_patches * W_patches)
    d, h, w = attn.separate_positions(ids, H_patches, W_patches)
    return {"T_slots": T, "frame_pos_unique": sorted(set(d.tolist())), "height_pos_range": [int(h.min()), int(h.max())],
            "width_pos_range": [int(w.min()), int(w.max())], "n_tokens": int(ids.numel())}


def batch_shape_dependence(backbone, frames_all: np.ndarray, device, n_windows: int = 8) -> dict:
    """Pooled-feature dependence on the batch shape and on the precision, measured on up to ``n_windows`` real windows
    of one clip (anchors as in ``extract``): the 2f pass at B = n_windows vs B = 1, the 8f pass at B = n_windows vs
    B = 1 (both for bf16 autocast and for fp32), and bf16 vs fp32 at the batched shape.  Relative L2 / cosine over the
    1024-d pooled vectors ``f2_mean``, ``f8_mean``, ``f8_last``."""
    import torch

    anchors = clip_anchors(int(frames_all.shape[0]), stride=8)[:n_windows]
    if len(anchors) == 0:
        return {"note": "clip too short for a window"}
    fr = torch.from_numpy(np.ascontiguousarray(frames_all)).to(device)
    a = torch.from_numpy(anchors).to(device)
    win2 = fr[(a[:, None] + torch.arange(-(N_FRAMES_2F - 1), 1, device=device)[None, :])]
    win8 = fr[(a[:, None] + torch.arange(-HISTORY, 1, device=device)[None, :])]
    B = int(win2.shape[0])

    def pooled(x, use_bf16, bs):
        return torch.cat([pool_slots(encode_tokens(backbone, x[i: i + bs], use_bf16=use_bf16)) for i in range(0, x.shape[0], bs)], 0)

    def dev(p, q):
        p, q = p.double(), q.double()
        rel_l2 = (p - q).norm(dim=-1) / q.norm(dim=-1)
        cosine = torch.nn.functional.cosine_similarity(p, q, dim=-1)
        return {"rel_l2_mean": float(rel_l2.mean()), "rel_l2_max": float(rel_l2.max()), "cosine_min": float(cosine.min())}

    feats = {}
    for prec, use in (("bf16", device.type == "cuda"), ("fp32", False)):
        f2_b, f2_1 = pooled(win2, use, B)[:, 0], pooled(win2, use, 1)[:, 0]
        s8_b, s8_1 = pooled(win8, use, B), pooled(win8, use, 1)
        feats[prec] = {"f2_mean": f2_b, "f8_mean": s8_b.mean(1), "f8_last": s8_b[:, -1]}
        feats[prec]["batch"] = {"f2_mean": dev(f2_1, f2_b), "f8_mean": dev(s8_1.mean(1), s8_b.mean(1)), "f8_last": dev(s8_1[:, -1], s8_b[:, -1])}
    out = {"n_windows": B, "anchors": anchors.tolist(), "tf32": bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
           "note": "batch: same windows at B = 1 vs B = n_windows (one forward); bf16_vs_fp32: batched bf16 autocast vs batched fp32 (TF32 off)"}
    for prec in ("bf16", "fp32"):
        out[f"{prec}_B1_vs_B{B}"] = feats[prec]["batch"]
    out["bf16_vs_fp32"] = {k: dev(feats["bf16"][k], feats["fp32"][k]) for k in ("f2_mean", "f8_mean", "f8_last")}
    return out


def run_check_t2(args) -> dict:
    import torch

    torch.set_num_threads(int(args.threads))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        set_strict_fp32()  # the fp32 reference below is the evidence protocol's fp32 (no TF32)
    enc8, rep8 = build_encoder(args.ckpt, device)
    dep2, rep2 = build_deployed_style_encoder(args.ckpt, device)
    paths = read_clip_list(args.list) if args.list else []
    frames = None
    decoded_path = None
    for p in paths[: 20]:
        if os.path.exists(sidecar_path(p)):
            try:
                frames, _ = decode_clip(p)
                decoded_path = p
                break
            except Exception:  # noqa: BLE001
                continue
    rng = np.random.default_rng(0)
    if frames is None or frames.shape[0] < 8:
        frames = rng.integers(0, 256, size=(8, IMG_HW[0], IMG_HW[1], 3), dtype=np.uint8)
        src = "random"
    else:
        src = decoded_path
    frames_all = frames
    frames = frames_all[20: 28] if frames_all.shape[0] >= 28 else frames_all[:8]  # 8 consecutive frames away from the clip start
    x8 = torch.from_numpy(np.ascontiguousarray(frames[:8]))[None].to(device)  # [1, 8, H, W, 3]
    x2 = x8[:, -2:]
    with torch.no_grad():
        tok2_ours = encode_tokens(enc8.backbone, x2, use_bf16=False)  # fp32 for an exact comparison
        tok2_dep = encode_tokens(dep2, x2, use_bf16=False)
        tok8_ours = encode_tokens(enc8.backbone, x8, use_bf16=False)
        tok2_bf16 = encode_tokens(enc8.backbone, x2, use_bf16=(device.type == "cuda"))
    last = tok8_ours[:, -TOKENS_PER_SLOT:]
    pooled_fp32, pooled_bf16 = tok2_ours.mean(1), tok2_bf16.mean(1)
    bf16_report = {
        "max_abs_token_diff": float((tok2_bf16 - tok2_ours).abs().max()),
        "token_rel_l2_mean": float(((tok2_bf16 - tok2_ours).norm(dim=-1) / tok2_ours.norm(dim=-1)).mean()),
        "token_cosine_mean": float(torch.nn.functional.cosine_similarity(tok2_bf16, tok2_ours, dim=-1).mean()),
        "pooled_mean_rel_l2": float(((pooled_bf16 - pooled_fp32).norm(dim=-1) / pooled_fp32.norm(dim=-1)).mean()),
        "pooled_mean_cosine": float(torch.nn.functional.cosine_similarity(pooled_bf16, pooled_fp32, dim=-1).mean()),
        "note": "deviation of a bf16-autocast forward from the fp32 (TF32 off) forward for one T=2 window; the evidence features are extracted in fp32, "
                "bf16 is only the fast option of `extract --precision bf16`",
    }
    cos = torch.nn.functional.cosine_similarity(last, tok2_ours, dim=-1)
    cos_mean_pooled = torch.nn.functional.cosine_similarity(last.mean(1), tok2_ours.mean(1), dim=-1)
    rel = ((last - tok2_ours).norm(dim=-1) / tok2_ours.norm(dim=-1))
    batch_dep = batch_shape_dependence(enc8.backbone, frames_all, device, n_windows=int(getattr(args, "n_windows", 8)))
    out = {
        "source_frames": src, "device": str(device),
        "ours": {"construction": "init_video_model(max_num_frames=8, use_rope=True, crop_size=(256,512)) + target_encoder (strip 'module.')", **{k: rep8[k] for k in ("missing", "unexpected")}},
        "deployed": {"construction": "vit_large(num_frames=2, use_rope=True) via evals/.../vit_encoder.init_module semantics + target_encoder (replace 'module.' and 'backbone.')", **rep2},
        "t2_tokens": int(tok2_ours.shape[1]), "t8_tokens": int(tok8_ours.shape[1]),
        "max_abs_diff_ours_vs_deployed_T2_fp32": float((tok2_ours - tok2_dep).abs().max()),
        "max_abs_token_fp32": float(tok2_ours.abs().max()),
        "max_abs_diff_bf16_vs_fp32_T2": float((tok2_bf16 - tok2_ours).abs().max()),
        "bf16_vs_fp32_T2": bf16_report,
        "pooled_feature_dependence": batch_dep,
        "context_effect_last_slot_vs_T2": {"token_cosine_mean": float(cos.mean()), "token_cosine_min": float(cos.min()),
                                           "token_rel_l2_mean": float(rel.mean()), "pooled_mean_cosine": float(cos_mean_pooled.mean())},
        "rope_positions": {"T2_input": rope_positions(1), "T8_input": rope_positions(4)},
        "notes": [
            "RoPE encodes positions as rotations of q/k, so attention depends only on relative (frame, row, col) offsets; a single tubelet gets frame position 0 for all tokens, identical to how the same tubelet is related to itself inside the 8-frame input.",
            "No absolute/learned positional parameters exist (pos_embed=None under use_rope), so num_frames (8 here vs 2 in the deployed init_module) does not change the forward for a T=2 input; it only sets the unused num_patches / grid_depth attributes.",
            "The deployed agent runs with double_image=true (scripts/evaluation/eval_drive_jepa_perception_free.sh, frontdrive e0 script): camera_feature = cat(frame t-0.5s, frame t) along the time axis, register_prehook=False, so the encoder really sees 2 distinct frames = 1 tubelet = 512 tokens.",
        ],
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    logger.info("check-t2: max|ours - deployed| = %.3g (tokens up to %.3g); last-slot-with-context vs T2 cosine %.3f",
                out["max_abs_diff_ours_vs_deployed_T2_fp32"], out["max_abs_token_fp32"], out["context_effect_last_slot_vs_T2"]["token_cosine_mean"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="clip list -> features npz")
    e.add_argument("--list", required=True, help="clip list, '<mp4> 0' per line")
    e.add_argument("--ckpt", required=True, help="e50 checkpoint (dict with target_encoder)")
    e.add_argument("--out", required=True, help="output .npz path")
    e.add_argument("--stride", type=int, default=8, help="anchor stride in kept frames (8 = 4 s)")
    e.add_argument("--max-clips", type=int, default=0, help="only the first N clips (0 = all)")
    e.add_argument("--batch", type=int, default=8, help="8-frame windows per forward pass (2-frame passes use 4x)")
    e.add_argument("--num-workers", type=int, default=4, help="decoding workers (<= 4 on the shared box)")
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--precision", choices=["bf16", "fp32"], default="fp32",
                   help="fp32 with TF32 disabled (default, the evidence protocol: batch-shape independent) or bf16 autocast (2-3x faster, "
                        "~1 %% pooled deviation that depends on the batch shape)")
    e.add_argument("--exclude-list", default=None, help="file with clip ids or mp4 paths to drop (e.g. the E2 continued-pretraining pool) "
                                                        "so the run is a held-out comparison; the ids are recorded in the meta")
    e.add_argument("--shard", default=None, help="'i/n': process only the i-th of n contiguous chunks of the (filtered) clip list; "
                                                 "join the outputs with merge-features")
    e.set_defaults(func=run_extract)
    m = sub.add_parser("merge-features", help="shard npz files from extract --shard -> one features npz")
    m.add_argument("--inputs", nargs="+", required=True, help="shard npz files (ordered by their shard index automatically)")
    m.add_argument("--out", required=True, help="merged .npz path")
    m.set_defaults(func=run_merge_features)
    r = sub.add_parser("retarget", help="recompute the targets of a features npz from the sidecars (features untouched)")
    r.add_argument("--features", required=True, help="features npz to retarget in place (atomic overwrite)")
    r.add_argument("--clips-root", default=DEFAULT_CLIPS_ROOT, help="root of the clips (<root>/<id[:2]>/<id>.can.npz layout)")
    r.add_argument("--changed", default=",".join(SIGN_FIX_CHANGED_TARGETS),
                   help="comma-separated targets whose definition changed ('all' = every target); every other target must come back bit-identical")
    r.add_argument("--plausible", action="store_true",
                   help="treat sidecar values outside PLAUSIBLE_RANGES as missing while recomputing (use with --changed all)")
    r.add_argument("--backup", default=None, help="optional npz that receives the previous targets / dt arrays")
    r.add_argument("--report", default=None, help="optional json path for the old-vs-new comparison report")
    r.set_defaults(func=run_retarget)
    q = sub.add_parser("probe", help="features npz -> results.json + table.md")
    q.add_argument("--features", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--n-boot", type=int, default=1000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--outer-folds", type=int, default=5)
    q.add_argument("--inner-folds", type=int, default=4)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--n-jobs", type=int, default=4, help="worker processes for the logistic inner CV (1 BLAS thread each; "
                                                        "1.5x faster than one 4-thread process on the shared box); 1 = sequential")
    q.add_argument("--targets", default=None, help="comma-separated subset of targets to probe (default: all)")
    q.add_argument("--merge", action="store_true", help="keep targets already present in <out-dir>/results.json that this call "
                                                       "does not recompute (run the probe in several shorter calls)")
    q.add_argument("--slots", action="store_true", help="also probe each single slot mean of the 8f pass (ridge R^2)")
    q.add_argument("--n-perm", type=int, default=8, help="label permutations per feature variant in the shuffled control (null mean / sd / "
                                                         "range; >= 5 so the clip-clustered null width is visible)")
    q.add_argument("--save-predictions", action="store_true", help="store pooled held-out predictions in results.json")
    q.set_defaults(func=run_probe)
    c = sub.add_parser("check-t2", help="deployed T=2 construction vs this construction")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--list", default=None, help="clip list; the first decodable clip provides the test frames")
    c.add_argument("--out", default=None, help="json output path")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--n-windows", type=int, default=8, help="windows of the test clip for the batch-shape / precision dependence")
    c.set_defaults(func=run_check_t2)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
