"""Ego-motion sidecars, camera-intrinsics bookkeeping and the two-view clip dataset for the
motion-constrained V-JEPA arm (plan v4.1 section 3.1: CAN-constrained cross-camera motion representation).

Everything here runs in DataLoader workers (numpy only); the heads live in ``pretrain/motion_heads.py``.

Conventions
-----------
Vehicle frame (comma/openpilot ``d_forward``/``d_left``/``d_yaw``): x forward [m], y left [m], yaw
counter-clockwise (left turn positive) [rad].  ``compose_se2`` and the ``dx``/``dy``/``dyaw`` targets are
expressed in the vehicle frame of the FIRST frame of the context slot (kept frame ``c_{tubelet*s}``).

Camera frame: x right, y down, z forward (OpenCV).  ``patch_center_rays`` returns unit vectors in it.

Pixel coordinates are continuous: pixel (row i, col j) covers [j, j+1) x [i, i+1), centre (j+0.5, i+0.5).
Under this convention a resize (W, H) -> (W', H') is exactly u' = u * W'/W (what
``F.interpolate(align_corners=False)`` implements) and a crop is a pure translation, so intrinsics update
by the affine map in ``crop_resize_intrinsics``.  Boxes are ``(x0, y0, w, h)`` in input pixels.

Token order: the vendored PatchEmbed3D flattens (slot, row, col) time-major, idx = slot*Hp*Wp + row*Wp + col;
all per-token masks here (``common_region_token_mask``) use the same row-major (row*Wp + col) order.

Yaw signs (sidecar -> this module).  The sidecar's ``d_yaw`` is the difference of the dead-reckoning heading
``psi``, which comma/openpilot express as a NED angle (clockwise-positive, z down), and ``wdev_z`` is the gyro
rate about the openpilot device-frame z axis, which also points DOWN; both are therefore right-turn-positive and
agree with each other (corr(wdev_z*dt, d_yaw) = +0.83 on the local clips), while ``d_left`` is left-positive.
Measured 2026-10-12 on 600 local sidecars (17,117 turning rows with |d_yaw| > 0.01 rad, tici/tizi/mici):
sign(d_left) == -sign(d_yaw) on 98.8 % of rows, corr(d_left, d_yaw) = -0.82, and the unicycle chord relation
holds as d_left = -d_forward*tan(d_yaw/2) (mean residual 0.012 m vs 0.237 m with the opposite sign).  This
module converts both to the CCW frame used everywhere else: ``dyaw = D_YAW_SIGN * d_yaw`` and
``yaw_rate = YAW_RATE_SIGN * wdev_z`` with ``D_YAW_SIGN = YAW_RATE_SIGN = -1``.  All ``dyaw``/``yaw_rate``
values emitted here (sequence columns and targets) are CCW (left turn positive) and consistent with ``d_left``.

Sidecar rows: one per kept 2 fps frame of the mp4 (row i <-> mp4 frame i).  ``d_*[i]`` is the motion from
kept frame i to kept frame i+1 in frame i's vehicle frame; the last row is NaN, as are rows where
dead-reckoning was invalid.  On most local clips row 0 of ``v_ego``/``a_ego``/``steer_deg`` is NaN.  One
local sidecar has fewer rows than the mp4 has frames; rows outside the sidecar are treated as missing.  A few
tizi sidecars have no ``wdev_z``/``yaw`` key at all: the yaw_rate column is then invalid (``seq_valid``) while
the other columns stay usable -- validity is tracked PER FEATURE, a row is masked only when every feature is
missing.  Kept-frame indices must be strictly increasing (the vendored sampler pads short videos by repeating
the last frame); a window with repeated frames gets no valid targets or sequence rows.

Public mixture sidecars (comma_export/docs/public_mixture_spec.md; ``device`` covla / dojo / openscene, written by
``comma_export/publicdata``) use the same keys and sign conventions (clockwise ``d_yaw`` / ``wdev_z``, left-positive
``d_left``; the exporters never pre-negate) and add ``K_clip`` (3x3 intrinsics of the exported 512x256 frames),
``fov_h_deg`` and ``src_fps``.  ``sidecar_intrinsics`` prefers a finite ``K_clip`` over the per-device fallback;
targets, sequence rows and validity are computed exactly as for comma clips (``steer_deg`` and the ADAS flags are NaN
there, so only the comma-only columns become invalid).  ``heading_src`` is never loaded, so its public values
(``pose``, ``ego2global``) need no handling.

Time: ``dt`` is the real ``t_sof`` difference (ns -> s) between the first frames of the two slots; when
``t_sof`` is missing or unusable it falls back to the nominal ``d * 1.0`` s (2 fps clips, tubelet 2).

Calibration: the sidecar's ``cal_pitch``/``cal_yaw`` are openpilot's ``liveCalibration.rpyCalib`` angles (euler
angles of ``device_from_calib``, roll unknown -> 0), i.e. the mounting of the device relative to the road
(calibrated) frame; they differ by several degrees between installs.  ``rays_to_calibrated_frame`` rotates the
per-token camera rays into the road-aligned camera frame (x right, y down, z = direction of travel) exactly as
openpilot's ``get_warp_matrix`` chain does, so that identical ego motion means identical image motion across
installs; the dataset applies the per-clip median calibration when ``rays_road_frame`` is on.

Units at the interface: everything this module emits is in RAW physical units -- the motion sequence ``seq``
(m/s, rad/s, m/s^2, m, m, rad, flags[, deg, flags]) as well as the targets (m, m, rad, m/s, rad/s, m/s, rad/s).
Input normalisation is the model's job: ``motion_heads.MotionSequenceEncoder`` divides the sequence features by
its fixed scales and ``motion_heads.readout_loss`` divides the targets by the target scales (per horizon, see
``DEFAULT_TARGET_SCALES`` / ``TARGET_SCALE_PER_SLOT``).  (Scaling on both sides would normalise twice; the data
side therefore never scales.)
"""
from __future__ import annotations

import logging
import math
import os
import pathlib
import zipfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

# Vendored V-JEPA 2 code (read-only).  ``vjepa2.*`` resolves through navsim_v1 on PYTHONPATH, ``app.*`` and
# ``src.*`` through navsim_v1/vjepa2 (see env/drivejepa_env.sh); app/vjepa/train.py uses the same mix.
from app.vjepa.transforms import _tensor_normalize_inplace
from vjepa2.src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from vjepa2.src.datasets.utils.video.transforms import _get_param_spatial_crop
from vjepa2.src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from vjepa2.src.datasets.video_dataset import VideoDataset

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------
# Camera geometry of the exported clips (commadata clips.py): full-res road camera -> 526x330 qcamera for every
# comma device (anisotropic for mici) -> rows 33:296 cropped -> resized to 512x256 (factor 512/526 on both axes).
# Public-mixture clips (comma_export/docs/public_mixture_spec.md: covla / dojo / openscene) are cropped from the full
# source frame and carry their exact intrinsics in the sidecar (``K_clip``, preferred by ``sidecar_intrinsics``); their
# DEVICE_CAMERAS rows are fallbacks only (``device_fallback_intrinsics``) for a public clip without a usable sidecar.
# --------------------------------------------------------------------------------------------------------------
QCAM_W, QCAM_H = 526, 330
CLIP_CROP_ROWS = (33, 296)
CLIP_W, CLIP_H = 512, 256
# device -> (full_w, full_h, focal_px); principal point at the image centre (openpilot common/transformations/camera.py)
DEVICE_CAMERAS: Dict[str, Tuple[int, int, float]] = {
    "tici": (1928, 1208, 2648.0),
    "tizi": (1928, 1208, 2648.0),
    "mici": (1344, 760, 1141.5),  # OS04C10 sensor assumption
    # public mixture (spec section 6), fallbacks only -- the sidecar's K_clip is authoritative
    "covla": (1928, 1208, 2648.0),  # comma three road camera (40 deg HFOV), exported as the centre 2:1 crop
    "dojo": (1920, 1080, 1005.7),  # DrivingDojo CameraFpgaP0H120; the export crops a 64 deg box, not the full width
    "openscene": (1920, 1080, 1545.0),  # nuPlan CAM_F0 undistorted model (principal point (960, 560), not centred)
}
PUBLIC_DEVICES = ("covla", "dojo", "openscene")  # sidecar ``device`` values written by comma_export/publicdata
# ids are consumed by the heads as an embedding index: existing comma ids must never change
DEVICE_IDS = {"": 0, "tici": 1, "tizi": 2, "mici": 3, "covla": 4, "dojo": 5, "openscene": 6}
# Sidecar yaw quantities are clockwise-positive (NED heading / device z down); -1 converts them to the CCW vehicle
# frame of d_left and compose_se2.  See the module docstring for the measurement on local clips.
D_YAW_SIGN = -1.0  # applied to d_yaw (sequence column and SE(2) composition)
YAW_RATE_SIGN = -1.0  # applied to wdev_z (yaw_rate column and yaw_rate_mean / dyaw_rate targets)
# openpilot common/transformations/camera.py: view (camera) frame x right, y down, z forward = device y, z, x
VIEW_FRAME_FROM_DEVICE_FRAME = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# Per-frame motion sequence fed to the heads (rows = kept frames, see motion_targets) and the readout targets.
MOTION_SEQ_FEATURES = ["v_ego", "yaw_rate", "a_ego", "d_forward", "d_left", "d_yaw", "standstill"]
COMMA_ONLY_FEATURES = ["steer_deg", "cc_enabled", "cc_lat_active", "cc_long_active"]
# Sequence features stay in raw units (m/s, rad/s, m/s^2, m, m, rad, flag[, deg, flags]); the per-feature input
# scales live with the consumer (motion_heads._SEQ_FEATURE_SCALES), see the module docstring.
MOTION_SEQ_MAX_LEN = 7  # default sequence length: tubelet*(n_slots-1)+1 kept frames for 8-frame clips (c_0 .. c_6)
# Readout targets.  dx/dy/dyaw: pose of the target slot's first frame in the context slot's vehicle frame (pair
# geometry); v_mean/yaw_rate_mean: window means (one-slot quantities x dt explain them almost entirely); dv/dyaw_rate:
# PAIR-SPECIFIC differences (target-slot tubelet mean minus context-slot tubelet mean) that no single tubelet gives --
# added so that the readout cannot be solved from the target-encoder tokens alone (review 2026-10-12).
MOTION_TARGETS = ["dx", "dy", "dyaw", "v_mean", "yaw_rate_mean", "dv", "dyaw_rate"]
# Normalisation scales = empirical spread on local sidecars with the CCW composition (2026-10-12, 400 clips, 2,400
# windows per horizon): dx std 10.9*d m, dy 0.28/1.09/2.34 m (d = 1/2/3), dyaw 0.073*d rad, v_mean 10.9 m/s, yaw_rate
# 0.07 rad/s, dv 0.6*d m/s, dyaw_rate 0.04-0.08 rad/s.  Entries in TARGET_SCALE_PER_SLOT are multiplied by the slot
# gap d (per-horizon scale), so every normalised target has std ~0.3-1 and the Huber loss is balanced across targets.
DEFAULT_TARGET_SCALES = {"dx": 10.0, "dy": 1.0, "dyaw": 0.15, "v_mean": 10.0, "yaw_rate_mean": 0.1, "dv": 1.0, "dyaw_rate": 0.07}
TARGET_SCALE_PER_SLOT = ("dx", "dy", "dyaw", "dv")  # scale grows linearly with the slot gap d
IMAGENET_NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ``heading_src`` (per-frame provenance of the heading: comma orientationNED / gyro_int / bicycle, public mixture pose /
# ego2global) is exporter QC documentation only: it is never loaded (object array in the comma files, needs pickle) and
# no value of it is interpreted anywhere in this module, so new sources need no consumer change.
_SIDECAR_SKIP_KEYS = ("heading_src",)


def motion_seq_len(frames_per_clip: int, tubelet: int = 2) -> int:
    """Rows of the motion sequence for a clip: kept frames c_0 .. c_{tubelet*(n_slots-1)} = tubelet*(n_slots-1)+1."""
    n_slots = int(frames_per_clip) // int(tubelet)
    return int(tubelet) * (n_slots - 1) + 1


# --------------------------------------------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------------------------------------------
def crop_resize_intrinsics(K: np.ndarray, box_xywh: Sequence[float], out_hw: Sequence[int]) -> np.ndarray:
    """K' for cropping ``box_xywh`` = (x0, y0, w, h) (input pixels) and resizing the crop to ``out_hw`` = (H, W).

    Continuous pixel convention: u' = (u - x0) * W/w, v' = (v - y0) * H/h.  Returns float32 (3, 3).
    """
    x0, y0, w, h = (float(v) for v in box_xywh)
    H, W = (float(v) for v in out_hw)
    sx, sy = W / w, H / h
    K = np.asarray(K, dtype=np.float64)
    out = np.array(
        [[K[0, 0] * sx, K[0, 1] * sx, (K[0, 2] - x0) * sx], [0.0, K[1, 1] * sy, (K[1, 2] - y0) * sy], [0.0, 0.0, 1.0]]
    )
    return out.astype(np.float32)


def _comma_clip_intrinsics(device: str) -> np.ndarray:
    """The commadata clips.py qcamera chain applied to the DEVICE_CAMERAS row of ``device`` (unknown -> tici)."""
    full_w, full_h, f = DEVICE_CAMERAS.get(str(device), DEVICE_CAMERAS["tici"])
    k_full = np.array([[f, 0.0, full_w / 2.0], [0.0, f, full_h / 2.0], [0.0, 0.0, 1.0]])
    k_q = crop_resize_intrinsics(k_full, (0, 0, full_w, full_h), (QCAM_H, QCAM_W))  # qcamera downscale
    r0, r1 = CLIP_CROP_ROWS
    return crop_resize_intrinsics(k_q, (0, r0, QCAM_W, r1 - r0), (CLIP_H, CLIP_W))  # clips.py crop + resize


def clip_intrinsics(device: str) -> np.ndarray:
    """Intrinsics (float32 3x3, pixels) of an exported 512x256 clip from the device string alone.

    comma devices ('tici'|'tizi'|'mici'; unknown or empty strings fall back to tici/tizi): the commadata clips.py
    qcamera chain -- tici fx~703.2 fy~704.1 cx=256.0 cy~128.5 (HFOV ~40 deg); mici fx~434.8 fy~482.5 (HFOV ~61 deg).
    Public devices (PUBLIC_DEVICES): ``device_fallback_intrinsics`` (the comma chain is not their export geometry).
    Prefer ``sidecar_intrinsics``, which uses the sidecar's ``K_clip`` when present (every public sidecar has one).
    """
    if str(device) in PUBLIC_DEVICES:
        return device_fallback_intrinsics(device)
    return _comma_clip_intrinsics(device)


def device_fallback_intrinsics(device: str) -> np.ndarray:
    """Intrinsics (float32 3x3) of a clip from its device string alone, i.e. when the sidecar has no usable ``K_clip``.

    comma devices and unknown strings: the clips.py qcamera chain (``clip_intrinsics``, unchanged behaviour).  Public
    devices (PUBLIC_DEVICES): the full-frame centred 2:1 crop of the DEVICE_CAMERAS frame resized to 512x256 with the
    principal point at the frame centre.  Exact for covla (rows 122:1086 of 1928x1208 -> fx = fy = 703.2, cx = 256,
    cy = 128, the spec's K_clip); approximate for openscene (the undistorted principal point sits at row 560, not 540:
    cy 128 instead of 133.3) and for dojo, whose export crops a 64 deg (1256 px wide) box instead of the full width
    (fx 268.2 here, ~410 in K_clip).  Every public sidecar carries K_clip, so this is only reached for a public clip
    without a sidecar."""
    device = str(device)
    if device not in PUBLIC_DEVICES:
        return _comma_clip_intrinsics(device)
    full_w, full_h, f = DEVICE_CAMERAS[device]
    k_full = np.array([[f, 0.0, full_w / 2.0], [0.0, f, full_h / 2.0], [0.0, 0.0, 1.0]])
    h = min(full_h, full_w // 2)  # publicdata.common.centre_box_2to1
    y0 = (full_h - h) // 2
    return crop_resize_intrinsics(k_full, (0, y0, full_w, h), (CLIP_H, CLIP_W))


def sidecar_intrinsics(side: Optional[dict], device: str) -> np.ndarray:
    """Intrinsics (float32 3x3, pixels of the 512x256 clip) of one clip: the sidecar's ``K_clip`` when it is a finite
    (3, 3) matrix with positive focal lengths (public-mixture sidecars, spec sections 2 / 6; a degenerate K would make
    ``patch_center_rays`` singular), else ``device_fallback_intrinsics(device)``.  Comma sidecars carry no ``K_clip``,
    so for them the result is exactly ``clip_intrinsics(device)`` as before."""
    k = None if side is None else side.get("K_clip")
    if k is not None:
        k = np.asarray(k, dtype=np.float64)
        if k.shape == (3, 3) and bool(np.isfinite(k).all()) and k[0, 0] > 0 and k[1, 1] > 0:
            return k.astype(np.float32)
    return device_fallback_intrinsics(device)


def patch_center_rays(K: np.ndarray, out_hw: Sequence[int], patch: int = 16) -> np.ndarray:
    """Unit viewing rays (x right, y down, z forward) through the patch centres of an (H, W) image with ``K``.

    Returns float32 (H//patch, W//patch, 3); row-major order matches the token order inside a slot.
    """
    H, W = (int(v) for v in out_hw)
    hp, wp = H // patch, W // patch
    us = (np.arange(wp, dtype=np.float64) + 0.5) * patch
    vs = (np.arange(hp, dtype=np.float64) + 0.5) * patch
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (hp, wp, 3) homogeneous pixel coordinates
    rays = pix @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.astype(np.float32)


def common_region_token_mask(box_a: Sequence[float], box_b: Sequence[float], out_hw: Sequence[int], patch: int = 16) -> np.ndarray:
    """Bool (H//patch * W//patch,) over view-a tokens whose patch centre, mapped back to input pixels, lies in
    box_a intersect box_b (half-open rectangle [x0, x1) x [y0, y1)).  Row-major token order (row*Wp + col)."""
    xa, ya, wa, ha = (float(v) for v in box_a)
    xb, yb, wb, hb = (float(v) for v in box_b)
    H, W = (int(v) for v in out_hw)
    hp, wp = H // patch, W // patch
    us = xa + (np.arange(wp, dtype=np.float64) + 0.5) * patch * (wa / W)  # view-a patch centres in input pixels
    vs = ya + (np.arange(hp, dtype=np.float64) + 0.5) * patch * (ha / H)
    x0, x1 = max(xa, xb), min(xa + wa, xb + wb)
    y0, y1 = max(ya, yb), min(ya + ha, yb + hb)
    in_x = (us >= x0) & (us < x1)
    in_y = (vs >= y0) & (vs < y1)
    return (in_y[:, None] & in_x[None, :]).reshape(-1)


def calib_rotation(pitch: float, yaw: float, roll: float = 0.0) -> np.ndarray:
    """openpilot ``rot_from_euler([roll, pitch, yaw])`` = Rz(yaw) @ Ry(pitch) @ Rx(roll): the ``device_from_calib``
    rotation (calibrated/road frame vectors -> device frame; device frame x forward, y right, z down) given the
    ``liveCalibration.rpyCalib`` angles (rad).  Float64 (3, 3)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def rays_to_calibrated_frame(rays: np.ndarray, cal_pitch: float, cal_yaw: float) -> np.ndarray:
    """Rotate camera-frame unit rays (..., 3) (x right, y down, z optical axis) into the road-aligned camera frame
    (x right, y down, z = direction of travel on a straight flat road): r' = V @ R^T @ V^T @ r with R =
    calib_rotation(cal_pitch, cal_yaw) (``device_from_calib``) and V = VIEW_FRAME_FROM_DEVICE_FRAME, the same chain as
    openpilot's ``get_warp_matrix`` (camera_from_calib = K @ V @ device_from_calib).  Non-finite angles count as 0.
    Returns float32 with the input shape; unit norm is preserved (pure rotation)."""
    p = float(cal_pitch) if math.isfinite(float(cal_pitch)) else 0.0
    y = float(cal_yaw) if math.isfinite(float(cal_yaw)) else 0.0
    if p == 0.0 and y == 0.0:
        return np.asarray(rays, dtype=np.float32)
    v = VIEW_FRAME_FROM_DEVICE_FRAME
    m = v @ calib_rotation(p, y).T @ v.T  # view <- device <- calib <- device <- view
    return (np.asarray(rays, dtype=np.float64) @ m.T).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Ego motion
# --------------------------------------------------------------------------------------------------------------
def compose_se2(d_forward, d_left, d_yaw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chain per-step SE(2) deltas (step k: pose k -> pose k+1, in pose k's frame: forward, left, CCW yaw).

    Returns (dx, dy, dyaw), each of length n+1: pose k relative to pose 0, expressed in pose 0's frame
    (x forward, y left, yaw CCW), k = 0..n, with (0, 0, 0) at k = 0.  NaN in step k makes every later pose
    NaN (cumulative sums propagate NaN)."""
    df, dl, dy = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (d_forward, d_left, d_yaw))
    yaw = np.concatenate([[0.0], np.cumsum(dy)])  # heading of pose k in frame 0
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])  # rotate step k's delta from frame k into frame 0
    x = np.concatenate([[0.0], np.cumsum(c * df - s * dl)])
    y = np.concatenate([[0.0], np.cumsum(s * df + c * dl)])
    return x, y, yaw


def sidecar_path(mp4_path: str) -> str:
    """``<clip>.mp4`` -> ``<clip>.can.npz`` (same directory)."""
    return os.path.splitext(mp4_path)[0] + ".can.npz"


def device_file_path(mp4_path: str) -> str:
    """``<clip>.mp4`` -> ``<clip>.device``: optional one-word text file naming the device of a clip WITHOUT a sidecar
    (the sidecar is the primary source; the export can write this file for ``.nocan`` clips so that the intrinsics
    are not guessed).  Absent on the current local export: those 54 clips fall back to '' (tici intrinsics)."""
    return os.path.splitext(mp4_path)[0] + ".device"


def clip_device(mp4_path: str, side: Optional[dict]) -> str:
    """Device string of a clip: the sidecar's ``device`` key, else the ``.device`` text file, else '' (unknown ->
    tici numbers in ``clip_intrinsics`` and device_id 0)."""
    if side is not None and side.get("device"):
        return str(side["device"])
    path = device_file_path(mp4_path)
    if os.path.exists(path):
        try:
            with open(path) as f:
                words = f.read().strip().split()
        except OSError:
            words = []
        dev = words[0] if words else ""
        return dev if dev in DEVICE_CAMERAS else ""
    return ""


def clip_calibration(side: Optional[dict]) -> Tuple[float, float]:
    """Per-clip (cal_pitch, cal_yaw) in rad = nanmedian of the sidecar rows (0.0 when missing or all NaN)."""
    out = []
    for key in ("cal_pitch", "cal_yaw"):
        arr = None if side is None else side.get(key)
        val = 0.0
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            if np.isfinite(arr).any():
                val = float(np.nanmedian(arr))
        out.append(val if math.isfinite(val) else 0.0)
    return out[0], out[1]


def load_sidecar(path) -> Optional[dict]:
    """Numeric keys of a ``<clip_id>.can.npz`` sidecar (allow_pickle=False), ``device`` as str, 0-d arrays as
    Python scalars; ``heading_src`` (object array) is never loaded.  None if the file is missing/unreadable."""
    if path is None or not os.path.exists(path):
        return None
    out = {}
    try:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key in _SIDECAR_SKIP_KEYS:
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


# Physical plausibility bounds per sidecar key (raw units, per 2 Hz sidecar row / 0.5 s step).  Values outside are
# treated as missing (NaN -> seq_valid / target_valid False), exactly like NaN dead-reckoning rows.  Motivation
# (2026-10-12 sidecar audit of subset v0): 34 clips carry isolated |d_yaw| / 0.5 s spikes of 1-4.4 rad/s and 22 clips
# |wdev_z| > 1 rad/s (heading resets / gyro glitches, not vehicle motion; a car tops out near 0.7 rad/s), 12 clips
# |a_ego| > 8 m/s^2.  Bounds are loose on purpose: they remove glitches, not rare but real manoeuvres.
# NOTE: must stay identical to comma_export/publicdata/common.PLAUSIBLE_RANGES -- the public exporters report the
# out-of-range fraction of every clip with the same bounds (spec section 2).  Nothing is imported from comma_export here;
# tests/test_motion_data.py::test_plausible_ranges_match_public_exporter asserts the equality.
PLAUSIBLE_RANGES = {
    "v_ego": (-2.0, 60.0),       # m/s
    "a_ego": (-12.0, 12.0),      # m/s^2
    "wdev_z": (-1.0, 1.0),       # rad/s
    "d_yaw": (-0.5, 0.5),        # rad per 0.5 s step (1 rad/s)
    "d_forward": (-5.0, 30.0),   # m per step (60 m/s)
    "d_left": (-6.0, 6.0),       # m per step
    "steer_deg": (-720.0, 720.0),
}


def _gather(side: dict, key: str, rows: np.ndarray, plausible: bool = True) -> np.ndarray:
    """``side[key][rows]`` as float64 with NaN where the key is missing, ``rows`` fall outside the array, or (when
    ``plausible``) the value lies outside ``PLAUSIBLE_RANGES[key]``."""
    arr = side.get(key)
    out = np.full(rows.shape, np.nan, dtype=np.float64)
    if arr is None:
        return out
    arr = np.asarray(arr)
    ok = (rows >= 0) & (rows < arr.shape[0])
    if ok.any():
        out[ok] = arr[rows[ok]].astype(np.float64)
    if plausible and key in PLAUSIBLE_RANGES:
        lo, hi = PLAUSIBLE_RANGES[key]
        bad = np.isfinite(out) & ((out < lo) | (out > hi))
        if bad.any():
            out[bad] = np.nan
    return out


def motion_targets(side: Optional[dict], clip_frame_indices, s: int, d: int, tubelet: int = 2, use_comma_only: bool = False,
                   seq_len: Optional[int] = None) -> dict:
    """Motion sequence, readout targets and dt for the slot pair (s, s+d) of one clip.

    ``clip_frame_indices``: kept-frame (mp4 frame == sidecar row) indices c_0..c_{T-1} of the clip.  Slot k is the
    tubelet of frames c_{tubelet*k} .. c_{tubelet*k+tubelet-1}; the pose of slot k is the pose at its first frame.
    ``seq_len``: rows of the sequence (default MOTION_SEQ_MAX_LEN; use ``motion_seq_len(fpc, tubelet)`` for other
    clip lengths -- a window longer than seq_len raises instead of being truncated).
    Returns numpy values (the dataset converts them to tensors):
      seq          float32 (seq_len, F): rows for kept frames c_{tubelet*s} .. c_{tubelet*(s+d)} inclusive (tubelet*d+1
                   rows, zero padded), features MOTION_SEQ_FEATURES (+ COMMA_ONLY_FEATURES when use_comma_only) in
                   RAW units (the heads normalise); yaw_rate = YAW_RATE_SIGN*wdev_z and d_yaw = D_YAW_SIGN*d_yaw are CCW.
                   Non-finite entries are zeroed and flagged in seq_valid.
      seq_valid    bool (seq_len, F): per-feature validity (e.g. a sidecar without wdev_z has a False yaw_rate column
                   but valid speed / displacement columns).
      seq_mask     bool (seq_len,): row has at least one valid feature (padding rows False).
      target       float32 (7,): MOTION_TARGETS = [dx, dy, dyaw, v_mean, yaw_rate_mean, dv, dyaw_rate]; dx/dy/dyaw =
                   pose at c_{tubelet*(s+d)} relative to the vehicle frame at c_{tubelet*s} (x forward m, y left m,
                   yaw CCW rad), composed from every sidecar step in between (exact also for strided clips); v_mean /
                   yaw_rate_mean = mean over the tubelet*d+1 kept frames (m/s, rad/s CCW); dv / dyaw_rate = mean over
                   the target slot's tubelet frames minus mean over the context slot's tubelet frames (m/s, rad/s).
      target_valid bool (7,): all contributing sidecar values finite and within PLAUSIBLE_RANGES (glitch filter).
      dt           float32 seconds between the first frames of the two slots from t_sof; nominal 1.0*d if unusable.
      s, d         int64;  speed float32: nanmean v_ego over the window (0 if unknown), for speed strata.
    With ``side`` None, or when the kept-frame indices of the window are not strictly increasing (padded short video):
    zeros, masks False, dt = 1.0*d, speed 0.
    """
    feats = list(MOTION_SEQ_FEATURES) + (list(COMMA_ONLY_FEATURES) if use_comma_only else [])
    L = MOTION_SEQ_MAX_LEN if seq_len is None else int(seq_len)
    nf, nt = len(feats), len(MOTION_TARGETS)
    out = {
        "seq": np.zeros((L, nf), dtype=np.float32), "seq_valid": np.zeros((L, nf), dtype=bool), "seq_mask": np.zeros(L, dtype=bool),
        "target": np.zeros(nt, dtype=np.float32), "target_valid": np.zeros(nt, dtype=bool),
        "dt": np.float32(1.0 * d), "s": np.int64(s), "d": np.int64(d), "speed": np.float32(0.0),
    }
    idx = np.asarray(clip_frame_indices, dtype=np.int64).reshape(-1)
    a, b = tubelet * int(s), tubelet * (int(s) + int(d))
    if not (0 <= a < b < len(idx)):
        raise ValueError(f"slot pair s={s}, d={d} does not fit {len(idx)} frames with tubelet {tubelet}")
    rows = idx[a : b + 1]  # kept frames of slots s .. s+d (first frame of the target slot included)
    if len(rows) > L:
        raise ValueError(f"window of {len(rows)} kept frames exceeds seq_len {L}; pass seq_len=motion_seq_len(fpc, tubelet)")
    if side is None:
        return out
    window = idx[a : min(b + tubelet, len(idx))]  # context slot .. target slot (both tubelets)
    if not np.all(np.diff(window) > 0):
        return out  # repeated kept frames (padded short video): zero motion is NOT a valid target
    i0, i1 = int(idx[a]), int(idx[b])

    # -- per-frame sequence with per-feature validity (a missing gyro only invalidates the yaw_rate column)
    cols = {name: _gather(side, name, rows) for name in feats if name != "yaw_rate"}
    cols["yaw_rate"] = YAW_RATE_SIGN * _gather(side, "wdev_z", rows)
    if "d_yaw" in cols:
        cols["d_yaw"] = D_YAW_SIGN * cols["d_yaw"]
    vals = np.stack([cols[name] for name in feats], axis=1)  # (len(rows), F), raw units
    valid_f = np.isfinite(vals)
    m = len(rows)
    out["seq"][:m] = np.where(valid_f, np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float32)
    out["seq_valid"][:m] = valid_f
    out["seq_mask"][:m] = valid_f.any(axis=1)

    # -- integrated pose c_{tubelet*s} -> c_{tubelet*(s+d)} over all sidecar steps i0 .. i1-1 (CCW yaw)
    steps = np.arange(i0, i1, dtype=np.int64)
    x, y, yaw = compose_se2(
        _gather(side, "d_forward", steps), _gather(side, "d_left", steps), D_YAW_SIGN * _gather(side, "d_yaw", steps)
    )
    v = _gather(side, "v_ego", rows)
    yr = YAW_RATE_SIGN * _gather(side, "wdev_z", rows)
    # pair-specific differences between the two tubelets (NaN when the target tubelet is incomplete)
    rows_s, rows_t = idx[a : a + tubelet], idx[b : b + tubelet]
    if len(rows_t) == tubelet:
        dv = _gather(side, "v_ego", rows_t).mean() - _gather(side, "v_ego", rows_s).mean()
        dyr = YAW_RATE_SIGN * (_gather(side, "wdev_z", rows_t).mean() - _gather(side, "wdev_z", rows_s).mean())
    else:
        dv = dyr = np.nan
    target = np.array([x[-1], y[-1], yaw[-1], v.mean(), yr.mean(), dv, dyr], dtype=np.float64)  # NaN if any input NaN
    valid = np.isfinite(target)
    out["target"] = np.where(valid, np.nan_to_num(target), 0.0).astype(np.float32)
    out["target_valid"] = valid
    out["speed"] = np.float32(np.nanmean(v)) if np.isfinite(v).any() else np.float32(0.0)

    # -- real elapsed time from start-of-frame timestamps (ns)
    t = side.get("t_sof")
    if t is not None and 0 <= i0 < len(t) and i1 < len(t):
        dt = (float(t[i1]) - float(t[i0])) * 1e-9
        if math.isfinite(dt) and 0.0 < dt < 60.0:
            out["dt"] = np.float32(dt)
    return out


def sample_slot_pair(rng: np.random.Generator, n_slots: int = 4) -> Tuple[int, int]:
    """Uniform over all ordered slot pairs (s, s+d) with s >= 0, d >= 1, s+d <= n_slots-1 (6 pairs for 4 slots)."""
    pairs = [(s, d) for s in range(n_slots - 1) for d in range(1, n_slots - s)]
    s, d = pairs[int(rng.integers(len(pairs)))]
    return s, d


# --------------------------------------------------------------------------------------------------------------
# Transform: view 1 reproduces the vendored VideoTransform pipeline bit for bit (same RNG draws, no flip);
# view 2 is a nested zoom crop of view 1 drawn from a private numpy Generator so the vendored RNG streams are untouched.
# --------------------------------------------------------------------------------------------------------------
class MotionTransform:
    """Random-resized-crop + ImageNet normalisation of a uint8 clip, returning the crop boxes for intrinsics.

    ``__call__(buffer [T,H,W,3] uint8, rng=None) -> {"view1": float32 [3,T,H',W'], "box1": (x0,y0,w,h), ...}``.
    view1 = torch.tensor(buffer) -> permute -> vendored ``_get_param_spatial_crop`` (python ``random`` + ``np.random``
    draws, exactly as ``VideoTransform``) -> slice -> bilinear resize (align_corners=False) -> normalise with mean*255 /
    std*255.  No horizontal flip (it would negate yaw/steer).  With ``view2_scale`` = (lo, hi): a second box with the
    output aspect ratio whose height is U(lo, hi) x box1 height, placed uniformly inside box1 ("zoom" factor 1/scale),
    sampled from ``rng`` (a ``np.random.Generator``; a fresh one if None) -> "view2", "box2".
    """

    def __init__(self, crop_size=(256, 512), random_resize_aspect_ratio=(1.8, 2.2), random_resize_scale=(0.5, 1.0),
                 normalize=IMAGENET_NORMALIZE, view2_scale=(0.6, 0.85)):
        if isinstance(crop_size, (tuple, list)):
            self.target_height, self.target_width = int(crop_size[0]), int(crop_size[1])
        else:
            self.target_height = self.target_width = int(crop_size)
        self.random_resize_aspect_ratio = tuple(random_resize_aspect_ratio)
        self.random_resize_scale = tuple(random_resize_scale)
        self.view2_scale = None if view2_scale is None else (float(view2_scale[0]), float(view2_scale[1]))
        # VideoTransform (auto_augment off) multiplies by 255 because the buffer stays in uint8 range
        self.mean = torch.tensor(normalize[0], dtype=torch.float32) * 255.0
        self.std = torch.tensor(normalize[1], dtype=torch.float32) * 255.0

    @property
    def out_hw(self) -> Tuple[int, int]:
        return self.target_height, self.target_width

    def _crop_resize_normalize(self, frames: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
        x0, y0, w, h = box
        cropped = frames[:, :, y0 : y0 + h, x0 : x0 + w]  # [C,T,h,w]
        resized = torch.nn.functional.interpolate(
            cropped, size=(self.target_height, self.target_width), mode="bilinear", align_corners=False
        )
        return _tensor_normalize_inplace(resized, self.mean, self.std)

    def sample_view2_box(self, box1: Tuple[int, int, int, int], rng: np.random.Generator) -> Tuple[int, int, int, int]:
        x1, y1, w1, h1 = box1
        lo, hi = self.view2_scale
        aspect = self.target_width / self.target_height
        h2 = max(1, int(round(float(rng.uniform(lo, hi)) * h1)))
        w2 = max(1, int(round(h2 * aspect)))
        if w2 > w1:  # box1 narrower than the output aspect allows: fit the width instead
            w2 = w1
            h2 = max(1, int(round(w2 / aspect)))
        h2 = min(h2, h1)
        x2 = x1 + int(rng.integers(0, w1 - w2 + 1))
        y2 = y1 + int(rng.integers(0, h1 - h2 + 1))
        return x2, y2, w2, h2

    def __call__(self, buffer, rng: Optional[np.random.Generator] = None) -> dict:
        if torch.is_tensor(buffer):
            frames = buffer.to(torch.float32)
        else:
            frames = torch.tensor(buffer, dtype=torch.float32)
        frames = frames.permute(3, 0, 1, 2)  # T H W C -> C T H W
        height, width = frames.shape[2], frames.shape[3]
        i, j, h, w = _get_param_spatial_crop(self.random_resize_scale, self.random_resize_aspect_ratio, height, width)
        box1 = (int(j), int(i), int(w), int(h))
        out = {"view1": self._crop_resize_normalize(frames, box1), "box1": box1}
        if self.view2_scale is not None:
            if rng is None:
                rng = np.random.default_rng()
            box2 = self.sample_view2_box(box1, rng)
            out["view2"] = self._crop_resize_normalize(frames, box2)
            out["box2"] = box2
        return out


# --------------------------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------------------------
class MotionVideoDataset(VideoDataset):
    """VideoDataset that also returns intrinsics, rays and CAN motion targets (and an optional second view).

    ``__getitem__`` -> ``(buffer_list, label, extra, clip_indices)``: ``buffer_list = [view1]`` and ``clip_indices``
    stay first/last so the vendored MaskCollator (``sample[-1]``, ``udata[0]``) works unchanged; ``extra`` holds only
    tensors (default_collate-able): K (3,3), rays (Hp,Wp,3), box1 (4,) int64, device_id, has_motion, and the keys
    of ``motion_targets``; with view 2: view2 [3,T,H,W], K2, rays2, box2, common_mask (Hp*Wp,) over view-1 tokens.

    Clip sampling, decoding and the view-1 crop consume exactly the RNG draws of the parent (``np.random`` for the
    clip start, python ``random``/``np.random`` inside the crop sampler); the slot pair (s, d) and the view-2 box
    come from a private ``np.random.Generator`` seeded from (seed, worker id, torch.initial_seed()) so that with
    ``view2_scale=None`` the clips/labels/clip_indices are identical to the vendored pipeline for the same seeds.
    Clips without ``<clip>.can.npz`` get has_motion False, zero targets and the device of an optional ``<clip>.device``
    file, else '' (tici intrinsics, device_id 0); ``require_can=True`` drops them from the sample list instead (the
    per-dataset counts / weights are rebuilt, so weighted sampling still works).  K of the 512x256 clip is the sidecar's
    ``K_clip`` when present (public-mixture sidecars), else ``device_fallback_intrinsics(device)`` -- for comma sidecars
    exactly ``clip_intrinsics(device)`` (``sidecar_intrinsics``).  Buffers that are not 512x256 (e.g. tiny test clips)
    are treated as per-axis rescaled 512x256 clips when deriving K.
    ``rays_road_frame`` (default True) rotates the per-token rays into the road-aligned camera frame with the clip's
    median ``cal_pitch``/``cal_yaw`` (``rays_to_calibrated_frame``); K itself stays the plain crop intrinsics.  The
    sequence length is derived from the clip length (``motion_seq_len``), so other dataset_fpcs work unchanged.
    """

    def __init__(self, *args, use_comma_only: bool = False, seed: int = 0, tubelet_size: int = 2, patch_size: int = 16,
                 rays_road_frame: bool = True, require_can: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_clips != 1:
            raise ValueError("MotionVideoDataset supports num_clips == 1 only")
        self.use_comma_only = bool(use_comma_only)
        self.seed = int(seed)
        self.tubelet_size = int(tubelet_size)
        self.patch_size = int(patch_size)
        self.rays_road_frame = bool(rays_road_frame)
        self.require_can = bool(require_can)
        if self.require_can:
            self._drop_clips_without_sidecar()
        self._rng_key = None
        self._rng = None

    def _drop_clips_without_sidecar(self) -> None:
        """Remove samples without ``<clip>.can.npz`` and rebuild the parent's per-dataset bookkeeping consistently."""
        samples, labels, counts, start = [], [], [], 0
        for n in self.num_samples_per_dataset:
            sel = [i for i in range(start, start + n) if os.path.exists(sidecar_path(str(self.samples[i])))]
            samples += [self.samples[i] for i in sel]
            labels += [self.labels[i] for i in sel]
            counts.append(len(sel))
            start += n
        dropped = len(self.samples) - len(samples)
        if dropped:
            logger.info("require_can: dropped %d of %d clips without a CAN sidecar", dropped, len(self.samples))
        self.samples, self.labels, self.num_samples_per_dataset = samples, labels, counts
        self.per_dataset_indices = ConcatIndices(counts)
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, counts):
                self.sample_weights += [dw / ns] * ns if ns else []

    def _get_rng(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        worker = -1 if info is None else int(info.id)
        key = (os.getpid(), worker, int(torch.initial_seed()))  # pid detects a fork; the seed material is reproducible
        if key != self._rng_key:
            self._rng = np.random.default_rng([self.seed, worker + 1, key[2]])
            self._rng_key = key
        return self._rng

    def get_item_video(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]
        buffer, clip_indices = self.loadvideo_decord(sample, fpc)  # [T H W 3] uint8
        if len(buffer) == 0:
            return None  # parent __getitem__ retries another index
        label = self.labels[index]
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        clip = buffer[:fpc]
        rng = self._get_rng()
        if isinstance(self.transform, MotionTransform):
            tf = self.transform(clip, rng=rng)
            out_hw = self.transform.out_hw
        else:  # plain callable (e.g. the vendored VideoTransform): no crop information
            view1 = self.transform(clip) if self.transform is not None else torch.tensor(clip).permute(3, 0, 1, 2).float()
            tf = {"view1": view1, "box1": (0, 0, int(clip.shape[2]), int(clip.shape[1]))}
            out_hw = (int(view1.shape[-2]), int(view1.shape[-1]))

        side = load_sidecar(sidecar_path(sample))
        device = clip_device(sample, side)
        n_slots = fpc // self.tubelet_size
        s, d = sample_slot_pair(rng, n_slots)
        mt = motion_targets(side, clip_indices[0], s, d, tubelet=self.tubelet_size, use_comma_only=self.use_comma_only,
                            seq_len=motion_seq_len(fpc, self.tubelet_size))
        cal_pitch, cal_yaw = clip_calibration(side) if self.rays_road_frame else (0.0, 0.0)

        def rays_of(k):  # per-patch unit rays of a view, in the road-aligned camera frame when enabled
            r = patch_center_rays(k, out_hw, self.patch_size)
            return torch.from_numpy(rays_to_calibrated_frame(r, cal_pitch, cal_yaw))

        # intrinsics of the 512x256 clip (the sidecar's K_clip when present, else the per-device fallback) in buffer
        # pixels (rescaled if the buffer is not 512x256), then per view
        k_buf = crop_resize_intrinsics(sidecar_intrinsics(side, device), (0, 0, CLIP_W, CLIP_H), (clip.shape[1], clip.shape[2]))
        k1 = crop_resize_intrinsics(k_buf, tf["box1"], out_hw)
        extra = {
            "K": torch.from_numpy(k1),
            "rays": rays_of(k1),
            "box1": torch.tensor(tf["box1"], dtype=torch.int64),
            "device_id": torch.tensor(DEVICE_IDS.get(device, 0), dtype=torch.int64),
            "has_motion": torch.tensor(side is not None, dtype=torch.bool),
            "cal_pitch_yaw": torch.tensor([cal_pitch, cal_yaw], dtype=torch.float32),
        }
        for key, val in mt.items():
            extra[key] = torch.as_tensor(np.asarray(val))
        if "view2" in tf:
            k2 = crop_resize_intrinsics(k_buf, tf["box2"], out_hw)
            extra["view2"] = tf["view2"]
            extra["K2"] = torch.from_numpy(k2)
            extra["rays2"] = rays_of(k2)
            extra["box2"] = torch.tensor(tf["box2"], dtype=torch.int64)
            extra["common_mask"] = torch.from_numpy(common_region_token_mask(tf["box1"], tf["box2"], out_hw, self.patch_size))
        return [tf["view1"]], label, extra, clip_indices


def make_motion_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=None,  # vendored default is 4; None here so fps-driven configs (fps=2) work without passing it
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    use_comma_only=False,
    seed=0,
    tubelet_size=2,
    patch_size=16,
    rays_road_frame=True,
    require_can=False,
):
    """Mirror of the vendored ``make_videodataset`` building a MotionVideoDataset; returns (dataset, loader, sampler).

    ``transform`` should be a MotionTransform; ``collator`` the unchanged MaskCollator (it default_collates the tuple
    including the ``extra`` dict).  Exactly one of fps / duration / frame_step must be set (VideoDataset rule).
    ``rays_road_frame`` / ``require_can`` are forwarded to MotionVideoDataset (see its docstring)."""
    dataset = MotionVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        use_comma_only=use_comma_only,
        seed=seed,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        rays_road_frame=rays_road_frame,
        require_can=require_can,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        dataset = MonitoredDataset(
            dataset=dataset, log_filename=str(log_dir / f"resource_file_{rank}_%w.csv"), log_interval=10.0, monitor_interval=5.0
        )

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader_cls = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    data_loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("MotionVideoDataset created with %d clips", len(dataset))
    return dataset, data_loader, dist_sampler
