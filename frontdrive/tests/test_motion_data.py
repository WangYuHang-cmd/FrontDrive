"""Data path of the motion arm (pretrain/motion_data.py): sidecar targets, MotionTransform, MotionVideoDataset.

Synthetic 512x256 2 fps clips are generated with PyAV into a tmp dir together with sidecars that follow the
<clip_id>.can.npz export convention (one row per kept frame; d_forward / d_left = motion from frame i to i+1 in frame
i's vehicle frame with d_left LEFT-positive; d_yaw / wdev_z / yaw CLOCKWISE-positive like comma's NED heading and the
device gyro z axis; last d_* row NaN; heading_src object array; 0-d device/wheelbase).  motion_data converts the yaw
quantities to the CCW frame (D_YAW_SIGN / YAW_RATE_SIGN = -1), which the tests assert.  Tests on the real local clips
are skipped when the clip directory is absent.  CPU only.
"""
import glob
import math
import os
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (_ROOT / "third_party/Drive-JEPA/navsim_v1/vjepa2", _ROOT / "third_party/Drive-JEPA/navsim_v1", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.vjepa.transforms import VideoTransform  # noqa: E402
from src.masks.multiseq_multiblock3d import MaskCollator  # noqa: E402
from vjepa2.src.datasets.video_dataset import VideoDataset  # noqa: E402

from pretrain import motion_data as md  # noqa: E402

torch.set_num_threads(2)  # CPU-only tests; avoids oversubscription against other jobs on the box

REAL_CLIP_ROOT = "/home/henry/Desktop/Qwen_Drive/benchmarks/pretrain/clips/subset_v0"
MASK_CFGS = [
    dict(aspect_ratio=[0.75, 1.5], num_blocks=8, spatial_scale=[0.15, 0.15], temporal_scale=[1.0, 1.0], max_temporal_keep=1.0),
    dict(aspect_ratio=[0.75, 1.5], num_blocks=2, spatial_scale=[0.7, 0.7], temporal_scale=[1.0, 1.0], max_temporal_keep=1.0),
]
IV = md.MOTION_SEQ_FEATURES.index  # feature column lookup


# ------------------------------------------------------------------------------------------------------ helpers
def make_sidecar(n, device="mici", v=10.0, omega=0.0, dt=0.5, nan_rows=(), t0=1_000_000_000_000, steer=0.0,
                 cal_pitch=0.0, cal_yaw=0.0):
    """Synthetic sidecar dict (numpy, same keys/dtypes as the real files) for constant speed v and a constant CCW
    (left-turn positive) yaw rate omega, stored in the EXPORT convention: d_left left-positive, d_yaw = -omega*dt and
    wdev_z = -omega (clockwise-positive NED heading / device z down), yaw clockwise."""
    th = omega * dt
    if omega != 0.0:
        R = v / omega
        df, dl = R * math.sin(th), R * (1 - math.cos(th))
    else:
        df, dl = v * dt, 0.0
    side = {
        "v_ego": np.full(n, v, np.float32), "a_ego": np.zeros(n, np.float32), "steer_deg": np.full(n, steer, np.float32),
        "yaw": (-np.arange(n) * th).astype(np.float32), "wdev_z": np.full(n, -omega, np.float32),  # clockwise-positive
        "cal_pitch": np.full(n, cal_pitch, np.float32), "cal_yaw": np.full(n, cal_yaw, np.float32),
        "cc_enabled": np.zeros(n, bool), "cc_lat_active": np.zeros(n, bool), "cc_long_active": np.zeros(n, bool),
        "standstill": np.full(n, v == 0.0, bool),
        "t_sof": (t0 + np.round(np.arange(n) * dt * 1e9)).astype(np.int64), "frame_idx": (np.arange(n) * 10).astype(np.int32),
        "d_forward": np.full(n, df, np.float32), "d_left": np.full(n, dl, np.float32), "d_yaw": np.full(n, -th, np.float32),
        "device": device, "wheelbase": 2.7,
    }
    for key in ("d_forward", "d_left", "d_yaw"):
        side[key][-1] = np.nan
        for r in nan_rows:
            side[key][r] = np.nan
    return side


def save_sidecar(side, path):
    arrays = {k: v for k, v in side.items() if k not in ("device", "wheelbase")}
    n = len(side["t_sof"])
    np.savez(path, **arrays, device=np.array(side["device"]), wheelbase=np.array(side["wheelbase"]),
             heading_src=np.array(["dr"] * n, dtype=object), notes=np.array([{"k": 1}] * n, dtype=object))


def write_clip(path, n_frames=24, size=(512, 256), seed=0):
    import av

    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=2)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n_frames):
            img = np.roll(base, shift=7 * i, axis=1).copy()  # moving texture
            img[:, : 8 * (i % 16)] = 255
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """3 synthetic clips: straight mici with sidecar, turning tici with sidecar (+ NaN rows), one without sidecar."""
    d = tmp_path_factory.mktemp("clips")
    specs = [("aaaa0000aaaa0000", dict(device="mici", v=10.0, omega=0.0)),
             ("bbbb1111bbbb1111", dict(device="tici", v=8.0, omega=0.2, nan_rows=(3,))), ("cccc2222cccc2222", None)]
    paths, sides = [], []
    for i, (cid, spec) in enumerate(specs):
        mp4 = d / f"{cid}.mp4"
        write_clip(mp4, seed=i)
        if spec is None:
            (d / f"{cid}.nocan").touch()
            sides.append(None)
        else:
            side = make_sidecar(24, **spec)
            save_sidecar(side, d / f"{cid}.can.npz")
            sides.append(side)
        paths.append(str(mp4))
    csv = d / "clips.csv"
    csv.write_text("".join(f"{p} 0\n" for p in paths))
    return {"csv": str(csv), "paths": paths, "sides": sides, "dir": d}


def _seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ----------------------------------------------------------------------------------------------- motion_targets
def test_motion_targets_straight_motion():
    side = make_sidecar(30, v=10.0, omega=0.0)
    frames = np.arange(5, 13)  # kept frames c_0..c_7 of an 8-frame clip
    out = md.motion_targets(side, frames, s=0, d=2)
    assert out["seq"].shape == (7, len(md.MOTION_SEQ_FEATURES)) and out["seq"].dtype == np.float32
    assert out["seq_valid"].shape == out["seq"].shape and out["seq_valid"].dtype == bool
    assert out["seq_mask"].tolist() == [True] * 5 + [False] * 2  # frames c_0..c_4 inclusive
    assert out["seq_valid"][:5].all() and not out["seq_valid"][5:].any()
    assert out["target_valid"].all() and out["target"].dtype == np.float32 and out["target"].shape == (7,)
    dx, dy, dyaw, v_mean, yr_mean, dv, dyr = out["target"]
    assert dx == pytest.approx(20.0, abs=1e-5) and dy == pytest.approx(0.0, abs=1e-6) and dyaw == 0.0  # 4 steps x 5 m
    assert v_mean == pytest.approx(10.0) and yr_mean == 0.0 and out["speed"] == pytest.approx(10.0)
    assert dv == 0.0 and dyr == 0.0  # constant speed / yaw rate: no change between the two tubelets
    assert out["dt"] == pytest.approx(2.0, abs=1e-6) and out["s"] == 0 and out["d"] == 2
    assert out["s"].dtype == np.int64 and out["dt"].dtype == np.float32
    # raw physical units (the heads normalise), padding rows zero
    assert np.allclose(out["seq"][:5, IV("v_ego")], 10.0)
    assert np.allclose(out["seq"][:5, IV("d_forward")], 5.0)
    assert np.all(out["seq"][5:] == 0)
    # last slot pair: rows c_4..c_6, two steps
    out = md.motion_targets(side, frames, s=2, d=1)
    assert out["seq_mask"].tolist() == [True] * 3 + [False] * 4 and out["target"][0] == pytest.approx(10.0, abs=1e-5)
    assert out["dt"] == pytest.approx(1.0, abs=1e-6)
    # pair-specific targets: speed ramps 10 -> 17 m/s over 8 frames -> dv = mean(v[c_2], v[c_3]) - mean(v[c_0], v[c_1])
    side["v_ego"] = np.arange(30, dtype=np.float32) + 10.0
    out = md.motion_targets(side, frames, s=0, d=1)
    assert out["target"][md.MOTION_TARGETS.index("dv")] == pytest.approx(2.0) and out["target_valid"].all()
    side["wdev_z"][:] = -np.linspace(0, 0.29, 30, dtype=np.float32)  # clockwise-stored ramp of a CCW yaw rate
    out = md.motion_targets(side, frames, s=1, d=2)
    expect = float(-side["wdev_z"][[11, 12]].mean() + side["wdev_z"][[7, 8]].mean())
    assert out["target"][md.MOTION_TARGETS.index("dyaw_rate")] == pytest.approx(expect, abs=1e-6) and expect > 0


def test_motion_targets_turning_arc_and_yaw_rate_sign():
    """A LEFT turn exported like clips.py (d_left > 0, d_yaw < 0, wdev_z < 0) must come out CCW-positive and
    consistent: dy > 0, dyaw > 0, yaw_rate > 0, and the unicycle chord relation d_left = d_forward*tan(dyaw_ccw/2)."""
    v, omega, dt = 8.0, 0.2, 0.5
    side = make_sidecar(30, device="tici", v=v, omega=omega, dt=dt)
    assert (side["d_yaw"][:-1] < 0).all() and (side["d_left"][:-1] > 0).all() and (side["wdev_z"] < 0).all()
    assert md.D_YAW_SIGN == -1.0 and md.YAW_RATE_SIGN == -1.0
    out = md.motion_targets(side, np.arange(10, 18), s=0, d=3)  # 6 steps: c_0 -> c_6
    R, th = v / omega, 6 * omega * dt
    dx, dy, dyaw, v_mean, yr_mean, dv, dyr = out["target"]
    assert out["target_valid"].all()
    assert dx == pytest.approx(R * math.sin(th), rel=1e-5) and dy == pytest.approx(R * (1 - math.cos(th)), rel=1e-5)
    assert dyaw == pytest.approx(th, rel=1e-5) and dy > 0 and dyaw > 0  # left turn: y left positive, yaw CCW positive
    assert yr_mean == pytest.approx(omega, rel=1e-6) and v_mean == pytest.approx(v) and dv == 0.0 and dyr == 0.0
    assert np.sign(yr_mean) == np.sign(dyaw) == np.sign(dy)  # gyro, dead-reckoning yaw and lateral motion agree
    assert out["dt"] == pytest.approx(3.0, abs=1e-6)
    assert np.allclose(out["seq"][:7, IV("yaw_rate")], omega)  # raw rad/s, CCW
    assert np.allclose(out["seq"][:7, IV("d_yaw")], omega * dt)  # per-step CCW yaw in the sequence
    assert np.allclose(out["seq"][:7, IV("d_left")], out["seq"][:7, IV("d_forward")] * np.tan(out["seq"][:7, IV("d_yaw")] / 2), atol=1e-6)
    # mirrored right turn
    out = md.motion_targets(make_sidecar(30, v=v, omega=-omega, dt=dt), np.arange(10, 18), s=0, d=3)
    assert out["target"][1] == pytest.approx(-dy, rel=1e-5) and out["target"][2] == pytest.approx(-dyaw, rel=1e-5)
    assert out["target"][4] == pytest.approx(-omega, rel=1e-6)


def test_motion_targets_dt_from_t_sof_or_nominal():
    side = make_sidecar(30, dt=0.52)
    for s, d in [(0, 1), (1, 2), (0, 3)]:
        assert md.motion_targets(side, np.arange(8), s, d)["dt"] == pytest.approx(0.52 * 2 * d, abs=1e-5)
    side.pop("t_sof")
    assert md.motion_targets(side, np.arange(8), 1, 2)["dt"] == 2.0  # nominal 1.0 s per slot
    side = make_sidecar(30)
    side["t_sof"][:] = 0  # unusable timestamps -> nominal
    assert md.motion_targets(side, np.arange(8), 0, 1)["dt"] == 1.0


def test_motion_targets_validity_with_nan_dead_reckoning():
    side = make_sidecar(30, v=10.0, omega=0.1, nan_rows=(7,))  # dead-reckoning invalid at kept frame 7
    frames = np.arange(5, 13)
    out = md.motion_targets(side, frames, s=0, d=2)  # steps 5..8 include row 7
    assert out["target_valid"].tolist() == [False, False, False, True, True, True, True]
    assert np.all(out["target"][:3] == 0) and out["target"][3] == pytest.approx(10.0)
    # per-feature validity: the row of frame 7 keeps its speed / gyro columns, only the d_* columns are invalid
    assert out["seq_mask"].tolist() == [True, True, True, True, True, False, False]
    assert out["seq_valid"][2, [IV("v_ego"), IV("yaw_rate")]].tolist() == [True, True]
    assert not out["seq_valid"][2, [IV("d_forward"), IV("d_left"), IV("d_yaw")]].any()
    assert np.all(out["seq"][2, [IV("d_forward"), IV("d_left"), IV("d_yaw")]] == 0) and out["speed"] == pytest.approx(10.0)
    out = md.motion_targets(side, frames, s=2, d=1)  # steps 9..10 are clean
    assert out["target_valid"].all()
    # d_yaw-only NaN at the first step: heading unknown afterwards -> dx/dy/dyaw all invalid
    side = make_sidecar(30)
    side["d_yaw"][5] = np.nan
    out = md.motion_targets(side, frames, s=0, d=1)
    assert out["target_valid"].tolist() == [False, False, False, True, True, True, True]
    # NaN speed on one row (as on real clips, row 0): v_mean and dv invalid but speed still known; the row stays in
    # the sequence with its v_ego entry flagged invalid
    side = make_sidecar(30)
    side["v_ego"][0] = np.nan
    out = md.motion_targets(side, np.arange(0, 8), s=0, d=1)
    assert out["target_valid"].tolist() == [True, True, True, False, True, False, True] and out["speed"] == pytest.approx(10.0)
    assert out["seq_mask"].tolist() == [True, True, True, False, False, False, False]
    assert not out["seq_valid"][0, IV("v_ego")] and out["seq_valid"][0, IV("d_forward")] and out["seq"][0, IV("v_ego")] == 0
    # a sidecar without the gyro key (12 local tizi files): only the yaw_rate column / yaw-rate targets are invalid
    side = make_sidecar(30)
    side.pop("wdev_z")
    out = md.motion_targets(side, frames, s=0, d=3)
    assert out["seq_mask"].tolist() == [True] * 7 and not out["seq_valid"][:, IV("yaw_rate")].any()
    assert out["seq_valid"][:, IV("v_ego")].all() and out["target_valid"].tolist() == [True, True, True, True, False, True, False]


def test_motion_targets_none_and_out_of_range_rows():
    out = md.motion_targets(None, np.arange(8), s=1, d=2)
    assert not out["seq_mask"].any() and not out["target_valid"].any() and np.all(out["seq"] == 0)
    assert out["dt"] == 2.0 and out["speed"] == 0.0 and out["s"] == 1 and out["d"] == 2 and not out["seq_valid"].any()
    side = make_sidecar(20)  # shorter than the video (a real local case)
    frames = np.arange(16, 24)
    out = md.motion_targets(side, frames, s=0, d=1)  # steps 16, 17 -> fine
    assert out["target_valid"].all()
    out = md.motion_targets(side, frames, s=0, d=3)  # steps 16..21 cross the sidecar end
    assert not out["target_valid"].any()
    assert out["seq_mask"].tolist() == [True, True, True, True, False, False, False]  # rows 16..19 exist (row 19: d_* NaN only)
    with pytest.raises(ValueError):
        md.motion_targets(side, np.arange(8), s=2, d=2)
    # repeated kept frames (the vendored sampler pads short videos with the last frame): zero motion is NOT valid
    side = make_sidecar(30, v=10.0, omega=0.2)
    out = md.motion_targets(side, np.array([0, 1, 2, 3, 4, 4, 4, 4]), s=2, d=1)
    assert not out["target_valid"].any() and not out["seq_mask"].any() and out["dt"] == 1.0
    out = md.motion_targets(side, np.array([0, 1, 2, 3, 4, 4, 4, 4]), s=0, d=1)  # window c_0..c_3 is clean
    assert out["target_valid"].all()
    # sequence length follows the clip length instead of being truncated silently
    assert md.motion_seq_len(8) == 7 and md.motion_seq_len(16) == 15 and md.motion_seq_len(8, tubelet=4) == 5
    out = md.motion_targets(side, np.arange(16), s=1, d=6, seq_len=md.motion_seq_len(16))
    assert out["seq"].shape[0] == 15 and out["seq_mask"].sum() == 13 and out["target_valid"].all()
    with pytest.raises(ValueError):
        md.motion_targets(side, np.arange(16), s=0, d=7)  # 15 rows do not fit the default 7-row sequence


def test_motion_targets_comma_only_features():
    side = make_sidecar(30, steer=45.0)
    side["cc_lat_active"][:] = True
    out = md.motion_targets(side, np.arange(8), s=0, d=1, use_comma_only=True)
    feats = md.MOTION_SEQ_FEATURES + md.COMMA_ONLY_FEATURES
    assert out["seq"].shape == (7, len(feats)) and len(feats) == 11
    assert np.allclose(out["seq"][:3, feats.index("steer_deg")], 45.0)  # raw degrees
    assert np.all(out["seq"][:3, feats.index("cc_lat_active")] == 1) and np.all(out["seq"][:3, feats.index("cc_enabled")] == 0)


def test_load_sidecar_numeric_only(tmp_path):
    side = make_sidecar(12, device="tizi")
    save_sidecar(side, tmp_path / "x.can.npz")
    loaded = md.load_sidecar(str(tmp_path / "x.can.npz"))
    assert "heading_src" not in loaded and "notes" not in loaded
    assert loaded["device"] == "tizi" and isinstance(loaded["device"], str)
    assert isinstance(loaded["wheelbase"], float) and loaded["wheelbase"] == 2.7
    assert loaded["t_sof"].dtype == np.int64 and loaded["standstill"].dtype == bool and len(loaded["d_yaw"]) == 12
    assert np.isnan(loaded["d_yaw"][-1])
    assert md.load_sidecar(str(tmp_path / "missing.can.npz")) is None and md.load_sidecar(None) is None
    assert md.sidecar_path("/a/b/0123.mp4") == "/a/b/0123.can.npz"


# ---------------------------------------------------------------------------------------------- MotionTransform
def test_motion_transform_shapes_normalisation_boxes():
    T, H, W = 8, 256, 512
    grey = np.full((T, H, W, 3), 128, np.uint8)
    tf = md.MotionTransform(crop_size=(256, 512), view2_scale=(0.6, 0.85))
    rng = np.random.default_rng(0)
    _seed_all(0)
    for _ in range(10):
        out = tf(grey, rng=rng)
        v1, (x0, y0, w, h) = out["view1"], out["box1"]
        assert v1.shape == (3, T, H, W) and v1.dtype == torch.float32
        for c, (m, s) in enumerate(zip(*md.IMAGENET_NORMALIZE)):  # constant input -> exact normalised constant
            assert torch.allclose(v1[c], torch.full_like(v1[c], (128 - 255 * m) / (255 * s)), atol=1e-4)
        assert 0 <= x0 and 0 <= y0 and x0 + w <= W and y0 + h <= H and w > 0 and h > 0
        assert 1.75 <= w / h <= 2.25 and 0.48 <= w * h / (W * H) <= 1.0
        x2, y2, w2, h2 = out["box2"]
        assert out["view2"].shape == (3, T, H, W)
        assert x0 <= x2 and y0 <= y2 and x2 + w2 <= x0 + w and y2 + h2 <= y0 + h  # nested
        assert abs(w2 / h2 - 2.0) < 0.05 and 0.55 <= h2 / h <= 0.9  # output aspect, zoom in the requested range
    # no view 2 when disabled; tiny output size
    out = md.MotionTransform(crop_size=(64, 128), view2_scale=None)(grey)
    assert set(out) == {"view1", "box1"} and out["view1"].shape == (3, T, 64, 128)


def test_motion_transform_matches_vendored_video_transform():
    buf = np.random.default_rng(5).integers(0, 256, (8, 256, 512, 3), dtype=np.uint8)
    vt = VideoTransform(random_horizontal_flip=False, random_resize_aspect_ratio=(1.8, 2.2), random_resize_scale=(0.5, 1.0), crop_size=(256, 512))
    mt = md.MotionTransform(crop_size=(256, 512), random_resize_aspect_ratio=(1.8, 2.2), random_resize_scale=(0.5, 1.0), view2_scale=None)
    mt2 = md.MotionTransform(crop_size=(256, 512), random_resize_aspect_ratio=(1.8, 2.2), random_resize_scale=(0.5, 1.0), view2_scale=(0.6, 0.85))
    for seed in range(4):
        _seed_all(seed)
        ref = vt(buf)
        state = (random.random(), np.random.rand())
        _seed_all(seed)
        got = mt(buf)["view1"]
        assert torch.equal(ref, got)  # bit-identical view 1 (same RNG draws, same ops)
        assert state == (random.random(), np.random.rand())  # global RNG streams advanced identically
        _seed_all(seed)
        got2 = mt2(buf, rng=np.random.default_rng(seed))
        assert torch.equal(ref, got2["view1"])  # the view-2 draws never touch random / np.random
        assert state == (random.random(), np.random.rand())


# -------------------------------------------------------------------------------------------- MotionVideoDataset
def _dataset(csv, view2=True, use_comma_only=False, seed=0, transform=None):
    tf = transform if transform is not None else md.MotionTransform(view2_scale=(0.6, 0.85) if view2 else None)
    return md.MotionVideoDataset(data_paths=[csv], frames_per_clip=8, dataset_fpcs=[8], fps=2, frame_step=None,
                                 transform=tf, use_comma_only=use_comma_only, seed=seed)


def _check_item(item, out_hw=(256, 512), view2=True):
    buffer_list, label, extra, clip_indices = item
    H, W = out_hw
    hp, wp = H // 16, W // 16
    assert isinstance(buffer_list, list) and len(buffer_list) == 1 and buffer_list[0].shape == (3, 8, H, W)
    assert label == 0 and len(clip_indices) == 1 and len(clip_indices[-1]) == 8
    assert all(torch.is_tensor(v) for v in extra.values())
    assert extra["K"].shape == (3, 3) and extra["K"].dtype == torch.float32
    assert extra["rays"].shape == (hp, wp, 3) and extra["device_id"].dtype == torch.int64 and extra["has_motion"].dtype == torch.bool
    assert extra["seq"].shape[0] == 7 and extra["seq_mask"].shape == (7,) and extra["target"].shape == (7,)
    assert extra["seq_valid"].shape == extra["seq"].shape and extra["seq_valid"].dtype == torch.bool
    assert extra["target_valid"].dtype == torch.bool and extra["dt"].dtype == torch.float32 and extra["speed"].dtype == torch.float32
    assert extra["cal_pitch_yaw"].shape == (2,) and torch.isfinite(extra["cal_pitch_yaw"]).all()
    s, d = int(extra["s"]), int(extra["d"])
    assert 0 <= s <= 2 and 1 <= d <= 3 - s
    box1 = tuple(extra["box1"].tolist())
    dev = {v: k for k, v in md.DEVICE_IDS.items()}[int(extra["device_id"])]
    k_buf = md.crop_resize_intrinsics(md.clip_intrinsics(dev), (0, 0, 512, 256), (256, 512))
    assert np.allclose(extra["K"].numpy(), md.crop_resize_intrinsics(k_buf, box1, out_hw), atol=1e-3)
    cal = extra["cal_pitch_yaw"].tolist()  # rays are rotated into the road-aligned camera frame with the clip calibration
    assert np.allclose(extra["rays"].numpy(), md.rays_to_calibrated_frame(md.patch_center_rays(extra["K"].numpy(), out_hw), *cal), atol=1e-6)
    assert np.allclose(np.linalg.norm(extra["rays"].numpy(), axis=-1), 1.0, atol=1e-5)
    if view2:
        box2 = tuple(extra["box2"].tolist())
        assert extra["view2"].shape == (3, 8, H, W) and extra["K2"].shape == (3, 3) and extra["rays2"].shape == (hp, wp, 3)
        assert np.allclose(extra["K2"].numpy(), md.crop_resize_intrinsics(k_buf, box2, out_hw), atol=1e-3)
        assert np.allclose(extra["rays2"].numpy(), md.rays_to_calibrated_frame(md.patch_center_rays(extra["K2"].numpy(), out_hw), *cal), atol=1e-6)
        assert extra["common_mask"].shape == (hp * wp,) and extra["common_mask"].dtype == torch.bool
        assert np.array_equal(extra["common_mask"].numpy(), md.common_region_token_mask(box1, box2, out_hw))
        assert 0 < int(extra["common_mask"].sum()) < hp * wp  # a strict zoom: some view-1 tokens fall outside
        # a 3D point seen in both views maps back to the same buffer pixel through K / K2
        P = np.array([1.5, 0.2, 30.0])
        u1 = (extra["K"].numpy() @ P)[:2] / P[2]
        u2 = (extra["K2"].numpy() @ P)[:2] / P[2]
        back1 = (box1[0] + u1[0] * box1[2] / W, box1[1] + u1[1] * box1[3] / H)
        back2 = (box2[0] + u2[0] * box2[2] / W, box2[1] + u2[1] * box2[3] / H)
        assert np.allclose(back1, back2, atol=1e-2)
    else:
        assert "view2" not in extra and "common_mask" not in extra
    return s, d


def test_dataset_items_on_synthetic_clips(synth):
    ds = _dataset(synth["csv"])
    assert len(ds) == 3
    _seed_all(0)
    items = [ds[i] for i in range(3)]
    for i, item in enumerate(items):
        s, d = _check_item(item)
        extra, ci = item[2], item[3][0]
        side = synth["sides"][i]
        assert bool(extra["has_motion"]) == (side is not None)
        assert int(extra["device_id"]) == md.DEVICE_IDS[side["device"] if side else ""]
        ref = md.motion_targets(side, ci, s, d)  # dataset targets == direct computation on the same clip frames
        for key in ("seq", "seq_valid", "seq_mask", "target", "target_valid", "dt", "speed"):
            assert np.array_equal(extra[key].numpy(), np.asarray(ref[key])), key
    # straight mici clip: dx = v * dt exactly (sidecar built with dt = 0.5 s)
    extra = items[0][2]
    assert extra["target_valid"].all() and float(extra["target"][0]) == pytest.approx(10.0 * float(extra["dt"]), rel=1e-5)
    assert float(extra["target"][2]) == 0.0 and float(extra["dt"]) == pytest.approx(float(extra["d"]), abs=1e-6)
    # turning tici clip (left turn stored clockwise in the sidecar): CCW dyaw / yaw rate and left-positive dy
    extra = items[1][2]
    if bool(extra["target_valid"][2]):
        assert float(extra["target"][1]) > 0 and float(extra["target"][2]) > 0 and float(extra["target"][4]) > 0
    # clip without a sidecar
    extra = items[2][2]
    assert not extra["has_motion"] and not extra["target_valid"].any() and float(extra["speed"]) == 0.0
    assert np.allclose(extra["K"].numpy()[0, 0] * extra["box1"][2].item() / 512, md.clip_intrinsics("")[0, 0], atol=1e-2)
    assert extra["cal_pitch_yaw"].tolist() == [0.0, 0.0]
    # view 2 off + comma-only fields
    item = _dataset(synth["csv"], view2=False, use_comma_only=True)[0]
    _check_item(item, view2=False)
    assert item[2]["seq"].shape == (7, 11) and item[2]["seq_valid"].shape == (7, 11)


def test_dataset_device_file_require_can_and_calibrated_rays(synth):
    """`.device` next to a sidecar-less clip names its camera; require_can drops such clips; cal_pitch / cal_yaw from
    the sidecar rotate the rays (and only the rays)."""
    nocan = pathlib.Path(synth["paths"][2])
    devfile = nocan.with_suffix(".device")
    devfile.write_text("mici\n")
    try:
        _seed_all(0)
        extra = _dataset(synth["csv"])[2][2]
        assert int(extra["device_id"]) == md.DEVICE_IDS["mici"] and not extra["has_motion"]
        assert np.allclose(extra["K"].numpy()[0, 0] * extra["box1"][2].item() / 512, md.clip_intrinsics("mici")[0, 0], atol=1e-2)
        devfile.write_text("eon\n")  # unknown device name -> fallback ''
        assert int(_dataset(synth["csv"])[2][2]["device_id"]) == 0
    finally:
        devfile.unlink()
    assert md.clip_device(str(nocan), None) == "" and md.clip_device(str(nocan), {"device": "tizi"}) == "tizi"
    ds = md.MotionVideoDataset(data_paths=[synth["csv"]], frames_per_clip=8, dataset_fpcs=[8], fps=2, frame_step=None,
                               transform=md.MotionTransform(view2_scale=None), require_can=True)
    assert len(ds) == 2 and ds.num_samples_per_dataset == [2] and all(os.path.exists(md.sidecar_path(p)) for p in ds.samples)
    ds_w = md.MotionVideoDataset(data_paths=[synth["csv"], synth["csv"]], datasets_weights=[0.5, 0.5], frames_per_clip=8,
                                 dataset_fpcs=[8, 8], fps=2, frame_step=None, transform=md.MotionTransform(view2_scale=None),
                                 require_can=True)
    assert len(ds_w) == 4 and ds_w.num_samples_per_dataset == [2, 2] and len(ds_w.sample_weights) == 4
    assert ds_w.per_dataset_indices[3] == (1, 1)
    # calibration: a pitched / yawed mount changes the rays but not K; rays_road_frame=False keeps the camera frame
    side = make_sidecar(24, device="tici", cal_pitch=0.05, cal_yaw=-0.02)
    save_sidecar(side, synth["dir"] / "dddd3333dddd3333.can.npz")
    write_clip(synth["dir"] / "dddd3333dddd3333.mp4", seed=9)
    csv = synth["dir"] / "cal.csv"
    csv.write_text(f"{synth['dir'] / 'dddd3333dddd3333.mp4'} 0\n")
    _seed_all(2)
    e_road = _dataset(str(csv), view2=False)[0][2]
    _seed_all(2)
    e_cam = md.MotionVideoDataset(data_paths=[str(csv)], frames_per_clip=8, dataset_fpcs=[8], fps=2, frame_step=None,
                                  transform=md.MotionTransform(view2_scale=None), rays_road_frame=False)[0][2]
    assert np.allclose(e_road["cal_pitch_yaw"].numpy(), [0.05, -0.02], atol=1e-6) and e_cam["cal_pitch_yaw"].tolist() == [0.0, 0.0]
    assert torch.equal(e_road["K"], e_cam["K"]) and not torch.allclose(e_road["rays"], e_cam["rays"])
    assert np.allclose(e_cam["rays"].numpy(), md.patch_center_rays(e_cam["K"].numpy(), (256, 512)), atol=1e-6)
    assert np.allclose(e_road["rays"].numpy(), md.rays_to_calibrated_frame(e_cam["rays"].numpy(), 0.05, -0.02), atol=1e-6)


def test_dataset_collates_through_mask_collator_and_loader(synth):
    ds = _dataset(synth["csv"])
    collator = MaskCollator(cfgs_mask=MASK_CFGS, dataset_fpcs=[8], crop_size=(256, 512), patch_size=(16, 16), tubelet_size=2)
    _seed_all(0)
    batches = collator([ds[0], ds[1]])
    assert len(batches) == 1
    udata, masks_enc, masks_pred = batches[0]
    assert udata[0][0].shape == (2, 3, 8, 256, 512) and udata[1].tolist() == [0, 0] and udata[3][0].shape == (2, 8)
    extra = udata[2]
    assert extra["K"].shape == (2, 3, 3) and extra["rays"].shape == (2, 16, 32, 3) and extra["seq"].shape == (2, 7, 7)
    assert extra["seq_valid"].shape == (2, 7, 7) and extra["target"].shape == (2, 7) and extra["cal_pitch_yaw"].shape == (2, 2)
    assert extra["view2"].shape == (2, 3, 8, 256, 512) and extra["common_mask"].shape == (2, 512)
    assert extra["s"].shape == (2,) and extra["dt"].dtype == torch.float32 and extra["has_motion"].tolist() == [True, True]
    assert len(masks_enc) == 2 and len(masks_pred) == 2
    for me, mp in zip(masks_enc, masks_pred):
        assert me.shape[0] == 2 and mp.shape[0] == 2 and me.dtype == torch.int64 and me.max() < 2048
    # the factory: unchanged collator, DistributedSampler, one batch from the loader
    dataset, loader, sampler = md.make_motion_videodataset(
        data_paths=[synth["csv"]], batch_size=2, dataset_fpcs=[8], fps=2, transform=md.MotionTransform(),
        collator=collator, num_workers=0, world_size=1, rank=0, drop_last=True, pin_mem=False, persistent_workers=False, seed=3)
    assert isinstance(dataset, md.MotionVideoDataset) and len(loader) == 1
    sampler.set_epoch(0)
    (udata, masks_enc, masks_pred), = next(iter(loader))
    assert udata[0][0].shape == (2, 3, 8, 256, 512) and udata[2]["target"].shape == (2, 7) and masks_enc[0].shape[0] == 2


def test_dataset_equivalent_to_vendored_pipeline(synth):
    """With view 2 off the clips, clip indices and labels are identical to VideoDataset + VideoTransform (flip off),
    and the global RNG streams end in the same state: the basis for the trainer's motion.enabled=false equivalence."""
    vt = VideoTransform(random_horizontal_flip=False, random_resize_aspect_ratio=(1.8, 2.2), random_resize_scale=(0.5, 1.0), crop_size=(256, 512))
    vds = VideoDataset(data_paths=[synth["csv"]], frames_per_clip=8, dataset_fpcs=[8], fps=2, frame_step=None, transform=vt)
    mds = _dataset(synth["csv"], view2=False)
    order = [0, 1, 2, 2, 0, 1]
    _seed_all(11)
    ref = [vds[i] for i in order]
    ref_state = (random.random(), np.random.rand(), torch.rand(1).item())
    _seed_all(11)
    got = [mds[i] for i in order]
    got_state = (random.random(), np.random.rand(), torch.rand(1).item())
    for (rb, rl, rc), (gb, gl, _extra, gc) in zip(ref, got):
        assert torch.equal(rb[0], gb[0]) and rl == gl and np.array_equal(rc[0], gc[0])
    assert ref_state == got_state


def test_dataset_private_rng_reproducible(synth):
    def draws(seed, torch_seed=0):
        ds = _dataset(synth["csv"], seed=seed)
        _seed_all(torch_seed)
        out = []
        for i in range(12):
            e = ds[i % 3][2]
            out.append((int(e["s"]), int(e["d"]), tuple(e["box2"].tolist())))
        return out

    a, b, c = draws(0), draws(0), draws(1)
    assert a == b and a != c  # same (seed, torch.initial_seed()) -> same slot pairs and view-2 boxes
    pairs = {(s, d) for s, d, _ in draws(0) + draws(1) + draws(2) + draws(3)}
    assert pairs <= {(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)} and len(pairs) >= 4


def test_dataset_in_forked_workers_reproducible(synth):
    """num_workers=2 (training path): per-worker private RNG streams, batches reproduce for the same seeds.  The
    vendored MaskCollator's shared step counter races between workers, so masks are excluded from the comparison."""
    csv6 = synth["dir"] / "clips6.csv"
    csv6.write_text(open(synth["csv"]).read() * 2)
    collator = MaskCollator(cfgs_mask=MASK_CFGS, dataset_fpcs=[8], crop_size=(256, 512), patch_size=(16, 16), tubelet_size=2)

    def run(seed):
        _seed_all(seed)
        _, loader, sampler = md.make_motion_videodataset(
            data_paths=[str(csv6)], batch_size=2, dataset_fpcs=[8], fps=2, transform=md.MotionTransform(), collator=collator,
            num_workers=2, drop_last=True, pin_mem=False, persistent_workers=False, seed=0)
        sampler.set_epoch(0)
        out = []
        for ((udata, masks_enc, masks_pred),) in loader:
            e = udata[2]
            assert udata[0][0].shape == (2, 3, 8, 256, 512) and masks_enc[0].shape[0] == 2 and e["view2"].shape[0] == 2
            out.append((udata[0][0].double().sum().item(), e["s"].tolist(), e["d"].tolist(), e["box1"].tolist(), e["box2"].tolist(),
                        udata[3][0].tolist(), e["target"].tolist(), e["has_motion"].tolist()))
        return out

    a, b, c = run(5), run(5), run(6)
    assert len(a) == 3 and a == b and a != c


# ------------------------------------------------------------------------------------------------- real clips
def _real_clips():
    mp4s = sorted(glob.glob(os.path.join(REAL_CLIP_ROOT, "*", "*.mp4")))
    with_side = [p for p in mp4s if os.path.exists(md.sidecar_path(p))]
    without = [p for p in mp4s if not os.path.exists(md.sidecar_path(p))]
    return with_side, without


@pytest.mark.skipif(not os.path.isdir(REAL_CLIP_ROOT), reason="local comma clips not present")
def test_real_local_clips(tmp_path):
    with_side, without = _real_clips()
    if len(with_side) < 2:
        pytest.skip("need at least two local clips with sidecars")
    paths = with_side[:2] + (without[:1] if without else with_side[2:3])
    csv = tmp_path / "real.csv"
    csv.write_text("".join(f"{p} 0\n" for p in paths))
    ds = _dataset(str(csv))
    _seed_all(1)
    for i, p in enumerate(paths):
        item = ds[i]
        s, d = _check_item(item)
        extra = item[2]
        side = md.load_sidecar(md.sidecar_path(p))
        if side is None:
            assert not extra["has_motion"] and int(extra["device_id"]) == 0
            continue
        assert extra["has_motion"] and int(extra["device_id"]) == md.DEVICE_IDS[side["device"]]
        assert side["device"] in md.DEVICE_CAMERAS and "heading_src" not in side
        assert torch.isfinite(extra["target"]).all() and torch.isfinite(extra["seq"]).all()
        # real kept-frame timing deviates by up to ~0.15 s from the nominal 0.5 s spacing on a few clips
        assert float(extra["dt"]) == pytest.approx(float(d), abs=0.2)
        if bool(extra["target_valid"][0]):
            assert abs(float(extra["target"][0])) < 50.0  # < 50 m in <= 3 s
            if bool(extra["target_valid"][3]):  # dx ~ v_mean * dt for near-constant speed
                assert float(extra["target"][0]) == pytest.approx(float(extra["target"][3]) * float(extra["dt"]), abs=3.0)
    collator = MaskCollator(cfgs_mask=MASK_CFGS, dataset_fpcs=[8], crop_size=(256, 512), patch_size=(16, 16), tubelet_size=2)
    (udata, masks_enc, masks_pred), = collator([ds[0], ds[1]])
    assert udata[0][0].shape == (2, 3, 8, 256, 512) and udata[2]["K"].shape == (2, 3, 3) and masks_enc[0].shape[0] == 2


def _real_sidecars_per_device(max_per_device=40):
    with_side, _ = _real_clips()
    out = {}
    for p in with_side:
        side = md.load_sidecar(md.sidecar_path(p))
        if side is None or "wdev_z" not in side:
            continue
        dev = side["device"]
        if len(out.setdefault(dev, [])) < max_per_device:
            out[dev].append(side)
        if all(len(v) >= max_per_device for v in out.values()) and len(out) == 3:
            break
    return out


@pytest.mark.skipif(not os.path.isdir(REAL_CLIP_ROOT), reason="local comma clips not present")
def test_real_sidecar_yaw_signs_per_device():
    """The documented sign conventions on real clips, per device (tici / tizi / mici are all present locally):
    (1) the gyro agrees with the dead-reckoning heading after the same sign flip (corr(YAW_RATE_SIGN*wdev_z*dt,
    D_YAW_SIGN*d_yaw) > 0.5); (2) the CCW yaw is consistent with the left-positive lateral motion: corr(d_left,
    D_YAW_SIGN*d_yaw) > 0.5 and the unicycle chord relation d_left = d_forward*tan(dyaw_ccw/2) holds far better than
    with the opposite sign (this is what the mirrored composition of the first version got wrong)."""
    per_dev = _real_sidecars_per_device()
    assert per_dev, "no sidecar with wdev_z found"
    for dev, sides in per_dev.items():
        xs, ys, dl, df, dyc = [], [], [], [], []
        for side in sides:
            w, dy, t = side["wdev_z"].astype(np.float64), side["d_yaw"].astype(np.float64), side["t_sof"].astype(np.float64)
            dt = np.diff(t) * 1e-9
            x, y = md.YAW_RATE_SIGN * w[:-1] * dt, md.D_YAW_SIGN * dy[:-1]
            m = np.isfinite(x) & np.isfinite(y)
            xs.append(x[m])
            ys.append(y[m])
            turning = np.isfinite(dy) & np.isfinite(side["d_left"]) & np.isfinite(side["d_forward"]) & (np.abs(dy) > 0.01)
            dl.append(side["d_left"].astype(np.float64)[turning])
            df.append(side["d_forward"].astype(np.float64)[turning])
            dyc.append(md.D_YAW_SIGN * dy[turning])
        X, Y = np.concatenate(xs), np.concatenate(ys)
        assert len(X) > 500 and np.corrcoef(X, Y)[0, 1] > 0.5, dev
        assert np.mean(np.abs(Y - X)) < np.mean(np.abs(Y + X)), dev
        DL, DF, DY = np.concatenate(dl), np.concatenate(df), np.concatenate(dyc)
        if len(DL) < 50:
            continue  # not enough turning rows in the first clips of this device
        assert np.corrcoef(DL, DY)[0, 1] > 0.5, dev
        res_ccw, res_mirror = np.mean(np.abs(DL - DF * np.tan(DY / 2))), np.mean(np.abs(DL + DF * np.tan(DY / 2)))
        assert res_ccw < 0.2 * res_mirror, (dev, res_ccw, res_mirror)
        # rows just above the 0.01 rad threshold are noisy (low speed): 88 % on the first mici clips, 99 % overall
        assert np.mean(np.sign(DL) == np.sign(DY)) > 0.8, dev
    # integrated targets on real turning 3-s windows: dy and dyaw share their sign (left turn -> both positive)
    agree = total = 0
    for sides in per_dev.values():
        for side in sides[:20]:
            for c0 in range(0, 100, 25):
                out = md.motion_targets(side, np.arange(c0, c0 + 8), 0, 3)
                if out["target_valid"][:3].all() and abs(out["target"][2]) > 0.1:
                    total += 1
                    agree += int(np.sign(out["target"][1]) == np.sign(out["target"][2]))
    assert total >= 20 and agree / total > 0.9, (agree, total)


def test_motion_targets_implausible_spikes_are_masked():
    """A heading-reset spike in d_yaw / wdev_z (observed in 34 / 22 subset-v0 clips) must not become a target."""
    from pretrain.motion_data import PLAUSIBLE_RANGES, MOTION_SEQ_FEATURES, MOTION_TARGETS, _gather

    side = make_sidecar(30, v=10.0, omega=0.1)
    side["d_yaw"] = np.asarray(side["d_yaw"], dtype=np.float64).copy()
    side["wdev_z"] = np.asarray(side["wdev_z"], dtype=np.float64).copy()
    side["d_yaw"][7] = 2.5     # 5 rad/s over one 0.5 s step: a glitch, not a manoeuvre
    side["wdev_z"][7] = 3.0    # rad/s
    idx = np.arange(30)
    # window of slots 2 -> 4 covers kept frames 4..8 (contains row 7); dyaw composes over steps 4..7
    out = md.motion_targets(side, idx, s=2, d=2)
    names = MOTION_TARGETS
    assert not out["target_valid"][names.index("dyaw")]
    assert not out["target_valid"][names.index("yaw_rate_mean")]
    # the spike sits in the LAST step of this window: the final position does not depend on the final heading change
    assert out["target_valid"][names.index("dx")] and out["target_valid"][names.index("dy")]
    assert out["target_valid"][names.index("v_mean")] and out["target_valid"][names.index("dv")]
    # spike in the middle of the chain (steps 6..9): every pose after it is unknown -> dx / dy / dyaw invalid
    out_mid = md.motion_targets(side, idx, s=3, d=2)
    assert not out_mid["target_valid"][[names.index(k) for k in ("dx", "dy", "dyaw", "yaw_rate_mean")]].any()
    assert out_mid["target_valid"][names.index("v_mean")]
    f_yr, f_dy, f_v = MOTION_SEQ_FEATURES.index("yaw_rate"), MOTION_SEQ_FEATURES.index("d_yaw"), MOTION_SEQ_FEATURES.index("v_ego")
    row = 7 - 4  # sequence row of kept frame 7
    assert not out["seq_valid"][row, f_yr] and not out["seq_valid"][row, f_dy] and out["seq_valid"][row, f_v]
    assert out["seq"][row, f_yr] == 0.0 and out["seq_mask"][row]
    # a window that does not touch row 7 is unaffected
    out2 = md.motion_targets(side, idx, s=6, d=2)
    assert out2["target_valid"].all()
    # the plain gather applies the same bounds; raw access keeps the value
    g = _gather(side, "d_yaw", np.array([7]))
    assert np.isnan(g[0]) and _gather(side, "d_yaw", np.array([7]), plausible=False)[0] == 2.5
    assert PLAUSIBLE_RANGES["wdev_z"] == (-1.0, 1.0)


# ------------------------------------------------------------------------------ public mixture sidecars (2026-10-13)
# comma_export/docs/public_mixture_spec.md section 6: K_clip preference, new devices, PLAUSIBLE_RANGES parity and a
# synthetic DrivingDojo sidecar built with the EXPORTER's own helpers going through load_sidecar / motion_targets / dataset.
COMMA_EXPORT_ROOT = _ROOT.parent / "comma_export"  # sibling checkout: /home/henry/Desktop/Qwen_Drive/comma_export
# approximate CameraFpgaP0H120 intrinsics (spec section 6 numbers); the local sample's camera_parameters.json has
# fx 1013.6 / fy 1014.3 / centre (953.9, 507.4) -- any K exercises the K_clip preference, the exact values do not matter
DOJO_K_SRC = np.array([[1005.7, 0.0, 1015.9], [0.0, 1005.7, 513.1], [0.0, 0.0, 1.0]])


def _public_common():
    """``publicdata.common`` of the comma_export checkout (skips when absent).  The path is appended, not prepended, so
    frontdrive's own top-level names (tests, config, ...) keep precedence over comma_export's."""
    if not (COMMA_EXPORT_ROOT / "publicdata" / "common.py").is_file():
        pytest.skip(f"comma_export checkout not found at {COMMA_EXPORT_ROOT}")
    if str(COMMA_EXPORT_ROOT) not in sys.path:
        sys.path.append(str(COMMA_EXPORT_ROOT))
    import importlib

    return importlib.import_module("publicdata.common")


def make_public_dojo_sidecar(pc, n=24, v=8.0, yaw_rate_ccw=0.2):
    """DrivingDojo-style sidecar built with the exporter's helpers: constant speed ``v`` on a turn of CCW yaw rate
    ``yaw_rate_ccw`` (> 0 = LEFT turn) sampled on the dojo 0.4 / 0.6 s frame grid.  Planar pose track ->
    ``steps_from_planar_pose`` (d_left left +, d_yaw clockwise +), clockwise heading psi = -yaw_ccw, wdev_z =
    ``yaw_rate_from_heading`` (clockwise +), K_clip from the 64 deg ``fov_box`` of the sample camera -- the spec's
    section 3.2 chain, nothing pre-negated.  Returns (sidecar dict, K_clip, frame times s, crop box)."""
    gaps = np.where(np.arange(n - 1) % 2 == 0, 0.4, 0.6)
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    th = yaw_rate_ccw * t  # CCW heading from +x (right-handed planar frame)
    R = v / yaw_rate_ccw
    xy = np.stack([R * np.sin(th), R * (1.0 - np.cos(th))], axis=1)  # circle through the origin, heading th
    d_fwd, d_left, d_yaw = pc.steps_from_planar_pose(xy, th)
    psi_cw = -th
    wdev_z = pc.yaw_rate_from_heading(psi_cw, t)
    box = pc.fov_box(DOJO_K_SRC, 1920, 1080, 64.0)
    K_clip = pc.crop_resize_intrinsics(DOJO_K_SRC, box)
    t_ns = (1_700_000_000 * 10**9 + np.round(t * 1e9)).astype(np.int64)
    side = pc.make_sidecar(n, "dojo", K_clip, 64.0, 10.0, t_ns, np.arange(n) * 2, np.full(n, v), psi_cw, wdev_z,
                           d_fwd, d_left, d_yaw, ["pose"] * n)
    return side, K_clip, t, box


def test_sidecar_intrinsics_prefers_finite_k_clip(tmp_path):
    """A finite (3, 3) ``K_clip`` wins over the device fallback; absent / NaN / inf / wrong shape / zero focal fall
    back; a comma sidecar without K_clip gives exactly clip_intrinsics(device), so comma training is unchanged."""
    k_clip = np.array([[409.6, 0.0, 256.3], [0.0, 409.6, 128.9], [0.0, 0.0, 1.0]], dtype=np.float32)
    side = make_sidecar(12, device="tici")
    assert "K_clip" not in side
    assert np.array_equal(md.sidecar_intrinsics(side, "tici"), md.clip_intrinsics("tici"))
    assert np.array_equal(md.sidecar_intrinsics(None, "mici"), md.clip_intrinsics("mici"))
    assert np.array_equal(md.sidecar_intrinsics({"device": "tizi"}, "tizi"), md.clip_intrinsics("tizi"))
    assert np.array_equal(md.sidecar_intrinsics(None, ""), md.clip_intrinsics("tici"))
    got = md.sidecar_intrinsics(dict(side, K_clip=k_clip), "tici")
    assert got.dtype == np.float32 and got.shape == (3, 3) and np.array_equal(got, k_clip)
    assert np.array_equal(md.sidecar_intrinsics({"K_clip": k_clip.astype(np.float64).tolist()}, "dojo"), k_clip)
    nan_k = k_clip.copy()
    nan_k[0, 2] = np.nan
    for broken in (nan_k, k_clip[:2], k_clip.reshape(9), np.zeros((3, 3), np.float32), np.full((3, 3), np.inf, np.float32)):
        assert np.array_equal(md.sidecar_intrinsics({"K_clip": broken}, "dojo"), md.device_fallback_intrinsics("dojo"))
    # through the file format: a comma-style sidecar with an extra K_clip loads it as a (3, 3) float32 array
    save_sidecar(dict(side, K_clip=k_clip), tmp_path / "k.can.npz")
    loaded = md.load_sidecar(str(tmp_path / "k.can.npz"))
    assert loaded["K_clip"].shape == (3, 3) and loaded["K_clip"].dtype == np.float32
    assert np.array_equal(md.sidecar_intrinsics(loaded, loaded["device"]), k_clip)


def test_public_device_ids_and_fallback_intrinsics():
    """covla / dojo / openscene = ids 4 / 5 / 6 with the comma ids unchanged; clip_intrinsics does not crash for them;
    the device fallback is the full-frame centred 2:1 crop (the spec's covla K_clip) and identical to clip_intrinsics
    for the comma devices and unknown strings."""
    assert md.DEVICE_IDS == {"": 0, "tici": 1, "tizi": 2, "mici": 3, "covla": 4, "dojo": 5, "openscene": 6}
    assert len(set(md.DEVICE_IDS.values())) == len(md.DEVICE_IDS) and set(md.DEVICE_CAMERAS) <= set(md.DEVICE_IDS)
    assert md.PUBLIC_DEVICES == ("covla", "dojo", "openscene") and set(md.PUBLIC_DEVICES) <= set(md.DEVICE_CAMERAS)
    assert md.DEVICE_CAMERAS["covla"] == (1928, 1208, 2648.0) and md.DEVICE_CAMERAS["dojo"] == (1920, 1080, 1005.7)
    assert md.DEVICE_CAMERAS["openscene"] == (1920, 1080, 1545.0)
    assert md.DEVICE_CAMERAS["tici"] == (1928, 1208, 2648.0) and md.DEVICE_CAMERAS["mici"] == (1344, 760, 1141.5)
    for dev in md.PUBLIC_DEVICES:
        k = md.clip_intrinsics(dev)  # the comma chain is well defined (no crash) but is not the export geometry
        assert k.shape == (3, 3) and np.isfinite(k).all()
        fb = md.device_fallback_intrinsics(dev)
        W, H, f = md.DEVICE_CAMERAS[dev]
        h = min(H, W // 2)
        assert fb.dtype == np.float32 and fb.shape == (3, 3)
        assert fb[0, 0] == pytest.approx(f * 512 / W, rel=1e-6) and fb[1, 1] == pytest.approx(f * 256 / h, rel=1e-6)
        assert fb[0, 2] == pytest.approx(256.0, abs=1e-4) and fb[1, 2] == pytest.approx(128.0, abs=1e-4)
        assert fb[0, 1] == 0 and fb[1, 0] == 0 and fb[2].tolist() == [0, 0, 1]
        assert math.degrees(2 * math.atan(256 / fb[0, 0])) > 39.0  # HFOV 40 (covla) / 63.7 (openscene) / 87 (dojo)
    assert np.allclose(md.device_fallback_intrinsics("covla"), [[703.2, 0, 256], [0, 703.2, 128], [0, 0, 1]], atol=0.05)
    assert md.device_fallback_intrinsics("openscene")[0, 0] == pytest.approx(412.0, abs=0.01)
    assert md.device_fallback_intrinsics("dojo")[0, 0] == pytest.approx(1005.7 * 512 / 1920, rel=1e-6)
    for dev in ("", "tici", "tizi", "mici", "eon"):
        assert np.array_equal(md.device_fallback_intrinsics(dev), md.clip_intrinsics(dev))
    # a `.device` file may name a public device
    assert md.clip_device("/nowhere/x.mp4", {"device": "openscene"}) == "openscene"


def test_plausible_ranges_match_public_exporter():
    """The exporter's QC bounds and device list are the consumer's; the fallback numbers equal the exporter's helpers."""
    pc = _public_common()
    assert pc.PLAUSIBLE_RANGES == md.PLAUSIBLE_RANGES
    assert set(pc.DEVICES) == set(md.PUBLIC_DEVICES)
    for dev in md.PUBLIC_DEVICES:
        W, H, f = md.DEVICE_CAMERAS[dev]
        k_full = [[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]]
        assert np.allclose(md.device_fallback_intrinsics(dev), pc.crop_resize_intrinsics(k_full, pc.centre_box_2to1(W, H)), atol=1e-4)
    # the two crop_resize_intrinsics implementations agree bit for bit
    box = (37, 122, 1850, 925)
    assert np.array_equal(pc.crop_resize_intrinsics(DOJO_K_SRC, box), md.crop_resize_intrinsics(DOJO_K_SRC, box, (256, 512)))


def test_public_dojo_sidecar_through_loader_and_motion_targets(tmp_path):
    """A dojo sidecar from publicdata.common.make_sidecar (left turn, 0.4 / 0.6 s grid): exporter QC passes (sign
    agreement 1.0, chord residual 0), load_sidecar keeps K_clip / fov_h_deg / src_fps and drops heading_src,
    sidecar_intrinsics returns K_clip, and motion_targets gives finite valid targets in the CCW frame (dy > 0,
    dyaw > 0, yaw_rate > 0) with dt from the real t_sof gaps."""
    pc = _public_common()
    n, v, w = 24, 8.0, 0.2
    side, K_clip, t, box = make_public_dojo_sidecar(pc, n=n, v=v, yaw_rate_ccw=w)
    # exporter side: clockwise-positive d_yaw / wdev_z are negative on a left turn, d_left positive, chord relation exact
    assert (side["d_yaw"][:-1] < 0).all() and (side["d_left"][:-1] > 0).all() and (side["wdev_z"] < 0).all()
    assert side["heading_src"][0] == "pose" and str(side["device"]) == "dojo" and side["K_clip"].shape == (3, 3)
    assert np.isnan(side["steer_deg"]).all() and np.isnan(side["cc_enabled"]).all() and side["standstill"].sum() == 0
    q = pc.summarize_sidecar(side)
    assert q["sign_agreement"] == 1.0 and q["n_turn_steps"] == n - 1 and q["chord_resid_med_m"] < 1e-4 and q["frac_out_of_range"] == 0.0
    assert q["v_mean"] == pytest.approx(v) and q["frac_standstill"] == 0.0
    path = tmp_path / "ee" / "eeee4444eeee4444.can.npz"
    pc.save_sidecar(path, side)
    loaded = md.load_sidecar(str(path))
    assert loaded is not None and loaded["device"] == "dojo" and "heading_src" not in loaded
    assert isinstance(loaded["fov_h_deg"], float) and loaded["fov_h_deg"] == pytest.approx(64.0) and loaded["src_fps"] == pytest.approx(10.0)
    assert math.isnan(loaded["wheelbase"]) and loaded["K_clip"].shape == (3, 3) and loaded["K_clip"].dtype == np.float32
    assert set(np.unique(np.diff(loaded["t_sof"])).tolist()) == {400_000_000, 600_000_000}
    assert np.array_equal(md.sidecar_intrinsics(loaded, loaded["device"]), K_clip)
    assert K_clip[0, 0] == pytest.approx(1005.7 * 512 / box[2], rel=1e-6) and 400 < K_clip[0, 0] < 420  # 64 deg box
    assert not np.allclose(md.sidecar_intrinsics(loaded, "dojo"), md.device_fallback_intrinsics("dojo"), atol=1.0)
    # consumer: exact arc targets in the CCW frame, dt from t_sof
    frames = np.arange(3, 11)
    R = v / w
    for s, d in [(0, 1), (0, 3), (1, 2), (2, 1)]:
        out = md.motion_targets(loaded, frames, s, d)
        a, b = 2 * s, 2 * (s + d)
        i0, i1 = frames[a], frames[b]
        assert out["target_valid"].all() and np.isfinite(out["target"]).all() and np.isfinite(out["seq"]).all()
        dx, dy, dyaw, v_mean, yr_mean, dv, dyr = out["target"]
        theta = w * (t[i1] - t[i0])
        assert dx == pytest.approx(R * math.sin(theta), rel=1e-4) and dy == pytest.approx(R * (1 - math.cos(theta)), rel=1e-4)
        assert dyaw == pytest.approx(theta, rel=1e-4) and dy > 0 and dyaw > 0
        assert yr_mean == pytest.approx(w, rel=1e-5) and yr_mean > 0 and v_mean == pytest.approx(v)
        assert dv == pytest.approx(0.0, abs=1e-6) and dyr == pytest.approx(0.0, abs=1e-6)
        assert out["dt"] == pytest.approx(t[i1] - t[i0], abs=1e-6) and out["speed"] == pytest.approx(v)
        m = b - a + 1
        assert out["seq_mask"].tolist() == [True] * m + [False] * (7 - m) and out["seq_valid"][:m].all()
        assert np.allclose(out["seq"][:m, IV("yaw_rate")], w) and (out["seq"][:m, IV("d_yaw")] > 0).all()
        assert (out["seq"][:m, IV("d_left")] > 0).all() and np.allclose(out["seq"][:m, IV("v_ego")], v)
        assert np.allclose(out["seq"][:m, IV("d_left")], out["seq"][:m, IV("d_forward")] * np.tan(out["seq"][:m, IV("d_yaw")] / 2), atol=1e-5)
    # comma-only columns (steer / ADAS flags) are NaN in public sidecars: invalid per feature, everything else valid
    out = md.motion_targets(loaded, frames, 0, 1, use_comma_only=True)
    feats = md.MOTION_SEQ_FEATURES + md.COMMA_ONLY_FEATURES
    assert out["seq_mask"].tolist() == [True] * 3 + [False] * 4 and out["target_valid"].all()
    assert not out["seq_valid"][:3, [feats.index(k) for k in md.COMMA_ONLY_FEATURES]].any()
    assert out["seq_valid"][:3, : len(md.MOTION_SEQ_FEATURES)].all()
    # mirrored right turn (DrivingDojo chain): dy < 0, dyaw < 0, yaw_rate < 0 after the CCW flip
    side_r, _, _, _ = make_public_dojo_sidecar(pc, n=n, v=v, yaw_rate_ccw=-w)
    assert (side_r["d_yaw"][:-1] > 0).all() and (side_r["d_left"][:-1] < 0).all() and (side_r["wdev_z"] > 0).all()
    out_r = md.motion_targets(side_r, frames, 0, 3)
    assert out_r["target_valid"].all() and out_r["target"][1] < 0 and out_r["target"][2] < 0 and out_r["target"][4] < 0
    # CoVLA chain (NED positions + clockwise heading via steps_from_ned_pose): a right turn is d_yaw > 0 / d_left < 0
    # in the sidecar and comes out dy < 0 / dyaw < 0 here -- the exporter must never pre-negate
    psi = np.linspace(0.0, 0.6, 13)  # clockwise-from-north heading, turning right
    ned = np.stack([R * np.sin(psi), R * (1.0 - np.cos(psi))], axis=1)  # (north, east)
    d_fwd, d_left, d_yaw = pc.steps_from_ned_pose(ned, psi)
    assert (d_yaw[:-1] > 0).all() and (d_left[:-1] < 0).all() and (d_fwd[:-1] > 0).all()
    x, y, yaw = md.compose_se2(d_fwd[:-1], d_left[:-1], md.D_YAW_SIGN * d_yaw[:-1])
    assert y[-1] < 0 and yaw[-1] == pytest.approx(-0.6, abs=1e-5) and x[-1] == pytest.approx(R * math.sin(0.6), rel=1e-4)


def test_dataset_uses_k_clip_of_public_sidecar(synth, tmp_path):
    """A dojo clip written with the public helpers (encode_mp4 + make_sidecar / save_sidecar) through the dataset:
    device_id 5, K / K2 / rays from the sidecar's K_clip (not the device fallback), finite CAN targets identical to the
    direct computation; the comma tici clip of the fixture (no K_clip) keeps the clip_intrinsics('tici') chain."""
    pc = _public_common()
    side, K_clip, _, _ = make_public_dojo_sidecar(pc, n=24)
    cid = "ffff5555ffff5555"
    mp4 = tmp_path / cid[:2] / f"{cid}.mp4"
    base = np.random.default_rng(3).integers(0, 256, (256, 512, 3), dtype=np.uint8)
    frames = [np.roll(base, shift=9 * i, axis=1).copy() for i in range(24)]
    pc.encode_mp4(mp4, frames)
    pc.save_sidecar(md.sidecar_path(str(mp4)), side)
    csv = tmp_path / "public.csv"
    csv.write_text(f"{mp4} 0\n")
    ds = _dataset(str(csv))
    _seed_all(0)
    buffer_list, label, extra, clip_indices = ds[0]
    assert buffer_list[0].shape == (3, 8, 256, 512) and label == 0 and len(clip_indices[0]) == 8
    assert int(extra["device_id"]) == md.DEVICE_IDS["dojo"] == 5 and bool(extra["has_motion"])
    out_hw = (256, 512)
    k_buf = md.crop_resize_intrinsics(K_clip, (0, 0, 512, 256), out_hw)  # 512x256 buffer: K_clip itself
    assert np.allclose(k_buf, K_clip, atol=1e-4)
    box1, box2 = tuple(extra["box1"].tolist()), tuple(extra["box2"].tolist())
    assert np.allclose(extra["K"].numpy(), md.crop_resize_intrinsics(k_buf, box1, out_hw), atol=1e-3)
    assert np.allclose(extra["K2"].numpy(), md.crop_resize_intrinsics(k_buf, box2, out_hw), atol=1e-3)
    wrong = md.crop_resize_intrinsics(md.device_fallback_intrinsics("dojo"), box1, out_hw)
    assert abs(float(extra["K"][0, 0]) - wrong[0, 0]) > 50.0  # 64 deg export box vs full-width fallback
    assert extra["cal_pitch_yaw"].tolist() == [0.0, 0.0]  # public sidecars carry zero calibration: plain camera rays
    assert np.allclose(extra["rays"].numpy(), md.patch_center_rays(extra["K"].numpy(), out_hw), atol=1e-6)
    assert np.allclose(extra["rays2"].numpy(), md.patch_center_rays(extra["K2"].numpy(), out_hw), atol=1e-6)
    assert torch.isfinite(extra["target"]).all() and torch.isfinite(extra["seq"]).all() and bool(extra["target_valid"].all())
    assert float(extra["target"][1]) > 0 and float(extra["target"][2]) > 0 and float(extra["target"][4]) > 0  # left turn, CCW
    assert float(extra["speed"]) == pytest.approx(8.0) and float(extra["dt"]) == pytest.approx(float(extra["d"]), abs=1e-6)
    ref = md.motion_targets(md.load_sidecar(md.sidecar_path(str(mp4))), clip_indices[0], int(extra["s"]), int(extra["d"]))
    for key in ("seq", "seq_valid", "seq_mask", "target", "target_valid", "dt", "speed"):
        assert np.array_equal(extra[key].numpy(), np.asarray(ref[key])), key
    # comma tici clip of the fixture (no K_clip): unchanged qcamera-chain intrinsics
    _seed_all(0)
    e_tici = _dataset(synth["csv"], view2=False)[1][2]
    assert int(e_tici["device_id"]) == md.DEVICE_IDS["tici"] and "K_clip" not in synth["sides"][1]
    k_buf_t = md.crop_resize_intrinsics(md.clip_intrinsics("tici"), (0, 0, 512, 256), out_hw)
    assert np.allclose(e_tici["K"].numpy(), md.crop_resize_intrinsics(k_buf_t, tuple(e_tici["box1"].tolist()), out_hw), atol=1e-3)
    assert np.array_equal(md.sidecar_intrinsics(synth["sides"][1], "tici"), md.clip_intrinsics("tici"))
