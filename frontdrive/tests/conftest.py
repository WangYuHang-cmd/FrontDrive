"""Shared pytest fixtures for frontdrive/tests: synthetic 2 fps clips + CAN sidecars that follow the comma export
spec (see pretrain/motion_data.py), generated once per cache directory and reused.

Cache location: $FRONTDRIVE_TEST_CACHE if set (e.g. the agent scratchpad), else pytest's per-session basetemp.
Markers: ``gpu`` (needs CUDA; the tests skip themselves when it is unavailable).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest

FRONTDRIVE_ROOT = pathlib.Path(__file__).resolve().parents[1]
NAVSIM_V1 = FRONTDRIVE_ROOT / "third_party" / "Drive-JEPA" / "navsim_v1"

# Make `pretrain.*` (frontdrive root), `vjepa2.*` (navsim_v1) and the vendored `app.*` / `src.*` (navsim_v1/vjepa2)
# importable even when PYTHONPATH is unset (env/drivejepa_env.sh sets the latter two); the test modules repeat these
# inserts so they also run stand-alone. Order mirrors the env script: vjepa2 first, then navsim_v1, then the root.
for _p in (NAVSIM_V1 / "vjepa2", NAVSIM_V1, FRONTDRIVE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test needs a CUDA device (skipped otherwise)")


def _write_clip(path: pathlib.Path, n_frames: int, size, rng: np.random.Generator, fps: int = 2) -> None:
    """H.264 mp4 (libx264, yuv420p, `fps`) of `n_frames` frames of a drifting striped pattern, size = (H, W)."""
    import av

    H, W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "18", "preset": "ultrafast"}
        ph, sp = rng.uniform(0, 6.28, 3), rng.uniform(0.05, 0.2, 3)
        for t in range(n_frames):
            img = np.stack([127 + 100 * np.sin(xx / 20 + ph[k] + sp[k] * t + yy / 30 * k) for k in range(3)], -1)
            img += rng.normal(0, 6, img.shape)
            frame = av.VideoFrame.from_ndarray(np.clip(img, 0, 255).astype(np.uint8), format="rgb24")
            for pk in s.encode(frame):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)


def write_synthetic_sidecar(path: pathlib.Path, n_frames: int, rng: np.random.Generator, device: str = "mici",
                            v0: float | None = None, yaw_rate: float | None = None) -> dict:
    """``<clip>.can.npz`` following the comma export (clips.py): one row per kept 2 fps frame, d_forward / d_left =
    motion from kept frame i to i+1 in frame i's vehicle frame (forward m, LEFT-positive m), d_yaw = Delta psi of the
    NED dead-reckoning heading (CLOCKWISE-positive rad), wdev_z = device gyro z (z down -> clockwise-positive rad/s),
    yaw = clockwise heading, last d_* row NaN, t_sof in ns, frame_idx = 10*i, heading_src stored as a unicode array
    (the loader skips it by name), device / wheelbase 0-d arrays.
    Motion: constant speed `v0` (m/s) on an arc of constant `yaw_rate` (rad/s, CCW / left-turn positive, i.e. the
    physical convention of motion_data); the export convention stores -yaw_rate*dt in d_yaw and -yaw_rate in wdev_z."""
    dt = 0.5
    v0 = float(rng.uniform(3.0, 25.0)) if v0 is None else float(v0)
    w = float(rng.uniform(-0.15, 0.15)) if yaw_rate is None else float(yaw_rate)
    v = np.full(n_frames, v0, dtype=np.float32)
    d_yaw = np.full(n_frames, -w * dt, dtype=np.float32)  # clockwise-positive, like the real sidecars
    # exact arc of a unicycle: chord components in the frame at the start of the step
    if abs(w) > 1e-9:
        d_forward = np.full(n_frames, v0 / w * np.sin(w * dt), dtype=np.float32)
        d_left = np.full(n_frames, v0 / w * (1.0 - np.cos(w * dt)), dtype=np.float32)
    else:
        d_forward, d_left = np.full(n_frames, v0 * dt, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)
    for a in (d_forward, d_left, d_yaw):
        a[-1] = np.nan
    yaw = np.concatenate([[0.0], np.cumsum(d_yaw[:-1])]).astype(np.float32)  # clockwise heading
    t0 = int(1e12)
    t_sof = (t0 + np.arange(n_frames) * int(dt * 1e9) + rng.integers(-2_000_000, 2_000_000, n_frames)).astype(np.int64)
    sidecar = dict(
        v_ego=v, a_ego=np.zeros(n_frames, np.float32), steer_deg=np.full(n_frames, np.degrees(w) * 10, np.float32),
        yaw=yaw, wdev_z=np.full(n_frames, -w, np.float32), cal_pitch=np.full(n_frames, 0.1, np.float32),
        cal_yaw=np.zeros(n_frames, np.float32), cc_enabled=np.zeros(n_frames, bool), cc_lat_active=np.zeros(n_frames, bool),
        cc_long_active=np.zeros(n_frames, bool), standstill=np.zeros(n_frames, bool), t_sof=t_sof,
        frame_idx=(np.arange(n_frames) * 10).astype(np.int32), d_forward=d_forward, d_left=d_left, d_yaw=d_yaw,
        heading_src=np.array(["synthetic"] * n_frames), device=np.array(device), wheelbase=np.array(2.7),
    )
    np.savez(str(path), **sidecar)
    return {"v": v0, "yaw_rate": w, "device": device}


def make_synthetic_clips(root: pathlib.Path, n_clips: int = 4, size=(128, 256), n_frames: int = 24, seed: int = 0,
                         n_without_sidecar: int = 1) -> dict:
    """Create (or reuse) `n_clips` synthetic clips under `root`; the last `n_without_sidecar` get a ``.nocan`` marker
    instead of a sidecar. Returns {"dir", "csv", "mp4s", "sidecars", "meta"}; csv lines are "<mp4> 0"."""
    root = pathlib.Path(root)
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            info = json.loads(manifest.read_text())
            if all(os.path.exists(p) for p in info["mp4s"]) and info.get("params") == [n_clips, list(size), n_frames, seed]:
                return info
        except (ValueError, KeyError):
            pass
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mp4s, sidecars, meta = [], [], []
    for i in range(n_clips):
        clip = root / f"{i:016x}.mp4"
        _write_clip(clip, n_frames, size, rng)
        mp4s.append(str(clip))
        if i < n_clips - n_without_sidecar:
            side = root / f"{i:016x}.can.npz"
            meta.append(write_synthetic_sidecar(side, n_frames, rng, device=("mici" if i % 2 else "tici")))
            sidecars.append(str(side))
        else:
            (root / f"{i:016x}.nocan").touch()
            sidecars.append(None)
            meta.append(None)
    csv = root / "clips.csv"
    csv.write_text("".join(f"{p} 0\n" for p in mp4s))
    info = {"dir": str(root), "csv": str(csv), "mp4s": mp4s, "sidecars": sidecars, "meta": meta,
            "params": [n_clips, list(size), n_frames, seed]}
    manifest.write_text(json.dumps(info))
    return info


@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory) -> pathlib.Path:
    """Directory for generated test data: $FRONTDRIVE_TEST_CACHE or pytest's session basetemp."""
    env = os.environ.get("FRONTDRIVE_TEST_CACHE")
    path = pathlib.Path(env) if env else tmp_path_factory.getbasetemp() / "frontdrive_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def synthetic_clips_factory(test_cache_dir):
    """``factory(name, n_clips=4, size=(128, 256), n_frames=24, seed=0, n_without_sidecar=1) -> info dict``."""

    def factory(name: str = "default", **kwargs) -> dict:
        n = kwargs.get("n_clips", 4)
        h, w = kwargs.get("size", (128, 256))
        t = kwargs.get("n_frames", 24)
        return make_synthetic_clips(test_cache_dir / f"synthetic_{name}_{n}x{t}_{h}x{w}", **kwargs)

    return factory


@pytest.fixture(scope="session")
def tiny_synthetic_clips(synthetic_clips_factory) -> dict:
    """4 clips of 24 frames at 128x256 (source size for the 64x128 tiny config), 3 with sidecars, 1 without."""
    return synthetic_clips_factory("tiny", n_clips=4, size=(128, 256), n_frames=24, seed=0, n_without_sidecar=1)
