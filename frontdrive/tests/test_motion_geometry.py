"""Geometry helpers of pretrain/motion_data.py: intrinsics update, rays, common-region token mask, SE(2) chaining.

Pixel convention: continuous coordinates, pixel (i, j) covers [j, j+1) x [i, i+1); vehicle frame x forward, y left,
yaw CCW; camera frame x right, y down, z forward.  CPU only.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (_ROOT / "third_party/Drive-JEPA/navsim_v1/vjepa2", _ROOT / "third_party/Drive-JEPA/navsim_v1", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pretrain.motion_data import (  # noqa: E402
    CLIP_CROP_ROWS,
    CLIP_H,
    CLIP_W,
    DEVICE_CAMERAS,
    PUBLIC_DEVICES,
    QCAM_H,
    QCAM_W,
    VIEW_FRAME_FROM_DEVICE_FRAME,
    calib_rotation,
    clip_intrinsics,
    common_region_token_mask,
    compose_se2,
    crop_resize_intrinsics,
    device_fallback_intrinsics,
    patch_center_rays,
    rays_to_calibrated_frame,
)


def _project(K, P):
    p = np.asarray(K, np.float64) @ np.asarray(P, np.float64)
    return p[:2] / p[2]


# ---------------------------------------------------------------------------------------------------- intrinsics
def test_clip_intrinsics_documented_numbers_and_fallback():
    K = clip_intrinsics("tici")
    assert K.shape == (3, 3) and K.dtype == np.float32
    assert K[0, 0] == pytest.approx(703.2, abs=0.1) and K[1, 1] == pytest.approx(704.1, abs=0.1)
    assert K[0, 2] == pytest.approx(256.0, abs=1e-3) and K[1, 2] == pytest.approx(128.5, abs=0.05)
    Km = clip_intrinsics("mici")
    assert Km[0, 0] == pytest.approx(434.8, abs=0.1) and Km[1, 1] == pytest.approx(482.5, abs=0.1)
    # horizontal field of view of the 512-wide clip
    assert math.degrees(2 * math.atan(CLIP_W / 2 / K[0, 0])) == pytest.approx(40.0, abs=0.1)
    assert math.degrees(2 * math.atan(CLIP_W / 2 / Km[0, 0])) == pytest.approx(61.0, abs=0.1)
    # closed form of the clips.py chain for the comma devices: fx = f * 512/W ; fy = f * (330/H) * (512/526)
    for dev, (W, H, f) in DEVICE_CAMERAS.items():
        if dev in PUBLIC_DEVICES:  # public mixture devices: clip_intrinsics delegates to the centred 2:1 fallback
            Kd = clip_intrinsics(dev)
            assert np.array_equal(Kd, device_fallback_intrinsics(dev))
            assert Kd[0, 0] == pytest.approx(f * CLIP_W / W, rel=1e-5) and Kd[1, 2] == pytest.approx(CLIP_H / 2, rel=1e-5)
            continue
        Kd = clip_intrinsics(dev)
        assert Kd[0, 0] == pytest.approx(f * CLIP_W / W, rel=1e-5)
        assert Kd[1, 1] == pytest.approx(f * (QCAM_H / H) * (CLIP_W / QCAM_W), rel=1e-5)
        assert Kd[1, 2] == pytest.approx((QCAM_H / 2 - CLIP_CROP_ROWS[0]) * CLIP_W / QCAM_W, rel=1e-5)
    # unknown / empty device -> tici numbers
    assert np.array_equal(clip_intrinsics(""), K) and np.array_equal(clip_intrinsics("eon"), K)
    assert np.array_equal(clip_intrinsics("tizi"), K)


def test_crop_resize_intrinsics_projects_consistently():
    rng = np.random.default_rng(0)
    K = clip_intrinsics("mici")
    # identity crop + resize keeps K
    assert np.allclose(crop_resize_intrinsics(K, (0, 0, CLIP_W, CLIP_H), (CLIP_H, CLIP_W)), K, atol=1e-4)
    for _ in range(25):
        w, h = int(rng.integers(50, CLIP_W + 1)), int(rng.integers(30, CLIP_H + 1))
        x0, y0 = int(rng.integers(0, CLIP_W - w + 1)), int(rng.integers(0, CLIP_H - h + 1))
        out_hw = [(256, 512), (64, 128), (100, 300)][int(rng.integers(3))]
        K2 = crop_resize_intrinsics(K, (x0, y0, w, h), out_hw)
        for _ in range(5):
            P = np.array([rng.uniform(-20, 20), rng.uniform(-5, 5), rng.uniform(5, 80)])
            u, v = _project(K, P)
            u2, v2 = _project(K2, P)
            assert u2 == pytest.approx((u - x0) * out_hw[1] / w, abs=1e-3)
            assert v2 == pytest.approx((v - y0) * out_hw[0] / h, abs=1e-3)
    # composition: crop A then a crop B inside A's output == direct crop of the composed box
    KA = crop_resize_intrinsics(K, (40, 20, 400, 200), (256, 512))
    KAB = crop_resize_intrinsics(KA, (64, 32, 256, 128), (256, 512))
    # box B in buffer pixels: x = 40 + 64 * 400/512, w = 256 * 400/512 ; y = 20 + 32 * 200/256, h = 128 * 200/256
    K_direct = crop_resize_intrinsics(K, (40 + 64 * 400 / 512, 20 + 32 * 200 / 256, 256 * 400 / 512, 128 * 200 / 256), (256, 512))
    assert np.allclose(KAB, K_direct, rtol=1e-5, atol=1e-3)


# --------------------------------------------------------------------------------------------------------- rays
def test_patch_center_rays_unit_norm_and_centre():
    K = clip_intrinsics("tici")
    rays = patch_center_rays(K, (256, 512), patch=16)
    assert rays.shape == (16, 32, 3) and rays.dtype == np.float32
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-6)
    assert (rays[..., 2] > 0.9).all()  # 40 deg HFOV camera: every ray close to the optical axis
    # cx = 256 is the grid's mirror axis -> left/right symmetric x, identical y
    assert np.allclose(rays[:, :, 0], -rays[:, ::-1, 0], atol=1e-6) and np.allclose(rays[:, :, 1], rays[:, ::-1, 1], atol=1e-6)
    # the four central rays average to (almost) +z: y offset is the 0.5 px cy misalignment only
    centre = rays[7:9, 15:17].reshape(-1, 3).mean(0)
    assert abs(centre[0]) < 1e-6 and abs(centre[1]) < 2e-3 and centre[2] > 0.999
    # sign convention: right of the principal point -> x > 0, below -> y > 0; exact angle of the right-most column
    assert rays[0, 31, 0] > 0 and rays[15, 0, 1] > 0 and rays[0, 0, 0] < 0 and rays[0, 0, 1] < 0
    assert rays[8, 31, 0] / rays[8, 31, 2] == pytest.approx((504 - 256) / K[0, 0], rel=1e-5)
    # a principal point sitting exactly on a patch centre gives the exact +z ray there
    K_on_grid = np.array([[700.0, 0, 264.0], [0, 700.0, 136.0], [0, 0, 1]], np.float32)
    assert np.allclose(patch_center_rays(K_on_grid, (256, 512))[8, 16], [0, 0, 1], atol=1e-7)
    # small images / other patch grids
    assert patch_center_rays(K, (64, 128), patch=16).shape == (4, 8, 3)


def test_rays_to_calibrated_frame_follows_openpilot_calibration():
    """R = calib_rotation(pitch, yaw) is openpilot's device_from_calib (Rz Ry Rx); the direction of travel is the
    calibrated x axis, i.e. V @ R @ e_x in the camera frame, and it must map to the optical axis (0, 0, 1) of the
    road-aligned camera frame.  Pure rotation: norms and angles between rays are preserved; zero angles = identity."""
    V = VIEW_FRAME_FROM_DEVICE_FRAME
    assert np.array_equal(V @ np.array([1.0, 0, 0]), [0, 0, 1])  # device x (forward) -> camera z
    assert np.array_equal(V @ np.array([0, 1.0, 0]), [1, 0, 0]) and np.array_equal(V @ np.array([0, 0, 1.0]), [0, 1, 0])
    rng = np.random.default_rng(0)
    K = clip_intrinsics("mici")
    rays = patch_center_rays(K, (256, 512))
    assert np.array_equal(rays_to_calibrated_frame(rays, 0.0, 0.0), rays)
    assert np.array_equal(rays_to_calibrated_frame(rays, float("nan"), 0.0), rays)  # unknown calibration -> identity
    R = calib_rotation(0.0, 0.0)
    assert np.allclose(R, np.eye(3))
    for _ in range(10):
        p, y, r = rng.uniform(-0.12, 0.12), rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05)
        R = calib_rotation(p, y, r)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        # Rz(yaw) @ Ry(pitch) @ Rx(roll) convention (aerospace / openpilot rot_from_euler)
        cz, sz, cy_, sy_, cx, sx = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        assert np.allclose(R, Rz @ Ry @ Rx)
        R2 = calib_rotation(p, y)  # roll unknown in the sidecar -> 0
        travel_cam = V @ R2 @ np.array([1.0, 0.0, 0.0])  # road-forward direction seen in the device camera frame
        assert np.allclose(rays_to_calibrated_frame(travel_cam, p, y), [0, 0, 1], atol=1e-12)
        out = rays_to_calibrated_frame(rays, p, y)
        assert out.dtype == np.float32 and out.shape == rays.shape
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-6)
        assert np.allclose(out.reshape(-1, 3) @ out.reshape(-1, 3).T, rays.reshape(-1, 3) @ rays.reshape(-1, 3).T, atol=1e-5)
    # a device pitched so that its optical axis is BELOW the direction of travel: the optical axis then points down (+y)
    # in the road frame (travel maps to +z, so the optical axis sits on the other side)
    p = 0.05
    travel = V @ calib_rotation(p, 0.0) @ np.array([1.0, 0, 0])
    optical = rays_to_calibrated_frame(np.array([0.0, 0.0, 1.0]), p, 0.0)
    assert np.sign(optical[1]) == -np.sign(travel[1]) and abs(optical[1]) == pytest.approx(abs(np.sin(p)), abs=1e-6)
    # yaw moves rays sideways only
    opt_y = rays_to_calibrated_frame(np.array([0.0, 0.0, 1.0]), 0.0, 0.03)
    assert abs(opt_y[1]) < 1e-9 and abs(opt_y[0]) == pytest.approx(abs(np.sin(0.03)), abs=1e-6)


# ---------------------------------------------------------------------------------------------- common region
def _brute_force_mask(box_a, box_b, out_hw, patch=16):
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    H, W = out_hw
    hp, wp = H // patch, W // patch
    x0, x1, y0, y1 = max(xa, xb), min(xa + wa, xb + wb), max(ya, yb), min(ya + ha, yb + hb)
    m = np.zeros(hp * wp, bool)
    for r in range(hp):
        for c in range(wp):
            u = xa + (c + 0.5) * patch * wa / W
            v = ya + (r + 0.5) * patch * ha / H
            m[r * wp + c] = (x0 <= u < x1) and (y0 <= v < y1)
    return m


def test_common_region_token_mask_nested_and_overlapping():
    out = (256, 512)
    full = (0, 0, 512, 256)
    assert common_region_token_mask(full, full, out).all()
    assert common_region_token_mask(full, (0, 0, 512, 256), out).shape == (512,)
    # nested, axis aligned with the patch grid: right half of the image -> columns 16..31
    m = common_region_token_mask(full, (256, 0, 256, 256), out).reshape(16, 32)
    assert m[:, 16:].all() and not m[:, :16].any()
    # half-open boundary: a centre at x = 248 is inside [248, 512) but outside [248.5, 512)
    assert common_region_token_mask(full, (248, 0, 264, 256), out).reshape(16, 32)[:, 15].all()
    assert not common_region_token_mask(full, (248.5, 0, 263.5, 256), out).reshape(16, 32)[:, 15].any()
    # nested in a zoomed view a (centres spaced 9.375 input px): analytic column/row sets
    box_a, box_b = (100, 50, 300, 150), (150, 80, 100, 60)
    m = common_region_token_mask(box_a, box_b, out).reshape(16, 32)
    rows, cols = np.where(m)
    assert set(cols) == set(range(5, 16)) and set(rows) == set(range(3, 10))
    assert np.array_equal(m.reshape(-1), _brute_force_mask(box_a, box_b, out))
    # partially overlapping (box b sticks out of box a to the top-left)
    box_b = (0, 0, 200, 120)
    m = common_region_token_mask(box_a, box_b, out).reshape(16, 32)
    rows, cols = np.where(m)
    assert cols.max() == 10 and rows.max() == 6 and cols.min() == 0 and rows.min() == 0
    assert np.array_equal(m.reshape(-1), _brute_force_mask(box_a, box_b, out))
    # disjoint -> nothing; random boxes agree with brute force; tiny grids work
    assert not common_region_token_mask(box_a, (0, 0, 50, 40), out).any()
    rng = np.random.default_rng(1)
    for _ in range(20):
        ba = (rng.uniform(0, 100), rng.uniform(0, 60), rng.uniform(100, 400), rng.uniform(60, 190))
        bb = (rng.uniform(0, 300), rng.uniform(0, 150), rng.uniform(50, 300), rng.uniform(30, 150))
        assert np.array_equal(common_region_token_mask(ba, bb, out), _brute_force_mask(ba, bb, out))
    assert common_region_token_mask((0, 0, 128, 64), (64, 0, 64, 64), (64, 128)).reshape(4, 8)[:, 4:].all()


# ------------------------------------------------------------------------------------------------------ SE(2)
def test_compose_se2_straight_and_arc():
    x, y, yaw = compose_se2([5.0] * 6, [0.0] * 6, [0.0] * 6)
    assert x.shape == (7,) and np.allclose(x, 5.0 * np.arange(7)) and np.allclose(y, 0) and np.allclose(yaw, 0)
    # circular arc: v = 10 m/s, omega = 0.3 rad/s (left turn), dt = 0.5 s, chord deltas in the current frame
    v, omega, dt, n = 10.0, 0.3, 0.5, 8
    R, th = v / omega, omega * dt
    x, y, yaw = compose_se2([R * math.sin(th)] * n, [R * (1 - math.cos(th))] * n, [th] * n)
    k = np.arange(n + 1)
    assert np.allclose(x, R * np.sin(k * th), atol=1e-9) and np.allclose(y, R * (1 - np.cos(k * th)), atol=1e-9)
    assert np.allclose(yaw, k * th, atol=1e-12) and (y[1:] > 0).all()  # left turn -> y (left) positive
    # right turn mirrors y and yaw
    xr, yr, yawr = compose_se2([R * math.sin(th)] * n, [-R * (1 - math.cos(th))] * n, [-th] * n)
    assert np.allclose(xr, x) and np.allclose(yr, -y) and np.allclose(yawr, -yaw)


def test_compose_se2_matches_direct_pose_difference():
    # random absolute poses -> per-step deltas in the current vehicle frame -> chain back and compare with the direct
    # relative pose R(-psi_0) (p_k - p_0), psi_k - psi_0
    rng = np.random.default_rng(3)
    n = 12
    psi = np.cumsum(rng.uniform(-0.4, 0.4, n + 1))
    p = np.cumsum(rng.uniform(-3, 6, (n + 1, 2)), axis=0)
    dp = p[1:] - p[:-1]
    c, s = np.cos(psi[:-1]), np.sin(psi[:-1])
    d_forward = c * dp[:, 0] + s * dp[:, 1]
    d_left = -s * dp[:, 0] + c * dp[:, 1]
    d_yaw = psi[1:] - psi[:-1]
    x, y, yaw = compose_se2(d_forward, d_left, d_yaw)
    rel = p - p[0]
    c0, s0 = math.cos(psi[0]), math.sin(psi[0])
    assert np.allclose(x, c0 * rel[:, 0] + s0 * rel[:, 1], atol=1e-9)
    assert np.allclose(y, -s0 * rel[:, 0] + c0 * rel[:, 1], atol=1e-9)
    assert np.allclose(yaw, psi - psi[0], atol=1e-12)


def test_compose_se2_nan_propagation():
    nan = float("nan")
    x, y, yaw = compose_se2([1, 1, nan, 1], [0, 0, 0, 0], [0, 0, 0, 0])
    assert np.isfinite(x[:3]).all() and np.isnan(x[3:]).all() and np.isfinite(yaw).all()
    x, y, yaw = compose_se2([1, 1, 1, 1], [0, 0, 0, 0], [0, nan, 0, 0])
    assert np.isfinite(yaw[:2]).all() and np.isnan(yaw[2:]).all()
    assert np.isfinite(x[:3]).all() and np.isnan(x[3:]).all()  # step 2 needs cos(yaw_2), which is NaN
    assert np.isnan(y[3:]).all()
    # empty input -> the origin only
    x, y, yaw = compose_se2([], [], [])
    assert x.shape == (1,) and x[0] == 0 and y[0] == 0 and yaw[0] == 0
