"""CPU tests for diagnostics/d2_info_gap.py (plan v4.1 section 3.1 D2): window / target construction on synthetic
sidecars (comma export conventions: d_yaw / wdev_z clockwise +, d_left left +; the targets must come out CCW / left +),
SE(2) composition, probes (linear recovery + multi-permutation shuffled control), GroupKFold clip integrity, pooling,
clip-list exclusion / sharding, shard merging, ``retarget`` on a synthetic features npz, an end-to-end ``probe`` run on a
synthetic features npz, and one check of the yaw sign convention on real local sidecars (skipped when absent).
No GPU and no checkpoint needed."""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np
import pytest

FRONTDRIVE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(FRONTDRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTDRIVE_ROOT))

from diagnostics import d2_info_gap as d2  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def make_sidecar(n: int = 40, v0: float = 10.0, acc: float = 1.0, yaw_rate: float = 0.1, decel_rows=(), standstill_rows=(),
                 nan_dyaw_rows=(), steer: float = 2.5) -> dict:
    """Unicycle on an arc of constant ``yaw_rate`` (PHYSICAL: CCW / left-turn positive, rad/s) with linearly increasing
    speed; d_forward / d_left are exact chords of each 0.5 s step (constant speed within the step, taken at the step
    start) so SE(2) composition can be checked analytically.  Stored with the comma export conventions (see
    pretrain/motion_data.py and the sign measurement in d2_info_gap): d_forward / d_left forward / LEFT positive,
    ``d_yaw = -yaw_rate*dt`` and ``wdev_z = -yaw_rate`` CLOCKWISE positive, ``yaw`` a clockwise heading."""
    dt = 0.5
    t = np.arange(n) * dt
    v = (v0 + acc * t).astype(np.float32)
    a = np.full(n, acc, dtype=np.float32)
    for r in decel_rows:
        a[r] = -3.0
    w = yaw_rate
    d_yaw = np.full(n, -w * dt, dtype=np.float32)  # clockwise-positive, like the real sidecars (D_YAW_SIGN flips it)
    if abs(w) > 1e-12:
        d_forward = (v / w * np.sin(w * dt)).astype(np.float32)
        d_left = (v / w * (1.0 - np.cos(w * dt))).astype(np.float32)  # left turn (w > 0) -> positive d_left
    else:
        d_forward, d_left = (v * dt).astype(np.float32), np.zeros(n, np.float32)
    for arr in (d_forward, d_left, d_yaw):
        arr[-1] = np.nan
    for r in nan_dyaw_rows:
        d_yaw[r] = np.nan
    standstill = np.zeros(n, bool)
    for r in standstill_rows:
        standstill[r] = True
    t_sof = (10**12 + (t * 1e9)).astype(np.int64)
    return dict(v_ego=v, a_ego=a, steer_deg=np.full(n, steer, np.float32), wdev_z=np.full(n, -w, np.float32),
                yaw=np.cumsum(np.nan_to_num(d_yaw)).astype(np.float32), standstill=standstill, t_sof=t_sof,
                frame_idx=(np.arange(n) * 10).astype(np.int32), d_forward=d_forward, d_left=d_left, d_yaw=d_yaw, device="mici")


def save_sidecar(path: pathlib.Path, side: dict) -> None:
    """Write a ``make_sidecar`` dict as a ``.can.npz`` like the export does (0-d device, object heading_src)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v for k, v in side.items() if k != "device"}, device=np.array(side["device"]),
             heading_src=np.array(["x"] * len(side["v_ego"]), dtype=object), wheelbase=np.array(2.7))


def se2_chain(df, dl, dy):
    """Reference composition with explicit 3x3 homogeneous matrices."""
    T = np.eye(3)
    for f, l, y in zip(df, dl, dy):
        c, s = np.cos(y), np.sin(y)
        step = np.array([[c, -s, f], [s, c, l], [0, 0, 1.0]])
        T = T @ step
    return T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])


# ----------------------------------------------------------------------------------------------------------------
# windows / targets
# ----------------------------------------------------------------------------------------------------------------
def test_clip_anchors():
    a = d2.clip_anchors(120, stride=8)
    assert a.tolist() == list(range(7, 112, 8)) and len(a) == 14
    assert (a + d2.HORIZON < 120).all() and (a - d2.HISTORY >= 0).all()
    assert d2.clip_anchors(12, stride=8).tolist() == [7]  # 7 + 4 < 12
    assert d2.clip_anchors(11, stride=8).tolist() == []  # 7 + 4 == 11 is not < 11
    assert d2.clip_anchors(120, stride=1).tolist() == list(range(7, 116))
    with pytest.raises(ValueError):
        d2.clip_anchors(120, stride=0)


def test_wrap_angle():
    x = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi + 1e-9, 2 * np.pi + 0.5, np.nan])
    w = d2.wrap_angle(x)
    assert np.allclose(w[:-1], [0.0, np.pi, np.pi, np.pi, -np.pi + 1e-9, 0.5], atol=1e-9)
    assert (w[:-1] > -np.pi).all() and (w[:-1] <= np.pi).all()
    assert np.isnan(w[-1])


def test_compose_se2_square_and_arc():
    # forward 1, turn left 90 deg, forward 1 -> (1, 1, pi/2) in the start frame
    x, y, yaw = d2.compose_se2([1.0, 1.0], [0.0, 0.0], [np.pi / 2, 0.0])
    assert np.allclose([x, y, yaw], [1.0, 1.0, np.pi / 2])
    # a right turn (CW, negative yaw) ends on the right (negative y)
    x, y, yaw = d2.compose_se2([1.0, 1.0], [0.0, 0.0], [-np.pi / 2, 0.0])
    assert np.allclose([x, y, yaw], [1.0, -1.0, -np.pi / 2])
    # constant-curvature arc: chords compose exactly to the analytic arc end point
    v, w, dt, k = 10.0, 0.2, 0.5, 4
    df = np.full(k, v / w * np.sin(w * dt))
    dl = np.full(k, v / w * (1 - np.cos(w * dt)))
    dy = np.full(k, w * dt)
    x, y, yaw = d2.compose_se2(df, dl, dy)
    T = k * dt
    assert np.allclose([x, y, yaw], [v / w * np.sin(w * T), v / w * (1 - np.cos(w * T)), w * T])
    # random chain against homogeneous matrices
    rng = np.random.default_rng(0)
    df, dl, dy = rng.normal(5, 1, 6), rng.normal(0, 0.3, 6), rng.normal(0, 0.4, 6)
    assert np.allclose(d2.compose_se2(df, dl, dy)[:2], se2_chain(df, dl, dy)[:2])
    # NaN step -> NaN result; empty -> identity
    assert all(np.isnan(d2.compose_se2([1, np.nan], [0, 0], [0, 0])))
    assert d2.compose_se2([], [], []) == (0.0, 0.0, 0.0)


def test_window_targets_known_values():
    v0, acc, w = 10.0, 1.0, 0.1
    side = make_sidecar(n=40, v0=v0, acc=acc, yaw_rate=w, decel_rows=(18,), standstill_rows=(23,), steer=2.5)
    a = 7
    tg = d2.window_targets(side, a)
    dt = 0.5
    assert tg["v_now"] == pytest.approx(v0 + acc * a * dt, rel=1e-6)
    assert tg["yaw_rate_now"] == pytest.approx(w, rel=1e-6)
    assert tg["a_now"] == pytest.approx(acc)
    assert tg["dv_2s"] == pytest.approx(acc * 4 * dt, rel=1e-5)  # 2 s of 1 m/s^2
    assert tg["dyaw_2s"] == pytest.approx(4 * w * dt, rel=1e-5)
    # lateral displacement: exact chain of the 4 chords (speed changes per step, so use the reference composer) with the
    # sidecar's clockwise d_yaw converted to CCW
    steps = np.arange(a, a + 4)
    x_ref, y_ref, _ = se2_chain(side["d_forward"][steps], side["d_left"][steps], d2.D_YAW_SIGN * side["d_yaw"][steps])
    assert tg["dleft_2s"] == pytest.approx(y_ref, rel=1e-5)
    assert y_ref > 0  # left turn -> positive lateral displacement
    assert tg["decel_event"] == 0.0  # the -3 m/s^2 row (18) is outside a+1..a+4 = 8..11
    assert tg["steer_now"] == pytest.approx(2.5)
    assert tg["standstill_now"] == 0.0
    assert tg["dt_hist"] == pytest.approx(3.5) and tg["dt_fut"] == pytest.approx(2.0)
    # anchors that see the events
    assert d2.window_targets(side, 15)["decel_event"] == 1.0  # rows 16..19 include 18
    assert d2.window_targets(side, 14)["decel_event"] == 1.0  # rows 15..18 include 18
    assert d2.window_targets(side, 18)["decel_event"] == 0.0  # rows 19..22 do not
    assert d2.window_targets(side, 23)["standstill_now"] == 1.0
    # dyaw_2s wraps into (-pi, pi]
    big = make_sidecar(n=20, yaw_rate=4.0)  # 2 rad per step -> 8 rad over 2 s
    assert d2.window_targets(big, 7)["dyaw_2s"] == pytest.approx(d2.wrap_angle(8.0))


def test_window_targets_nan_exclusion_and_short_sidecar():
    side = make_sidecar(n=40, nan_dyaw_rows=(9,))
    tg = d2.window_targets(side, 7)  # steps 7..10 include the NaN d_yaw at 9
    assert np.isnan(tg["dyaw_2s"]) and np.isnan(tg["dleft_2s"])
    assert np.isfinite(tg["v_now"]) and np.isfinite(tg["dv_2s"]) and np.isfinite(tg["decel_event"])
    tg = d2.window_targets(side, 15)  # steps 15..18 are fine
    assert np.isfinite(tg["dyaw_2s"]) and np.isfinite(tg["dleft_2s"])
    # NaN acceleration inside the future window -> decel_event NaN (excluded), a_now at the anchor still fine
    side2 = make_sidecar(n=40)
    side2["a_ego"][10] = np.nan
    tg = d2.window_targets(side2, 7)
    assert np.isnan(tg["decel_event"]) and np.isfinite(tg["a_now"])
    # the last sidecar row is NaN for d_* by construction -> last anchors lose the pose targets but keep v targets
    short = make_sidecar(n=20)  # pretend the video has 40 frames but the sidecar only 20 rows
    anchors = d2.clip_anchors(40, stride=8)  # 7, 15, 23, 31
    tg, dth, dtf = d2.targets_matrix(short, anchors)
    assert tg.shape == (4, len(d2.TARGET_NAMES)) and dth.shape == (4,) and dtf.shape == (4,)
    names = d2.TARGET_NAMES
    assert np.isfinite(tg[0, names.index("v_now")]) and np.isfinite(tg[0, names.index("dv_2s")])
    assert np.isnan(tg[2:, names.index("v_now")]).all()  # rows 23, 31 beyond the sidecar
    assert np.isfinite(tg[1, names.index("dv_2s")])  # a=15 -> a+4 = 19 is the last sidecar row, still available
    assert np.isfinite(tg[1, names.index("dyaw_2s")])  # steps 15..18 are all finite (the NaN last row is step 19)


def test_window_targets_short_sidecar_boundary():
    short = make_sidecar(n=20)
    names = d2.TARGET_NAMES
    tg = d2.window_targets(short, 15)
    assert np.isfinite(tg["dv_2s"])  # a+4 = 19 is the last row
    assert np.isfinite(tg["dyaw_2s"])  # steps 15..18 are finite
    tg = d2.window_targets(short, 16)
    assert np.isnan(tg["dv_2s"]) and np.isfinite(tg["v_now"])  # a+4 = 20 is outside the sidecar
    assert np.isnan(tg["dyaw_2s"]) and np.isnan(tg["dleft_2s"])  # steps 16..19 include the NaN last row
    tg_m, _, _ = d2.targets_matrix(short, [15, 16])
    assert np.isfinite(tg_m[0, names.index("dv_2s")]) and np.isnan(tg_m[1, names.index("dv_2s")])


def test_window_targets_sidecar_yaw_sign_convention():
    """The sidecar stores d_yaw / wdev_z clockwise-positive and d_left left-positive (sign fix 2026-10-12): for a LEFT
    turn (d_yaw < 0, wdev_z < 0, d_left > 0) the targets must be dyaw_2s > 0, yaw_rate_now > 0, dleft_2s > 0 and equal
    the analytic unicycle arc; the mirrored right turn flips all three."""
    assert d2.D_YAW_SIGN == -1.0 and d2.YAW_RATE_SIGN == -1.0
    v, w, dt, a = 10.0, 0.2, 0.5, 7  # constant speed so the chords compose exactly to the arc
    for sign in (+1.0, -1.0):  # left turn, then the mirrored right turn
        wp = sign * w
        side = make_sidecar(n=40, v0=v, acc=0.0, yaw_rate=wp)
        steps = np.arange(a, a + 4)
        assert (np.sign(side["d_yaw"][steps]) == -sign).all() and np.sign(side["wdev_z"][a]) == -sign  # clockwise +
        assert (np.sign(side["d_left"][steps]) == sign).all()  # left +
        tg = d2.window_targets(side, a)
        assert np.sign(tg["dyaw_2s"]) == sign and np.sign(tg["yaw_rate_now"]) == sign and np.sign(tg["dleft_2s"]) == sign
        assert tg["yaw_rate_now"] == pytest.approx(wp, rel=1e-6) and tg["dyaw_2s"] == pytest.approx(4 * wp * dt, rel=1e-5)
        T = 4 * dt
        assert tg["dleft_2s"] == pytest.approx(v / wp * (1.0 - np.cos(wp * T)), rel=1e-4)  # y of the arc after 2 s
        # consistent with the pretrain motion arm's composition of the same sidecar steps
        x_ref, y_ref, yaw_ref = se2_chain(side["d_forward"][steps], side["d_left"][steps], d2.D_YAW_SIGN * side["d_yaw"][steps])
        assert tg["dleft_2s"] == pytest.approx(y_ref, rel=1e-6) and tg["dyaw_2s"] == pytest.approx(yaw_ref, rel=1e-6)
    # the pre-fix construction (raw clockwise d_yaw into the CCW composer) puts a left turn on the WRONG side
    side = make_sidecar(n=40, v0=v, acc=0.0, yaw_rate=w)
    steps = np.arange(a, a + 4)
    _, y_mirrored, _ = d2.compose_se2(side["d_forward"][steps], side["d_left"][steps], side["d_yaw"][steps])
    assert y_mirrored < 0 < d2.window_targets(side, a)["dleft_2s"]
    # targets_matrix applies the same conversion
    tg_m, _, _ = d2.targets_matrix(side, [a])
    assert tg_m[0, d2.TARGET_NAMES.index("dleft_2s")] == pytest.approx(d2.window_targets(side, a)["dleft_2s"], rel=1e-6)
    assert tg_m[0, d2.TARGET_NAMES.index("yaw_rate_now")] == pytest.approx(w, rel=1e-6)


def test_real_sidecars_d_left_vs_signed_d_yaw():
    """On >= 30 local comma sidecars, d_left correlates POSITIVELY with D_YAW_SIGN * d_yaw over turning frames
    (|d_yaw| > 0.005 rad) and the unicycle chord relation d_left = -d_forward * tan(d_yaw / 2) holds with the converted
    sign, i.e. the stored d_yaw is clockwise-positive while d_left is left-positive."""
    root = pathlib.Path(d2.DEFAULT_CLIPS_ROOT)
    paths = sorted(root.glob("*/*.can.npz"))[:60] if root.exists() else []
    if len(paths) < 30:
        pytest.skip("local comma sidecars not available")
    dl, dy, df, used = [], [], [], 0
    for p in paths:
        side = d2.load_sidecar(str(p))
        if side is None or any(k not in side for k in ("d_left", "d_yaw", "d_forward")):
            continue
        a, b, c = (np.asarray(side[k], dtype=np.float64) for k in ("d_left", "d_yaw", "d_forward"))
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (np.abs(b) > 0.005)
        dl.append(a[ok]), dy.append(b[ok]), df.append(c[ok])
        used += 1
    dl, dy, df = (np.concatenate(x) for x in (dl, dy, df))
    assert used >= 30 and len(dl) >= 500
    corr = float(np.corrcoef(dl, d2.D_YAW_SIGN * dy)[0, 1])
    assert corr > 0.5, f"corr(d_left, D_YAW_SIGN*d_yaw) = {corr:.3f} over {len(dl)} turning rows of {used} clips"
    assert np.mean(np.sign(dl) == np.sign(d2.D_YAW_SIGN * dy)) > 0.9
    resid = np.mean(np.abs(dl - df * np.tan(d2.D_YAW_SIGN * dy / 2)))
    resid_mirrored = np.mean(np.abs(dl + df * np.tan(d2.D_YAW_SIGN * dy / 2)))
    assert resid < 0.25 * resid_mirrored, f"chord residual {resid:.4f} m vs mirrored {resid_mirrored:.4f} m"
    # and on real windows the lateral displacement has the sign of the yaw change (left turn -> left displacement)
    agree, n = 0, 0
    for p in paths:
        side = d2.load_sidecar(str(p))
        if side is None:
            continue
        for a in d2.clip_anchors(d2.sidecar_n_rows(side), stride=8):
            tg = d2.window_targets(side, int(a))
            if np.isfinite(tg["dyaw_2s"]) and np.isfinite(tg["dleft_2s"]) and abs(tg["dyaw_2s"]) > 0.05:
                n += 1
                agree += int(np.sign(tg["dyaw_2s"]) == np.sign(tg["dleft_2s"]))
    assert n >= 20 and agree / n > 0.9, f"{agree}/{n} turning windows have dleft_2s on the side of dyaw_2s"


def test_retarget_features_recomputes_changed_targets_only(tmp_path):
    """``retarget`` recomputes the stored targets from the sidecars, insists that the untouched targets / dt arrays
    are bit-identical, overwrites the npz atomically and records the sign constants in the meta."""
    root = tmp_path / "clips"
    ids = ["00aa000000000001", "00bb000000000002", "01cc000000000003"]
    sides = [make_sidecar(n=40, v0=8.0 + 2 * i, acc=0.5, yaw_rate=(-1) ** i * 0.15 * (i + 1), decel_rows=(18,)) for i in range(3)]
    for cid, side in zip(ids, sides):
        save_sidecar(root / cid[:2] / f"{cid}.can.npz", side)
    anchors = d2.clip_anchors(40, stride=8)  # 7, 15, 23, 31
    rows = [(cid, side) for cid, side in zip(ids, sides)]
    good, dth, dtf = zip(*[d2.targets_matrix(side, anchors) for _, side in rows])
    good, dth, dtf = np.concatenate(good), np.concatenate(dth), np.concatenate(dtf)
    N = len(good)
    clip_id = np.repeat(ids, len(anchors))
    anchor = np.tile(anchors, len(ids)).astype(np.int32)
    # the pre-fix targets: raw (clockwise) wdev_z / d_yaw and the mirrored SE(2) composition
    stale = good.copy()
    for j, name in enumerate(d2.TARGET_NAMES):
        if name in ("yaw_rate_now", "dyaw_2s"):
            stale[:, j] = -good[:, j]
    for r, (cid, a) in enumerate(zip(clip_id, anchor)):
        side = sides[ids.index(cid)]
        steps = np.arange(a, a + 4)
        stale[r, d2.TARGET_NAMES.index("dleft_2s")] = d2.compose_se2(side["d_forward"][steps], side["d_left"][steps], side["d_yaw"][steps])[1]
    assert not np.allclose(stale[:, d2.TARGET_NAMES.index("dleft_2s")], good[:, d2.TARGET_NAMES.index("dleft_2s")], equal_nan=True)
    D = 6
    rng = np.random.default_rng(0)
    feats = {k: rng.normal(size=(N, D)).astype(np.float16) for k in ("f2_mean", "f8_mean", "f8_last")}
    meta = {"stride": 8, "history": 7, "horizon": 4, "n_windows": N}
    npz = tmp_path / "feat.npz"
    np.savez(npz, **feats, f8_slots=rng.normal(size=(N, 4, D)).astype(np.float16), clip_id=clip_id, anchor=anchor,
             n_frames=np.full(N, 40, np.int32), sidecar_rows=np.full(N, 40, np.int32), device=np.array(["mici"] * N),
             targets=stale.astype(np.float32), target_names=np.array(d2.TARGET_NAMES), dt_hist=dth, dt_fut=dtf,
             meta_json=np.array(json.dumps(meta)))
    before = npz.read_bytes()
    # a corrupted "unchanged" target is refused and nothing is written
    bad = tmp_path / "bad.npz"
    bad_t = stale.copy()
    bad_t[3, d2.TARGET_NAMES.index("v_now")] += 1.0
    with np.load(npz, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["targets"] = bad_t
    np.savez(bad, **arrs)
    bad_bytes = bad.read_bytes()
    with pytest.raises(AssertionError, match="v_now"):
        d2.retarget_features(str(bad), str(root))
    assert bad.read_bytes() == bad_bytes and not (tmp_path / "bad.npz.tmp").exists() and not (tmp_path / "bad.npz.meta.json").exists()
    # a clip without a sidecar is a hard error
    with pytest.raises(RuntimeError, match="no readable sidecar"):
        d2.retarget_features(str(npz), str(tmp_path / "empty_root"))
    assert npz.read_bytes() == before
    # the real thing
    rep = d2.retarget_features(str(npz), str(root), backup=str(tmp_path / "bk" / "old.npz"), report_path=str(tmp_path / "rep.json"))
    assert rep["n_windows"] == N and rep["n_clips"] == 3 and rep["max_abs_diff_unchanged"] == 0.0
    assert all(c["bit_identical"] for c in rep["unchanged"].values()) and set(rep["unchanged"]) == set(d2.TARGET_NAMES) - set(d2.SIGN_FIX_CHANGED_TARGETS)
    assert all(rep["dt"][k]["bit_identical"] for k in ("dt_hist", "dt_fut"))
    for t in ("yaw_rate_now", "dyaw_2s"):
        assert rep["changed"][t]["exact_negation"] and rep["changed"][t]["corr_old_new"] == pytest.approx(-1.0)
    assert rep["changed"]["dleft_2s"]["max_abs_diff"] > 0.1 and rep["changed"]["dleft_2s"]["corr_old_new"] < 0
    assert rep["sign_constants"] == {"D_YAW_SIGN": -1.0, "YAW_RATE_SIGN": -1.0}
    with np.load(npz, allow_pickle=False) as z:
        assert np.array_equal(z["targets"], good, equal_nan=True) and z["targets"].dtype == np.float32
        assert np.array_equal(z["dt_hist"], dth) and np.array_equal(z["dt_fut"], dtf)
        for k, v in feats.items():
            assert np.array_equal(z[k], v)  # features untouched
        assert z["clip_id"].astype(str).tolist() == clip_id.tolist()
        m = json.loads(str(z["meta_json"]))
    assert m["retargeted"] is True and m["retarget"]["D_YAW_SIGN"] == -1.0 and m["retarget"]["YAW_RATE_SIGN"] == -1.0
    assert m["retarget"]["changed_targets"] == list(d2.SIGN_FIX_CHANGED_TARGETS) and m["stride"] == 8
    assert m["retarget"]["target_definitions"]["dleft_2s"] == d2.TARGET_DESCRIPTIONS["dleft_2s"] and "retarget_history" not in m
    assert json.loads((tmp_path / "feat.npz.meta.json").read_text())["retargeted"] is True
    assert json.loads((tmp_path / "rep.json").read_text())["n_windows"] == N
    with np.load(tmp_path / "bk" / "old.npz", allow_pickle=False) as bz:
        assert np.array_equal(bz["targets"], stale.astype(np.float32), equal_nan=True) and bz["clip_id"].astype(str).tolist() == clip_id.tolist()
    assert not (tmp_path / "feat.npz.tmp").exists()
    # idempotent through the CLI: everything is now bit-identical, and the previous retarget block moves to the history
    assert d2.main(["retarget", "--features", str(npz), "--clips-root", str(root)]) == 0
    with np.load(npz, allow_pickle=False) as z:
        m2 = json.loads(str(z["meta_json"]))
        assert np.array_equal(z["targets"], good, equal_nan=True)
    assert len(m2["retarget_history"]) == 1 and m2["retarget_history"][0]["date"] == m["retarget"]["date"]
    assert all(c["max_abs_diff"] == 0.0 for c in m2["retarget"]["changed"].values())
    # sidecar lookup: export layout, flat layout, and the walk index
    flat = tmp_path / "flat"
    save_sidecar(flat / "zz00000000000009.can.npz", sides[0])
    deep = tmp_path / "deep"
    save_sidecar(deep / "x" / "y" / "zz00000000000010.can.npz", sides[0])
    assert d2.find_sidecar(ids[0], str(root)) == str(root / ids[0][:2] / f"{ids[0]}.can.npz")
    assert d2.find_sidecar("zz00000000000009", str(flat)) == str(flat / "zz00000000000009.can.npz")
    assert d2.find_sidecar("zz00000000000010", str(deep)) is None
    assert d2.find_sidecar("zz00000000000010", str(deep), d2.build_sidecar_index(str(deep))) == str(deep / "x" / "y" / "zz00000000000010.can.npz")


def test_load_sidecar_skips_object_arrays(tmp_path):
    side = make_sidecar(n=12)
    p = tmp_path / "abc.can.npz"
    np.savez(p, **{k: v for k, v in side.items() if k != "device"}, device=np.array("mici"),
             heading_src=np.array(["x"] * 12, dtype=object), wheelbase=np.array(2.7))
    loaded = d2.load_sidecar(str(p))
    assert loaded is not None and "heading_src" not in loaded and loaded["device"] == "mici" and loaded["wheelbase"] == 2.7
    assert d2.sidecar_n_rows(loaded) == 12
    assert d2.load_sidecar(str(tmp_path / "missing.can.npz")) is None
    assert d2.sidecar_path("/x/y/clip.mp4") == "/x/y/clip.can.npz" and d2.clip_id_of("/x/y/clip.mp4") == "clip"


# ----------------------------------------------------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------------------------------------------------
def _linear_problem(n_clips=40, per_clip=10, d=16, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_clips * per_clip, d))
    X[:, 0] *= 5.0  # un-standardised scale to exercise the standardisation
    w = rng.normal(size=d)
    y = X @ w + noise * rng.normal(size=len(X))
    groups = np.repeat([f"clip{i:03d}" for i in range(n_clips)], per_clip)
    return X, y, groups


def test_ridge_probe_recovers_linear_target_and_shuffled_control():
    X, y, groups = _linear_problem()
    r = d2.probe_regression(X, y, groups)
    assert np.isfinite(r["pred"]).all() and (r["fold"] >= 0).all()
    assert d2.r2_score_pooled(y, r["pred"]) > 0.9
    assert d2.r2_score_pooled(y, r["baseline_mean"]) <= 0.0 + 1e-9
    rs = d2.probe_regression(X, y, groups, shuffle_train_labels=True, rng=np.random.default_rng(1))
    assert d2.r2_score_pooled(y, rs["pred"]) <= 0.05
    assert len(r["lambdas"]) == 5 and all(l in set(d2.RIDGE_LAMBDAS.tolist()) for l in r["lambdas"])


def test_ridge_path_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    y = rng.normal(size=50)
    mu, sd = d2.standardize_fit(X)
    Xs = (X - mu) / sd
    lams = np.array([0.01, 1.0])
    W, yb = d2.ridge_path(Xs, y, lams)
    for j, lam in enumerate(lams):
        w_ref = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(6), Xs.T @ (y - y.mean()))
        assert np.allclose(W[:, j], w_ref)
    assert yb == pytest.approx(y.mean())


def test_logistic_probe_and_shuffled_control():
    rng = np.random.default_rng(0)
    n_clips, per_clip, d = 40, 10, 8
    X = rng.normal(size=(n_clips * per_clip, d))
    w = rng.normal(size=d)
    y = (X @ w + 0.3 * rng.normal(size=len(X)) > 0).astype(int)
    groups = np.repeat([f"c{i}" for i in range(n_clips)], per_clip)
    r = d2.probe_logistic(X, y, groups)
    assert np.isfinite(r["score"]).all()
    assert d2.auroc(y, r["score"]) > 0.9 and d2.auprc(y, r["score"]) > 0.9
    rs = d2.probe_logistic(X, y, groups, shuffle_train_labels=True, rng=np.random.default_rng(2))
    assert abs(d2.auroc(y, rs["score"]) - 0.5) < 0.12


def test_groupkfold_keeps_clips_intact():
    X, y, groups = _linear_problem(n_clips=23, per_clip=7)
    r = d2.probe_regression(X, y, groups)
    assert len(r["folds"]) == 5
    for tr, te in r["folds"]:
        assert not (set(groups[tr]) & set(groups[te]))
    # every window of a clip lands in the same outer fold
    for g in np.unique(groups):
        assert len(np.unique(r["fold"][groups == g])) == 1
    # all windows are held out exactly once
    assert sorted(np.concatenate([te for _, te in r["folds"]]).tolist()) == list(range(len(y)))
    # inner splits respect groups too
    for tr, va in d2._group_kfold(groups, 4):
        assert not (set(groups[tr]) & set(groups[va]))


def test_paired_bootstrap_sanity():
    X, y, groups = _linear_problem(n_clips=30, per_clip=5, noise=1.0)
    good = d2.probe_regression(X, y, groups)["pred"]
    noisy = good + np.random.default_rng(3).normal(scale=2.0, size=len(y))
    preds = {"f2_mean": noisy, "f8_mean": good, "f8_last": good}
    out = d2.paired_clip_bootstrap(groups, y, preds, "r2", n_boot=200, seed=0)
    for k in preds:
        lo, hi = out["estimates"][k]["ci95"]
        assert lo <= out["estimates"][k]["value"] <= hi
    d_same = out["differences"]["f8_last-f2_mean"]
    d_zero = d2.paired_clip_bootstrap(groups, y, {"f8_mean": good, "f2_mean": good}, "r2", n_boot=50)["differences"]["f8_mean-f2_mean"]
    assert d_zero["value"] == 0.0 and d_zero["ci95"] == [0.0, 0.0]
    assert d_same["value"] > 0 and d_same["ci95"][0] > 0  # the clean predictor is clearly better than the noisy one
    assert out["n_clips"] == 30 and out["n_boot"] == 200
    m = d2.paired_clip_bootstrap(groups, y, preds, "mae", n_boot=50)
    assert m["differences"]["f8_mean-f2_mean"]["value"] < 0


def test_pool_slots_and_normalise_frames():
    torch = pytest.importorskip("torch")
    B, D = 2, 8
    tok = torch.randn(B, 4 * d2.TOKENS_PER_SLOT, D)
    slots = d2.pool_slots(tok)
    assert slots.shape == (B, 4, D)
    assert torch.allclose(slots[:, 3], tok[:, 3 * d2.TOKENS_PER_SLOT:].mean(1))
    assert torch.allclose(slots.mean(1), tok.mean(1), atol=1e-6)  # equal slot sizes -> mean of means = global mean
    with pytest.raises(ValueError):
        d2.pool_slots(tok[:, :-1])
    frames = torch.randint(0, 256, (1, 2, 32, 64, 3), dtype=torch.uint8)
    x = d2.normalize_frames(frames)
    assert x.shape == (1, 3, 2, 32, 64) and x.dtype == torch.float32
    ref = (frames[0, 1, 5, 7, 2].float() / 255 - 0.406) / 0.225
    assert torch.isclose(x[0, 2, 1, 5, 7], ref)


def test_imagenet_constants_match_vendored_transform():
    inspect = pytest.importorskip("inspect")
    try:
        from app.vjepa.transforms import VideoTransform  # vendored; needs env/drivejepa_env.sh PYTHONPATH
    except Exception:  # noqa: BLE001
        pytest.skip("vendored vjepa2 app not importable")
    default = inspect.signature(VideoTransform.__init__).parameters["normalize"].default
    assert tuple(default[0]) == d2.IMAGENET_MEAN and tuple(default[1]) == d2.IMAGENET_STD


def test_strip_module_prefix():
    sd = {"module.backbone.a": 1, "backbone.b": 2, "module.c": 3}
    assert d2.strip_module_prefix(sd) == {"backbone.a": 1, "backbone.b": 2, "c": 3}


# ----------------------------------------------------------------------------------------------------------------
# end-to-end probe on a synthetic features npz
# ----------------------------------------------------------------------------------------------------------------
def test_probe_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n_clips, per_clip, D = 40, 6, 12
    N = n_clips * per_clip
    clip_id = np.repeat([f"{i:016x}" for i in range(n_clips)], per_clip)
    latent = rng.normal(size=(N, D))
    f8_mean = latent + 0.1 * rng.normal(size=(N, D))
    f2_mean = latent.copy()
    f2_mean[:, :6] = rng.normal(size=(N, 6))  # the 2f features lose half of the latent dims
    f8_last = latent + 0.2 * rng.normal(size=(N, D))
    w = rng.normal(size=D)
    targets = np.full((N, len(d2.TARGET_NAMES)), np.nan, dtype=np.float32)
    lin = latent @ w
    for name in d2.CONTINUOUS_TARGETS:
        targets[:, d2.TARGET_NAMES.index(name)] = lin + 0.1 * rng.normal(size=N)
    targets[::7, d2.TARGET_NAMES.index("dleft_2s")] = np.nan  # some excluded windows
    targets[:, d2.TARGET_NAMES.index("decel_event")] = (lin > np.percentile(lin, 70)).astype(np.float32)
    targets[:, d2.TARGET_NAMES.index("standstill_now")] = (lin < np.percentile(lin, 10)).astype(np.float32)
    npz = tmp_path / "feat.npz"
    np.savez(npz, f2_mean=f2_mean.astype(np.float16), f8_mean=f8_mean.astype(np.float16), f8_last=f8_last.astype(np.float16),
             f8_slots=np.stack([f8_mean] * 4, 1).astype(np.float16), clip_id=clip_id, anchor=np.tile(np.arange(per_clip) * 8 + 7, n_clips).astype(np.int32),
             targets=targets, target_names=np.array(d2.TARGET_NAMES), device=np.array(["mici"] * N),
             meta_json=np.array(json.dumps({"stride": 8, "devices": {"mici": N}})))
    out_dir = tmp_path / "probe"
    args = types.SimpleNamespace(features=str(npz), out_dir=str(out_dir), n_boot=30, seed=0, outer_folds=5, inner_folds=4, threads=2,
                                 slots=True, save_predictions=False, n_perm=3, n_jobs=1)
    res = d2.run_probe(args)
    assert (out_dir / "results.json").exists() and (out_dir / "table.md").exists()
    loaded = json.loads((out_dir / "results.json").read_text())
    assert loaded["n_windows_total"] == N and loaded["n_clips_total"] == n_clips
    v = res["targets"]["v_now"]
    assert v["n_windows"] == N
    r2 = {k: v["bootstrap"]["r2"]["estimates"][k]["value"] for k in d2.FEATURE_VARIANTS}
    assert r2["f8_mean"] > 0.9 and r2["f8_mean"] > r2["f2_mean"]
    gap = v["bootstrap"]["r2"]["differences"]["f8_mean-f2_mean"]
    assert gap["value"] > 0 and gap["ci95"][0] > 0
    for k in d2.FEATURE_VARIANTS:
        sh = v["shuffled_control"][k]
        assert sh["n_perm"] == 3 and len(sh["r2_per_perm"]) == 3 and len(sh["lambda_per_fold_per_perm"]) == 3
        assert sh["r2"] <= 0.05 and all(x <= 0.05 for x in sh["r2_per_perm"])  # the mean and every permutation are at chance
        assert sh["r2_range"][0] <= sh["r2"] <= sh["r2_range"][1] and sh["r2_sd"] >= 0
    assert res["targets"]["dleft_2s"]["n_windows"] == N - len(range(0, N, 7))
    dec = res["targets"]["decel_event"]
    assert dec["bootstrap"]["auroc"]["estimates"]["f8_mean"]["value"] > 0.9
    assert dec["baseline"]["auprc"] == pytest.approx(dec["distribution"]["prevalence"])
    assert dec["distribution"]["n_pos_clips"] == len(np.unique(clip_id[targets[:, d2.TARGET_NAMES.index("decel_event")] > 0.5]))
    shd = dec["shuffled_control"]["f8_mean"]
    assert shd["n_perm"] == 3 and len(shd["auroc_per_perm"]) == 3 and len(shd["auprc_per_perm"]) == 3
    assert abs(shd["auroc"] - 0.5) < 0.2 and shd["auroc_range"][0] <= shd["auroc"] <= shd["auroc_range"][1]
    assert shd["C_per_fold"] == dec["regularisation"]["f8_mean"]["C_per_fold"]
    assert "slot_analysis" in res and "v_now" in res["slot_analysis"]["targets"]
    assert res["probe_calls"][0]["n_perm"] == 3 and "3 independent label permutations" in res["protocol"]["shuffled_control"]
    table = (out_dir / "table.md").read_text()
    assert "| v_now" in table and "decel_event" in table and "gap f8_mean-f2" in table
    assert "Shuffled-label control, continuous targets" in table and "Shuffled-label control, binary targets" in table


# ----------------------------------------------------------------------------------------------------------------
# compute-sharing paths added in the second pass: ridge fold cache, fixed-C control, parallel inner CV
# ----------------------------------------------------------------------------------------------------------------
def test_ridge_cache_matches_uncached_and_guards_folds():
    X, y, groups = _linear_problem(n_clips=30, per_clip=6, d=10)
    cache = d2.RidgeFoldCache()
    r_plain = d2.probe_regression(X, y, groups)
    r_cached = d2.probe_regression(X, y, groups, cache=cache, feature_key="X")
    n_entries = len(cache)
    assert n_entries > 0
    assert np.allclose(r_plain["pred"], r_cached["pred"], rtol=1e-8, atol=1e-8) and r_plain["lambdas"] == r_cached["lambdas"]
    # a second target on the same features reuses every decomposition (no new entries) and equals the uncached result
    y2 = X @ np.arange(X.shape[1]) + 0.05 * np.random.default_rng(5).normal(size=len(y))
    r2_cached = d2.probe_regression(X, y2, groups, cache=cache, feature_key="X")
    assert len(cache) == n_entries
    assert np.allclose(d2.probe_regression(X, y2, groups)["pred"], r2_cached["pred"], rtol=1e-8, atol=1e-8)
    # the shuffled control through the cache is still at chance
    rs = d2.probe_regression(X, y, groups, shuffle_train_labels=True, rng=np.random.default_rng(1), cache=cache, feature_key="X")
    assert d2.r2_score_pooled(y, rs["pred"]) <= 0.05
    # reusing a key with a different group structure (different folds) is refused
    other_groups = np.repeat([f"g{i}" for i in range(len(y) // 5)], 5)
    with pytest.raises(ValueError):
        d2.probe_regression(X, y, other_groups, cache=cache, feature_key="X")
    # decompose + solve == ridge_path
    mu, sd = d2.standardize_fit(X)
    Xs = (X - mu) / sd
    ev, V = d2.ridge_decompose(Xs)
    W1, b1 = d2.ridge_solve(ev, V, Xs, y, d2.RIDGE_LAMBDAS)
    W2, b2 = d2.ridge_path(Xs, y, d2.RIDGE_LAMBDAS)
    assert np.allclose(W1, W2) and b1 == b2


def test_logistic_fixed_C_and_parallel_inner_cv():
    rng = np.random.default_rng(0)
    n_clips, per_clip, d = 30, 8, 6
    X = rng.normal(size=(n_clips * per_clip, d))
    w = rng.normal(size=d)
    y = (X @ w + 0.3 * rng.normal(size=len(X)) > 0).astype(int)
    groups = np.repeat([f"c{i}" for i in range(n_clips)], per_clip)
    r1 = d2.probe_logistic(X, y, groups, n_jobs=1)
    r2 = d2.probe_logistic(X, y, groups, n_jobs=2)  # worker processes, same selection rule
    assert r1["Cs"] == r2["Cs"]
    assert abs(d2.auroc(y, r1["score"]) - d2.auroc(y, r2["score"])) < 0.01
    # fixed C per outer fold: no inner CV, the given C is used and reported (0.01 is an interior grid value)
    fixed = [0.01] * 5
    rf = d2.probe_logistic(X, y, groups, fixed_Cs=fixed)
    assert rf["Cs"] == fixed and rf["C_at_grid_edge"] == [False] * 5
    with pytest.raises(ValueError):
        d2.probe_logistic(X, y, groups, fixed_Cs=[0.01, 0.1])
    # the shuffled control with the real run's C stays at chance
    rs = d2.probe_logistic(X, y, groups, shuffle_train_labels=True, rng=np.random.default_rng(3), fixed_Cs=r1["Cs"])
    assert abs(d2.auroc(y, rs["score"]) - 0.5) < 0.12


def test_probe_targets_subset_and_merge(tmp_path):
    rng = np.random.default_rng(1)
    n_clips, per_clip, D = 20, 4, 8
    N = n_clips * per_clip
    clip_id = np.repeat([f"{i:016x}" for i in range(n_clips)], per_clip)
    latent = rng.normal(size=(N, D))
    lin = latent @ rng.normal(size=D)
    targets = np.full((N, len(d2.TARGET_NAMES)), np.nan, dtype=np.float32)
    for name in d2.CONTINUOUS_TARGETS:
        targets[:, d2.TARGET_NAMES.index(name)] = lin + 0.1 * rng.normal(size=N)
    targets[:, d2.TARGET_NAMES.index("decel_event")] = (lin > np.percentile(lin, 60)).astype(np.float32)
    targets[:, d2.TARGET_NAMES.index("standstill_now")] = (lin < np.percentile(lin, 40)).astype(np.float32)
    npz = tmp_path / "feat.npz"
    f16 = lambda a: a.astype(np.float16)  # noqa: E731
    np.savez(npz, f2_mean=f16(latent + rng.normal(size=(N, D))), f8_mean=f16(latent), f8_last=f16(latent + 0.1 * rng.normal(size=(N, D))),
             f8_slots=f16(np.stack([latent] * 4, 1)), clip_id=clip_id, anchor=np.tile(np.arange(per_clip) * 8 + 7, n_clips).astype(np.int32),
             targets=targets, target_names=np.array(d2.TARGET_NAMES), device=np.array(["mici"] * N), meta_json=np.array(json.dumps({"stride": 8})))
    out_dir = tmp_path / "probe"
    base = dict(features=str(npz), out_dir=str(out_dir), n_boot=10, seed=0, outer_folds=5, inner_folds=3, threads=2, save_predictions=False,
                n_perm=2)
    res1 = d2.run_probe(types.SimpleNamespace(**base, slots=True, targets="v_now,a_now", merge=False, n_jobs=1))
    assert list(res1["targets"]) == ["v_now", "a_now"] and list(res1["slot_analysis"]["targets"]) == ["v_now", "a_now"]
    res2 = d2.run_probe(types.SimpleNamespace(**base, slots=False, targets="decel_event", merge=True, n_jobs=1))
    assert list(res2["targets"]) == ["v_now", "a_now", "decel_event"]  # canonical order, earlier targets kept
    assert list(res2["slot_analysis"]["targets"]) == ["v_now", "a_now"]
    assert len(res2["probe_calls"]) == 2 and res2["probe_calls"][1]["targets"] == ["decel_event"]
    assert res2["probe_wall_time_s"] >= res1["probe_wall_time_s"]
    on_disk = json.loads((out_dir / "results.json").read_text())
    assert list(on_disk["targets"]) == ["v_now", "a_now", "decel_event"]
    table = (out_dir / "table.md").read_text()
    assert "| v_now" in table and "| decel_event" in table and "yaw_rate_now" not in table.split("## Chosen")[0]
    # without --merge a later call replaces the file
    res3 = d2.run_probe(types.SimpleNamespace(**base, slots=False, targets="standstill_now", merge=False, n_jobs=1))
    assert list(res3["targets"]) == ["standstill_now"] and list(json.loads((out_dir / "results.json").read_text())["targets"]) == ["standstill_now"]
    with pytest.raises(ValueError):
        d2.run_probe(types.SimpleNamespace(**base, slots=False, targets="nope", merge=False, n_jobs=1))


# ----------------------------------------------------------------------------------------------------------------
# review fixes: multi-permutation null, clip exclusion / sharding, shard merging
# ----------------------------------------------------------------------------------------------------------------
def test_shuffled_logistic_controls_chance_and_deterministic():
    rng = np.random.default_rng(0)
    n_clips, per_clip, d = 30, 8, 6
    X = rng.normal(size=(n_clips * per_clip, d))
    y = (X @ rng.normal(size=d) + 0.3 * rng.normal(size=len(X)) > 0).astype(int)
    groups = np.repeat([f"c{i}" for i in range(n_clips)], per_clip)
    Cs = [0.1] * 5
    s1 = d2.shuffled_logistic_controls(X, y, groups, Cs, n_perm=4, rng=np.random.default_rng(7), n_jobs=1)
    assert s1.shape == (4, len(y)) and np.isfinite(s1).all()
    aucs = [d2.auroc(y, s) for s in s1]
    assert all(abs(a - 0.5) < 0.15 for a in aucs)
    assert len({round(a, 6) for a in aucs}) == 4  # distinct permutations
    # worker processes give the same scores (seeds are drawn up front), and a different rng gives a different null draw
    s2 = d2.shuffled_logistic_controls(X, y, groups, Cs, n_perm=4, rng=np.random.default_rng(7), n_jobs=2)
    assert np.allclose(s1, s2, atol=1e-6)
    s3 = d2.shuffled_logistic_controls(X, y, groups, Cs, n_perm=4, rng=np.random.default_rng(8), n_jobs=1)
    assert not np.allclose(s1, s3)
    # a degenerate (NaN) C gives constant scores for that fold; a wrong number of Cs is refused
    s4 = d2.shuffled_logistic_controls(X, y, groups, [float("nan")] + Cs[1:], n_perm=1, rng=np.random.default_rng(0))
    folds = d2._group_kfold(groups, 5)
    assert (s4[0, folds[0][1]] == 0).all()
    with pytest.raises(ValueError):
        d2.shuffled_logistic_controls(X, y, groups, [0.1, 0.1], n_perm=1, rng=np.random.default_rng(0))
    # the real probe clears the null range by a wide margin on this separable problem
    real = d2.probe_logistic(X, y, groups, fixed_Cs=Cs)
    summ = d2.summarize_null("auroc", aucs)
    assert summ["auroc_range"][1] < d2.auroc(y, real["score"])
    assert summ["auroc_range"][0] <= summ["auroc"] <= summ["auroc_range"][1] and summ["auroc_per_perm"] == aucs
    nan_summ = d2.summarize_null("r2", [float("nan")])
    assert np.isnan(nan_summ["r2"]) and len(nan_summ["r2_per_perm"]) == 1
    assert d2.summarize_null("r2", [0.3])["r2_sd"] == 0.0


def test_exclude_list_and_shards(tmp_path):
    paths = [f"/data/{i:02x}/clip{i:04d}.mp4" for i in range(10)]
    lst = tmp_path / "clips.csv"
    lst.write_text("".join(f"{p} 0\n" for p in paths))
    assert d2.read_clip_list(str(lst)) == paths
    excl = tmp_path / "exclude.txt"
    excl.write_text("# E2 training pool\nclip0003\n/other/root/clip0007.mp4 1\n\nclip9999\n")
    ids = d2.read_exclude_list(str(excl))
    assert ids == {"clip0003", "clip0007", "clip9999"}
    kept, dropped = d2.apply_exclusions(paths, ids)
    assert dropped == ["clip0003", "clip0007"] and len(kept) == 8 and all(d2.clip_id_of(p) not in ids for p in kept)
    assert d2.apply_exclusions(paths, set()) == (paths, [])
    # shards partition the (filtered) list into contiguous near-equal chunks, in order
    shards = [d2.shard_paths(kept, f"{i}/3") for i in range(3)]
    assert sum(shards, []) == kept and max(map(len, shards)) - min(map(len, shards)) <= 1
    assert d2.shard_paths(kept, None) == kept and d2.shard_paths(kept, "") == kept
    assert d2.parse_shard("2/3") == (2, 3) and d2.parse_shard(None) is None
    for bad in ("3/3", "a/b", "-1/2", "1", "0/0"):
        with pytest.raises(ValueError):
            d2.parse_shard(bad)


def _shard_npz(path, clip_ids, shard, n_target=len(d2.TARGET_NAMES), D=6, extra_meta=None, seed=0):
    rng = np.random.default_rng(seed)
    N = len(clip_ids)
    meta = {"list": "/l.csv", "ckpt": "/c.pt", "stride": 8, "history": 7, "horizon": 4, "precision": "fp32", "tf32": False, "batch_8f": 8,
            "batch_2f": 32, "exclude_list": None, "n_clips_listed": 6, "n_clips_excluded": 0, "feature_dtype": "float16", "embed_dim": D,
            "input_norm": {"scale": "1/255"}, "shard": shard, "n_clips_processed": len(set(clip_ids)), "n_clips_with_windows": len(set(clip_ids)),
            "n_windows": N, "gpu_forward_time_s": 10.0, "wall_time_s": 20.0, "peak_gpu_mem_gib": 1.0 + seed, "decoders": {"decord": len(set(clip_ids))},
            "skipped_clip_ids": {"no_sidecar": [f"skip{seed}"]}, "n_clips_skipped": {"no_sidecar": 1}}
    meta.update(extra_meta or {})
    f16 = lambda a: a.astype(np.float16)  # noqa: E731
    np.savez(path, f2_mean=f16(rng.normal(size=(N, D))), f8_mean=f16(rng.normal(size=(N, D))), f8_last=f16(rng.normal(size=(N, D))),
             f8_slots=f16(rng.normal(size=(N, 4, D))), clip_id=np.array(clip_ids), anchor=np.arange(N, dtype=np.int32),
             n_frames=np.full(N, 120, np.int32), sidecar_rows=np.full(N, 120, np.int32), device=np.array(["mici"] * N),
             targets=rng.normal(size=(N, n_target)).astype(np.float32), target_names=np.array(d2.TARGET_NAMES),
             dt_hist=np.full(N, 3.5, np.float32), dt_fut=np.full(N, 2.0, np.float32), meta_json=np.array(json.dumps(meta)))
    return meta


def test_merge_features_shards(tmp_path):
    a = tmp_path / "s0.npz"
    b = tmp_path / "s1.npz"
    _shard_npz(a, ["c0"] * 3 + ["c1"] * 2, "0/2", seed=0)
    _shard_npz(b, ["c2"] * 4, "1/2", seed=1)
    out = tmp_path / "merged.npz"
    meta = d2.merge_features([str(b), str(a)], str(out))  # wrong order on purpose: shards are re-ordered by index
    z = np.load(out, allow_pickle=False)
    assert z["clip_id"].astype(str).tolist() == ["c0"] * 3 + ["c1"] * 2 + ["c2"] * 4
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    assert np.array_equal(z["f8_slots"], np.concatenate([za["f8_slots"], zb["f8_slots"]]))
    assert np.array_equal(z["targets"], np.concatenate([za["targets"], zb["targets"]])) and z["target_names"].tolist() == d2.TARGET_NAMES
    assert meta["n_windows"] == 9 and meta["n_clips_with_windows"] == 3 and meta["n_clips_processed"] == 3
    assert meta["gpu_forward_time_s"] == 20.0 and meta["wall_time_s"] == 40.0 and meta["peak_gpu_mem_gib"] == 2.0
    assert meta["shards"] == ["0/2", "1/2"] and meta["shard"] is None and meta["devices"] == {"mici": 9}
    assert meta["skipped_clip_ids"] == {"no_sidecar": ["skip0", "skip1"]} and meta["n_clips_skipped"] == {"no_sidecar": 2}
    assert meta["decoders"] == {"decord": 3} and meta["precision"] == "fp32" and meta["dt_hist_mean_s"] == pytest.approx(3.5)
    assert json.loads((tmp_path / "merged.npz.meta.json").read_text())["n_windows"] == 9
    assert json.loads(str(z["meta_json"]))["merged_from"] == [str(a.resolve()), str(b.resolve())]
    # a clip in two shards, or shards with different precision, are refused
    c = tmp_path / "dup.npz"
    _shard_npz(c, ["c1"] * 2, "1/2", seed=2)
    with pytest.raises(ValueError):
        d2.merge_features([str(a), str(c)], str(tmp_path / "x.npz"))
    d = tmp_path / "bf16.npz"
    _shard_npz(d, ["c5"] * 2, "1/2", seed=3, extra_meta={"precision": "bf16 autocast"})
    with pytest.raises(ValueError):
        d2.merge_features([str(a), str(d)], str(tmp_path / "y.npz"))
    with pytest.raises(ValueError):
        d2.merge_features([], str(tmp_path / "z.npz"))


def test_retarget_plausible_mask_nans_spike_windows(tmp_path):
    """``retarget --plausible`` treats sidecar values outside PLAUSIBLE_RANGES (heading resets / gyro spikes) as missing:
    only the windows touching a spike lose the affected targets (NaN), every other value is unchanged, the module flag is
    restored afterwards, the report / meta record the mask, and the ranges equal pretrain.motion_data.PLAUSIBLE_RANGES."""
    from pretrain.motion_data import PLAUSIBLE_RANGES as MD_RANGES
    assert d2.PLAUSIBLE_RANGES == MD_RANGES
    root = tmp_path / "clips"
    cid = "00aa000000000001"
    side = make_sidecar(n=40, v0=8.0, acc=0.5, yaw_rate=0.15)
    side["d_yaw"][16] = 2.5  # heading reset inside the window of anchor 15 (steps 15..18)
    side["wdev_z"][7] = 3.0  # gyro spike at anchor 7
    save_sidecar(root / cid[:2] / f"{cid}.can.npz", side)
    anchors = d2.clip_anchors(40, stride=8)  # 7, 15, 23, 31
    good, dth, dtf = d2.targets_matrix(side, anchors)  # unmasked: the spikes enter the targets
    assert np.isfinite(good).all()
    N, D = len(good), 4
    rng = np.random.default_rng(0)
    npz = tmp_path / "feat.npz"
    np.savez(npz, **{k: rng.normal(size=(N, D)).astype(np.float16) for k in ("f2_mean", "f8_mean", "f8_last")},
             clip_id=np.repeat([cid], N), anchor=anchors.astype(np.int32), n_frames=np.full(N, 40, np.int32),
             sidecar_rows=np.full(N, 40, np.int32), device=np.array(["mici"] * N), targets=good.astype(np.float32),
             target_names=np.array(d2.TARGET_NAMES), dt_hist=dth, dt_fut=dtf,
             meta_json=np.array(json.dumps({"stride": 8, "history": 7, "horizon": 4, "n_windows": N})))
    rep = d2.retarget_features(str(npz), str(root), changed=["all"], plausible=True)
    assert d2.PLAUSIBLE_MASK is False and rep["plausible_mask"] is True and rep["changed_targets"] == list(d2.TARGET_NAMES)
    with np.load(npz, allow_pickle=False) as z:
        new = z["targets"]
        meta = json.loads(str(z["meta_json"]))
    assert meta["retarget"]["plausible_mask"] is True and meta["retarget"]["plausible_ranges"]["d_yaw"] == [-0.5, 0.5]
    j = {k: d2.TARGET_NAMES.index(k) for k in d2.TARGET_NAMES}
    assert np.isnan(new[1, j["dyaw_2s"]]) and np.isnan(new[1, j["dleft_2s"]])  # anchor 15: both yaw-dependent targets
    assert np.isnan(new[0, j["yaw_rate_now"]])  # anchor 7: the gyro spike
    masked = {(1, j["dyaw_2s"]), (1, j["dleft_2s"]), (0, j["yaw_rate_now"])}
    for r in range(N):
        for c in range(len(d2.TARGET_NAMES)):
            if (r, c) not in masked:
                assert new[r, c] == good[r, c].astype(np.float32), (r, d2.TARGET_NAMES[c])
    assert rep["changed"]["dyaw_2s"]["n_newly_nan"] == 1 and rep["changed"]["yaw_rate_now"]["n_newly_nan"] == 1
    assert rep["changed"]["v_now"]["n_newly_nan"] == 0 and rep["changed"]["v_now"]["bit_identical"]
    # without the flag the same call reproduces the stored (unmasked) targets exactly
    np.savez(npz, **{k: rng.normal(size=(N, D)).astype(np.float16) for k in ("f2_mean", "f8_mean", "f8_last")},
             clip_id=np.repeat([cid], N), anchor=anchors.astype(np.int32), n_frames=np.full(N, 40, np.int32),
             sidecar_rows=np.full(N, 40, np.int32), device=np.array(["mici"] * N), targets=good.astype(np.float32),
             target_names=np.array(d2.TARGET_NAMES), dt_hist=dth, dt_fut=dtf,
             meta_json=np.array(json.dumps({"stride": 8, "history": 7, "horizon": 4, "n_windows": N})))
    rep2 = d2.retarget_features(str(npz), str(root), changed=["all"])
    assert all(c["bit_identical"] for c in rep2["changed"].values()) and rep2["plausible_mask"] is False
