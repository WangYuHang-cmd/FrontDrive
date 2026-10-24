"""probe_compare_encoders: synthetic features/results -> known deltas, alignment checks, metric sanity."""
import json
import numpy as np
import pytest

from diagnostics import probe_compare_encoders as pce


def _make(tmp_path, label, noise, seed=0, n_clips=40, per_clip=10):
    rng = np.random.default_rng(seed)
    clip = np.repeat([f"c{i:03d}" for i in range(n_clips)], per_clip)
    anchor = np.tile(np.arange(per_clip) * 8, n_clips)
    y_cont = rng.normal(size=len(clip)); y_cont[::7] = np.nan           # some invalid windows
    y_bin = (rng.random(len(clip)) < 0.3).astype(float)
    targets = np.stack([y_cont, y_bin], axis=1)
    npz = tmp_path / f"{label}.npz"
    np.savez(npz, clip_id=clip, anchor=anchor, targets=targets, target_names=np.array(["v_now", "decel_event"]), device=np.array(["tici"] * len(clip)))
    ok = np.isfinite(y_cont)
    rng2 = np.random.default_rng(seed + 100)
    preds = {fv: (y_cont[ok] + rng2.normal(scale=noise, size=ok.sum())).astype(np.float32).tolist() for fv in pce.FEATURE_VARIANTS}
    scores = {fv: (y_bin + rng2.normal(scale=noise, size=len(clip))).astype(np.float32).tolist() for fv in pce.FEATURE_VARIANTS}
    res = {"targets": {"v_now": {"kind": "continuous", "held_out_predictions": preds}, "decel_event": {"kind": "binary", "held_out_scores": scores}}}
    js = tmp_path / f"{label}.json"; js.write_text(json.dumps(res))
    return str(npz), str(js)


def test_known_delta_and_markdown(tmp_path):
    a = _make(tmp_path, "A", noise=0.7)   # R^2 ~ 0.67
    b = _make(tmp_path, "B", noise=0.3)   # R^2 ~ 0.92
    runs = {"A": pce.load_run(a), "B": pce.load_run(b)}
    res = pce.compare(runs, [("B", "A")], n_boot=300, seed=0)
    v = res["targets"]["v_now"]["variants"]["f2_mean"]
    assert v["point"]["r2"]["B"] > v["point"]["r2"]["A"] > 0.3
    d = v["pairs"]["B-A"]["r2"]
    assert d["ci_low"] > 0 and d["verdict"] == "better"
    assert v["pairs"]["B-A"]["mae"]["verdict"] == "better"        # lower MAE is better
    b_ = res["targets"]["decel_event"]["variants"]["f8_mean"]
    assert b_["point"]["auroc"]["B"] > b_["point"]["auroc"]["A"] and b_["pairs"]["B-A"]["auprc"]["verdict"] == "better"
    md = pce.to_markdown(res, "t")
    assert "| B-A |" in md and "v_now" in md


def test_misaligned_windows_rejected(tmp_path):
    a = _make(tmp_path, "A", noise=0.7)
    c = _make(tmp_path, "C", noise=0.7, seed=1)     # different clip ids / targets
    with pytest.raises(ValueError):
        pce.compare({"A": pce.load_run(a), "C": pce.load_run(c)}, [("C", "A")], n_boot=10, seed=0)


def test_metrics_sanity():
    y = np.array([0, 0, 1, 1, 1.0]); s = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    assert pce.auroc(y, s) == pytest.approx(5 / 6)
    assert 0.8 < pce.auprc(y, s) <= 1.0
    yy = np.arange(10.0); assert pce.r2_pooled(yy, yy) == 1.0 and pce.mae(yy, yy + 1) == 1.0
    assert np.isnan(pce.auroc(np.array([1.0, 1.0]), np.array([0.2, 0.3])))  # one class only -> undefined
