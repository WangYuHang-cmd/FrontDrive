"""Unit tests for the frontdrive ④(b) learning-rate schedule knob in the vendored perception-based agent."""
import math

import pytest

from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import make_lr_lambda
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig


def test_default_config_is_upstream_constant_lr():
    cfg = DriveJEPAConfig()
    assert cfg.lr_schedule == "none" and cfg.lr_total_steps == 0


def test_cosine_endpoints_and_monotone():
    lam = make_lr_lambda("cosine", total_steps=1000, min_mult=0.1)
    assert lam(0) == pytest.approx(1.0)
    assert lam(500) == pytest.approx(0.55, abs=1e-6)      # halfway: 0.1 + 0.9 * 0.5
    assert lam(1000) == pytest.approx(0.1)
    assert lam(5000) == pytest.approx(0.1)                # clamped after the end
    vals = [lam(s) for s in range(0, 1001, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_warmup_then_cosine():
    lam = make_lr_lambda("cosine", total_steps=1000, min_mult=0.0, warmup_steps=100)
    assert lam(0) == pytest.approx(0.01)
    assert lam(99) == pytest.approx(1.0)
    assert lam(100) == pytest.approx(1.0)
    assert lam(550) == pytest.approx(0.5, abs=1e-6)      # midpoint of the 900 cosine steps


def test_step_schedule_and_errors():
    lam = make_lr_lambda("step", total_steps=100, min_mult=0.1)
    assert lam(0) == 1.0 and lam(74) == 1.0 and lam(75) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        make_lr_lambda("cosine", total_steps=0)
    with pytest.raises(ValueError):
        make_lr_lambda("bogus", total_steps=10)(0)


def test_navtrain_total_steps_match_release():
    # train_pb.sh: 85,109 // 64 = 1,329 steps per epoch (released ckpt: 17,277 steps = 13 epochs); 20 epochs -> 26,580
    assert 85109 // 64 == 1329 and 85109 // 64 * 20 == 26580 and 1329 * 13 == 17277
