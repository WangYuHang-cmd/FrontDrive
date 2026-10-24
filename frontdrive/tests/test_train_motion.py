"""End-to-end tests of pretrain/train_motion.py (the fork of the vendored app/vjepa/train.py).

GPU-marked (skipped without CUDA): the vendored trainer hardcodes cuda:0 and wraps the models in DDP (NCCL).
1. Equivalence: motion.enabled=false on the tiny config for 3 steps vs the vendored main on the same synthetic clips
   and seed -> per-step losses equal to 1e-4 (same data order, crops, masks, init and optimizer path).
2. Motion arm: motion.enabled=true (P_phi + R_psi + consistency every 2nd step) for 3 steps -> finite losses, extra
   CSV columns populated, accounting.json written, heads in the checkpoint.
3. Controls + meta.max_steps (cond_motion=false, shuffle_motion=true, use_readout=false).
4. Resume: a second launch in the same folder auto-resumes from latest.pt, reloads both heads strictly and continues
   the cumulative accounting; a truncated latest.pt falls back to latest.prev.pt (atomic saves keep it).
5. CLI: python -m pretrain.train_motion --fname <yaml> --devices cuda:0 (process_main mirrors app/main.py).
Each run happens in a fresh subprocess (tests/_run_trainer.py) so module-level seeding and CUDA state do not leak.
All checkpoints written under run_root are deleted when the module finishes (the root disk is nearly full).
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml

TESTS_DIR = pathlib.Path(__file__).resolve().parent
FRONTDRIVE_ROOT = TESTS_DIR.parent
NAVSIM_V1 = FRONTDRIVE_ROOT / "third_party" / "Drive-JEPA" / "navsim_v1"
TINY_CFG = FRONTDRIVE_ROOT / "pretrain" / "configs" / "tiny_motion_test.yaml"
DRIVER = TESTS_DIR / "_run_trainer.py"

pytestmark = pytest.mark.gpu


def _env() -> dict:
    env = dict(os.environ)
    paths = [str(FRONTDRIVE_ROOT), str(NAVSIM_V1), str(NAVSIM_V1 / "vjepa2")]
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _run(trainer: str, cfg: dict, folder: pathlib.Path, timeout: int = 600) -> pathlib.Path:
    """Write cfg (with `folder`) and run the trainer in a subprocess; returns the run folder."""
    folder.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["folder"] = str(folder)
    cfg_path = folder / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cmd = [sys.executable, str(DRIVER), "--trainer", trainer, "--config", str(cfg_path)]
    res = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=timeout)
    (folder / "stdout.log").write_text(res.stdout)
    (folder / "stderr.log").write_text(res.stderr)
    assert res.returncode == 0, f"{trainer} trainer failed (rc={res.returncode}):\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return folder


def _run_expect_failure(cfg: dict, folder: pathlib.Path, needle: str, timeout: int = 300) -> None:
    """Run the fork and require a non-zero exit whose output contains `needle` (config-check tests)."""
    folder.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["folder"] = str(folder)
    cfg_path = folder / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cmd = [sys.executable, str(DRIVER), "--trainer", "motion", "--config", str(cfg_path)]
    res = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=timeout)
    assert res.returncode != 0, "expected the trainer to refuse this config"
    assert needle in res.stdout + res.stderr, (res.stdout[-2000:], res.stderr[-2000:])


def _read_log(folder: pathlib.Path) -> list[dict]:
    # CSVLogger appends: a resumed run writes its header line again, so repeated header rows are dropped
    rows = [r for r in csv.DictReader(open(folder / "log_r0.csv")) if r["epoch"] != "epoch"]
    assert rows, f"empty log in {folder}"
    return rows


@pytest.fixture(scope="module")
def tiny_cfg(tiny_synthetic_clips) -> dict:
    cfg = yaml.safe_load(TINY_CFG.read_text())
    cfg["data"]["datasets"] = [tiny_synthetic_clips["csv"]]
    return cfg


@pytest.fixture(scope="module")
def run_root(tmp_path_factory) -> pathlib.Path:
    root = tmp_path_factory.mktemp("train_motion_runs")
    yield root
    # every run leaves ~0.1 GB of tiny-model checkpoints (latest.pt / latest.prev.pt / e*.pt / *.tmp / *.broken):
    # never keep them in pytest's basetemp (the root disk runs at 99 %)
    for p in list(root.rglob("*.pt")) + list(root.rglob("*.pt.*")):
        p.unlink()


DIAG_COLUMNS = ("loss_pred_vis", "loss_readout_noctx", "grad_ratio_ctx", "e_std", "frac_ctx_ok", "ctx_vis_mean",
                "frac_mismatch", "huber_dx", "huber_dy", "huber_dyaw", "huber_v_mean", "huber_yaw_rate_mean", "huber_dv",
                "huber_dyaw_rate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (vendored trainer hardcodes cuda:0 + NCCL DDP)")
def test_disabled_fork_matches_vendored_trainer(tiny_cfg, run_root):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["motion"]["enabled"] = False
    vendored_cfg = copy.deepcopy(cfg)
    vendored_cfg.pop("motion")  # the vendored trainer ignores unknown keys, but keep its config pristine
    f_vendored = _run("vendored", vendored_cfg, run_root / "vendored")
    f_motion = _run("motion", cfg, run_root / "fork_disabled")
    rows_v, rows_m = _read_log(f_vendored), _read_log(f_motion)
    assert len(rows_v) == len(rows_m) == 3, (len(rows_v), len(rows_m))
    for rv, rm in zip(rows_v, rows_m):
        assert (rv["epoch"], rv["itr"]) == (rm["epoch"], rm["itr"])
        lv, lm = float(rv["loss"]), float(rm["loss"])
        assert math.isfinite(lv) and math.isfinite(lm)
        assert abs(lv - lm) <= 1e-4, f"step {rm['itr']}: vendored {lv} vs fork {lm}"
        # disabled arm: extra terms are exactly zero and the JEPA term is the whole loss
        assert float(rm["loss_jepa"]) == lm and float(rm["loss_pred"]) == 0.0 == float(rm["loss_cons"])
        assert int(rm["tokens_step"]) > 0 and float(rm["flops_step"]) > 0
        assert all(float(rm[k]) == 0.0 for k in DIAG_COLUMNS)  # diagnostics columns exist and are zero
    acc = json.loads((f_motion / "accounting.json").read_text())
    assert acc["steps"] == 3 and acc["batch_size"] == 2 and acc["n_patches_per_clip"] == 4 * 4 * 8
    assert acc["clip_tokens"] == 3 * 2 * 128 and acc["tokens"] > 0 and acc["flops_est"] > 0 and acc["wall_s"] > 0
    assert len(acc["config_hash"]) == 12 and acc["exit_status"] == "ok"
    assert (f_motion / "latest.pt").exists() and not (f_motion / "latest.pt.tmp").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_motion_arm_runs_and_accounts(tiny_cfg, run_root):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["motion"].update(enabled=True, cons_weight=0.1, cons_every=2)
    folder = _run("motion", cfg, run_root / "fork_motion")
    rows = _read_log(folder)
    assert len(rows) == 3
    for r in rows:
        for key in ("loss", "loss_jepa", "loss_pred", "loss_readout", "loss_cons", "loss_pred_vis", "loss_readout_noctx",
                    "grad_ratio_ctx", "e_std", "frac_ctx_ok", "ctx_vis_mean", "frac_mismatch"):
            assert math.isfinite(float(r[key])), (key, r[key])
        assert float(r["loss_pred"]) > 0 and float(r["loss_readout"]) >= 0 and float(r["loss_readout_noctx"]) >= 0
        assert 0.0 <= float(r["frac_ctx_ok"]) <= 1.0 and float(r["frac_mismatch"]) == 0.0 and float(r["ctx_vis_mean"]) >= 0
        assert float(r["grad_ratio_ctx"]) >= 0.0 and float(r["e_std"]) >= 0.0
        # per-target Huber terms: finite or "nan" (no valid entry for that target in the batch), never missing
        assert all(r[f"huber_{k}"] != "" for k in ("dx", "dy", "dyaw", "v_mean", "yaw_rate_mean", "dv", "dyaw_rate"))
    # consistency pass runs on global steps 0 and 2 (cons_every=2); the view-2 pass adds tokens on those steps
    assert float(rows[0]["loss_cons"]) > 0 and float(rows[2]["loss_cons"]) > 0 and float(rows[1]["loss_cons"]) == 0.0
    assert int(rows[0]["tokens_step"]) > int(rows[1]["tokens_step"])
    # matched seeds across arms: heads are built under a forked RNG, so step 0 sees the same clips / crops / masks
    # and the same backbone init as the disabled arm -> identical JEPA term before the first update
    disabled = run_root / "fork_disabled" / "log_r0.csv"
    if disabled.exists():
        assert abs(float(_read_log(disabled.parent)[0]["loss_jepa"]) - float(rows[0]["loss_jepa"])) <= 1e-4
    acc = json.loads((folder / "accounting.json").read_text())
    assert acc["steps"] == 3 and acc["motion"]["enabled"] is True and acc["exit_status"] == "ok"
    assert acc["params"]["motion_pred"] > 0 and acc["params"]["motion_readout"] > 0
    assert set(acc["flop_meter"]["by_name"]) >= {"encoder", "predictor", "target_encoder", "motion_pred", "motion_readout",
                                                 "motion_readout_diag"}
    ck = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["motion_heads"]["pred"] and ck["motion_heads"]["readout"] and ck["motion_cfg"]["enabled"] is True
    assert ck["accounting"]["steps"] == 3 and ck["weights_only"] is False
    # the heads' parameters live in two extra AdamW groups (decay / no-decay) after the four vendored ones
    assert len(ck["opt"]["param_groups"]) == 6
    assert ck["opt"]["param_groups"][5]["weight_decay"] == 0 and ck["opt"]["param_groups"][5].get("WD_exclude") is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_max_steps_and_controls(tiny_cfg, run_root):
    """meta.max_steps stops early; cond_motion=false / shuffle_motion=true / use_readout=false paths run; the
    consistency term without the read-out (constant solution) and a heavy mask config are refused at start-up."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["optimization"]["ipe"] = 5
    cfg["meta"]["max_steps"] = 2
    cfg["motion"].update(enabled=True, cond_motion=False, shuffle_motion=True, use_readout=False, cons_weight=0.0, pred_masked_only=False)
    folder = _run("motion", cfg, run_root / "fork_controls")
    rows = _read_log(folder)
    assert len(rows) == 2 and all(math.isfinite(float(r["loss"])) for r in rows)
    assert all(float(r["loss_readout"]) == 0.0 == float(r["loss_cons"]) for r in rows)  # read-out and consistency off
    assert all(float(r["loss_pred"]) > 0 and 0.0 <= float(r["frac_mismatch"]) <= 1.0 for r in rows)
    acc = json.loads((folder / "accounting.json").read_text())
    assert acc["steps"] == 2 and acc["exit_status"] == "ok"
    bad = copy.deepcopy(cfg)
    bad["motion"].update(use_readout=False, cons_weight=0.1)
    _run_expect_failure(bad, run_root / "fork_bad_cons", "cons_weight > 0 requires use_readout")
    bad = copy.deepcopy(cfg)
    bad["motion"].update(mask_config=1)  # 2 blocks @ 70 %: the context slots run empty
    _run_expect_failure(bad, run_root / "fork_bad_mask", "light context mask")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resume_restores_heads_and_accounting(tiny_cfg, run_root):
    """Epoch 1 (3 steps) -> latest.pt; a second launch in the same folder with epochs=2 and load_checkpoint=false must
    auto-resume from latest.pt (vendored r_path=None quirk fixed in the fork), reload both heads strictly and continue
    the cumulative accounting (steps, tokens, FLOPs, wall-clock) instead of restarting it."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["motion"].update(enabled=True, cons_weight=0.1, cons_every=2)
    folder = run_root / "fork_resume"
    _run("motion", cfg, folder)
    acc1 = json.loads((folder / "accounting.json").read_text())
    ck1 = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    assert ck1["epoch"] == 1 and acc1["steps"] == 3
    cfg2 = copy.deepcopy(cfg)
    cfg2["optimization"]["epochs"] = 2
    cfg2["meta"]["load_checkpoint"] = False  # auto-resume path: latest.pt already exists in the folder
    _run("motion", cfg2, folder)
    out = (folder / "stdout.log").read_text()
    assert "resuming from it" in out
    assert "[motion] loaded pred head from epoch 1: <All keys matched successfully>" in out
    assert "[motion] loaded readout head from epoch 1: <All keys matched successfully>" in out
    rows = _read_log(folder)
    assert [int(r["epoch"]) for r in rows] == [1, 1, 1, 2, 2, 2]
    assert all(math.isfinite(float(r["loss"])) for r in rows)
    acc2 = json.loads((folder / "accounting.json").read_text())
    assert acc2["steps"] == 6 and acc2["clip_tokens"] == 6 * 2 * 128
    assert acc2["tokens"] > acc1["tokens"] and acc2["flops_est"] > acc1["flops_est"] and acc2["wall_s"] > acc1["wall_s"]
    assert acc2["flop_meter"]["steps"] == 3  # the meter covers this process only; the totals above are cumulative
    ck2 = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    assert ck2["epoch"] == 2 and ck2["accounting"]["steps"] == 6 and len(ck2["opt"]["param_groups"]) == 6
    # the reloaded heads kept training: their weights moved on from the epoch-1 snapshot
    assert not torch.equal(ck1["motion_heads"]["pred"]["proj.weight"], ck2["motion_heads"]["pred"]["proj.weight"])
    assert not torch.equal(ck1["motion_heads"]["readout"]["query"], ck2["motion_heads"]["readout"]["query"])
    # atomic save kept the previous checkpoint; a truncated latest.pt (SIGKILL / ENOSPC mid-write) must not block the
    # next launch: it falls back to latest.prev.pt (epoch 1) and continues from there
    prev = folder / "latest.prev.pt"
    assert prev.exists() and torch.load(prev, map_location="cpu", weights_only=False)["epoch"] == 1
    latest = folder / "latest.pt"
    data = latest.read_bytes()
    latest.write_bytes(data[: len(data) // 2])
    cfg3 = copy.deepcopy(cfg2)
    cfg3["optimization"]["epochs"] = 3
    _run("motion", cfg3, folder)
    out = (folder / "stdout.log").read_text()
    assert "could not resume from" in out and "latest.prev.pt" in out and "loaded pred head from epoch 1" in out
    rows = _read_log(folder)
    assert [int(r["epoch"]) for r in rows] == [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
    acc3 = json.loads((folder / "accounting.json").read_text())
    assert acc3["steps"] == 9 and acc3["exit_status"] == "ok"  # 3 restored from latest.prev.pt + 6 new
    ck3 = torch.load(latest, map_location="cpu", weights_only=False)
    assert ck3["epoch"] == 3 and not (folder / "latest.pt.tmp").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cli_launcher(tiny_cfg, run_root):
    """python -m pretrain.train_motion --fname <yaml> --devices cuda:0: writes params-pretrain.yaml like app/main.py,
    stops after meta.max_steps and leaves the CSV log and accounting.json behind."""
    cfg = copy.deepcopy(tiny_cfg)
    cfg["optimization"]["ipe"] = 5
    cfg["meta"]["max_steps"] = 2
    cfg["motion"].update(enabled=True)
    folder = run_root / "cli"
    folder.mkdir(parents=True, exist_ok=True)
    cfg["folder"] = str(folder)
    cfg_path = folder / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cmd = [sys.executable, "-m", "pretrain.train_motion", "--fname", str(cfg_path), "--devices", "cuda:0"]
    res = subprocess.run(cmd, env=_env(), cwd=str(FRONTDRIVE_ROOT), capture_output=True, text=True, timeout=600)
    (folder / "stdout.log").write_text(res.stdout)
    (folder / "stderr.log").write_text(res.stderr)
    assert res.returncode == 0, f"CLI failed (rc={res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    assert (folder / "params-pretrain.yaml").exists()
    rows = _read_log(folder)
    assert len(rows) == 2 and all(math.isfinite(float(r["loss"])) for r in rows)
    acc = json.loads((folder / "accounting.json").read_text())
    assert acc["steps"] == 2 and acc["motion"]["enabled"] is True
