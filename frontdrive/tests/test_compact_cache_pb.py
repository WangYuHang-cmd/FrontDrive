"""Tests for navsim_ext/compact_cache_pb.py (derived perception-based training cache) and the vendored pb flags
(`DriveJEPAConfig.pretrain_pt_path`, `DriveJEPAConfig.side_cams_from_calibration`, `DriveJEPAConfig.encoder_activation_checkpointing`,
`SensorConfig.keep_calibration`).

Run with either local venv (absolute paths, see docs/evidence/2026-10-17_pb_recipe/README.md):
  env/drivejepa/bin/python -m pytest tests/test_compact_cache_pb.py -q            venv-pf (torch 2.8, no mmcv): the
                                                                                   feature-builder / agent tests skip
  env/drivejepa_pb_local/bin/python -m pytest tests/test_compact_cache_pb.py -q   CPU venv with mmcv_full: everything
Needs the local navtrain compact cache, the trainval log pickles and the dev CAM_F0 blobs (the module skips otherwise).
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

F = Path(__file__).resolve().parents[1]
NAVSIM_V1 = F / "third_party" / "Drive-JEPA" / "navsim_v1"
for _p in (NAVSIM_V1 / "vjepa2", NAVSIM_V1, F):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NAVSIM_LOCAL = Path(os.environ.get("NAVSIM_LOCAL", "/home/henry/Desktop/Qwen_Drive/benchmarks/navsim"))
COMPACT = NAVSIM_LOCAL / "cache" / "navtrain_compact"
LOGS = NAVSIM_LOCAL / "dataset" / "navsim_logs" / "trainval"
BLOBS = NAVSIM_LOCAL / "dataset" / "sensor_blobs" / "trainval"
MAPS = NAVSIM_LOCAL / "dataset" / "maps"
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(NAVSIM_LOCAL / "dataset"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(MAPS))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(NAVSIM_LOCAL / "exp"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(NAVSIM_V1))

pytestmark = pytest.mark.skipif(not (COMPACT.is_dir() and LOGS.is_dir()), reason="local navtrain compact cache / log pickles missing")

import torch  # noqa: E402

from navsim_ext import compact_cache_pb as pb  # noqa: E402


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


needs_mmcv = pytest.mark.skipif(not _has("mmcv"), reason="the pb feature builder imports mmcv (use env/drivejepa_pb_local)")


def _load_log(log: str):
    with open(LOGS / f"{log}.pkl", "rb") as f:
        frames = pickle.load(f)
    return frames, {fr["token"]: i for i, fr in enumerate(frames)}


def _dev_case():
    """First dev log (local blobs) with a navtrain token whose CAM_F0 frames p-1 and p exist locally."""
    if not BLOBS.is_dir():
        return None, None
    for log in sorted(os.listdir(BLOBS)):
        if not (COMPACT / log).is_dir() or not (LOGS / f"{log}.pkl").exists():
            continue
        frames, _ = _load_log(log)
        toks = set(os.listdir(COMPACT / log))
        for p in range(3, len(frames) - 10):
            if frames[p]["token"] in toks and all((BLOBS / frames[q]["cams"]["CAM_F0"]["data_path"]).exists() for q in (p - 1, p)):
                return log, frames[p]["token"]
    return None, None


@pytest.fixture(scope="module")
def dev_case():
    log, token = _dev_case()
    if log is None:
        pytest.skip("no dev log with local CAM_F0 frames")
    return log, token


@pytest.fixture(scope="module")
def pb_cache(tmp_path_factory, dev_case):
    """Derived pb cache of ONE dev log (all its navtrain tokens), built in-process with the default (auto) engine."""
    log, _ = dev_case
    out = tmp_path_factory.mktemp("navtrain_compact_pb")
    summary = pb.build(str(COMPACT), str(LOGS), str(out), workers=1, log_names=[log])
    assert summary["failed"] == 0 and summary["missing_pickle"] == 0 and summary["written"] > 0, summary
    return out, log


# ----------------------------------------------------------------------------------------------------------------------
def test_shape_and_projection_math():
    assert pb.padded_img_shape() == (448, 768, 3)  # 1080x1920 -> int(.4*) = 432x768 -> pad /32 = 448x768
    assert pb.padded_img_shape((1080, 1920)) == (448, 768, 3)
    rng = np.random.RandomState(0)
    from scipy.spatial.transform import Rotation

    for _ in range(20):
        R = Rotation.random(random_state=rng).as_matrix()
        t = rng.randn(3) * 2
        K = np.array([[1545.0, 0, 960.0], [0, 1545.0, 560.0], [0, 0, 1]])
        ours = pb.scaled_lidar2img(pb.lidar2img_from_calibration(R, t, K))
        ref = pb.independent_lidar2img({"sensor2lidar_rotation": R, "sensor2lidar_translation": t, "cam_intrinsic": K})
        assert np.allclose(ours, ref, rtol=1e-10, atol=1e-8)
        assert np.allclose(ours[3], [0, 0, 0, 1])


def test_split_resolution():
    """--split restricts logs AND tokens from the vendored scene-filter yamls (official split = plan's fit/dev split)."""
    logs_v, toks_v = pb.split_members("navtrain_val_official")
    logs_t, toks_t = pb.split_members("navtrain_train_official")
    assert len(logs_v) == 214 and len(toks_v) == 18179 and len(logs_t) == 978 and len(toks_t) == 85109
    assert not (set(logs_v) & set(logs_t)) and not (toks_v & toks_t)
    logs_all, toks_all = pb.split_members("navtrain")
    assert len(logs_all) == 1192 and len(toks_all) == 103288
    assert set(logs_v) | set(logs_t) == set(logs_all) and toks_v | toks_t == toks_all
    rec = json.load(open(F / "navsim_ext/configs/navtrain_official_split.json"))
    assert sorted(rec["dev_logs"]) == logs_v and rec["n_tokens_dev"] == len(toks_v) and rec["n_tokens_fit"] == len(toks_t)
    assert pb.split_members("all") == (None, None) and pb.split_members(None) == (None, None)
    with pytest.raises(AssertionError):
        pb.split_members("no_such_split")
    cache_logs = set(pb.list_cache_logs(COMPACT))
    assert set(logs_all) <= cache_logs
    log = logs_v[0]  # a val log's compact-cache tokens are all val tokens (log-disjoint split)
    assert {d.name for d in (COMPACT / log).iterdir() if d.is_dir()} <= toks_v


def test_build_matches_pickles_and_pf_cache(pb_cache):
    out, log = pb_cache
    n_tokens = sum(1 for d in (COMPACT / log).iterdir() if d.is_dir())
    rep = pb.verify(str(COMPACT), str(LOGS), str(out), n=0, log_names=[log], mode="pf")
    assert rep["mode"] == "pf" and rep["fail"] == 0, rep["fails_head"]
    assert rep["ok"] == rep["tokens_in_out"] == n_tokens
    assert rep["marker"] is True
    assert pb.count_tokens(str(out)) == (n_tokens, n_tokens)
    assert (Path(str(out) + "_info.txt")).exists() and (Path(str(out) + "_build_summary.jsonl")).exists()
    again = pb.build(str(COMPACT), str(LOGS), str(out), workers=1, log_names=[log])  # resumable: nothing rewritten
    assert again["written"] == 0 and again["skipped"] == n_tokens and again["failed"] == 0
    # split restriction: the dev log is a val log -> the train split derives nothing from it
    nothing = pb.build(str(COMPACT), str(LOGS), str(out), workers=1, log_names=[log], split="navtrain_train_official")
    assert nothing["n_logs_requested"] == 0 and nothing["written"] == 0
    val = pb.build(str(COMPACT), str(LOGS), str(out), workers=1, log_names=[log], split="navtrain_val_official")
    assert val["n_logs_requested"] == 1 and val["skipped"] == n_tokens and val["not_in_split"] == 0 and val["n_split_tokens_in_cache"] == n_tokens


def test_compact_dataset_and_collate(pb_cache):
    from torch.utils.data import default_collate

    from navsim_ext.compact_cache import CompactCacheOnlyDataset

    out, log = pb_cache

    class _Stub:  # CacheOnlyDataset only needs the builders' unique names
        def __init__(self, name):
            self._n = name

        def get_unique_name(self):
            return self._n

    ds = CompactCacheOnlyDataset(cache_path=str(out), feature_builders=[_Stub(pb.PB_FEATURE)], target_builders=[_Stub(pb.PB_TARGET)], log_names=[log])
    assert len(ds) == sum(1 for d in (COMPACT / log).iterdir() if d.is_dir()) >= 2
    feats, tgts = ds[0]
    for k in ("camera_feature_1", "camera_feature_2"):
        assert isinstance(feats[k], torch.Tensor) and feats[k].shape == (3, 256, 512) and feats[k].dtype == torch.float32
        assert 0.0 <= float(feats[k].min()) and float(feats[k].max()) <= 1.0
    batch_f, batch_t = default_collate([ds[0], ds[1]])
    assert set(batch_f) == set(pb.FEATURE_KEYS) and set(batch_t) == set(pb.TARGET_KEYS)
    assert batch_f["img_shape"].shape == (2, 4, 3) and batch_f["img_shape"].dtype == torch.float32
    assert batch_f["lidar2img"].shape == (2, 4, 4, 4) and batch_f["lidar2img"].dtype == torch.float32
    assert batch_f["ego_status"].shape == (2, 4, 11) and batch_f["ego_status"].dtype == torch.float32
    assert batch_f["camera_feature_1"].shape == (2, 3, 256, 512)
    assert batch_t["trajectory"].shape == (2, 8, 3) and batch_t["trajectory"].dtype == torch.float64
    assert isinstance(batch_t["token"], list) and len(batch_t["token"]) == 2 and all(isinstance(t, str) for t in batch_t["token"])
    assert batch_f["lidar2img"][:, 1:2].shape == (2, 1, 4, 4)  # drive_jepa_model.py slices the F0 row
    assert torch.equal(batch_f["img_shape"][0, 0], torch.tensor([448.0, 768.0, 3.0]))
    # the F0 row (index 1) is the front camera of the token's current frame
    frames, index = _load_log(log)
    for b, tok in enumerate(batch_t["token"]):
        ref = pb.independent_lidar2img(frames[index[tok]]["cams"]["CAM_F0"])
        assert np.allclose(batch_f["lidar2img"][b, 1].numpy(), ref, rtol=1e-5, atol=1e-2)


def test_dataclass_defaults_are_upstream(dev_case):
    from navsim.common.dataclasses import Camera, Cameras, SensorConfig

    assert [f.name for f in dataclasses.fields(SensorConfig)] == ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0", "lidar_pc"]
    sc = SensorConfig.build_no_sensors()
    assert "keep_calibration" not in dataclasses.asdict(sc) and sc.keep_calibration is False and SensorConfig.keep_calibration is False
    log, token = dev_case
    frames, index = _load_log(log)
    cams = frames[index[token]]["cams"]
    up = Cameras.from_camera_dict(BLOBS, cams, ["cam_f0"])  # upstream call (no keyword)
    assert up.cam_f0.image.shape == (1080, 1920, 3) and up.cam_f0.intrinsics is not None
    for c in (up.cam_l0, up.cam_r0, up.cam_b0, up.cam_l1, up.cam_l2, up.cam_r1, up.cam_r2):
        assert c == Camera()  # all fields None, exactly as upstream
    kept = Cameras.from_camera_dict(BLOBS, cams, ["cam_f0"], keep_calibration=True)
    assert np.array_equal(kept.cam_f0.image, up.cam_f0.image)
    for name in ("cam_l0", "cam_r0", "cam_b0"):
        c = getattr(kept, name)
        assert c.image is None and c.intrinsics.shape == (3, 3) and c.sensor2lidar_rotation.shape == (3, 3) and c.sensor2lidar_translation.shape == (3,)
        assert np.array_equal(c.intrinsics, cams[name.upper()]["cam_intrinsic"])


def test_encoder_kwargs_defaults_and_activation_checkpointing():
    """ImgEncoder's init kwargs: upstream values by default; pretrain_pt_path / encoder_activation_checkpointing overrides reach the ViT."""
    from navsim.agents.drive_jepa_perception_based.bevformer.simple_image_encoder import encoder_init_kwargs
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import vjepa2.src.models.vision_transformer as vit

    cfg = DriveJEPAConfig()
    assert cfg.pretrain_pt_path == "" and cfg.side_cams_from_calibration is False and cfg.encoder_activation_checkpointing is None
    path, mk, wk = encoder_init_kwargs(cfg)
    assert path == f"{os.environ['NAVSIM_EXP_ROOT']}/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
    assert mk["encoder"]["use_activation_checkpointing"] is True  # in1k.yaml default kept
    assert mk["encoder"]["model_name"] == "vit_large" and mk["encoder"]["checkpoint_key"] == "target_encoder" and wk["img_as_video_nframes"] == 2
    assert mk["encoder"]["patch_size"] == 16 and mk["encoder"]["tubelet_size"] == 2 and mk["encoder"]["use_rope"] is True
    path2, mk2, wk2 = encoder_init_kwargs(DriveJEPAConfig(pretrain_pt_path="/x/enc.pt", encoder_activation_checkpointing=False))
    assert path2 == "/x/enc.pt" and mk2["encoder"]["use_activation_checkpointing"] is False and wk2 == wk
    _, mk3, _ = encoder_init_kwargs(DriveJEPAConfig(encoder_activation_checkpointing=True))
    assert mk3["encoder"]["use_activation_checkpointing"] is True
    # the same kwargs path as init_module (vit_tiny instead of the 300M-param vit_large; identical constructor plumbing)
    m_off = vit.vit_tiny(input_size=(256, 512), num_frames=wk2["img_as_video_nframes"], **mk2["encoder"])
    m_on = vit.vit_tiny(input_size=(256, 512), num_frames=wk["img_as_video_nframes"], **mk["encoder"])
    assert m_off.use_activation_checkpointing is False and m_on.use_activation_checkpointing is True
    with torch.no_grad():  # identical forward math with and without checkpointing
        m_off.load_state_dict(m_on.state_dict())
        x = torch.randn(1, 3, 2, 256, 512)
        assert torch.allclose(m_on.eval()(x), m_off.eval()(x), atol=1e-5)


def test_hydra_overrides_instantiate_config():
    """The wrappers' '+agent.config.*' overrides compose and instantiate (hydra 1.2/1.3, omegaconf 2.3)."""
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = NAVSIM_V1 / "navsim/planning/script/config/training"
    overrides = ["agent=drive_jepa_perception_based_agent", "+agent.config.side_cams_from_calibration=true",
                 "+agent.config.encoder_activation_checkpointing=false", "+agent.config.pretrain_pt_path=/x/enc.pt"]
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)
        conf = instantiate(cfg.agent.config)
        assert conf.side_cams_from_calibration is True and conf.encoder_activation_checkpointing is False and conf.pretrain_pt_path == "/x/enc.pt"
        assert conf.trajectory_sampling.num_poses == 8
        base = instantiate(compose(config_name="default_training", overrides=["agent=drive_jepa_perception_based_agent"]).agent.config)
        assert base.side_cams_from_calibration is False and base.encoder_activation_checkpointing is None and base.pretrain_pt_path == ""
    del hydra


@needs_mmcv
def test_agent_sensor_config_flag():
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.common.dataclasses import SensorConfig

    cfg = DriveJEPAConfig()
    assert cfg.pretrain_pt_path == "" and cfg.side_cams_from_calibration is False
    sc = DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=cfg))  # the method only reads self._config
    assert dataclasses.asdict(sc) == dict(cam_f0=[2, 3], cam_l0=[3], cam_l1=[], cam_l2=[], cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[])
    assert sc.keep_calibration is False
    assert [sc.get_sensors_at_iteration(i) for i in range(4)] == [[], [], ["cam_f0"], ["cam_f0", "cam_l0", "cam_r0", "cam_b0"]]
    sc2 = DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=DriveJEPAConfig(side_cams_from_calibration=True)))
    assert dataclasses.asdict(sc2) == dict(cam_f0=[2, 3], cam_l0=[], cam_l1=[], cam_l2=[], cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[], lidar_pc=[])
    assert sc2.keep_calibration is True and SensorConfig.keep_calibration is False and SensorConfig.build_no_sensors().keep_calibration is False
    assert [sc2.get_sensors_at_iteration(i) for i in range(4)] == [[], [], ["cam_f0"], ["cam_f0"]]
    sc3 = pb._flagged_sensor_config(DriveJEPAConfig(side_cams_from_calibration=True))
    assert dataclasses.asdict(sc3) == dataclasses.asdict(sc2) and sc3.keep_calibration is True


@needs_mmcv
def test_builder_engine_equals_numpy_engine(dev_case, pb_cache):
    """The derivation through the patched vendored builder code (engine=builder) is bit-identical to the numpy mirror,
    and the fixture cache (engine=auto -> builder here) holds exactly those values."""
    assert pb.builder_available() and pb.resolve_engine("auto") == "builder"
    log, _ = dev_case
    out, _ = pb_cache
    frames, index = _load_log(log)
    toks = sorted(d.name for d in (COMPACT / log).iterdir() if d.is_dir())[:6]
    assert pb.padded_img_shape_builder() == pb.padded_img_shape() == (448, 768, 3)
    for tok in toks:
        history = pb._history(frames, index, tok)
        agree = pb.engines_agree(history)
        assert all(agree.values()), (tok, agree)
        b = pb.bev_meta_builder(history)
        cached = pb.gz_load(out / log / tok / f"{pb.PB_FEATURE}.gz")
        for k in pb.META_KEYS:
            assert b[k].dtype == np.float32 and np.array_equal(b[k], cached[k]), (tok, k)
    summary = json.loads(Path(str(out) + "_build_summary.jsonl").read_text().splitlines()[0])
    assert summary["engine"] == "builder" and summary["selfcheck"] is True
    assert pb.resolve_engine("numpy") == "numpy" and pb.resolve_engine("builder") == "builder"
    with pytest.raises(AssertionError):
        pb.resolve_engine("no_such_engine")


@needs_mmcv
def test_feature_builder_calibration_only_matches_upstream_and_cache(dev_case, pb_cache):
    """Flag on: AgentInput without side images -> features with the upstream keys/shapes/dtypes, lidar2img/img_shape/ego_status
    identical to the derived cache, F0 row identical to the upstream image-based arithmetic and to an independent projection."""
    from navsim.agents.drive_jepa_perception_based.bevformer.bev_feature_build import LoadMultiViewImageFromFiles
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder, DriveJEPATargetBuilder
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim_ext.compact_cache import _decode_jpeg

    log, token = dev_case
    out, _ = pb_cache
    frames, index = _load_log(log)
    cfg = DriveJEPAConfig(side_cams_from_calibration=True)
    sensor_config = DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=cfg))
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log], tokens=[token])
    loader = SceneLoader(data_path=LOGS, sensor_blobs_path=BLOBS, scene_filter=scene_filter, sensor_config=sensor_config)
    assert loader.tokens == [token]
    agent_input = loader.get_agent_input_from_token(token)  # run_pdm_score path: no side JPEG is opened
    assert agent_input.cameras[-1].cam_l0.image is None and agent_input.cameras[-1].cam_l0.intrinsics is not None
    assert agent_input.cameras[-1].cam_f0.image.shape == (1080, 1920, 3) and agent_input.cameras[-2].cam_f0.image.shape == (1080, 1920, 3)

    feats = DriveJEPAFeatureBuilder(cfg).compute_features(agent_input)
    assert set(feats) == {"img_shape", "lidar2img", "camera_feature_1", "camera_feature_2", "ego_status"}
    assert feats["camera_feature_1"].shape == (3, 256, 512) and feats["camera_feature_1"].dtype == torch.float32
    assert feats["camera_feature_2"].shape == (3, 256, 512) and feats["camera_feature_2"].dtype == torch.float32
    assert feats["img_shape"].shape == (4, 3) and feats["img_shape"].dtype == torch.float32
    assert feats["lidar2img"].shape == (4, 4, 4) and feats["lidar2img"].dtype == torch.float32
    assert feats["ego_status"].shape == (4, 11) and feats["ego_status"].dtype == torch.float32
    assert torch.equal(feats["img_shape"], torch.full((4, 3), 1.0) * torch.tensor([448.0, 768.0, 3.0]))

    cached = pb.gz_load(out / log / token / f"{pb.PB_FEATURE}.gz")
    assert np.array_equal(feats["lidar2img"].numpy(), cached["lidar2img"])
    assert np.array_equal(feats["img_shape"].numpy(), cached["img_shape"])
    assert np.array_equal(feats["ego_status"].numpy(), cached["ego_status"])

    # F0 row: independent projection from the pickle calibration, and bit-identical to the upstream image-based arithmetic
    cur = frames[index[token]]
    assert np.allclose(feats["lidar2img"][1].numpy(), pb.independent_lidar2img(cur["cams"]["CAM_F0"]), rtol=1e-5, atol=1e-2)
    upstream = LoadMultiViewImageFromFiles(agent_input)  # flag off on the same input: only the F0 image is present -> 1 row
    flagged = LoadMultiViewImageFromFiles(agent_input, side_cams_from_calibration=True)
    assert len(upstream["lidar2img"]) == 1 and len(flagged["lidar2img"]) == 4
    assert np.array_equal(upstream["lidar2img"][0], flagged["lidar2img"][1])
    assert np.array_equal(upstream["camera_intrinsics"][0], flagged["camera_intrinsics"][1])
    for row, name in enumerate(pb.CAM_ORDER):
        assert np.allclose(feats["lidar2img"][row].numpy(), pb.independent_lidar2img(cur["cams"][name]), rtol=1e-5, atol=1e-2)

    # the cached JPEG frames are the same crops (PSNR >= 35 dB vs the exact float features)
    for k in ("camera_feature_1", "camera_feature_2"):
        dec = _decode_jpeg(cached[k])
        mse = float(((dec - feats[k]) ** 2).mean())
        assert 10 * math.log10(1.0 / max(mse, 1e-12)) >= 35.0, k

    # target: the pf trajectory stored in the derived cache equals the pb target builder (needs the map api -> Scene)
    if MAPS.is_dir():
        scene = loader.get_scene_from_token(token)
        tgt = DriveJEPATargetBuilder(cfg).compute_targets(scene)
        cached_t = pb.gz_load(out / log / token / f"{pb.PB_TARGET}.gz")
        assert tgt["token"] == cached_t["token"] == token
        assert np.array_equal(tgt["trajectory"].numpy(), cached_t["trajectory"])
        feats_scene = DriveJEPAFeatureBuilder(cfg).compute_features(scene.get_agent_input())  # the caching path (Dataset)
        for k in ("lidar2img", "img_shape", "ego_status"):
            assert np.array_equal(feats_scene[k].numpy(), cached[k]), k


@needs_mmcv
def test_flag_off_with_four_images_equals_flag_on(dev_case):
    """Review 2026-10-17 (dossier (b), Patch-report open issue 3): the UPSTREAM path (flag off, CAM_L0/R0/B0 frame-3 images
    present) and the flag-on calibration-only path give identical features on the same frame. No side JPEG exists locally, so
    the side images are synthetic 1080x1920 uint8 arrays attached to the REAL calibration of the pickle (pixel content never
    enters img_shape / lidar2img, only the image size does; nuPlan's eight cameras share the 1920x1080 sensor). F0 = real JPEGs."""
    import copy

    from navsim.agents.drive_jepa_perception_based.bevformer.bev_feature_build import LoadMultiViewImageFromFiles, _get_bev_feature
    from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder
    from navsim.common.dataclasses import Camera, SceneFilter
    from navsim.common.dataloader import SceneLoader

    log, token = dev_case
    frames, index = _load_log(log)
    pos = index[token]
    cfg_on, cfg_off = DriveJEPAConfig(side_cams_from_calibration=True), DriveJEPAConfig()
    sc_on = DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=cfg_on))
    sc_off = DriveJEPAAgent.get_sensor_config(types.SimpleNamespace(_config=cfg_off))  # upstream: cam_l0/r0/b0 = [3]
    assert [sc_off.get_sensors_at_iteration(i) for i in range(4)] == [[], [], ["cam_f0"], ["cam_f0", "cam_l0", "cam_r0", "cam_b0"]]
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log], tokens=[token])
    ai_on = SceneLoader(data_path=LOGS, sensor_blobs_path=BLOBS, scene_filter=scene_filter, sensor_config=sc_on).get_agent_input_from_token(token)
    # the AgentInput the upstream loader builds when the side JPEGs exist: Camera() for every unrequested camera, image +
    # calibration for the requested ones (cam_f0 at frames 2,3 = the real JPEGs; cam_l0/r0/b0 at frame 3 = synthetic pixels)
    rng = np.random.RandomState(0)
    ai_off = copy.deepcopy(ai_on)
    for fi, cameras in enumerate(ai_off.cameras):
        requested = sc_off.get_sensors_at_iteration(fi)
        for name in ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"):
            if name not in requested:
                setattr(cameras, name, Camera())
            elif name != "cam_f0":
                c = frames[pos - 3 + fi]["cams"][name.upper()]
                setattr(cameras, name, Camera(image=rng.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
                                              sensor2lidar_rotation=c["sensor2lidar_rotation"], sensor2lidar_translation=c["sensor2lidar_translation"],
                                              intrinsics=c["cam_intrinsic"], distortion=c["distortion"]))
    assert ai_off.cameras[-1].cam_l0.image.shape == (1080, 1920, 3) and ai_off.cameras[-1].cam_f0.image.shape == (1080, 1920, 3)
    assert ai_off.cameras[-1].cam_l1 == Camera() and ai_off.cameras[0].cam_f0 == Camera()
    assert ai_on.cameras[-1].cam_l0.image is None and ai_on.cameras[-1].cam_l0.intrinsics is not None
    # raw loader rows: 4 in both, same order [b0, f0, l0, r0], bit-identical
    up, fl = LoadMultiViewImageFromFiles(ai_off), LoadMultiViewImageFromFiles(ai_on, side_cams_from_calibration=True)
    assert len(up["lidar2img"]) == len(fl["lidar2img"]) == 4 and len(up["img"]) == 4 and len(fl["img"]) == 1
    for key in ("lidar2img", "cam_intrinsic", "lidar2cam", "camera_intrinsics"):
        for row in range(4):
            assert up[key][row].dtype == fl[key][row].dtype and np.array_equal(up[key][row], fl[key][row]), (key, row)
    # full feature builders: flag off = the upstream code path on the 4 images, flag on = calibration-only
    np.random.seed(0)
    f_off = DriveJEPAFeatureBuilder(cfg_off).compute_features(ai_off)
    np.random.seed(0)
    f_on = DriveJEPAFeatureBuilder(cfg_on).compute_features(ai_on)
    assert set(f_off) == set(f_on) == {"img_shape", "lidar2img", "camera_feature_1", "camera_feature_2", "ego_status"}
    for k in f_off:
        assert f_off[k].dtype == f_on[k].dtype and f_off[k].shape == f_on[k].shape and torch.equal(f_off[k], f_on[k]), k
    assert tuple(f_off["img_shape"].shape) == (4, 3) and tuple(f_off["lidar2img"].shape) == (4, 4, 4)
    assert torch.equal(f_off["lidar2img"][1:2], f_on["lidar2img"][1:2])  # the row drive_jepa_model.py slices ([:, 1:2] after collation)
    # the single size assumption of the flagged path, made explicit: a side image of another size would change upstream's
    # img_shape row 2 (never lidar2img) -- and upstream's _get_bev_feature cannot even np.stack images of two sizes, so the
    # upstream pipeline itself presupposes one sensor size for the four cameras
    from navsim.agents.drive_jepa_perception_based.bevformer import bev_feature_build as bfb

    ai_bad = copy.deepcopy(ai_off)
    ai_bad.cameras[-1].cam_l0.image = rng.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    bad = bfb.PadMultiViewImage(bfb.RandomScaleImageMultiViewImage(bfb.NormalizeMultiviewImage(LoadMultiViewImageFromFiles(ai_bad))))
    assert np.array_equal(np.stack(bad["lidar2img"]).astype(np.float32), f_on["lidar2img"].numpy())
    assert bad["img_shape"][1] == (448, 768, 3) and bad["img_shape"][2] == (288, 512, 3)
    with pytest.raises(ValueError):
        _get_bev_feature(ai_bad)


@needs_mmcv
def test_verify_builder_mode_on_5_dev_tokens(pb_cache):
    """`verify --mode builder --n 5`: CompactCacheOnlyDataset output of 5 dev tokens vs the patched upstream builders."""
    out, log = pb_cache
    rep = pb.verify(str(COMPACT), str(LOGS), str(out), n=5, mode="builder", blobs=str(BLOBS), log_names=[log])
    assert rep["mode"] == "builder" and rep["checked"] == 5 and rep["ok"] == 5 and rep["fail"] == 0, rep
    assert rep["exact"] == {"img_shape": 5, "lidar2img": 5, "ego_status": 5, "trajectory": 5}
    assert rep["psnr_min"] >= 35.0 and rep["marker"] is True
    if MAPS.is_dir():
        assert rep["target_ref"] == {"scene": 5, "pf": 0} and rep["scene_error"] is None
    auto = pb.verify(str(COMPACT), str(LOGS), str(out), n=2, mode="auto", blobs=str(BLOBS), log_names=[log])
    assert auto["mode"] == "builder" and auto["ok"] == 2  # auto picks the builder mode when mmcv + blobs are present
    nowhere = pb.verify(str(COMPACT), str(LOGS), str(out), n=2, mode="auto", blobs="/nonexistent", log_names=[log])
    assert nowhere["mode"] == "pf" and nowhere["ok"] == 2


def test_coverage_check(pb_cache, tmp_path):
    out, log = pb_cache
    cache = tmp_path / "cache"
    shutil.copytree(out, cache)
    tokens = sorted(d.name for d in (cache / log).iterdir() if d.is_dir())
    assert len(tokens) >= 3
    root = tmp_path / "Drive-JEPA-cache"
    mc = root / "train_metric_cache"
    (mc / "metadata").mkdir(parents=True)
    rows = []
    for tok in tokens[1:]:  # tokens[0] is uncovered
        p = mc / log / "unknown" / tok / "metric_cache.pkl"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        rows.append(str(p))
        if tok != tokens[1]:  # tokens[1] has no anchors index
            a = root / "anchors_scores_index" / log / "unknown" / tok / "metric_cache.pkl.npy"
            a.parent.mkdir(parents=True)
            a.write_bytes(b"")
    (mc / "metadata" / "x_node_0.csv").write_text("file_name\n" + "\n".join(rows) + "\n")
    rep = pb.coverage(str(cache), str(mc))
    assert rep["n_cache_tokens"] == len(tokens) and rep["n_metric_tokens"] == len(tokens) - 1 and rep["n_uncovered"] == 1
    assert rep["uncovered_head"] == [f"{log}/{tokens[0]}"] and rep["n_missing_anchor"] is None
    rep = pb.coverage(str(cache), str(mc), check_anchors=True)
    assert rep["n_missing_anchor"] == 1 and rep["missing_anchor_head"] == [f"{log}/{tokens[1]}"]
    aside = tmp_path / "aside"
    rep = pb.coverage(str(cache), str(mc), move_aside=str(aside), check_anchors=True)
    assert rep["moved_aside"] == 2
    assert not (cache / log / tokens[0]).exists() and not (cache / log / tokens[1]).exists()
    assert (aside / log / tokens[0] / f"{pb.PB_FEATURE}.gz").exists() and (aside / log / tokens[1] / f"{pb.PB_TARGET}.gz").exists()
    rep = pb.coverage(str(cache), str(mc), check_anchors=True)
    assert rep["n_uncovered"] == 0 and rep["n_missing_anchor"] == 0 and rep["n_cache_tokens"] == len(tokens) - 2
