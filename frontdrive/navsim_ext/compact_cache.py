"""Compact (JPEG-bytes) training cache for the Drive-JEPA perception-free agent (plan v4 §4.3).

The upstream cache stores float tensors [3,256,512] per frame (~3.1 MB/sample, ~300 GB for navtrain). We keep the
exact directory layout `<cache>/<log>/<token>/<builder_unique_name>.gz` so `CacheOnlyDataset` discovery works,
but the feature pickle holds JPEG bytes of the already cropped/resized 512x256 frames; `CompactCacheOnlyDataset`
decodes them to the float CHW tensors the model expects (same as torchvision ToTensor on the resized image).
Target pickles are unchanged (TrajectoryTargetBuilder output). A marker file `COMPACT_JPEG` at the cache root tells
the patched run_training.py to use this dataset class.

Build (needs the devkit env + sparse CAM_F0 mirror):
    python -m navsim_ext.compact_cache build --split navtrain --cache /path/cache --workers 8 [--quality 95]
"""
from __future__ import annotations

import argparse
import json
import gzip
import io
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "/home/henry/Desktop/Qwen_Drive/frontdrive/third_party/Drive-JEPA/navsim_v1"))
sys.path.insert(0, str(DEVKIT))
from navsim.planning.training.dataset import CacheOnlyDataset, load_feature_target_from_pickle, dump_feature_target_to_pickle  # noqa: E402

MARKER = "_COMPACT_JPEG"  # a directory (CacheOnlyDataset iterates cache-root entries as log dirs)
JPEG_KEYS = ("camera_feature_1", "camera_feature_2")


def _encode_jpeg(img_hwc_uint8: np.ndarray, quality: int) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def _decode_jpeg(b: bytes) -> torch.Tensor:
    import cv2
    arr = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)  # == torchvision ToTensor


class CompactCacheOnlyDataset(CacheOnlyDataset):
    """CacheOnlyDataset that decodes JPEG-bytes camera features on the fly."""

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        token_path = self._valid_cache_paths[token]
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict = load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz"))
            for k, v in data_dict.items():
                features[k] = _decode_jpeg(v) if isinstance(v, (bytes, bytearray)) else v
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            targets.update(load_feature_target_from_pickle(token_path / (builder.get_unique_name() + ".gz")))
        return features, targets


def _log_loader(log: str, tokens, data_path: str, blobs_path: str, filter_cfg: dict):
    """SceneLoader restricted to ONE log (loads one pickle): workers stay small even for the 1,192-log navtrain split."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    scene_filter = instantiate(OmegaConf.create(filter_cfg))
    scene_filter.log_names = [log]
    scene_filter.tokens = list(tokens) if tokens is not None else None
    return SceneLoader(data_path=Path(data_path), sensor_blobs_path=Path(blobs_path), scene_filter=scene_filter,
                       sensor_config=SensorConfig(cam_f0=[2, 3], cam_l0=False, cam_r0=False, cam_l1=False, cam_r1=False, cam_l2=False, cam_r2=False, cam_b0=False, lidar_pc=False))


def _builders():
    from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import TrajectoryTargetBuilder
    from navsim.agents.drive_jepa_perception_free.drive_jepa_features import DriveJEPAFeatureDIBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return DriveJEPAFeatureDIBuilder(front_only=True), TrajectoryTargetBuilder(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))


def _build_log(args):
    """Worker: compute features/targets for all tokens of ONE log and write compact pickles. Returns (log, n_tokens, jpeg_bytes, n_failed)."""
    log, tokens, cache_path, quality, data_path, blobs_path, filter_cfg = args
    import cv2  # noqa: F401
    loader = _log_loader(log, tokens, data_path, blobs_path, filter_cfg)
    fb, tb = _builders()
    n_ok = n_fail = 0; total = 0
    for token in loader.tokens:
        try:
            scene = loader.get_scene_from_token(token)
            feats = fb.compute_features(scene.get_agent_input())
            out_feats = {}
            for k, v in feats.items():
                if k in JPEG_KEYS:
                    img = (v.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                    out_feats[k] = _encode_jpeg(img, quality)
                else:
                    out_feats[k] = v
            tgts = tb.compute_targets(scene)
            token_dir = Path(cache_path) / log / token
            token_dir.mkdir(parents=True, exist_ok=True)
            dump_feature_target_to_pickle(token_dir / (fb.get_unique_name() + ".gz"), out_feats)
            dump_feature_target_to_pickle(token_dir / (tb.get_unique_name() + ".gz"), tgts)
            total += sum(len(v) for v in out_feats.values() if isinstance(v, bytes)); n_ok += 1
        except Exception as e:  # keep going; the parent reports the count
            print(f"[compact_cache] {log}/{token} failed: {type(e).__name__}: {str(e)[:100]}", flush=True); n_fail += 1
    return log, n_ok, total, n_fail


def _tokens_by_log(data_path: Path, log_names, token_set):
    """Map the split's tokens to their logs by reading each log pickle once (parent process)."""
    import pickle
    out = {}
    for log in log_names:
        fp = data_path / f"{log}.pkl"
        if not fp.exists():
            continue
        toks = [fr["token"] for fr in pickle.load(open(fp, "rb")) if fr["token"] in token_set]
        if toks:
            out[log] = toks
    return out


def build(split: str, cache: str, workers: int, quality: int, limit: int) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    cfg_dir = DEVKIT / "navsim/planning/script/config/common/train_test_split"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=split)
    data_split = cfg.data_split
    root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_path = root / "navsim_logs" / data_split
    blobs_path = root / "sensor_blobs" / data_split
    filter_cfg = OmegaConf.to_container(cfg.scene_filter, resolve=True)
    from hydra.utils import instantiate
    sf = instantiate(cfg.scene_filter)
    log_names = sf.log_names or sorted(p.stem for p in data_path.glob("*.pkl"))
    if sf.tokens:  # explicit token list (navtrain): no need to materialise every scene in the parent (filter_scenes held ~10 GB)
        tokens = list(sf.tokens)
    else:
        from navsim.common.dataloader import filter_scenes
        tokens = list(filter_scenes(data_path, sf).keys())
    if limit:
        tokens = tokens[:limit]
    by_log = _tokens_by_log(data_path, log_names, set(tokens))
    tokens = [t for toks in by_log.values() for t in toks]  # only tokens whose log pickle exists
    Path(cache).mkdir(parents=True, exist_ok=True); (Path(cache) / MARKER).mkdir(exist_ok=True)  # must stay EMPTY: CacheOnlyDataset treats its entries as tokens
    (Path(str(cache).rstrip("/") + "_info.txt")).write_text("compact jpeg cache; quality %d; CAM_F0 [2,3] 512x256; split %s\n" % (quality, split))
    print(f"[compact_cache] split={split} tokens={len(tokens)} logs={len(by_log)} -> {cache}", flush=True)
    done = n_fail = 0; total_bytes = 0; done_logs = 0
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:  # spawn: workers do not inherit the parent's heap
        futs = [ex.submit(_build_log, (log, toks, cache, quality, str(data_path), str(blobs_path), filter_cfg)) for log, toks in by_log.items()]
        for f in as_completed(futs):
            try:
                _, n_ok, nb, nf = f.result(); total_bytes += nb; done += n_ok; n_fail += nf
            except Exception as e:
                print("[compact_cache] log failed:", type(e).__name__, str(e)[:120], flush=True)
            done_logs += 1
            if done_logs % 50 == 0:
                print(f"[compact_cache] logs {done_logs}/{len(by_log)} tokens {done} failed {n_fail} jpeg_bytes_total={total_bytes/1e9:.2f} GB", flush=True)
    print(f"[compact_cache] done {done} tokens ({n_fail} failed) in {done_logs} logs, jpeg bytes {total_bytes/1e9:.2f} GB", flush=True)


def _exact_token(args):
    """Worker: exact (uncompressed) features/targets for one token (single-log loader), nothing written."""
    token, log, data_path, blobs_path, filter_cfg = args
    loader = _log_loader(log, [token], data_path, blobs_path, filter_cfg)
    fb, tb = _builders()
    scene = loader.get_scene_from_token(token)
    feats = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in fb.compute_features(scene.get_agent_input()).items()}
    tgts = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in tb.compute_targets(scene).items()}
    return token, log, feats, tgts


def verify(split: str, cache: str, limit: int, workers: int, min_psnr: float = 35.0, only_cached: bool = False) -> int:
    """Compare the compact cache with exact features on `limit` tokens: PSNR of decoded JPEG keys, exact equality of the
    rest and of the targets. Returns 0 if every token passes (PSNR >= min_psnr, non-image features and targets identical)."""
    import random
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataloader import filter_scenes
    cfg_dir = DEVKIT / "navsim/planning/script/config/common/train_test_split"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=split)
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); data_path = root / "navsim_logs" / cfg.data_split; blobs_path = root / "sensor_blobs" / cfg.data_split
    filter_cfg = OmegaConf.to_container(cfg.scene_filter, resolve=True)
    split_tokens = set(filter_scenes(data_path, instantiate(cfg.scene_filter)).keys())
    cached = [(d.name, log.name) for log in Path(cache).iterdir() if log.is_dir() and log.name != MARKER for d in log.iterdir()]  # (token, log)
    pairs = [(t, l) for t, l in cached if t in split_tokens]
    missing = 0 if only_cached else len(split_tokens) - len(pairs)
    random.Random(0).shuffle(pairs); pairs = pairs[:limit]
    fb, tb = _builders(); fb_name, tb_name = fb.get_unique_name(), tb.get_unique_name()
    psnrs, n_ok, n_fail = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_exact_token, (t, l, str(data_path), str(blobs_path), filter_cfg)) for t, l in pairs]):
            token, log, feats, tgts = f.result()
            tdir = Path(cache) / log / token
            cf = load_feature_target_from_pickle(tdir / (fb_name + ".gz")); ct = load_feature_target_from_pickle(tdir / (tb_name + ".gz"))
            ok = True
            for k, v in feats.items():
                cv = cf[k]
                if isinstance(cv, (bytes, bytearray)):
                    dec = _decode_jpeg(cv).numpy(); mse = float(np.mean((dec - v) ** 2)); psnr = 10 * np.log10(1.0 / max(mse, 1e-12)); psnrs.append(psnr)
                    ok &= dec.shape == v.shape and psnr >= min_psnr
                else:
                    ok &= np.array_equal(np.asarray(cv), np.asarray(v))
            for k, v in tgts.items():
                ok &= np.array_equal(np.asarray(ct[k]), np.asarray(v))
            n_ok += int(bool(ok)); n_fail += int(not ok)
    rep = {"split": split, "cache": cache, "checked": n_ok + n_fail, "ok": n_ok, "fail": n_fail, "missing_from_cache": int(missing),
           "psnr_min": round(min(psnrs), 2) if psnrs else None, "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None, "min_psnr_required": min_psnr}
    print(json.dumps(rep), flush=True)
    return 0 if (n_fail == 0 and missing == 0 and n_ok > 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--split", default="navtrain"); b.add_argument("--cache", required=True); b.add_argument("--workers", type=int, default=8); b.add_argument("--quality", type=int, default=95); b.add_argument("--limit", type=int, default=0)
    v = sub.add_parser("verify"); v.add_argument("--split", default="navtrain"); v.add_argument("--cache", required=True); v.add_argument("--limit", type=int, default=60); v.add_argument("--workers", type=int, default=4); v.add_argument("--min-psnr", type=float, default=35.0); v.add_argument("--only-cached", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, a.cache, a.workers, a.quality, a.limit)
    elif a.cmd == "verify":
        return verify(a.split, a.cache, a.limit, a.workers, a.min_psnr, a.only_cached)
    return 0


if __name__ == "__main__":
    sys.exit(main())
