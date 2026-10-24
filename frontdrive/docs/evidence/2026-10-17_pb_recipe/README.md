# 2026-10-17 — pb (perception-based) recipe: side-camera-free training, swappable encoder init, derived training cache

Scope (plan v4.3 §2 ②/③, appendix H.5 "备选 = 按紧凑缓存思路派生 pb 缓存"): make the Drive-JEPA perception-based agent
trainable without CAM_L0/R0/B0 JPEGs, allow a different encoder initialisation than the e50 file, derive its training cache
from the existing pf compact JPEG cache + log pickles (no image I/O), and provide the one-GPU wrappers (cache build, training
smoke with the micro-batch pushed up, training). All work done locally on 2026-10-17 02:28–03:20 EDT, no GPU used.

`F` = `/home/henry/Desktop/Qwen_Drive/frontdrive`, `V1` = `F/third_party/Drive-JEPA/navsim_v1/navsim`.

## 1. Vendored-tree changes (defaults = upstream behaviour; patch `F/third_party_patches/drive_jepa_548bb82.patch`, regenerated)

| file | lines | change |
|---|---|---|
| `V1/agents/drive_jepa_perception_based/drive_jepa_config.py` | 52–62 (fields at 56 and 62) | `pretrain_pt_path: str = ""`, `side_cams_from_calibration: bool = False` (both documented; hydra needs the `+` append prefix because the keys are absent from `config/common/agent/drive_jepa_perception_based_agent.yaml`) |
| `V1/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py` | 26–27, 29–30, 41 | encoder file = `config.pretrain_pt_path` when non-empty, else the upstream `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt`; `print("[ImgEncoder] loaded encoder weights from …")`; the CWD-relative `./vjepa2/configs/eval/vitl/in1k.yaml` falls back to the vendored file when the CWD is not the repo root (upstream would raise FileNotFoundError there) |
| `V1/agents/drive_jepa_perception_based/bevformer/bev_feature_build.py` | 16–30, 33, 46–53, 91–93, 99–108 | `_lidar2img_from_calibration(cam)` (the upstream arithmetic as a helper); `LoadMultiViewImageFromFiles(agent_input, side_cams_from_calibration=False)` adds a calibration-only row for a camera whose image was not requested; `_get_bev_feature(..., side_cams_from_calibration=False)` replicates the padded F0 shape to those rows and asserts 4 rows `[b0,f0,l0,r0]`; asserts the flag is not combined with the ResNet 4-camera path |
| `V1/agents/drive_jepa_perception_based/drive_jepa_features.py` | 69–70 | passes `self._config.side_cams_from_calibration` |
| `V1/agents/drive_jepa_perception_based/drive_jepa_agent.py` | 79–95 | `get_sensor_config()`: `cam_l0/r0/b0 = []` and `sensor_config.keep_calibration = True` when the flag is on; otherwise the upstream `SensorConfig(cam_f0=[2,3], cam_l0=[3], …, cam_b0=[3])` |
| `V1/common/dataclasses.py` | 3, 62, 69, 85–91, 212, 453, 522–527 | `Cameras.from_camera_dict(..., keep_calibration=False)`: unloaded cameras keep `sensor2lidar_rotation/translation/intrinsics/distortion` (image None) only when asked; the two call sites pass `getattr(sensor_config, "keep_calibration", False)`; `SensorConfig.keep_calibration: ClassVar[bool] = False` (not a dataclass field → `asdict()` / `get_sensors_at_iteration()` unchanged) |

The 8 untracked split yamls are listed in `F/third_party_patches/drive_jepa_548bb82.untracked_files.txt` (sizes + sha256 prefixes,
how to regenerate them with `navsim_ext/make_dev_split.py`). Deviation from the task text: it named
`agents/pad/bevformer/{simple_image_encoder,bev_feature_build}.py`; `pad/bevformer/simple_image_encoder.py` does not exist — the
pb agent imports its own copies under `agents/drive_jepa_perception_based/bevformer/` (the dossier's citations), which is what
was edited; `agents/pad/` is untouched.

## 2. New frontdrive files

- `F/navsim_ext/compact_cache_pb.py` — `build` / `verify` / `coverage`. Derives `<out>/<log>/<token>/{drive_jepa_feature,drive_jepa_target}.gz`
  (+ empty `_COMPACT_JPEG` marker, `<out>_info.txt`, `<out>_build_summary.json`) from the pf compact cache + `<log>.pkl`:
  `camera_feature_1/2` = the pf JPEG bytes (same `[28:-28]` crop → 512×256), `img_shape` (4,3) f32 = (448,768,3) rows,
  `lidar2img` (4,4,4) f32 = `diag(.4,.4,1,1) @ K4 @ lidar2cam^T` per `[CAM_B0, CAM_F0, CAM_L0, CAM_R0]`, `ego_status` (4,11) f32 via
  the devkit's `AgentInput.from_scene_dict_list` (no sensors), `trajectory` (8,3) f64 = the pf target, `token`. Arrays are numpy
  (no torch objects → identical pickles from torch 2.8 and 2.1); `default_collate` yields the upstream dtypes. Atomic writes, resumable.
  `coverage` implements dossier pitfall 2 (cache tokens ⊆ `train_metric_cache` metadata, optional anchors-index check, `--move-aside`).
- `F/tests/test_compact_cache_pb.py` — 7 tests (details in §3).
- `F/scripts/h100/build_pb_cache.sh [workers]` — venv-pf (or `PB_CACHE_ENV=pb`), pre-checks S1 (206,576 files) / S8 (a pickle per cache log), build, count, `verify --limit 300`.
- `F/scripts/h100/train_pb.sh <tag> [seed] [overrides]` — venv-pb, upstream recipe on one GPU: `BATCH=64 ACCUM=1` (effective 64, no accumulation),
  20 epochs, `16-mixed`, `strategy=auto`, 6 loader workers, `+agent.config.side_cams_from_calibration=true`, optional
  `FRONTDRIVE_PRETRAIN_PT` → `+agent.config.pretrain_pt_path=…`, exports (5,10,15,16–20), `last.ckpt`, resume via `FRONTDRIVE_RESUME_CKPT`;
  pre-checks (venv, e50 file, cache marker, anchors npy, `train_metric_cache` metadata path validity, anchors dir, token coverage);
  VRAM sampler of our PID (`gpu_mem.csv`) and `accounting.json` (effective batch, steps/epoch, peak VRAM, s/step, wall, rc, OOM flag, exports).
- `F/scripts/h100/smoke_pb_train.sh [sizes]` — `64 96 128` × (200 train batches + 20 val batches, 1 epoch) → `pb_smoke/summary.tsv`
  (rc, OOM, s/step, peak VRAM, last.ckpt); stops at the first OOM; resumes the first fitting size from its `last.ckpt` for a second epoch.
- `F/env/drivejepa_pb_local/` — local CPU test venv (python 3.10.22, torch 2.1.0+cpu, torchvision 0.16.0+cpu, mmcv-full 1.7.2 **CPU** wheel
  from `download.openmmlab.com/mmcv/dist/cpu/torch2.1`, mmdet 2.28.2, numpy 1.24.0, hydra 1.2.0, PL 2.2.1, nuplan-devkit ce3c323,
  omegaconf pinned 2.3.1, setuptools 65.5.1; 2.8 GB; built with uv in ~3 min from the cache). Freeze: `F/env/drivejepa_pb_local_freeze_2026-10-17.txt`.
  `import mmcv.ops` works on CPU; the full pb agent import chain imports.

## 3. Verification

| check | result |
|---|---|
| `py_compile` of the 6 edited vendored files + the 2 new python files (both venvs) | ok |
| `git apply --check -R` of the regenerated patch against the working tree | consistent (13 files, 147+/24−, sha256[:12] 8e8440412903, 480 lines) |
| encoder layouts: `torch.load(mmap)` of e50 and `e2_Cp_comma_plain_from_e50_e40_target_encoder.pt` | both: `target_encoder` with 292 keys, all `module.backbone.*`, same first/last keys, `patch_embed.proj.weight` (1024,3,2,16,16), float32 → drop-in |
| `ImgEncoder(DriveJEPAConfig(pretrain_pt_path=<C′ export>))` on CPU from CWD `/tmp` (yaml fallback) | `loaded pretrained model with msg: <All keys matched successfully>`, `[ImgEncoder] loaded encoder weights from …e2_Cp…pt`, 303.9 M params, 6.0 s, RSS 2.8 GB |
| defaults identical to upstream (CPU venv): `DriveJEPAConfig()` → `""`/`False`; `get_sensor_config()` → `{cam_f0:[2,3], cam_l0:[3], …, cam_b0:[3], lidar_pc:[]}`, `keep_calibration False`, `asdict` has the 9 upstream keys; flag on → side cams `[]`, instance `keep_calibration True`, class attr stays False | ok (test `test_agent_sensor_config_flag`, `test_dataclass_defaults_are_upstream`) |
| `torch.FloatTensor(float64)` == `astype(float32)` bitwise; `mmcv.imresize`/`impad_to_multiple` of 1080×1920 → (432,768) → (448,768) | ok |
| hydra `compose` + `instantiate(cfg.agent.config)` with `+agent.config.side_cams_from_calibration=true +agent.config.pretrain_pt_path=…` (omegaconf 2.3.1) | ok; without `+` hydra refuses (`Key … is not in struct`) → wrappers use `+` |
| `pytest tests/test_compact_cache_pb.py` in `env/drivejepa_pb_local` | **7 passed** (3.1 s) — `pytest_drivejepa_pb_local.txt` |
| same in `env/drivejepa` (venv-pf, no mmcv) | 5 passed, 2 skipped (feature-builder / agent tests) — `pytest_drivejepa_pf.txt` |
| derived cache CLI on dev log `2021.06.07.11.59.52_veh-35_02283_02464` in both venvs | 17/17 tokens written, `verify` 17 ok / 0 fail, decompressed contents identical across venvs (34 files) |
| build throughput (local, `--limit-logs 24 --workers 2`, nice) | 1,931 tokens in 4.5 s (~430 tokens/s); 224 MB → ~116 KB/token → ≈12 GB for 103,288 tokens, ≈4 min at 2 workers (`build_summary_24logs_2workers.json`) |
| camera model assumption | all 8 cameras in 40 dev logs (542 sampled frames each): `fx=fy=1545, cx=960, cy=560`; local CAM_F0 JPEGs are 1920×1080 → the replicated `img_shape` (448,768,3) is the upstream value for every row |
| `push.sh scan` (leak gate over the code set incl. the new files) | clean (76 code files) |

What `test_feature_builder_calibration_only_matches_upstream_and_cache` establishes (mmcv venv, dev token with local CAM_F0 frames 2,3):
`SceneLoader` with the flagged sensor config opens no side JPEG (`cam_l0.image is None`, calibration present); `DriveJEPAFeatureBuilder`
output keys `{img_shape, lidar2img, camera_feature_1, camera_feature_2, ego_status}` with shapes `(4,3) f32 / (4,4,4) f32 / [3,256,512] f32 ×2 /
(4,11) f32`; `lidar2img`, `img_shape`, `ego_status` **exactly equal** the derived cache; the F0 row equals an independent
`S @ K4 @ [R^T | −R^T t]` projection from the pickle (float32 tolerance) and is **bit-identical** to the upstream image-based arithmetic
(`LoadMultiViewImageFromFiles(agent_input)` with the flag off on the same input gives the single F0 row); the cached JPEG frames reach
PSNR ≥ 35 dB vs the exact float features; via the map-backed `Scene` path (what `run_dataset_caching.py` uses) the pb `DriveJEPATargetBuilder`
target equals the derived `trajectory`/`token` and `Scene.get_agent_input()` features equal the cache too.

## 4. How to run on the H100 (after S1 + S8 are pushed; HF `train_metric_cache.tar` + `anchors_scores_index.tar` extracted; both venvs built)

```bash
F=/home/henry/Desktop/Qwen_Drive/frontdrive; QD=/home/henry/Desktop/Qwen_Drive
bash $F/scripts/h100/build_pb_cache.sh 8                                   # venv-pf, ~5 min, ~12 GB -> benchmarks/navsim/cache/navtrain_compact_pb
bash $F/scripts/h100/smoke_pb_train.sh 64 96 128                           # H.8 (iv): rc/OOM, s/step, peak VRAM per micro-batch, resume check
bash $F/scripts/h100/train_pb.sh pb_e50_s0 0                               # ② baseline: e50 init, effective batch 64 x 1, 20 epochs
FRONTDRIVE_PRETRAIN_PT=$QD/benchmarks/pretrain/exports/e2_Cp_comma_plain_from_e50_e40_target_encoder.pt bash $F/scripts/h100/train_pb.sh pb_Cp_s0 0   # ③ C'
# dev eval of an export: bash $F/scripts/h100/eval_pb.sh pb_e50_s0_ep18 $NAVSIM_EXP_ROOT/pb_e50_s0/exports/epoch_18.ckpt navtrain_val_official 4
#   (needs PB_REQUIRE_SIDE_CAMS=0 and `+agent.config.side_cams_from_calibration=true` once eval_pb.sh is updated, see open issue 2)
```
Henry's rule (10-17 01:40): largest micro-batch that fits, no gradient accumulation; the baseline keeps effective batch 64 = `BATCH=64 ACCUM=1`
(the wrapper prints a NOTE when `BATCH*ACCUM != 64`); a larger micro-batch found by the smoke is a §2-④ knowledge point with the lr scaling recorded.

## 5. Open issues

1. **omegaconf ≥ 2.4 breaks the devkit configs** (`YAML node expansion exceeds the configured limit of 10000` on
   `default_train_val_test_log_split.yaml`, 14,561 logs; the 103k-token scene filters too). The upstream `requirements.txt` pins
   hydra-core 1.2.0 (`omegaconf~=2.2`) but not omegaconf, and an unpinned install resolved 2.4.0 here. `scripts/h100/bootstrap_pb_env.sh`
   (not in this task's edit list) must add `'omegaconf<2.4'` (2.3.1 verified); `train_pb.sh` additionally exports
   `OMEGACONF_MAX_YAML_EXPANDED_NODES=10000000` as a mitigation, but `eval_pb.sh` / the upstream scripts do not. venv-pf has 2.3.1 (fine).
2. `eval_pb.sh` (not in the edit list) still requires side-camera JPEGs (`PB_REQUIRE_SIDE_CAMS=1`) and does not pass
   `+agent.config.side_cams_from_calibration=true`; the new config keys need hydra's `+` prefix (or adding them to
   `config/common/agent/drive_jepa_perception_based_agent.yaml`, also outside the edit list).
3. The flag-OFF path with all four images could not be exercised locally (no CAM_L0/R0/B0 JPEGs anywhere locally); equivalence is shown
   at the arithmetic level (F0 row bit-identical, identical `SensorConfig`, identical dataclass defaults, upstream `img_shape` value).
   A one-off cross-check on the server/data host (a few test-split tokens that have all 8 cameras: features with and without the flag) would close it.
4. Side-camera image size 1920×1080 is inferred from the identical intrinsics of all 8 cameras (and nuPlan's documentation), not from side
   JPEGs; the flagged path asserts only the loaded images' size.
5. `train_metric_cache` coverage of the 103,288 cache tokens (and the anchors index) is unknown until the HF tar is extracted; `train_pb.sh`
   refuses to start on gaps unless `PB_COVERAGE_MOVE_ASIDE=1` (moves offenders to `navtrain_compact_pb_uncovered/`) or `PB_SKIP_COVERAGE_CHECK=1`.
6. Peak VRAM is sampled from `nvidia-smi --query-compute-apps` (10 s in the smoke, 30 s in training), not `torch.cuda.max_memory_allocated`
   (a callback would need `run_training.py`, outside the edit list); the smoke's s/step is the PL progress bar's last it/s (includes the online scorer).
7. The derived cache stores numpy arrays and JPEG bytes: it is only valid through `CompactCacheOnlyDataset` (marker + `FRONTDRIVE_NAVSIM_EXT`,
   both set by `train_pb.sh`); the upstream `CacheOnlyDataset` would hand bytes to the model.
8. Nothing was committed (main agent commits); nothing ran on the GPU.

---

# TASK B addendum (2026-10-17 11:00–12:00 EDT): split-restricted derived cache, builder engine, builder-mode verify, spec-conformant wrappers

Same conventions as above (`F`, `V1`). Local only, no GPU, nothing committed. Files of this addendum: `pytest_*_taskB.txt`,
`verify_builder_cli_taskB.txt`, `engine_equality_74_tokens.txt`, `build_summary_3val_3train_logs_builder_engine.jsonl`,
`smoke_pick_synthetic.txt`, `patch_sha_taskB.txt`.

## B.1 Vendored-tree changes on top of §1 (defaults unchanged; patch regenerated, see `patch_sha_taskB.txt`)

| file | lines | change |
|---|---|---|
| `V1/agents/drive_jepa_perception_based/drive_jepa_config.py` | 2, 62–65 | `encoder_activation_checkpointing: Optional[bool] = None` — None keeps the vjepa2 yaml value (`in1k.yaml:162` `use_activation_checkpointing: true`); `False` = the ViT keeps its activations (more VRAM, faster, identical math); hydra `+agent.config.encoder_activation_checkpointing=false` |
| `V1/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py` | 12–33, 47–51 | `encoder_init_kwargs(config)` = the upstream `__init__` kwargs construction (yaml, `img_as_video_nframes=2`, `model_name=vit_large`) + the two frontdrive overrides, extracted so it is testable without the 5 GB weights; `ImgEncoder.__init__` calls it; the provenance print now also states `activation_checkpointing=` |
| `V1/agents/drive_jepa_perception_based/drive_jepa_features.py` | 29–47, 94 | `_ego_status_feature(agent_input)` = the unchanged ego-status loop of `compute_features`, extracted so the cache derivation runs the builder's own code; `compute_features` calls it |
| `V1/agents/{drive_jepa_perception_based,pad}/bevformer/{decoder,spatial_cross_attention}.py` | +`import os`, one condition each | **not TASK B**: written 11:43 EDT by the concurrent H100-migration work (`MSDA_PYTORCH=1` → exact PyTorch `ms_deform_attn` fallback); they are in the working tree, so the regenerated patch (= `git diff` of the vendored repo, 17 files) carries them — flagged for the main agent's review |

Checked on CPU with the real e50 file (venv `drivejepa_pb_local`): `ImgEncoder(DriveJEPAConfig(encoder_activation_checkpointing=False))` →
`img_backbone.use_activation_checkpointing is False`, 304.1 M params, print `(activation_checkpointing=False)`; default → `True`;
backbone forward with checkpointing off vs on (same weights, eval, no grad) **max abs diff 0.0**, and the fp32 gradient of
`blocks[0].attn.qkv.weight` off vs on **max abs diff 0.0** → the knob only trades VRAM for speed.

## B.2 `F/navsim_ext/compact_cache_pb.py` (rewritten, 753 lines)

- **Two derivation engines** (`--engine auto|builder|numpy`, `resolve_engine` :188): `builder` (:197–244) = the patched vendored code
  itself — devkit `AgentInput.from_scene_dict_list` with calibration-only cameras (`SensorConfig.keep_calibration`),
  `bev_feature_build.LoadMultiViewImageFromFiles(side_cams_from_calibration=True)` (4 rows, no image), the vendored
  `NormalizeMultiviewImage`/`RandomScaleImageMultiViewImage`/`PadMultiViewImage` transforms (scale applied to every row), `torch.FloatTensor`
  like `_get_bev_feature`, the padded shape from the same transforms on a blank 1080×1920 image (`padded_img_shape_builder` :210), and
  `drive_jepa_features._ego_status_feature` → identical to the training-time builder by construction (needs mmcv → venv-pb);
  `numpy` (:103–175) = the torch/mmcv-free mirror of §2 (venv-pf). `engines_agree` (:246) is asserted on the first derived token of
  every log when the builder engine runs (`--no-selfcheck` disables).
- **`--split`** (`split_members` :91): logs **and** tokens from the vendored `scene_filter/<split>.yaml` (PyYAML CSafeLoader, 0.17 s for the
  103k-token navtrain yaml; no omegaconf → immune to the ≥2.4 node limit); per-log token subsets are computed in the parent (:333–341) so
  workers never receive the 100k-token set. `build` summary now records `split`, `engine`, `n_split_tokens_in_cache`, `not_in_split`; summaries
  append to `<out>_build_summary.jsonl`, provenance lines to `<out>_info.txt`.
- **`verify --n N --mode auto|builder|pf`** (:476–668): `builder` samples N derived tokens of logs whose CAM_F0 frames p-1, p exist under
  `--blobs` (spread over ≤10 logs, `_sample_builder_tokens` :525), loads them **through `CompactCacheOnlyDataset`** (what the training
  DataLoader sees) and compares with the patched upstream `DriveJEPAFeatureBuilder` (flagged `SensorConfig` from `DriveJEPAAgent.get_sensor_config`)
  and `DriveJEPATargetBuilder` on the map-backed `Scene`: **exact** `img_shape`/`lidar2img`/`ego_status`/`trajectory`+token (dtype-checked),
  **PSNR ≥ 35 dB** for `camera_feature_1/2`; without maps the trajectory falls back to the pf target (`target_ref` counts both).
  `auto` = builder when mmcv and blobs are present, else `pf` (the §2 checks, now also against the numpy engine). `count` prints complete token dirs.

## B.3 Tests — `F/tests/test_compact_cache_pb.py` (12 tests, 5 new)

`test_split_resolution` (214/978 logs, 18,179/85,109 tokens, disjoint, union = navtrain 1,192/103,288, equals the JSON record),
`test_encoder_kwargs_defaults_and_activation_checkpointing` (upstream kwargs by default; overrides reach `vit_tiny` with identical forward),
`test_hydra_overrides_instantiate_config` (`+agent.config.{side_cams_from_calibration,encoder_activation_checkpointing,pretrain_pt_path}`
compose+instantiate; defaults `False/None/""`), `test_builder_engine_equals_numpy_engine` (mmcv; bit-identical on 6 dev tokens, fixture cache
built by the builder engine with selfcheck), `test_verify_builder_mode_on_5_dev_tokens` (mmcv; `verify(mode="builder", n=5)` → 5/5 exact, PSNR ≥ 35,
`target_ref scene 5`; `auto` picks builder with blobs and pf without).
Results: **12 passed** (`env/drivejepa_pb_local`, 19.7 s, `pytest_drivejepa_pb_local_taskB.txt`); **8 passed, 4 skipped** (mmcv) in `env/drivejepa`
(`pytest_drivejepa_pf_taskB.txt`).

## B.4 Wrappers (spec of TASK B)

- `scripts/h100/build_pb_cache.sh [workers]` — sources **venv-pb** by default (`PB_CACHE_ENV=pf` alternative), asserts the venv interpreter is the
  `python` on PATH (:31–34; the first local run slipped through to miniconda's python → fixed), pre-checks S1 (206,576 files) / S8 (a pickle per log),
  builds `--split navtrain_val_official` then `navtrain_train_official` into `$NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache` (upstream
  `cache_path`; `PB_CACHE_OUT` overrides), `count` ≥ 103,288, `verify --n 50 --mode auto` (builder mode on the dev logs with CAM_F0 blobs), and an
  informative `coverage` run when `train_metric_cache` is present.
- `scripts/h100/train_pb.sh <tag> [seed] [overrides]` — knobs `TAG SEED BATCH=64 ACCUM=1 EPOCHS=20 WORKERS=6 PRECISION SIDE_CALIB=1 ACT_CKPT=1 PRETRAIN_PT
  EXPORT_EVERY=5 EXPORT_FROM=16 SAVE_LAST=1 RESUME CKPT_EVERY CACHE` (FRONTDRIVE_* spellings accepted); **refuses `BATCH*ACCUM != 64` unless
  `ALLOW_NONSTANDARD_BATCH=1`** (:45–48, verified locally: `BATCH=96` → REFUSED rc 2, `BATCH=32 ACCUM=2` passes the guard); `params.json` at start
  (all knobs, `git describe` or `no-git`, patch/run_training sha12, torch/PL/GPU) (:96–117); `python … > >(tee -a train.log) 2>&1 &` so `$!` is the
  trainer PID for the `nvidia-smi --query-compute-apps` sampler (`gpu_mem.csv`); `accounting.json` (peak VRAM of our PID, GPU total, steps/epoch,
  s/step, OOM flag, exports). `ACT_CKPT=0` → `+agent.config.encoder_activation_checkpointing=false` = the recipe-preserving way to raise VRAM use.
- `scripts/h100/smoke_pb_train.sh [sizes]` — `64 96 128` × `limit_train_batches=30`, `limit_val_batches=5`, `max_epochs=1` into `pb_smoke/<batch>/`,
  plus `pb_smoke/64_noackpt/` (ACT_CKPT=0); sizes ≠ 64 run with `ALLOW_NONSTANDARD_BATCH=1` (throughput only); stops at the first OOM; `summary.tsv`
  + `pick.json`: **largest micro-batch with rc 0, no OOM, peak ≤ CAP_MIB (60000)** and the **baseline recommendation** (`BATCH=64 ACCUM=1`,
  `ACT_CKPT=0` if that row fits, else 1); resume check of the baseline from `last.ckpt`. The embedded python of both wrappers was run on synthetic
  run dirs (`smoke_pick_synthetic.txt`: 64 → 31,000 MiB fits, 64_noackpt 52,000 fits, 96 47,000 fits, 128 OOM → pick 96, recommend `ACT_CKPT=0`).

## B.5 CLI verification (local)

| run | result |
|---|---|
| venv `drivejepa_pb_local`: `build --split navtrain_val_official --limit-logs 3 --workers 2` → engine **builder** | 74 tokens / 3 logs, 2.1 s, selfcheck on, 0 failed |
| same, `--split navtrain_train_official --limit-logs 3` | 86 tokens of 3 *train* logs (split restriction works), `count` → 160/160 complete |
| `verify --n 5 --mode builder` / `--n 50` | 5/5 and 48/48 (3 logs × ≤17 per log) ok, exact 100 % for the four keys, PSNR min 38.9 / mean 40.1 dB, `target_ref scene` for all, 4 s / 10 s |
| `verify --n 50 --mode pf` | 50/50 ok |
| venv `drivejepa` (no mmcv): same val build → engine **numpy**; `verify --mode auto` falls back to `pf` (50/50); `--engine builder` refused with a clear RuntimeError | ok |
| builder-engine vs numpy-engine caches (74 tokens, 148 files) | **decompressed bytes identical 148/148**, all keys equal (`engine_equality_74_tokens.txt`) |
| size | 113 KB/token → **≈11.7 GB** for 103,288 tokens |
| `bash -n` (3 wrappers), `py_compile` (both venvs, 8 files), `push.sh scan` | ok / ok / clean (84 code files) |
| accidental full run of `build_pb_cache.sh` with `PB_CACHE_ENV=pf` (venv-pf exists locally) | ran end-to-end into a scratch dir until stopped during the train split (val split done): the wrapper works in venv-pf; artefacts removed, `$NAVSIM_EXP_ROOT` clean |

## B.6 Open issues (delta to §5)

1. §5-1 (omegaconf ≥ 2.4) unchanged for `run_training.py`/`eval_pb.sh`; `compact_cache_pb.py` itself no longer depends on omegaconf.
2. §5-2 (`eval_pb.sh`) unchanged — still needs `+agent.config.side_cams_from_calibration=true` and `PB_REQUIRE_SIDE_CAMS=0`.
3. The 4 `MSDA_PYTORCH` hunks in the patch are not TASK B's; if the migration agent regenerates the patch concurrently the file may churn — the content
   is the same `git diff`.
4. `verify --mode builder` on the server needs the dev CAM_F0 blobs (S3) and maps (`NUPLAN_MAPS_ROOT`, pushed by S3) for the `Scene` target; without maps it
   reports `target_ref pf` + `scene_error` (still a pass if the pf target matches).
5. Peak VRAM remains nvidia-smi sampling (5 s in the smoke); `train_metric_cache` coverage of the 103,288 tokens still unknown until the HF tar is extracted.
6. The online scorer (16 forked processes, each lzma-loading `metric_cache.pkl` per sample) is CPU/RAM-bound and not measured locally; the smoke's
   `wall_s`/`s_per_step` include it.
