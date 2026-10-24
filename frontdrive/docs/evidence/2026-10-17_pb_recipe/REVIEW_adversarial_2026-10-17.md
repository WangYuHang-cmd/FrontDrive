# Adversarial review of the pb recipe work (TASK P + TASK B) — 2026-10-17 12:00–12:40 EDT, local only, no GPU

`F` = `/home/henry/Desktop/Qwen_Drive/frontdrive`, `V1` = `F/third_party/Drive-JEPA/navsim_v1/navsim`. Vendored tree NOT modified by
this review (patch still byte-identical to `git diff`, sha256[:12] `6cbe2e6c56a0`, 645 lines, 17 files).

## Findings

| # | severity | file:line | finding | status |
|---|---|---|---|---|
| 1 | MAJOR | `F/scripts/h100/train_pb.sh:27-30` (before) | `train_pb.sh <tag> <hydra override>` consumed the first override as the seed (`SEED=${2}` → `seed=trainer.params…`, hydra error); `TAG=<env> train_pb.sh <override>` dropped `$1` by the unconditional `shift` | **fixed** (:27-32: numeric-2nd-arg rule; env TAG/SEED leave positionals to the overrides; 6 cases exercised) |
| 2 | MAJOR | `train_pb.sh` accounting block (before :143-148) | peak VRAM = `None` when nvidia-smi has no per-process accounting (`[N/A]`) or our PID never appeared → the smoke's `fits=0` / "NO micro-batch fits" although the run succeeded; a `[N/A]` per-PID field also dropped the GPU-total sample of that line | **fixed**: per-column parsing, fallback `max(memory.used) − gpu_before.txt`, `peak_vram_source` + `gpu_used_mib_before_launch` recorded (synthetic dirs: per-PID 33000; N/A → 48000−15000=33000 with the fallback source; empty → None) |
| 3 | MAJOR | `train_pb.sh:60-62` | several `train_metric_cache/metadata/*.csv` (HF tar may ship `_node_*.csv`): `MetricCacheLoader` (`V1/common/dataloader.py:177`) reads an arbitrary `iterdir()[0]`, the coverage check `sorted()[0]` → KeyError mid-training on tokens of the other csv(s) | **fixed**: refuses unless exactly one csv (merge with `scripts/fix_metric_cache_paths.sh`, which also moves parts to `_parts/`) |
| 4 | MINOR | `train_pb.sh` accounting (before :152) | progress-line regex required `v_num` (absent when the logger has no version) → `s_per_step=None` | **fixed**: `Epoch \d+:` + `it/s|s/it` |
| 5 | MINOR | `F/scripts/h100/build_pb_cache.sh:54` | success check ignored token dirs that lack a pickle (interrupted write); `CacheOnlyDataset._load_valid_caches` (`V1/planning/training/dataset.py:99-107`) lists every token dir → FileNotFoundError at training | **fixed**: `token_dirs == tokens_complete` (:55-56), re-run heals (skip needs both files) |
| 6 | MINOR | `F/scripts/h100/smoke_pb_train.sh:21` | 5 s VRAM sampling for a ~1 min (30-step) run → peak likely missed | **fixed**: default 2 s |
| 7 | MINOR (open issue 3 of the Patch report) | `V1/agents/drive_jepa_perception_based/bevformer/bev_feature_build.py` | flag-OFF path with all four images never exercised | **closed**: `tests/test_compact_cache_pb.py::test_flag_off_with_four_images_equals_flag_on` — real calibration of the pickle, synthetic 1080×1920 side pixels (only the size enters `img_shape`): 4 raw rows `[b0,f0,l0,r0]` bit-identical (`lidar2img/cam_intrinsic/lidar2cam/camera_intrinsics`), all 5 `compute_features` keys `torch.equal` with equal dtype/shape, `[1:2]` row equal; also shows upstream's `_get_bev_feature` cannot even `np.stack` two image sizes (ValueError) — the single-sensor-size assumption is upstream's own |
| 8 | MINOR | `F/docs/evidence/2026-10-17_pb_recipe/README.md` §2, §4 | stale vs the TASK B addendum: output dir `navtrain_compact_pb`, venv-pf default, `verify --limit 300`, 200/20 smoke steps | open (documentation only; B.4/B.5 are current) |
| 9 | INFO | `V1/.../bevformer/simple_image_encoder.py:21-22,52-53` | the only default-off behavioural deltas: one provenance `print` line (upstream already `print(model)`s the whole ViT) and the CWD-relative yaml falling back to the vendored file (upstream raised FileNotFoundError off-root) | accepted |
| 10 | INFO | patch | 4 `MSDA_PYTORCH` hunks of the concurrent migration work (`{pb,pad}/bevformer/{decoder,spatial_cross_attention}.py`): `not os.environ.get("MSDA_PYTORCH")` → default unchanged | accepted (not this task's) |
| 11 | BLOCKER (outside the edit list) | `F/scripts/h100/bootstrap_pb_env.sh:25` | unpinned `pip install -r requirements.txt` resolves omegaconf 2.4.0 whose YAML node limit (10,000) rejects `default_train_val_test_log_split.yaml` / `scene_filter/navtrain*.yaml`. Verified in the cached 2.4.0 wheel (`omegaconf/_yaml.py:23`): the knob `OMEGACONF_MAX_YAML_EXPANDED_NODES` is real → `train_pb.sh:79` is a working mitigation for `run_training.py`; `eval_pb.sh` / upstream scripts are unprotected | open → main agent: add `'omegaconf<2.4'` to bootstrap_pb_env.sh (and/or export the variable in `env/drivejepa_pb_env.sh`) |
| 12 | MAJOR (outside the edit list) | `F/scripts/h100/eval_pb.sh:22-24,29-39` | no `+agent.config.side_cams_from_calibration=true`; demands CAM_L0 blobs (`PB_REQUIRE_SIDE_CAMS=1`) → dev/navtest evals of the self-trained pb ckpts need side JPEGs that are not planned to be pushed | open → main agent |

## Refutation attempts (what was checked, how)

1. **Default-off equivalence** — every hunk of the 6 pb files traced: `LoadMultiViewImageFromFiles` takes the upstream branch when the flag is off
   (`bev_feature_build.py:45-54`); `_get_bev_feature`'s extra `assert not (return_images and side_cams…)` is a no-op; `get_sensor_config` returns
   the upstream `SensorConfig` (`drive_jepa_agent.py:81-95`); `Cameras.from_camera_dict(..., keep_calibration=False)` → `Camera()` as upstream;
   `SensorConfig.keep_calibration` is a `ClassVar` under `from __future__ import annotations` and IS excluded from `fields()`/`asdict()`
   (`test_dataclass_defaults_are_upstream` asserts the 9 upstream fields; `get_sensors_at_iteration` unchanged); `_ego_status_feature` = the
   same ops; `encoder_init_kwargs` = the same yaml/kwargs (`use_activation_checkpointing` untouched when `None`). Deltas: #9 only. Not refuted.
2. **lidar2img / img_shape identity with the flag on** — new test #7 (flag off with 4 images vs flag on, same frame) + the existing F0-row
   bit-identity; scale 0.4 is applied to every `lidar2img` row by `RandomScaleImageMultiViewImage` (`transform3d.py:343`) regardless of the
   image list, `PadMultiViewImage` (`:73`) yields `(448,768,3)` for 1080×1920 (432×768 → /32), `torch.FloatTensor` → float32 for both keys
   (`bev_feature_build.py:114-115`), row order `[b0,f0,l0,r0]` from the single loop (`:44`), `drive_jepa_model.py:55` slices row 1 = F0. Not refuted.
   Residual: side sensors are 1920×1080 (nuPlan spec; the test shows upstream presupposes one size anyway).
3. **Cache format** — `test_compact_dataset_and_collate` re-run: `CompactCacheOnlyDataset` + `default_collate` give `img_shape (B,4,3) f32`,
   `lidar2img (B,4,4,4) f32`, `ego_status (B,4,11) f32`, `camera_feature_1/2 (B,3,256,512) f32` in [0,1] (RGB, == ToTensor), `trajectory (B,8,3)
   f64` (upstream `torch.tensor(poses)` is f64 too, `drive_jepa_features.py:153`), `token` list[str]; marker is a directory
   (`run_training.py:118` uses `os.path.exists`); frame order pf `camera_feature_1/2` = frames 3/2 = pb (`drive_jepa_features.py:67-76` vs pf `:90-99`);
   numpy pickles written by venv-pf (numpy 1.26.4) and venv-pb-local (numpy 1.24.0) are byte-identical and load in both. Not refuted
   (valid only through `CompactCacheOnlyDataset`, as documented).
4. **Builder resumability / parallel safety** — atomic `os.replace` writes, skip requires both pickles, one log per spawn worker, splits run
   sequentially into the same out dir, `_info.txt`/`_build_summary.jsonl` append-only; stray dirs from a kill now caught by #5. Not refuted.
5. **Wrappers** — the exact OVR list of `train_pb.sh` (+ the smoke's `limit_*`) composed with hydra 1.2.0/omegaconf 2.3.1 (6.9 s), every
   `trainer.params.*`/`dataloader.params.*` key exists in `config/training/default_training.yaml` (:26-29, :33-52), `pl.Trainer(**params)` →
   `SingleDeviceStrategy`, accumulate 1, limits 30/5; `DataLoader(**params)` ok; `+` is required for the three `agent.config` keys; the
   effective-batch guard (`:45-48`) read and the smoke passes `ALLOW_NONSTANDARD_BATCH=1` only for sizes ≠ 64; `$!` is the trainer PID
   (simple command in background with process substitution). Fixed #1-#4, #6.
6. **Patch** — `git -C <vendored> diff` is byte-identical to the patch file; `git show 548bb82:<path>` copies of the 17 files in a scratch
   git repo: `patch -p1 --dry-run` rc 0, `git apply --check` ok, after `git apply` every file is identical to the live vendored tree.

## Verification (final)

```
bash -n scripts/h100/{build_pb_cache,train_pb,smoke_pb_train}.sh                        -> ok ×3
env/drivejepa_pb_local/bin/python -m py_compile tests/test_compact_cache_pb.py navsim_ext/compact_cache_pb.py  -> ok (also venv-pf)
env/drivejepa_pb_local/bin/python -m pytest tests/test_compact_cache_pb.py -q           -> 13 passed in 20.65s
env/drivejepa/bin/python -m pytest tests/test_compact_cache_pb.py -q                    -> 8 passed, 5 skipped (mmcv) in 18.37s
cmp <(git -C third_party/Drive-JEPA diff) third_party_patches/drive_jepa_548bb82.patch  -> identical (6cbe2e6c56a0, 645 lines, 17 files)
patch -p1 --dry-run (fresh 548bb82 copies)                                              -> rc 0; git apply --check ok; post-apply == live tree
nvidia-smi memory.used                                                                   -> 15 MiB (no GPU use); no stray processes; $NAVSIM_EXP_ROOT has no pb artefacts
```
