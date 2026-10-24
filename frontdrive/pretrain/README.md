# pretrain/ — continued V-JEPA pretraining on comma clips (plan v4 §3, §5 E2)

- `configs/comma_vitl_256x512_8f.yaml`: arm C/A base config (plain V-JEPA objective) for the vendored app in
  `third_party/Drive-JEPA/navsim_v1/vjepa2` (`app/vjepa/train.py`). Needs the frontdrive patches: rectangular crop in
  `app/vjepa/transforms.py`, `random_horizontal_flip` config key, and `meta.init_from` (weights-only init) in `train.py`.
- Clip lists: `<mp4> 0` per line, produced from `commadata.clips` manifests (`status == ok`).
- Arms (plan §3.1 2x2): A/B public mixture only, C/D public ⊕ comma 1:1; B/D add the motion heads (`motion_heads.py` + `train_motion.py`, see below).
- Accounting: every run logs steps, tokens (= steps × batch × 2048), FLOPs estimate and wall time to the ledger.

## Measured (2026-10-11, RTX 5090, evidence `docs/evidence/2026-10-11_E2_smoke/`)
- Vendored app runs with the patches: `cd third_party/Drive-JEPA/navsim_v1/vjepa2 && python -m app.main --fname <cfg> --devices cuda:0`
  (PYTHONPATH from `env/drivejepa_env.sh` also allows running from any directory).
- Batch 16, 8x256x512, activation checkpointing, bf16: **~1.45 s/step, 11.4 GB peak** → 100k steps ≈ 40 h per arm
  (plan v4 §5 assumed 20 h). Headroom: batch 32 or no activation checkpointing when the GPU is exclusive.
- Two patch fixes found by the smoke test: YAML `crop_size: [256, 512]` must be converted to a tuple before the mask
  generator; `meta.init_from` must strip only the DDP `module.` prefix (the wrappers keep `backbone.`). Verified e50 load
  through `train_motion.py` (it loads into the unwrapped modules; in the vendored app the same block runs after the DDP
  wrapping and loads nothing, see the caveat below): encoder / target_encoder missing 0 unexpected 0; predictor missing 0
  unexpected 8 (e50 has 10 mask tokens, we use 2).
- Checkpoint `latest.pt` = {encoder, predictor, target_encoder, opt, scaler, epoch, ...} with `backbone.*` keys →
  export `{target_encoder}` for `vit_encoder.init_module`.
- Motion arm code (B/D): `motion_data.py`, `motion_heads.py`, `train_motion.py` — see the section below and `tests/`.

## Motion arm (plan v4.1 §3.1): `train_motion.py`
- Launch: `PYTHONPATH=$FRONTDRIVE:$PYTHONPATH python -m pretrain.train_motion --fname pretrain/configs/comma_vitl_256x512_8f_motion.yaml --devices cuda:0`
  (launcher mirrors `app/main.py`; fork-only config keys: the `motion` block, `meta.device`, `meta.max_steps`).
- Fork of the patched `app/vjepa/train.py`: every change is a `# [motion]` block (`diff -u` against the vendored file). With
  `motion.enabled: false` data, masks, losses and RNG streams are identical to the vendored trainer — `tests/test_train_motion.py`
  compares the two step for step (`tiny_motion_test.yaml`, synthetic clips; losses agree to all logged decimals).
- Heads P_phi / R_psi are built after the backbones under a forked RNG (matched seeds: arm C and D see the same clips, crops, masks
  and backbone init) and get two extra AdamW groups (decay / no-decay with `WD_exclude`) that follow the predictor's lr / wd schedule.
- Units at the data/heads interface: `motion_data.py` emits RAW physical units (sequence: m/s, rad/s, m/s², m, m, rad, flags[, deg];
  targets `[dx, dy, dyaw, v_mean, yaw_rate_mean, dv, dyaw_rate]`: m, m, rad, m/s, rad/s, m/s, rad/s; dt in s). Normalisation happens
  once, in the heads (`MotionSequenceEncoder._SEQ_FEATURE_SCALES`, `readout_loss` / `motion.target_scales`, per horizon: dx/dy/dyaw/dv
  scales × slot gap d). **Yaw signs**: the sidecar's `d_yaw` (Δ of the NED heading) and `wdev_z` (device gyro, z down) are both
  clockwise-positive while `d_left` is left-positive; `motion_data` flips both (`D_YAW_SIGN = YAW_RATE_SIGN = -1`) so every emitted
  yaw quantity is CCW and consistent with `d_left` (verified 2026-10-12 on 600 local sidecars: `d_left = -d_forward·tan(d_yaw/2)`,
  sign agreement 98.8 %; with the earlier `+1` convention the integrated dy had the wrong sign in 97 % of turning 3-s windows).
  Per-token rays are rotated into the road-aligned camera frame with the clip's median `cal_pitch`/`cal_yaw` (`rays_road_frame`).
- Checkpoint `latest.pt` = vendored keys (`encoder`, `predictor`, `target_encoder` as DDP state dicts `module.backbone.*`, `opt`,
  `scaler`, `epoch`, ...) + `motion_heads: {pred, readout}` (plain state dicts), `motion_cfg`, `config_hash`, `accounting`,
  `weights_only`. Saves are atomic (`.tmp` → `os.replace`), keep the previous file as `latest.prev.pt`, refuse to write when the
  free space is < 2× the estimated size (fail loudly instead of training on without checkpoints); periodic `e{N}.pt` are
  weights-only (~2.5 GB for ViT-L vs ~5.3 GB with the optimizer). An existing `latest.pt` in `folder` is auto-resumed (the vendored
  `r_path=None` crash is fixed in the fork), an unreadable one falls back to `latest.prev.pt` (else a loud fresh start with
  `init_from`), heads are reloaded strictly and the accounting continues. Resuming a motion run from a plain-JEPA checkpoint is not
  supported (use `init_from`). `meta.device: cuda:N` with CUDA unavailable is an error, never a silent CPU run.
- Logging: `log_r<rank>.csv` columns epoch, itr, loss, loss_jepa, loss_pred, loss_readout, loss_cons, iter_ms, gpu_ms, data_ms,
  tokens_step, flops_step, then the diagnostics loss_pred_vis (L1 on target positions the online encoder saw), loss_readout_noctx
  (L_readout with the online context removed), grad_ratio_ctx (‖∂L_readout/∂z_ctx‖ / ‖∂L_readout/∂z_tgt‖), e_std (collapse monitor
  of R_ψ's embedding), frac_ctx_ok, ctx_vis_mean, frac_mismatch (shuffle control) and the per-target Huber terms huber_<target>;
  `<folder>/accounting.json` (cumulative steps, tokens, FLOPs estimate = 2·params·tokens, ×3 with backward, wall-clock, config hash,
  per-module meter, `exit_status` ok / nan_loss / data_error / aborted) every epoch and at exit (also via `atexit` on an abnormal
  exit) — heads make "same steps ≠ same compute".
- Shortcut counter-measures (review 2026-10-12, `motion.*` keys): the target-encoder tokens are a whole-clip representation and the
  one-slot targets (dx ≈ v·dt, v_mean) are readable from them alone, so `R_ψ(z_s, sg(z̄_{s+d}))` could ignore the online branch.
  Therefore: pair-specific targets dv / dyaw_rate; per-item **target-branch dropout** `readout_tgt_drop` (default 0.5: the online
  tokens become the only input, the deployed encoder necessarily receives the readout gradient); `pred_masked_only` (L_pred only on
  target positions the online encoder did not see — the tube masks keep the same spatial positions in every slot); `min_ctx_tokens`
  (items whose context slot was emptied by the collator's end-truncation are excluded); the CAN-mismatch control permutes within
  (speed bin × slot gap) strata among items with CAN and masks the fixed points; `cons_weight > 0` requires `use_readout`;
  `mask_config` must be the light mask. `tests/test_motion_heads.py::test_readout_shortcut_is_detected_and_target_dropout_restores_online_gradient`
  reproduces the shortcut on synthetic data and shows the dropout restores the online-side gradient.
- Measured (2026-10-12 integration smoke, first 200 local clips, batch 4, 4 workers, 20 steps, bf16 + activation checkpointing,
  e50 init; the GPU was shared with another user's job, so timings are upper bounds): plain arm (`motion.enabled: false`)
  9.9 GB peak (`torch.cuda.max_memory_allocated`), median 0.42 s/step, 25.9k tokens / 14.3 TFLOP per step (estimate), loss
  0.510 → 0.494; motion arm (`cons_weight 0.5, cons_every 2`) 10.2 GB peak, median 0.92 s (plain steps) / 1.09 s (consistency
  steps) under contention, 39.8k tokens / 19.2 TFLOP per step on average (31.1k / 14.1 on plain steps, 48.5k / 24.3 on
  consistency steps), all terms finite (jepa 0.524 → 0.494, pred 0.736 → 0.626, readout 0.092 → 0.008, cons 0.097 → 0.013).
  Equivalence of the disabled fork with the vendored trainer (3 steps, same seed, `num_workers 0`): tiny config (fp32) and
  ViT-L (fp32, 20 real clips, batch 2) identical to all 5 logged decimals; ViT-L bf16 identical at step 0 and within 1e-5 at
  steps 1-2, where two vendored runs also differ from each other by 1e-5 (non-deterministic bf16 kernels, not a fork effect).
  Note: with `num_workers > 0` the vendored `MaskCollator` step counter races across workers, so masks (and therefore step-0
  losses) are not reproducible run to run.
- Caveat (2026-10-11): the vendored `meta.init_from` patch runs after the DDP wrapping and loads `backbone.*` keys into a module whose
  keys are `module.backbone.*`, so in the vendored app it matches nothing (log: `encoder missing=292 unexpected=292`, weights stay
  random). The fork loads into the unwrapped modules and raises if encoder / target_encoder keys do not match exactly (verified: e50
  292/292 keys match; predictor tolerates the 8 extra e50 mask tokens). Plain-JEPA arms through the vendored app need that fix in
  the vendored file first (or run them as `train_motion.py` with `motion.enabled: false`).

### Encoder checkpoints on this machine (2026-10-12)
| file | what | sha256 (16) | notes |
|---|---|---|---|
| `benchmarks/navsim/exp/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt` | Drive-JEPA continued-pretrained ViT-L ("e50", epoch 51) | — | init of E1 / arms C', D' |
| `benchmarks/vjepa2/vitl.pt` | original V-JEPA 2 ViT-L/16 (Meta, MIT; epoch 40, lr 5.25e-4 cooldown ckpt, 5.13 GB incl. optimizer) | `5346856ec9df6948` | same key layout as e50 (`module.backbone.*`, 292 / 160 / 292 keys, 2 predictor mask tokens); target_encoder differs from e50 by mean abs 0.019 per weight; init of E1 ranking encoder `vitl` and arm E_v2 (`configs/e2_Ev2_comma_motion_from_vitl.yaml`) |

## comma subset v0 — frozen clip list (2026-10-12)
- Clip list `benchmarks/pretrain/lists/comma_v0_clips.csv` (6,592 clips, 109.87 h at 2 fps, 11.4 GB; 6,538 with CAN sidecars;
  tici 2,258 / tizi 2,299 / mici 1,981 / unknown 54 = clips without sidecar), list sha256
  `d7701f3b53789459f89dd3ad70f5b72513d03863c075ffbef3cf1fd6c9910e94`.
- Frozen manifest with per-clip sha256 (clip ids only): `pretrain/manifests/comma_v0_manifest.csv` — this is the future
  public pretraining subset (plan §4.4); the raw manifest with route ids lives outside the repo (`manifest_raw/subset_v0/`).
- Built with `python -m pretrain.make_clip_list --clips <dir> --out <list> --manifest <csv>`.

### Sidecar revision 2026-10-12 (yaw sign of gyro-integrated headings)
- Audit of all 6,526 sidecars with `wdev_z`: in clips whose heading comes from `orientationNED` (77 %), `wdev_z` and the
  pose-derived `d_yaw` agree in sign on 98 % of turning frames (both clockwise-positive). In clips whose heading was
  gyro-integrated (`heading_src == gyro_int`, 1,474 clips / 24.6 h, 22 %) the exporter's default gyro sign was wrong
  (`-wdev_z`, assuming device z up) and was only repaired when a correlation test against the steer model / pose could run;
  89 of the 434 gyro_int clips with turning frames had a mirrored heading (d_yaw and d_left sign-flipped).
- Fix: `comma_export` commit b49510f (`trajectory.dead_reckon`: default `+wdev_z`, flip only on corr < -0.6;
  `clips.py --sidecar-only`). The 1,474 gyro_int sidecars were rewritten on the data host (2 min) and pulled here;
  **videos, clip ids and the clip manifest (`comma_v0_manifest.csv`, sha256 over the mp4) are unchanged**. After the fix:
  2 of 4,175 turning clips disagree (frames-weighted agreement 99.1 %). Pre-fix copies: `clips/subset_v0/_sidecar_backup_pre_gyrofix/`.
- Sidecar integrity list: `manifests/comma_v0_sidecar_manifest.csv` (clip_id, has_can, heading_src_main, n_rows, sha256 of
  the `.can.npz`), 6,538 sidecars.
- Remaining glitches (not fixable at export): 34 clips with isolated |d_yaw|/0.5 s > 1 rad/s and 22 with |wdev_z| > 1 rad/s
  (heading resets / gyro spikes), 12 with |a_ego| > 8 m/s^2. `pretrain/motion_data.py` now masks such values via
  `PLAUSIBLE_RANGES` (treated like NaN rows: per-feature `seq_valid`, `target_valid`).
- D2 note: the by-device probes (`docs/evidence/2026-10-12_D2_by_device/`) were run on the pre-fix sidecars; their 8f-vs-2f
  gap verdicts are unaffected (both inputs see the same targets) but absolute yaw-target R^2 of the full mici run is
  depressed by the spike clips.

### Public mixture sidecars (2026-10-13)
- The public control-mixture clips (CoVLA / DrivingDojo / OpenScene navtrain; exporters in `comma_export/publicdata/`, format
  fixed by `comma_export/docs/public_mixture_spec.md`) are byte-compatible with the comma v0 clips: `<id[:2]>/<id>.mp4` + `.can.npz`,
  512x256 at 2 fps, the same sidecar keys and the same sign conventions (`d_yaw` / `wdev_z` clockwise-positive, `d_left` left-positive;
  the exporters never pre-negate, `motion_data` keeps flipping both with `D_YAW_SIGN = YAW_RATE_SIGN = -1`). The mixture is a list of
  clip roots in the data config; nothing else in the trainer changes.
- New sidecar keys: **`K_clip`** (3x3 f32 pinhole intrinsics of the exported 512x256 frames, computed by the exporter from the source
  K and the crop box actually used), `fov_h_deg`, `src_fps`. `motion_data.sidecar_intrinsics(side, device)` prefers a finite (3, 3)
  `K_clip` over the per-device fallback, and the dataset derives `K` / `K2` / rays from it. Comma sidecars have no `K_clip`, so their
  intrinsics are unchanged (`clip_intrinsics(device)`, qcamera chain).
- New devices in `DEVICE_CAMERAS` / `DEVICE_IDS` (comma ids unchanged): `covla` (id 4, 1928x1208, f 2648), `dojo` (id 5, 1920x1080,
  f 1005.7), `openscene` (id 6, 1920x1080, f 1545). Their fallback `device_fallback_intrinsics(device)` (full-frame centred 2:1 crop,
  centre principal point) is only reached for a public clip without a sidecar: exact for covla (`[[703.2, 0, 256], [0, 703.2, 128]]`),
  approximate for openscene (cy 128 vs 133.3 from the off-centre principal point) and for dojo (the export crops a 64 deg, 1256 px
  box: fx ~410 in `K_clip` vs 268.2 full-width).
- `heading_src` values `pose` (DrivingDojo) / `ego2global` (OpenScene) need no consumer change: the key is skipped at load time
  (`_SIDECAR_SKIP_KEYS`) and no value of it is interpreted anywhere in `motion_data`.
- `PLAUSIBLE_RANGES` must stay identical to `comma_export/publicdata/common.PLAUSIBLE_RANGES` (the exporters report the out-of-range
  fraction with the same bounds); `tests/test_motion_data.py::test_plausible_ranges_match_public_exporter` asserts the equality, and a
  synthetic `dojo` sidecar built with `publicdata.common.make_sidecar` is checked end to end through `load_sidecar` / `motion_targets` /
  `MotionVideoDataset` (left turn -> dy > 0, dyaw > 0, yaw_rate > 0 after the CCW flip; chord residual 0; sign agreement 1.0).
