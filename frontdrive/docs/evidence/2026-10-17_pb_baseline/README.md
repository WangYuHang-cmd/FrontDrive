# Perception-based baseline on the H100 (plan v4.3 §2 ② → ③) — evidence log

Started 2026-10-17 14:53 EDT (18:53Z) with `scripts/h100/phase5_pb_pipeline.sh` in tmux `fdpb` on the shared H100
(launched by Henry; Claude's session cannot run `ssh`, see `../2026-10-17_h100_migration/README.md`). Environment: conda
`cvpr` (Python 3.10, torch 2.1.2+cu121, upstream Drive-JEPA pins, mmcv-full 1.7.2 built for sm_90).

## Recipe (upstream `train_drive_jepa_perception_based.sh` on one GPU)

| item | value |
|---|---|
| agent | `drive_jepa_perception_based_agent` (vendored navsim_v1 @548bb82 + `third_party_patches/drive_jepa_548bb82.patch`) |
| effective batch | **64 = micro-batch 64 × accumulation 1** (official: 32 × 2 GPUs); larger micro-batches only as throughput measurements |
| epochs / optimizer | 20; Adam 1e-4, backbone 0.1×; precision 16-mixed; no scheduler / wd / clipping (upstream) |
| training cache | derived from the JPEG compact cache + log pickles (`navsim_ext/compact_cache_pb.py`, byte-identical to the patched upstream builder on 148 dev tokens); 103,288 tokens = 18,179 val + 85,109 train |
| side cameras | `side_cams_from_calibration=true` (lidar2img from the pickles; no CAM_L0/R0/B0 pixels; upstream never feeds side-camera pixels to the network) |
| scorer | online PDM scorer over `train_metric_cache` (v1, 22 GB) + `anchors_scores_index` (upstream, 16 processes) |
| encoder init | ② `vitl_merge_3dataset_e50.pt` (upstream default); ③ `PRETRAIN_PT` = C′ / D′ exports |
| VRAM knob | `ACT_CKPT=0` (encoder activation checkpointing off: same math, more VRAM, faster) if the smoke shows it fits under 60 GB |
| exports | fp16 agent weights at epochs 5, 10, 15 and 16–20 (`EXPORT_EVERY=5`, `EXPORT_FROM=16`); dev evals of every export |
| dev eval | `eval_pb.sh <tag> <ckpt> navtrain_val_official 4` with `PB_SIDE_CALIB=1` (18,157 scenes) |

## Chain log (fill in as stages complete; times UTC)

| stage | result |
|---|---|
| 1 cache | run 1 (18:53:50Z): val split 18,179 tokens in 0.2 min, train split 85,109 tokens by 18:55:09Z (engine = patched upstream builder, 12 workers; `tokens_complete 103288 = token_dirs`); **verify FAILED 50/50** with img_shape / lidar2img / ego_status exact and PSNR ≥ 37.6 but `trajectory` 0/50 bit-identical to the scene target recomputed on the server → chain aborted 18:55:50Z. Diagnosis: the derived trajectory is the pf target computed on the local machine; the scene target is recomputed on the H100 host (SE(2) transform with sin/cos) → last-ulp CPU / numpy-SIMD differences; the old check demanded `array_equal`. Fix 59fd092: tolerance 1e-9 (exact count still reported, max abs diff printed). **Run 2 (19:43:54Z → 19:45:56Z): verify 50/50 OK, trajectory bit-identical 0/50 but all within tolerance, `trajectory_max_abs_diff = 7.1e-15`** (hypothesis confirmed), img_shape / lidar2img / ego_status exact 50/50, PSNR min 37.63 / mean 40.14; coverage: 103,288 cache tokens = 103,288 train_metric_cache tokens, anchors present; cache 11 GB ✅ |
| 2 smoke 64 / 96 / 128 (+64 without act-ckpt, + resume check) | 19:45:56Z → 19:55:13Z (`pb_smoke/summary.tsv`, 30 steps + 5 val batches each): **64 / act-ckpt on: 2.86 s/step, peak 11,878 MiB**; **64 / act-ckpt off: 2.86 s/step, peak 41,116 MiB** (no speed-up from keeping the activations → the step is not GPU-bound; see the steady-state check below); **96 / act-ckpt on: rc 1 (not OOM) after 75 s** → sweep stopped, 128 not run. Cause (`pb_smoke/96/train.log`): the mmcv kernel asserts `batch % im2col_step == 0` in `ms_deform_attn_cuda.cu:244` with the upstream `im2col_step = 64` → a pb micro-batch must be a multiple of 64 (128 would have passed the assert; untested, irrelevant for the effective-batch-64 baseline, relevant only for §2-④ large-batch knowledge points); resume check of 64 from `last.ckpt`: OK (2.78 s/step). Pick: baseline BATCH 64 × ACCUM 1, **ACT_CKPT 0** (41 GB ≤ 60 GB cap) |
| 3 training `pb_e50_s0` | started 19:55:13Z: 64 × 1, 20 epochs, act-ckpt off, 6 loader workers, upstream 16-process scorer; 85,109 train samples → 1,329 steps/epoch. **Steady state (epoch 0, step 603, 20:21Z): 0.40 it/s = 2.5 s/step, our VRAM 41,116 MiB, GPU util alternating 92 % / 0 % (mean ≈ 39 %)** → the step is CPU-bound (online PDM scorer, 20 vCPU), not GPU-bound, which is why act-ckpt off gave no speed-up; ≈ 56 min/epoch + validation ⇒ ≈ 20–22 h for 20 epochs (ETA ≈ 2026-10-18 17–19Z). Measured with the reference eval (4 processes) running concurrently for the first 25 min. **Actual: `exports/epoch_5.ckpt` at 23:52Z = 5 epochs (incl. validation) in 3 h 57 min → 47.5 min/epoch → 20 epochs ≈ 15.8 h, ETA ≈ 2026-10-18 11:45Z (07:45 EDT)**; load1 during training ≈ 13. Online-scorer validation proxy after epoch 5 (progress bar, 18,179 val tokens vs `train_metric_cache`): `val/score_epoch 0.912` (collision 0.991, dac 0.971, progress 0.866, ttc 0.966, comfort 0.998, l2 1.04 m) — a proxy only; the dev PDMS comes from `eval_pb.sh` on the exports 🔄 |
| 3b training `pb_e50_s0` with C′ beside it | exports ep15 05:01Z, ep16 06:02Z, ep17 07:04Z, ep18 08:05Z, ep19 09:06Z → **61–62 min/epoch** with the second trainer (47.5 solo); epoch 20 at 49 % 13:34Z, ETA ≈ 14:03Z (10:03 EDT). Online-scorer proxy by completed epoch: 9 → 0.920, 12 → 0.926, **14 → 0.928 (max)**, 15 → 0.921, 16 → 0.924, 17 → 0.919, 18 → 0.925, 19 → 0.917 (fluctuating ≈ 0.92; formal PDMS of ep5 was proxy + 0.45). C′ proxy at the same epochs: 5 → 0.908, 9 → 0.920 (= e50 at 9) |
| 3c training `pb_e50_s0` **finished** | 20 epochs done 14:07:45Z (10-18): `exports/epoch_20.ckpt`, `checkpoints/last.ckpt` (+ Lightning `best_epoch=13.ckpt`); **wall 18 h 12 min** (19:55Z → 14:08Z; 4 h alone at 47.5 min/epoch, 14 h beside C′ at 61–62 min/epoch); last epoch 1,329 steps in 1:00:55 = 2.75 s/step; peak VRAM 41,138 MiB (act-ckpt off). The chain logged `finished rc=141`: the `train_pb.sh` **wrapper** had died of SIGPIPE at 23:58Z on 10-17 (`gpu_mem.csv` stops there, 486 samples) — its monitor loop used an early-exit `awk … exit` on `nvidia-smi --query-compute-apps`, which can SIGPIPE nvidia-smi as soon as more than one compute process is listed (first time: the ep5 dev eval's 4 processes, started 23:53Z); under `set -e -o pipefail` that ended the wrapper. The trainer (a background child) was unaffected and completed normally; lost: `accounting.json` and the VRAM samples after 23:58Z (numbers above reconstructed from the export mtimes and the progress bar). Fixed in `train_pb.sh` (no early-exit pipes, `set +e` around the loop) and the chain now waits for a live trainer pid instead of relaunching when the wrapper exits early. C′'s wrapper is alive (its `gpu_mem.csv` keeps growing) — the race only bites when nvidia-smi has several lines left to write |
| 4 dev evals (official ckpt reference + exports) | **`pb_official_v1_dev` 19:55Z → 20:20Z: PDMS 93.46** (18,157 scenes all valid; concurrent with training) → the dev reference for ②/③ (G-B: late mean of the self-trained baseline within 0.5 of it); exports: ep5 = 91.57 (00:19Z, auto-ingested by `pull.sh` → `eval/pb_ingest.sh`); ep10 / 15 / 16–20 pending. Sequence `phase5_sequence.sh` launched by Henry in tmux `fdpbseq` 00:25Z (② drain → C′ → D′) |

## Concurrency decision (2026-10-17 23:27 EDT, Henry: "服务器还有 GPU 可以利用")

The step is CPU-bound (GPU util ≈ 30 %, 41 GB used of 81.5 GB), so the spare GPU is used by running the ③ encoder-swap
training **beside** ② instead of after it. New runs default to activation checkpointing **on** (12 GB each; the smoke showed no
speed difference), so e50 (41 GB, started with act-ckpt off) + C′ (12 GB) = 53 GB stay under the 60 GB self-limit with
≥ 28 GB free for the other user. `phase5_pb_pipeline.sh` got a per-run flock (48e9541) so the sequence's own C′ item skips
while this C′ chain runs and D′ (12 GB) starts right after ②'s evals drain → two trainings at a time from then on.
`pb_Cp_s0` chain launched by Henry in tmux `fdpb_pb_Cp_s0` at 03:27:43Z (10-18): init `e2_Cp_comma_plain_from_e50_e40_target_encoder.pt`,
64 × 1, 20 epochs, act-ckpt on, exports / dev evals for epochs 16–20 only (`PB_EXPORT_EVERY=20`). The slowdown of ② from the
shared CPU is measured below once both run at steady state. First snapshot 03:29Z: GPU 41,138 + 11,858 MiB (both ours),
load1 21.1, ② at epoch 10 step 820 running 1.92 s/step, C′ at its first steps 3.45 s/step (warm-up). **Steady state 03:46Z
(both running): ② 2.13 s/step, C′ 2.63 s/step, load1 23.6 → combined ≈ 0.85 steps/s vs ≈ 0.40–0.50 for ② alone (+70 %);
②'s ETA moved from ≈ 07:45 to ≈ 08:11 EDT, C′ ETA ≈ 19:51 EDT 10-18** (page snapshot pasted by Henry).

**10-18 14:55:44Z: `pb_Dp_s0` chain launched by Henry in tmux `fdpb_pb_Dp_s0`** (Henry: the H100 still had room — 21 GB used by
C′ + one eval): init `e2_Dp_comma_motion_from_e50_e40_target_encoder.pt`, 64 × 1, 20 epochs, act-ckpt on, late-window exports;
runs beside C′ (epoch 11) and the ② eval drain; the sequence's D′ item will skip (lock). CPU, not VRAM, is the bottleneck
(load ≈ 30 with one trainer + one eval), so per-step times of C′/D′ are expected to rise; the trade is total throughput.

**10-18 18:23–18:29Z (Henry: no VRAM self-cap, his jobs have priority):** `gpu_policy.sh` (whole card, no reservation, MAX_PB 3);
declarative queue `pb_queue.sh` (tmux `fdqueue`) launched **`pb_e50_s1`** (second e50 seed, for the pb seed variance) at 18:28Z
beside C′ and D′ (GPU 50 GB in use at launch); `phase6_pf_fillers.sh` (tmux `fdpf`, uv venv torch 2.8 / PL 2.5.6) resumed
`e1_Cp_e40_pf_s0` for epoch 40 at 18:24Z (2.0 it/s) → dev evals 38/39/40, then `e1b_e50_pf_s0` epochs 42–45 → evals 41–45.
Eval streams `eval_helper.sh` for C′ and D′ (epochs 16–20, LOAD_MAX 40) wait for the exports.

**10-19 04:05Z status:** load1 **57–59** with 3 pb trainers + the pf E1b trainer + 2 eval streams → pb step times 5.9–7.7 s
(C′ epoch 19, ETA 09:23Z; D′ epoch 8, ETA 10-20 07:52Z; e50 s1 epoch 6, ETA 10-20 13:34Z): the summed pb throughput
(≈ 0.47 steps/s) fell below the two-trainer value (0.85) — the host thrashes beyond ~2 trainers + 1 eval. Policy change
(`gpu_policy.sh` LOAD_MAX_LAUNCH=30): the queue and the pf fillers launch nothing new while load1 > 30; evals keep LOAD_MAX 40.
Preliminary ③-C′ (pb): ep16 **92.64** (vs e50 ep16 −0.23 [−0.62, +0.15]), ep17 **92.96** (vs e50 ep17 +0.15 [−0.21, +0.50]);
pf side complete: C′ E1 epochs 36–40 = 87.46 / 83.05 / 85.96 / 85.31 / 85.77 (late-5 mean 85.51, SD 1.6) vs E1b (e50) ep40
paired −0.74 [−2.09, +0.65] → no pf-side gain either.

Live progress page (`scripts/h100/progress_web.sh`, tmux `fdweb`, launched by Henry 03:29Z): `state/www/index.html`
regenerated every 60 s by `progress_page.py`, served by Python's http.server bound to the server's Tailscale address, port
8787 (read-only; no other hosts' addresses; other users' processes shown by name only); `boot.sh` relaunches it.

## Results (dev = navtrain official val, 18,157 scenes; PDMS; paired Δ = log-clustered bootstrap, 10,000 draws)

| eval tag | ckpt | PDMS | NC | DAC | EP | TTC | C | DDC | paired Δ vs official pb dev ref | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| pb_official_v1_dev | official v1 ckpt (sha8 b3e1d4ab) | **93.46** | 99.06 | 98.09 | 90.76 | 96.55 | 99.86 | 89.47 | — (reference) | dev reference for ②/③; navtest of the same ckpt = 93.69 (`../2026-10-17_h100_migration`); side cams from calibration; ledger `20261017-2020_pb_official_v1_dev_dev` |
| pb_e50_s0_ep5 | `pb_e50_s0/exports/epoch_5.ckpt` (sha8 8008b5a8) | 91.57 | 99.11 | 97.55 | 86.91 | 96.79 | 99.51 | 94.65 | −1.88 [−2.31, −1.48] | epoch 5 of 20 (learning-curve point); evaluated beside the training (23:53Z → 00:19Z, 26 min); online-scorer proxy for the same epoch 0.912; ledger `20261018-0019_pb_e50_s0_ep5_dev` |
| pb_e50_s0_ep10 | `pb_e50_s0/exports/epoch_10.ckpt` | 92.68 | 98.88 | 98.34 | 89.60 | 95.52 | 99.92 | 91.62 | −0.78 [−1.12, −0.45] | epoch 10 of 20 (learning-curve point); first eval of the drain (14:09Z → 14:36Z, 27 min, beside C′); proxy 0.920; ledger `20261018-1436_pb_e50_s0_ep10_dev` |
| pb_e50_s0_ep15 | `exports/epoch_15.ckpt` | 92.57 | | | 91.36 | | | | −0.88 [−1.29, −0.49] | learning-curve point; ledger `20261018-1509_pb_e50_s0_ep15_dev` |
| pb_e50_s0_ep16 | `exports/epoch_16.ckpt` | **92.88** | 98.40 | 98.18 | 90.63 | 95.48 | 99.76 | 88.86 | −0.58 [−0.96, −0.22] | late window (best late export); helper stream 17:23Z → 17:59Z; ledger `20261018-1759_pb_e50_s0_ep16_dev` |
| pb_e50_s0_ep17 | `exports/epoch_17.ckpt` | 92.81 | 98.33 | 97.85 | 91.58 | 94.76 | 99.87 | 88.44 | −0.65 [−1.01, −0.29] | late window; ledger `20261018-1723_pb_e50_s0_ep17_dev` |
| pb_e50_s0_ep18 | `exports/epoch_18.ckpt` | **92.86** | 98.39 | 97.98 | 91.41 | 94.88 | 99.82 | 88.76 | −0.60 [−1.02, −0.20] | late window (best late export); ledger `20261018-1647_pb_e50_s0_ep18_dev` |
| pb_e50_s0_ep19 | `exports/epoch_19.ckpt` | 92.42 | 98.12 | 97.57 | 91.74 | 94.24 | 99.65 | 87.87 | −1.04 [−1.50, −0.59] | late window; ledger `20261018-1610_pb_e50_s0_ep19_dev` |
| pb_e50_s0_ep20 | `exports/epoch_20.ckpt` | 92.57 | 98.07 | 97.86 | 91.39 | 94.52 | 99.88 | 87.86 | −0.88 [−1.31, −0.47] | late window (last epoch); ledger `20261018-1534_pb_e50_s0_ep20_dev` |
| **pb_e50_s0_best13** | `checkpoints/best_epoch=13.ckpt` (sha8 4dc39756; Lightning best `val/score_epoch` = 0.928 after epoch 14) | **93.34** | 98.82 | 98.43 | 90.25 | 96.40 | 99.87 | 90.13 | **−0.12 [−0.40, 0.17]** | the like-for-like counterpart of the released ckpt (both dev-selected); 18:12Z → 18:57Z beside 3 trainers; ledger `20261018-1857_pb_e50_s0_best13_dev` |
| pb_e50_s0_best13 **navtest** | same ckpt | **93.50** | 98.83 | 98.47 | 91.17 | 95.83 | 99.73 | 94.44 | official ckpt navtest 93.69 (−0.19) | navtest eval #4 of 8 (plan §5 item ②: frozen self-trained baseline, 1 eval; not used for selection); 19:58Z → 20:38Z; a pure reproduction already lands in the Drive-JEPA band of the public board (93.3–93.7); ledger `20261018-2038_pb_e50_s0_best13_navtest` |

### ② late window (epochs 16–20, final) — `late_window_pb_e50_s0.json`

- Late mean **92.71**, SD 0.20 (min 92.42 at ep19, max 92.88 at ep16); per-scene mean of the five exports paired against the official ckpt on dev: **−0.75 [−1.11, −0.40]** (18,157 scenes, 214 log clusters, 10,000 draws).
- Sub-score deltas of the late average vs the official ckpt: NC −0.79, DAC −0.21, **EP +0.59**, **TTC −1.77**, C −0.06 (DDC −1.11, not in v1 PDMS): our self-trained model drives more progressively and less cautiously than the released one.
- **G-B rule (late mean ≥ 92.96 = reference − 0.5): not met, margin −0.25** → per plan §6 "基线差 >0.5 → 先排查配方 / 缓存等价，不下编码器结论". Budget: 20 epochs, 26,580 optimizer steps, wall 18 h 12 min (4 h solo + 14 h shared), peak VRAM 41.1 GB (act-ckpt off), 1 seed.
- **Official checkpoint metadata (`scripts/h100/ckpt_meta.sh`, 10-18 18:10Z):** `epoch 12` (0-based → 13 epochs completed),
  `global_step 17277` = 13 × 1,329 → **1,329 optimizer steps per epoch = effective batch 64 on the 85,109 train tokens, identical
  to ours**; Adam param groups lr 1e-5 (backbone) / 1e-4, no weight decay; PyTorch Lightning 2.2.1; `ModelCheckpoint(monitor=
  'val/score_epoch', mode='max')` with **`best_model_score 0.9298`** — i.e. the released weights are the best-validation
  checkpoint (selected on the 214 val logs = our dev split), not the last epoch. Our run's online val score peaked at **0.928
  after epoch 14** (`best_epoch=13.ckpt`), within 0.002 of the official 0.9298 → the training recipe is reproduced on the
  metric the authors optimised; the formal like-for-like dev comparison is `pb_e50_s0_best13` (eval started 18:10Z).
- **G-B verdict (10-18 19:00Z): PASSED on the like-for-like comparison.** Our dev-selected checkpoint `best_epoch=13.ckpt`
  scores **93.34** vs the released (also dev-selected) checkpoint's 93.46: paired per-scene Δ **−0.12 [−0.40, +0.17]**
  (18,157 scenes, 214 log clusters, 10,000 draws) — the interval covers 0 and |Δ| is a quarter of the 0.5 tolerance;
  sub-scores NC −0.24, DAC +0.34, EP −0.51, TTC −0.15, C +0.01 (DDC +0.66). The unselected late-window mean (92.71,
  −0.75) is reported alongside as the honest no-selection number and remains the protocol for ③/④ comparisons; the
  per-epoch SD 0.20 and the best-vs-late gap of 0.6 quantify how much dev selection is worth here.
- **Confound found while investigating:** the upstream training config (`default_training.yaml`: `monitor: val/score_epoch`, `save_top_k: 1`, `filename: best_{epoch}`) keeps only the **best checkpoint by the online validation score on the 214 val logs = our dev split**. The released checkpoint is therefore (very likely) dev-selected, while our late-window numbers are unselected; selection on a noisy metric with SD ≈ 0.2–0.5 per export is worth a few tenths. Our run's like-for-like counterpart is `checkpoints/best_epoch=13.ckpt` (val/score_epoch peaked after epoch 14 at 0.928) → to be evaluated on dev (`pb_e50_s0_best13`). The ③ encoder comparisons are paired against our own e50 late window under identical (unselected) conditions and are unaffected by this confound.

### ③-C′ `pb_Cp_s0` (comma plain-JEPA encoder C′, everything else identical to ②) — late window 16–20, final (10-19 08:28Z)

| export | PDMS | NC | DAC | EP | TTC | C | DDC | paired Δ vs e50 same epoch | paired Δ vs official |
|---|---|---|---|---|---|---|---|---|---|
| ep16 | 92.64 | 98.35 | 97.87 | 91.38 | 94.66 | 99.68 | 88.09 | −0.23 [−0.62, +0.15] | −0.82 [−1.23, −0.42] |
| ep17 | 92.96 | 98.57 | 97.92 | 91.01 | 95.50 | 99.93 | 89.32 | +0.15 [−0.21, +0.50] | −0.50 [−0.87, −0.14] |
| ep18 | 92.46 | 98.24 | 97.76 | 90.76 | 94.84 | 99.90 | 87.96 | −0.40 [−0.78, −0.00] | −1.00 [−1.43, −0.60] |
| ep19 | 92.54 | 98.49 | 97.70 | 91.30 | 94.56 | 99.86 | 88.07 | +0.13 [−0.25, +0.51] | −0.92 [−1.31, −0.53] |
| ep20 | 92.88 | 98.60 | 97.87 | 90.65 | 95.75 | 99.92 | 88.89 | +0.31 [−0.08, +0.72] | −0.58 [−0.95, −0.21] |
| **late mean** | **92.70** (SD 0.22) | | | | | | | **−0.01 [−0.26, +0.24]** (5-export per-scene means, 214 log clusters, 10,000 draws; `late_window_pb_Cp_s0_vs_e50.json`) | −0.76 [−1.08, −0.44] |

- Sub-score deltas of the C′ late average vs the e50 late average: NC +0.19, TTC +0.29, EP −0.33, DAC −0.06, C +0.06 (DDC +0.11):
  C′ is marginally more cautious and makes marginally less progress; everything within noise.
- **Verdict (plan §2 ③ rule: ≥ +0.3 with CI lower bound > 0): NO signal** — the 110 h comma plain-JEPA continued pretraining
  (100k steps, batch 8, e50 init) leaves the full perception-based model statistically indistinguishable from the e50 baseline.
  Consistent with the pf side (C′ E1 late-5 mean 85.51 vs E1b ≈ 86.8; paired vs E1b ep40 −0.74 [−2.09, +0.65]).
- Training: 20 epochs in 29 h 00 min wall (03:27Z → 08:28Z incl. the eval drain; beside 1–3 other trainers), act-ckpt on (12.6 GB),
  Lightning best val/score_epoch checkpoint = `best_epoch=11.ckpt` (12th epoch).
- **Dev-selected comparison (10-19 12:17Z):** `pb_Cp_s0_best11` = **92.99** (NC 98.97, DAC 98.35, EP 89.51, TTC 96.35, C 99.91, DDC
  90.49) vs e50 `best13` 93.34: paired **−0.35 [−0.69, −0.03]** (EP −0.74, DAC −0.08); vs the official ckpt −0.47 [−0.79, −0.17].
  So under selection C′ is marginally *behind* e50 (the interval just excludes 0), driven by lower progress; together with the
  unselected parity (−0.01) the overall read is C′ ≤ e50. Ledger `20261019-1217_pb_Cp_s0_best11_dev`;
  `report_pb_e50_s0_best13_vs_pb_Cp_s0_best11.json`.
- Still open for ③: D′ (comma + CAN motion objective; ETA 10-19 ≈ 20:30Z + evals) and the e50 second seed (ETA ≈ 23:00Z)
  for the pb seed variance.
- **pf fillers complete (10-19 13:48Z, `phase6_pf_fillers.sh`, uv venv torch 2.8 / PL 2.5.6, cross-hardware continuation
  disclosed in the ledger notes):** e50 E1b s0 epochs 41–45 (constant-lr extension) = 80.95 / 87.94 / 88.05 / 87.20 / 85.89
  (mean 86.01, SD 2.95; 87.27 without the ep41 dip — the C-29 "rare 5-point dip" again); C′ 36–40 mean 85.51 (SD 1.59); D′ 36–40
  mean 85.68 (SD 1.56). Under the perception-free protocol both comma encoders sit ≈ 1 point below e50; same direction as pb.

### ③-D′ `pb_Dp_s0` (comma + CAN motion-objective encoder D′, everything else identical to ②) — final (10-19 21:38Z)

| export | PDMS | NC | DAC | EP | TTC | C | DDC | paired Δ vs e50 same epoch | paired Δ vs official |
|---|---|---|---|---|---|---|---|---|---|
| ep16 | 92.35 | 98.38 | 97.92 | 90.26 | 94.99 | 99.68 | 88.62 | −0.53 [−0.86, −0.20] | −1.11 [−1.48, −0.76] |
| ep17 | 92.59 | 98.43 | 97.64 | 90.67 | 95.78 | 99.12 | 88.34 | −0.22 [−0.58, +0.13] | −0.86 [−1.28, −0.46] |
| ep18 | 92.74 | 98.36 | 98.03 | 90.46 | 95.49 | 99.85 | 88.42 | −0.12 [−0.51, +0.27] | −0.72 [−1.13, −0.33] |
| ep19 | 91.60 | 97.94 | 97.27 | 90.82 | 93.61 | 99.80 | 88.18 | −0.82 [−1.29, −0.35] | −1.86 [−2.40, −1.35] |
| ep20 | 92.24 | 98.29 | 97.46 | 90.51 | 95.04 | 99.82 | 87.45 | −0.33 [−0.81, +0.14] | −1.21 [−1.70, −0.76] |
| **late mean** | **92.30** (SD 0.44) | | | | | | | **−0.40 [−0.69, −0.12]** (5-export per-scene means; `late_window_pb_Dp_s0_vs_e50.json`) | −1.15 [−1.53, −0.80] |
| best10 (dev-selected, val/score peak after epoch 11) | 92.95 | | | | | | | vs e50 best13 93.34: −0.39 [−0.74, −0.04] | −0.51 [−0.80, −0.22] |
| soup(16–20) | 92.91 | | | | | | | vs e50 soup 93.10: −0.19 [−0.58, +0.18]; **vs its own late average +0.61 [+0.45, +0.76]** | −0.55 [−0.96, −0.16] |

- Sub-score deltas of the D′ late average vs e50: **EP −0.81**, DAC −0.23, C −0.15, TTC +0.21, NC +0.02: the CAN-objective encoder makes the
  planner noticeably less progressive without a compensating safety gain.
- **Verdict: negative** — D′ is significantly *below* e50 in the unselected window (CI entirely negative) and under selection; the
  ③ rule (≥ +0.3, lower bound > 0) is missed by a wide margin. Together with C′ (parity) and the pf side (both ≈ −1), the 110 h comma
  subset with either objective at ~1 % of Drive-JEPA's pretraining compute does not improve the full model. ③ is closed; ④ goes
  to planner recipe / zero-cost candidates (see below), and the comma pretraining line is reported as a negative result.
- Training: 20 epochs 14:55Z (10-18) → 20:39Z (10-19) incl. the eval drain, act-ckpt on; `best_epoch=10.ckpt`.
- **Soup replication:** on D′ the late-window average gains +0.61 [+0.45, +0.76] over its late mean (e50: +0.39) → the ④ finding
  holds on a second, independent run (larger here because D′'s window is noisier, SD 0.44).

### e50 second seed `pb_e50_s1` (seed variance of the pb protocol) — final (10-20 02:40Z)

| export | PDMS | NC | DAC | EP | TTC | C | DDC | paired Δ vs s0 same epoch |
|---|---|---|---|---|---|---|---|---|
| ep16 | 92.80 | 98.55 | 97.98 | 91.09 | 95.02 | 99.78 | 89.80 | −0.08 [−0.39, +0.23] |
| ep17 | 93.16 | 98.69 | 98.01 | 91.37 | 95.49 | 99.72 | 89.83 | +0.35 [+0.03, +0.67] |
| ep18 | 92.80 | 98.47 | 98.13 | 90.34 | 95.55 | 99.79 | 88.96 | −0.06 [−0.40, +0.29] |
| ep19 | 92.75 | 98.40 | 98.00 | 91.14 | 94.96 | 99.55 | 87.63 | +0.34 [+0.00, +0.67] |
| ep20 | 92.31 | 97.92 | 97.82 | 90.77 | 94.62 | 99.88 | 87.41 | −0.26 [−0.61, +0.08] |
| **late mean** | **92.76** (SD 0.30) | | | | | | | **+0.06 [−0.15, +0.26]** vs s0 92.71 (`late_window_pb_e50_s1_vs_s0.json`); vs official −0.69 [−1.02, −0.38] |
| best12 (dev-selected) | 93.20 | | | | | | | vs s0 best13 93.34: −0.14 [−0.46, +0.17]; vs official −0.26 [−0.55, +0.03] |
| soup(16–20) | **93.19** | | | | | | | vs its late average **+0.42 [+0.29, +0.56]**; vs s0 soup 93.10: +0.09 [−0.22, +0.39]; vs official −0.27 [−0.61, +0.05] |

- **Seed effect ≈ 0** (+0.06 with a ±0.2 interval): the late-window protocol resolves differences of ≈ 0.2 between runs; single
  exports still swing ±0.3–0.4 between seeds (ep17/19 +0.35/+0.34, ep20 −0.26) — why single-export comparisons are not used.
- **Soup replicated a third time** (+0.39 s0, +0.61 D′, +0.42 s1). Soups of the two seeds: 93.10 / 93.19 (official 93.46).

### ③ decision table (dev; unselected late mean 16–20 / dev-selected best / soup; paired = per-scene, 214 log clusters, 10,000 draws)

| run | late mean (SD) | Δ late vs e50 s0 | best ckpt | soup | verdict |
|---|---|---|---|---|---|
| e50 s0 (②) | 92.71 (0.20) | — | 93.34 | 93.10 | baseline; navtest best13 93.50 |
| e50 s1 | 92.76 (0.30) | +0.06 [−0.15, +0.26] | 93.20 | 93.19 | seed effect ≈ 0 |
| C′ (comma plain JEPA) | 92.70 (0.22) | −0.01 [−0.26, +0.24] | 92.99 | 93.19 | no signal |
| D′ (comma + CAN motion) | 92.30 (0.44) | −0.40 [−0.69, −0.12] | 92.95 | 92.91 | worse |
| official ckpt | — | — | 93.46 (dev-selected) | — | navtest 93.69 |

**③ closed (10-20, three days ahead of the 10-23 decision point): neither comma encoder improves the full model; ④ proceeds on the
planner recipe (cosine lr, `pb_e50_cos_s0` running, `pb_e50_cos_s1` queued) and zero-cost candidates (soup confirmed ×3).**

### ④(b) cosine lr decay (`pb_e50_cos_s0` / `pb_e50_cos_s1`: upstream recipe + per-step cosine decay to 0.1×; final 10-21 09:45Z)

| run | late 16–20 exports | late mean (SD) | paired Δ late vs constant-lr run of the same seed | best ckpt | soup | soup − late |
|---|---|---|---|---|---|---|
| cos s0 | 93.34 / 93.06 / 93.18 / 92.90 / 93.13 | **93.12** (0.16) | **+0.42 [+0.16, +0.66]** vs s0 92.71 | best11 **93.45** (vs s0 best13 +0.10 [−0.15, +0.37]; vs official −0.01 [−0.26, +0.23]) | 93.20 (vs s0 soup +0.10 [−0.22, +0.43]) | +0.08 [−0.02, +0.18] |
| cos s1 | 93.02 / 92.86 / 92.99 / 93.05 / 92.90 | **92.96** (0.08) | +0.20 [−0.09, +0.48] vs s1 92.76 | best12 93.00 (vs s1 best12 −0.20 [−0.52, +0.11]) | 93.08 (vs s1 soup −0.11 [−0.40, +0.20]) | +0.12 [+0.02, +0.21] |
| **pooled (2 seeds)** | | 93.04 vs 92.74 | **+0.31 [+0.11, +0.51]** | — | soups 93.14 vs 93.14: **−0.00 [−0.23, +0.25]** | |

- Sub-score deltas (late average, cosine − constant): **TTC +0.78 / +0.57, NC +0.40 / +0.36, DDC +0.44 / +0.36**, EP −0.02 / −0.03, DAC +0.08 / −0.14:
  the decayed end of training is *safer at equal progress*, and the late window is far steadier (SD 0.16 / 0.08 vs 0.20 / 0.30).
- **Reading:** cosine decay reliably improves the quality of the *unselected final epochs* (+0.31 pooled, CI > 0; seed 1 alone +0.20 does not
  reach the +0.3 bar), but it does **not raise the ceiling**: the soup and best-checkpoint forms end at the same level as the constant-lr run
  (soups 93.14 = 93.14; bests 93.45 / 93.00 vs 93.34 / 93.20). Cosine ≈ a built-in soup: the soup adds only +0.08 / +0.12 on top of it.
  All our candidate forms sit −0.3 below the released checkpoint's dev score (2-seed soups −0.32 [−0.64, +0.00] / [−0.58, −0.06]); cos_s0
  best11 (93.45) equals it, but that is one dev-selected checkpoint of one seed.
- Consequence for ⑤: the recipe knobs tried so far (encoder, lr schedule, soup) converge to dev ≈ 93.1–93.4; a higher ceiling needs a
  different lever (candidate selection / ensembling at inference, or the pretraining scale test Cp4/A4 now running).
- Ledger rows `*_pb_e50_cos_s{0,1}_*`; `late_window_pb_e50_cos_s{0,1}_vs_pb_e50_s{0,1}.json`.

### ④ first finding — weight averaging of the late exports ("soup", 10-19 14:35Z; `eval/soup_exports.py`, `scripts/h100/soup_eval.sh`)

| candidate | dev PDMS | NC | DAC | EP | TTC | C | DDC | paired Δ (per scene, 214 log clusters, 10,000 draws) |
|---|---|---|---|---|---|---|---|---|
| e50 soup(16–20) = uniform average of the five fp16 exports | **93.10** | 98.22 | 98.12 | 91.87 | 94.84 | 99.85 | 87.88 | vs e50 late average **+0.39 [+0.23, +0.55]**; vs e50 best13 −0.24 [−0.58, +0.08]; vs official −0.36 [−0.73, +0.02] |
| C′ soup(16–20) | 93.19 | 98.54 | 98.05 | 91.61 | 95.30 | 99.93 | 88.38 | vs e50 soup +0.09 [−0.23, +0.40] (encoder parity holds for soups too) |

- A deterministic transform of the already-trained run, **no dev selection involved** → +0.39 over the unselected late window
  with a CI well clear of 0 (the ④ bar is +0.3 with lower bound > 0). It recovers most of what checkpoint selection buys (best13
  93.34) without looking at dev, which makes it the natural submission candidate form (⑤): "soup of the final run's late window".
- Gains come from EP (+0.5 vs the late exports) with DAC held; NC is a little lower (98.22 vs ≈ 98.3) — same trade-off direction as
  training itself. To be confirmed on the second seed (`pb_e50_s1` soup vs its late average) and on D′ when they finish.
- Ledger `20261019-1412_pb_e50_s0_soup16_20_dev`, `20261019-1434_pb_Cp_s0_soup16_20_dev`; soup checkpoints kept at
  `exp/<run>/exports/soup_16_20.ckpt` (same fp16 export format, loadable by `eval_pb.sh`).

Ingestion: `eval/pb_ingest.sh <tag> <init_label> <encoder_pt|e50>` (copies the report here, writes `report_<ref>_vs_<tag>.{md,json}`, appends the ledger row).

### ④(c) inference-time ensemble of pb checkpoints (design 2026-10-21 23:30 EDT; **dev result 2026-10-22 02:45 EDT: closed negative** — `navsim_ext/pb_ensemble_agent.py`, `navsim_ext/configs/agent/drive_jepa_pb_ensemble_agent.yaml`, `scripts/h100/eval_pb_ensemble.sh`, `scripts/h100/ens_chain_1021.sh`)
- Motivation (per-scene CSVs already on disk, 10-21): oracle-of-two over {cos_s0 soup, cos_s1 soup} ≈ **95.4** vs 93.1–93.2 for the singles;
  ~55–61 % of scenes differ between two models, ~500 scenes with |Δ| ≥ 0.5 → real headroom for a selection rule that needs no training.
- Rule: every member is a complete `DriveJEPAAgent` with its own checkpoint; per scene each member outputs its trajectory and the predicted
  PDM score of that trajectory (max over its own proposal scores, sigmoid in [0, 1]). `argmax_conf` = the member with the highest
  predicted score wins; `mean_if_close` = xy averaged (circular-mean heading) when all members agree within 0.5 m at every pose, else
  argmax_conf; `first` = member 0 only (plumbing check). One trajectory per scene, no oracle, no dev selection, vendored tree untouched.
  Cross-scoring another member's proposals is not possible (the scorer reads per-proposal query features of its own model) → selection
  happens at trajectory level. Hydra: `agent=drive_jepa_pb_ensemble_agent` via `hydra.searchpath=[pkg://…common,file://…/navsim_ext/configs]`
  (composition + instantiation verified locally with stubbed members; 5 unit tests `tests/test_pb_ensemble_agent.py`).
- Evaluations queued in `ens_chain_1021.sh` (tmux `fdens`, sequential): smoke `ens_smoke_cos01_first` (16 scenes, mode first — must reproduce
  the cos_s0 soup per-scene scores exactly; checked locally after pull) → `pb_ens_cos01_argmax`, `pb_ens_cos01_meanclose`
  ({cos_s0 soup, cos_s1 soup}) → `pb_ens_cos01e50_argmax` (+ e50_s0 soup). Reports `eval_pb/pb_ens_*` are ingested as `ensemble`
  rows (BEST=3; paired vs the official reference by `pull.sh`); paired vs the best single member is computed by hand
  (`eval/ens_pick_analysis.py` / `diagnostics/d5_error_buckets.py`). Cost: inference only, each eval worker loads M members (≈ 3 GB × M GPU),
  ≈ 40–60 min per dev eval on the shared GPU. Readout: ensemble − best member on dev, paired CI; a selection rule that captures a
  fraction of the oracle gap is a ⑤ candidate form (members must then all be public checkpoints).
- **Result (2026-10-22 02:45 EDT; chain `fdens` ran 04:06–06:40Z on the shared GPU next to the Cp4 pb trainer and the A4 pretraining; smoke
  `first` reproduced the cos_s0 soup per-scene scores on all 16 scenes exactly — 7 score columns, max |Δ| = 0 → plumbing correct): closed negative.**

  | tag | members | mode | dev PDMS | paired vs best member (cos_s0 soup 93.20) | paired vs official 93.46 | wall |
  |---|---|---|---|---|---|---|
  | `pb_ens_cos01_argmax` | cos_s0 soup16-20 + cos_s1 soup16-20 (93.08) | argmax_conf | **93.29** | +0.08 [−0.09, +0.26] (EP +0.19, DAC +0.03, NC −0.03, TTC −0.04); vs cos_s1 soup +0.21 [−0.02, +0.45]; vs e50_s0 soup +0.19 [−0.15, +0.53] | −0.17 [−0.48, +0.13] | 45 min |
  | `pb_ens_cos01_meanclose` | same | mean_if_close (0.5 m) | **93.29** | +0.09 [−0.09, +0.26]; vs argmax 0.00 [−0.02, +0.02] (736 scenes differ, 386 better / 351 worse) | −0.17 [−0.47, +0.13] | 46 min |
  | `pb_ens_cos01e50_argmax` | + e50_s0 soup16-20 (constant lr, 93.10) | argmax_conf | **93.08** | −0.12 [−0.36, +0.11] (NC −0.40, TTC −0.58, EP +0.34); vs the 2-member ensemble −0.20 [−0.41, +0.00]; vs e50_s0 soup −0.01 [−0.24, +0.21] | −0.37 [−0.70, −0.05] | 62 min |

  Readout (pre-stated criterion: ensemble ≥ best member + 0.3 with CI lower bound > 0): **not met**. The two cosine members score differently on
  10,596 / 18,157 scenes (58.4 %); `argmax_conf` picks the better one in 57.6 % of them (6,106 vs 4,490; chance 50 %) — a real but tiny signal:
  +0.08 (+0.15 above a random pick, 93.14) out of an oracle-of-2 gap of +2.23 (95.43). Averaging close trajectories changes nothing. Adding the
  constant-lr member makes it worse: its score head out-voted the cosine members on 3,258 scenes where it scored 91.7 on average vs 94.8 for the
  best cosine member on the same scenes → the predicted PDM scores of independently trained runs are **not calibrated against each other**, and
  cross-run argmax picks the more confident run, not the better trajectory (the more aggressive constant-lr trajectories win the vote: NC/TTC drop).
  Oracle-of-3 would be 96.28; where the three members differ (61.3 % of scenes) the vote hits a best-scoring member in 45.7 % and the worst in 28.1 %.
  Conclusion: the models' own confidence is not a usable cross-model selector; ④(c) is closed, no ensemble in ⑤ (2–3× inference cost for ≤ +0.1),
  the ⑤ candidate form stays a single checkpoint (cosine run late soup / best). The oracle gap between seeds (2–3 PDMS) remains the largest untapped
  headroom and would need a separately trained candidate selector (trainable on navtrain-train only, never on dev) — a research item outside the
  leaderboard main line. Cost on the server: 4 eval workers × 4.4 GB; no navtest run (budget still 4/8).
  Reports (this dir): `report_pb_e50_cos_s0_soup16_20_vs_pb_ens_*`, `report_pb_e50_cos_s1_soup16_20_vs_pb_ens_cos01_argmax`,
  `report_pb_e50_s0_soup16_20_vs_pb_ens_*`, `report_pb_ens_cos01_argmax_vs_pb_ens_cos01_meanclose`, `report_pb_ens_cos01_argmax_vs_pb_ens_cos01e50_argmax`,
  `report_pb_official_v1_dev_vs_pb_ens_*`, `pb_ens_*_report.json`; ledger rows `*_pb_ens_*_dev` (encoder_init `ensemble`, objective `ensemble`).
