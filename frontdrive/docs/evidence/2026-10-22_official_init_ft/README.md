# ⑤ Continued training from the released pb weights + final models on all navtrain logs (design 2026-10-22 03:10 EDT)

**Decision (Henry, 2026-10-22 03:10 EDT, after ④(c) closed negative):** "用全部 navtrain 训练最终模型、从官方权重继续训练" — both levers
of the plan v4.3 ⑤ discussion: (1) the final submission candidates are trained on **all 1,192 navtrain logs** (train + the 214 val logs
that so far served only as dev), and (2) training **continues from the released Drive-JEPA pb v1 checkpoint** (public weights, HF
`LinhanWang/Drive-JEPA/drive_jepa_perception_based_agent_vitl.ckpt`, sha256 `b3e1d4ab…`, our navtest 93.69 / dev 93.46) instead of
starting from the e50 encoder. Everything else (compact pb cache, side cams from calibration, effective batch 64, act-ckpt on, online
scorer) is the ② recipe. Rules: public data (navtrain) + public weights only; no navtest data touches training or selection; the
leaderboard needs the code + weights published (plan §0).

## Why this and not more of ④
All candidate forms of ②/④ stopped at dev 93.1–93.45 (official dev-selected 93.46): soup +0.39 over the late mean, cosine lr +0.31
on the unselected window but no higher ceiling, comma pretraining ≤ e50, confidence ensembles +0.08. The two remaining compliant
levers change the protocol (no dev readout for a model trained on all logs; the released weights as init), so they needed Henry's call.
Expected sizes: +21 % training data (85,109 → 103,288 tokens) and "the cosine tail + soup" applied to the released weights (our cosine
runs gained +0.3–0.4 exactly there); gains of +0.1–0.4 dev-equivalent are plausible, nothing is guaranteed.

## Implementation (commit of 2026-10-22; all knobs are environment variables read by `scripts/h100/train_pb.sh`)
| knob | effect | where |
|---|---|---|
| `PB_INIT_CKPT=<agent .ckpt>` | load **all** agent tensors from a full checkpoint before `trainer.fit` (keys `agent._pad_model.*` → `_pad_model.*`, strict equality enforced, mismatch raises) | `run_training.py` (vendored patch, `FRONTDRIVE_INIT_CKPT`) |
| `PB_TRAIN_LOGS=all` | train scene filter = navtrain ∩ (train_logs ∪ val_logs) = every cache token; val set unchanged (now in-sample, monitoring only); cosine step count uses 103,288 samples | `run_training.py` (`FRONTDRIVE_TRAIN_LOGS`), `train_pb.sh` |
| `PB_LR=3e-5` | Adam peak lr (`agent.lr`; backbone 0.1×) | `train_pb.sh` → Hydra override |
| `PB_LR_SCHED=cosine PB_LR_MIN_MULT=0.1 PB_LR_WARMUP=200` | ④(b) per-step schedule with 200 linear warm-up steps | existing |
| `PB_EPOCHS=5 PB_EXPORT_EVERY=1 PB_EXPORT_FROM=1` | 5 epochs, fp16 export every epoch, soup window 1–5 | existing chain knobs |
| `PB_SKIP_DEV_EVAL=1` | chain builds `exports/soup_1_5.ckpt` but dispatches **no** dev evals (dev is in the training set) | `phase5_pb_pipeline.sh` |
Also: the chain's reboot marker now carries every PB_* knob (before, a reboot would have relaunched the cosine runs without their
schedule); `pb_queue.sh` treats `exports/epoch_<PB_EPOCHS>.ckpt` or `.phase5.done` as finished (was hard-coded `epoch_20`; the running
`fdqueue` must be restarted once to pick this up); `gpu_policy.sh` `MAX_PB=2` while the A4 pretraining arm runs (2 pb + arm = load ≈ 28,
a third pb would thrash — observed 2026-10-19; superseded at 09:15 EDT by 3, see Timing). `pull.sh`/`eval/pb_ingest.sh`: tag prefix `pb_off_ft_s<N>_*` → ledger `encoder_init
offinit`, objective `official_ft`, sha from the download list; `pb_off_ft_all_*` is never ingested (in-sample dev).
Coverage prerequisite verified from the ② launch log: the token coverage check (train_metric_cache + anchors index ⊇ all 103,288
cache tokens) passed on 2026-10-17 without move-aside, so the val logs need no extra preprocessing for training.

## Jobs (state/pb_queue.txt, launched by `fdqueue` when a pb slot is free; order = priority)
1. **Probe `pb_off_ft_s0`** — released ckpt → 978 train logs, 5 epochs (6,645 steps), 3e-5 cosine → 3e-6, warm-up 200; dev evals of
   `ep1..ep5`, `best<N>`, `soup1_5` (ingested as `offinit`, paired vs `pb_official_v1_dev`). Dev-verifiable because dev is held out.
2. **Finals `pb_off_ft_all_s0/s1/s2`** — same recipe on all 1,192 logs (8,065 steps each), no dev evals; product = `exports/soup_1_5.ckpt`
   per seed → one navtest eval each (`eval_pb.sh … navtest`, `NAVTEST_PURPOSE=final_candidate_seed<N>`; budget 4 → 7 of 8) → multi-seed
   leaderboard submission (the leaderboard shows max/mean/std; no seed selection needed).
Timing: Cp4 ends 10-23 ≈ 02:00 EDT (the 03:30 EDT message mis-read the progress page as 10-22 06:11), so with `MAX_PB=2` the probe
would have waited ≈ 20 h behind Cp4 / cos_s2. `MAX_PB` was therefore set back to 3 at 09:15 EDT 10-22 and the probe launched beside Cp4,
cos_s2 and the A4 arm (expect ≈ 6–7 s/step for every trainer, load ≈ 35–45; the queue line of cos_s2 was moved below the ⑤ jobs so that it
resumes last if it is ever stopped). Probe ≈ 13–15 h → dev readout ≈ 10-23 00:00–03:00 EDT; the finals take the slots as Cp4
(10-23 ≈ 05:00 after its eval drain) and the probe finish → all three by ≈ 10-24/25.

## Readout rules (pre-stated)
- Probe vs official dev reference (93.46, itself dev-selected → biased upward by ≈ 0.1–0.25 vs an unselected export): `soup1_5` or the
  late exports **≥ +0.2 with CI lower bound > 0** → continued training helps → keep the finals queued. Between −0.2 and +0.2 → neutral;
  the finals still run (the +21 % data is a separate lever) but the submission decision waits for navtest. **≤ −0.2 with CI upper bound < 0**
  → continued training hurts → delete the three final lines from the queue (rsync) and fall back to e50-cosine on all logs.
- Finals: no dev number exists; the three navtest values are reported as they come (mean/std), none is selected on navtest. Comparison
  anchors: official navtest 93.69, our best13 navtest 93.50.
- Caveat for the paper narrative: the model is "released weights + 5 epochs of our recipe on more data"; the contribution is the recipe
  (cosine tail, soup, all-log training) and the open, reproducible pipeline, not a new encoder.

## Files
`scripts/h100/train_pb.sh`, `scripts/h100/phase5_pb_pipeline.sh`, `scripts/h100/pb_queue.sh`, `scripts/h100/gpu_policy.sh`,
`third_party_patches/drive_jepa_548bb82.patch` (run_training.py hunks), `scripts/h100/pull.sh`, `eval/pb_ingest.sh`; queue lines in
`state/pb_queue.txt` (server + local copy). Results will be appended below as they land.

## Log
- 2026-10-22 09:29:44 EDT: queue launched `pb_off_ft_s0` (after `gpu_policy.sh` tunables were made re-readable — `X=${X:-d}` is not
  updated by a re-source; now `X=${X_OVERRIDE:-d}`). First trainer start died in 30 s (rc=1): the morning edit of `train_pb.sh`'s header
  log line had put the new fields into the `tee` file name and `pipefail` aborted the wrapper; fixed 09:31:56, the chain's automatic retry
  at 09:32:15 trained normally. Start-of-training loss 1.33–1.40 (step 17–27) = the converged level of the from-scratch runs → the released
  weights are in. Host with 3 pb trainers + the A4 arm: load1 41, 7.7 s/step per trainer (summed throughput −16 % vs 2 + arm), eval
  dispatch blocked at LOAD_MAX 40 → `MAX_PB` back to 2 at 09:36 (live). Probe ETA 10-22 23:57 EDT (earlier if cos_s2 is paused).
- 2026-10-22 13:42 EDT: probe **ep1 dev 93.31**, paired vs official 93.46 **−0.15 [−0.42, +0.12]** (online val proxy 0.927 vs the official's
  0.9298); first epoch = warm-up to 3e-5 then the start of the cosine decay, so the model sits at the official level after the lr jolt —
  no conclusion yet (the pre-stated rule applies to the late exports / soup). Host while the eval ran beside 3 trainers + the A4 arm:
  load1 41 / 5-min 48 / 15-min 60, trainers at 8.3–10 s/step (thrash) → cos_s2 pause recommended to Henry again.
- 2026-10-22 ≈16:05 EDT: Henry paused `pb_e50_cos_s2` (old recipe, fallback seed; was in epoch 6/20, resumes from last.ckpt = end of
  epoch 5 when the queue reaches its line at the bottom). Host before the pause: load1 63 / 15-min 71 with 3 trainers + arm + the ep2 dev
  eval, probe at 11.1 s/step, Cp4 7.7 s/step. Now 2 trainers (Cp4, probe) + arm + eval; `MAX_PB=2` keeps it that way until Cp4 finishes.
- 2026-10-22 22:30 EDT: probe exports ep2 **93.19** (−0.27 [−0.57, +0.03]), ep3 **93.18** (−0.27 [−0.56, +0.01]), ep4 **93.30**
  (−0.16 [−0.45, +0.12]) vs the official 93.46; all four unselected exports sit 0.15–0.27 below the dev-selected reference (whose selection
  bias is ≈ +0.1–0.25), consistently more aggressive (EP +0.9–1.1, NC −0.4, TTC −0.9–1.4, DAC −0.2). Online val proxy flat at 0.927.
  → so far the "neutral" zone of the pre-stated rule; soup_1_5 (after ep5, ≈ 01:00–02:00 EDT 10-23) decides. Host after the cos_s2 pause:
  load ≈ 28–32, probe 6.3 s/step, Cp4 5.6 s/step (ETA 10-23 10:02 EDT).
- 2026-10-23 00:38 EDT: **probe complete.** ep5 **93.29** (−0.17 [−0.42, +0.07]), Lightning best (epoch 4) **93.30** (−0.16 [−0.45, +0.12]),
  **soup_1_5 93.22 (−0.24 [−0.51, +0.03])** vs the official 93.46. All seven numbers lie 0.15–0.27 below the dev-selected reference with CIs
  touching 0; the soup does not help here (the five exports start from one point and stay within 0.13 of each other). Like-for-like
  (dev-selected best3 vs dev-selected official): −0.16 [−0.45, +0.12]. **Readout by the pre-stated rule: neutral zone** (not ≥ +0.2 with
  CI > 0, not ≤ −0.2 with CI < 0) → continued training at 3e-5 on the same data neither helps nor measurably hurts; the finals keep running
  (the +21 % data is a separate lever) and the submission decision waits for navtest. Observation: every export shows the same shift as all
  our from-scratch runs vs the official weights — EP +0.9–1.1, NC −0.4, TTC −0.9–1.4, DAC −0.2 — and it appears within the first epoch of
  continued training from the official weights, so it is produced by our training signal (online scorer targets / loss), not by the init.
  Candidate zero-training follow-up (plan ④(c) "评分/选择"): re-weight the proposal selection toward the predicted safety sub-scores and tune
  1–2 scalars on dev (design pending; see the ⑥ note below once written).
- 2026-10-23 00:40 EDT: queue launched `pb_off_ft_all_s0` (params: init_ckpt official, train_logs=all, lr 3e-5 cosine, warm-up 200, 5 epochs,
  exports every epoch, no dev evals).
- 2026-10-23 00:45 EDT: **bug — `pb_off_ft_all_s0` started on the train split only** (progress page: 1,329 steps/epoch instead of 1,613; no
  "FRONTDRIVE_TRAIN_LOGS=all" line in its Hydra log). Cause: the all-logs switch had been patched only into `build_datasets` (SceneLoader
  path), while our runs use `use_cache_without_dataset=True`, whose `CacheOnlyDataset` takes `log_names=cfg.train_logs` directly. Fix: the
  cache-only branch now builds the train set from train ∪ val logs when `FRONTDRIVE_TRAIN_LOGS=all`, logs the sample count, and raises if
  the all-logs set has < 100,000 samples (guard against a silent repeat). Pushed to the server 00:48 EDT; the running wrong-split chain must
  be killed and its run dir moved aside (card to Henry) — the queue then relaunches `pb_off_ft_all_s0` on the fixed code. Until then the
  run is a useless replicate of the probe (same seed, same split).
- 2026-10-23 01:50 EDT: Henry killed the wrong-split chain and moved its dir to `pb_off_ft_all_s0_wrongsplit_1023`; the queue relaunched
  `pb_off_ft_all_s0` at 05:50:15Z on the fixed code. Hydra log: "initialised all 495 agent tensors from …vitl.ckpt", "FRONTDRIVE_TRAIN_LOGS=all
  -> cache-only training set from 14561 logs" (the devkit list; ∩ cache = the 1,192 navtrain logs), **"train dataset: 103288 samples"** →
  the all-logs final run is now genuinely on train + val (1,613 steps/epoch, 8,065 cosine steps).

## navtest results (⑤ confirmation, 2026-10-23 23:46Z – 2026-10-24 01:15Z; chain `scripts/h100/navtest_finals_1023.sh`, budget 4 → 7 of 8)
| seed | candidate | navtest PDMS | vs official 93.69 (paired, scene-level bootstrap, 12,146 scenes) | vs our best13 93.50 | sub-score Δ vs official |
|---|---|---|---|---|---|
| s0 | `pb_off_ft_all_s0/exports/soup_1_5.ckpt` (sha8 ac0331f9) | **93.86** | +0.17 [−0.08, +0.43] | +0.36 [+0.09, +0.64] | NC −0.40, DAC +0.32, EP +1.04, TTC −0.86, DDC −1.07 |
| s1 | `pb_off_ft_all_s1/exports/soup_1_5.ckpt` (sha8 c7cb6ca5) | **93.78** | +0.09 [−0.16, +0.34] | +0.28 [+0.01, +0.56] | NC −0.44, DAC +0.38, EP +1.17, TTC −1.24, DDC −1.07 |
| s2 | `pb_off_ft_all_s2/exports/soup_1_5.ckpt` (sha8 fc0ca060) | **93.94** | +0.25 [−0.01, +0.51] | +0.44 [+0.17, +0.72] | NC −0.39, DAC +0.40, EP +0.98, TTC −0.76, DDC −0.91 |
| 3-seed | mean **93.86**, SD 0.08 | | +0.17 [−0.07, +0.41] (per-scene mean of the three) | | |
Reading: every seed beats the self-trained frozen baseline (plan §5 ②, 93.50) with CI > 0 and every seed is above the released
checkpoint (93.69); the gain over the released weights, +0.17 on average, is not individually significant at 12,146 scenes but has the
same sign in all three seeds and the same signature as on dev (more progress, slightly less NC/TTC/DDC). The CIs here are scene-level
(no log attribute file for navtest on this machine), so they are somewhat narrower than the log-clustered dev CIs. **G-D criterion
"navtest ≥ self-trained baseline" is met; the ⑤ candidate recipe is confirmed.** Submission form: all three seeds (multi-seed entry),
none selected on navtest. Budget state: 7 of 8 used (the last slot is reserved for a dev-validated selection rule, see
`../2026-10-23_selection_rules/`). Per-scene CSVs: `benchmarks/navsim/exp/eval_pb/pb_off_ft_all_s*_soup1_5_navtest/`.
