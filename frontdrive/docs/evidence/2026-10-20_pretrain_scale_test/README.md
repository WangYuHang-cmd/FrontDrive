# Pretraining scale test on the H100 (plan v4.3 ④, after ③ came out negative) — evidence log

Question: ③ showed that the 110 h comma subset with either objective at ~1 % of Drive-JEPA's continued-pretraining compute
(100k steps × batch 8 = 0.8 M clip draws) does not improve the full perception-based model (C′ parity, D′ worse,
`../2026-10-17_pb_baseline/`). Is that a data result or a compute floor? Two arms at **4× the clip draws** (batch 32, same 100k
steps, same masks/seed, lr sqrt-scaled 5e-5 → 1e-4), everything else identical to the batch-8 arms:

| arm | config | data | objective |
|---|---|---|---|
| **Cp4** | `pretrain/configs/e2_Cp4_comma_plain_from_e50_b32.yaml` | comma v0 (110 h) | plain V-JEPA |
| **A4** | `pretrain/configs/e2_A4_public_plain_from_e50_b32.yaml` | public mixture (OpenScene 0.40 / CoVLA 0.27 / DrivingDojo 0.33), e50-estimate weights | plain V-JEPA |

Readout (pb dev, unselected late mean + soup, paired vs e50): both ≈ e50 → compute floor, comma line closed; Cp4 > A4 with
CI lower bound > 0 → data effect at scale, expand to v1 (300 h); both > e50 but equal → continued pretraining helps regardless
of data (recipe knob, not a data result).

Chain: `scripts/h100/phase7_pretrain_arm.sh` (60-step smoke → full run with `latest.pt` resume and STOP → EMA encoder export →
`pb_<arm>_s0` appended to the pb queue → pb training / late evals / best / soup by the existing chain). Launched by Henry in tmux
`fdpretrain` 2026-10-20 03:49Z (Cp4 then A4, serial; uv venv torch 2.8 / decord). Software and hardware disclosed in the ledger
notes (arms on the H100 vs the batch-8 arms on the local 5090).

## Results

### Cp4 (comma clips, plain JEPA objective, batch 32 x 100k steps = 1.74e10 tokens, 4x the ③ arms) — pb run `pb_Cp4_s0`, 2026-10-23
| readout | Cp4 | e50 baseline (`pb_e50_s0`) | paired Δ (log-clustered bootstrap, 10,000) |
|---|---|---|---|
| late window ep16–20 (unselected) | 92.61 / 92.19 / 92.75 / 92.18 / 92.21 → **mean 92.39 (SD 0.27)** | 92.71 | **−0.32 [−0.60, −0.06]** (EP −0.68, NC +0.03, TTC −0.13, DAC −0.02) |
| Lightning best (dev-selected) | best14 **93.04** | best13 93.34 | −0.30 (vs official −0.42 [−0.82, −0.05]) |
| soup 16–20 | **92.75** | 93.10 | −0.35 (vs official −0.70 [−1.15, −0.27]) |
| vs official pb v1 dev ref 93.46 | late mean | | −1.07 [−1.46, −0.70] |
Readout: **4x the pretraining compute on comma-only clips makes the full model significantly worse than the e50 init** (−0.32 with the
CI below 0), in the same direction as D′ (−0.40) and below C′ (−0.01); the shift is again "more conservative" (EP −0.68, safety
sub-scores unchanged). The comma pretraining line therefore stays closed at 4x compute: continued JEPA pretraining on the comma
distribution alone drifts the encoder away from what the nuPlan planner needs. Whether the harm is the comma data or continued
pretraining per se is what `pb_A4_s0` (public mix, same compute; queued 2026-10-23 15:11Z from export sha 53d27af6cdb1) will show:
A4 ≈ e50 → data; A4 ≈ Cp4 → continued pretraining itself at this recipe. Files: `late_window_pb_Cp4_s0_vs_e50.json`,
`late_window_pb_Cp4_s0_vs_official.json`; ledger rows `*_pb_Cp4_s0_*_dev`; reports in `../2026-10-17_pb_baseline/`.

## Log

| time (UTC) | event |
|---|---|
| 10-20 03:49 | Cp4 smoke: 60 steps in 172 s wall (incl. start-up), peak **17.8 GB** |
| 10-20 03:53 | Cp4 full run started (attempt 1, fresh) |
| 10-20 04:01 | steady state (steps 177–226): **1.96 s/step = 1.84 s GPU + 0.11 s data**, ≈ 176k tokens/step (3.9× the batch-8 arms' 44.9k) → GPU-bound, not loader-bound; 100k steps ≈ **54 h** → Cp4 done ≈ 10-22 10:00Z, its pb training + evals ≈ 10-23; A4 ≈ 10-24 16:00Z (+ pb ≈ 10-25). The two concurrent pb trainers slowed from 2.13–2.6 to 2.63–2.70 s/step (≈ 15 %) |
| 10-20 13:16 | Cp4 at step ≈ 20,900 (epoch 9) after 9.4 h → 1.6–1.8 s/step with the GPU at 100 % util (44.6 GB in use); the pb trainers now run at ≈ 3.6 s/step (0.28 it/s, −35 %): `pb_e50_cos_s0` ETA ≈ 10-21 00:00Z, `pb_e50_cos_s1` ≈ 05:00Z. The progress page had been broken since 03:50Z by a variable-shadowing bug in the new pretraining-arms block (fixed 13:20Z) |
| 10-21 21:52 | Cp4 done (100k steps, 42 h wall); export `e2_Cp4_comma_plain_from_e50_b32_e40_target_encoder.pt` (sha b4217fdf); `pb_Cp4_s0` auto-launched 21:55; A4 started 21:55 |
| 10-23 15:11 | A4 done (100k steps); export `e2_A4_public_plain_from_e50_b32_e40_target_encoder.pt` (292 keys, 1.22 GB, sha 53d27af6cdb1); `pb_A4_s0` appended to the pb queue (after the ⑤ finals and cos_s2) |
| 10-23 ≈15:45 | `pb_Cp4_s0` chain done (20 epochs; 5–8 s/step while sharing the GPU with the A4 arm and the ⑤ runs): dev late 16–20 mean 92.39 vs e50 92.71, paired −0.32 [−0.60, −0.06]; best14 93.04; soup 92.75 → see Results |
