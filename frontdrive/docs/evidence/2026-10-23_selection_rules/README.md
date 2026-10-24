# ④(c) Training-free proposal-selection rules on one pb checkpoint (design 2026-10-23 20:00 EDT; Henry approved the direction 19:4x EDT)

**Motivation.** Every pb model we trained — from the e50 encoder, from the comma encoders, and the ⑤ probe continued from the released
weights — sits at **EP +0.9…+1.1, NC −0.4, TTC −0.9…−1.4, DAC −0.2** relative to the released checkpoint on dev, and the probe shows
this within its first epoch, so our training signal (online-scorer targets / loss) tilts the policy toward progress. The approved idea
was to re-weight the final selection with the model's own predicted *safety sub-scores*. **That is not possible as stated:** the
scorer's sub-score columns are untrained — `DriveJEPAConfig.sub_score_weight = 0` upstream (only the final-score column carries a loss;
no yaml/script overrides it) and every train log shows `sub_score_loss ≈ 0.69 = ln 2` throughout. The training-free levers that DO
exist in the forward output (`proposals`, `pred_logit`, `pred_logit2`, `pdm_score`) are the two final-score heads and the proposal
geometry; `navsim_ext/pb_select_agent.py` implements four rules (one trajectory per scene, no oracle, no training):

| mode | rule | what it tests |
|---|---|---|
| `argmax` | mean sigmoid of the two heads, argmax | = the model (smoke: must reproduce the plain evaluation per scene) |
| `minheads` | min of the two heads instead of the mean | both heads must agree a proposal is good (head disagreement as uncertainty) |
| `conservative` | among proposals within `eps` of the best mean score, the smallest final-pose displacement | "trade progress back for safety" with one dev-tuned scalar |
| `progressive` | same candidate set, the largest displacement | control: the opposite tilt must not also improve |

Base checkpoint = the dev-clean stand-in for the finals: `pb_off_ft_s0/exports/soup_1_5.ckpt` (released weights continued on the train
logs, dev **93.22** under argmax; per-scene CSV `eval_pb/pb_off_ft_s0_soup1_5`). Chain `scripts/h100/select_chain_1023.sh` (tmux `fdsel`):
smoke `sel_smoke_argmax` (16 scenes) → `pb_sel_offsoup_minheads` → `pb_sel_offsoup_cons002` (eps 0.02) → `pb_sel_offsoup_cons005`
(eps 0.05) → `pb_sel_offsoup_prog002` (control). Each dev eval ≈ 30–60 min with 4 workers; the chain waits for load1 ≤ LOAD_MAX and GPU
room (the navtest finals chain and two trainers share the host). Ingest: `pb_sel_*` rows (encoder_init `select`, BEST=4, paired vs the
official reference by `pull.sh`); the decisive paired comparison vs the same checkpoint's argmax (`pb_off_ft_s0_soup1_5` CSV) is computed
by hand with `diagnostics/d5_error_buckets.py`.

**Readout (pre-stated).** A rule beats argmax on the same checkpoint by **≥ +0.2 with CI low > 0** AND `progressive` does not also
improve → adopt it for the ⑤ submission (one navtest eval of final s0 + rule = the last budget slot, 8 of 8). Otherwise the finals are
submitted with the model's own argmax. The sub-scores of `conservative` vs argmax (EP down, NC/TTC up?) are reported either way:
they show whether the policy tilt is even reachable by re-selection among the model's proposals.

**Files.** `navsim_ext/pb_select_agent.py`, `navsim_ext/configs/agent/drive_jepa_pb_select_agent.yaml`, `scripts/h100/eval_pb_select.sh`,
`scripts/h100/select_chain_1023.sh`, `tests/test_pb_select_agent.py` (5 tests), ingest rules in `scripts/h100/pull.sh` / `eval/pb_ingest.sh`.
Results will be appended below.

## Results (chain `fdsel`, started 2026-10-23 23:25Z)
- Smoke `sel_smoke_argmax`: 16/16 scenes identical to `pb_off_ft_s0_soup1_5` (7 score columns, max |Δ| = 0) → the wrapper reproduces the model.
- `minheads`: **identical to argmax on all 18,157 scenes** (0 changed) → the two heads never disagree about the top proposal for this
  checkpoint (or the second head is absent): no lever here.
- `conservative`, eps 0.02: dev **92.44**, paired vs argmax **−0.78 [−1.07, −0.50]**; 69.6 % of scenes changed (785 better / 11,863 worse);
  sub-scores EP **−4.07**, NC +0.44, TTC +1.31, DAC +0.55, DDC +2.38. The safety gains ARE reachable by re-selection, but at four points of
  progress: within 0.02 of the best predicted score there are many proposals (the score surface is flat), and the most conservative of
  them is far too slow. Pending: eps 0.05 (will be worse) and the `progressive` control.
