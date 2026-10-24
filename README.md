# FrontDrive — single-front-camera planning on NAVSIM, built on Drive-JEPA

FrontDrive is the training recipe, evaluation tooling and evidence behind our NAVSIM v1 **navtest** entry: a Drive-JEPA
perception-based planner (ViT-L V-JEPA 2 encoder, BEVFormer-style head, 8192 anchored proposals) that sees **one front camera**
(CAM_F0 at t and t−0.5 s, 512×256) plus the ego status and emits one trajectory per scene. The side cameras enter only through their
calibration matrices (no side-camera pixels; verified bit-identical to the upstream path on navtest).

## Result (NAVSIM v1 navtest, PDMS, 12,146 scenes)

| model | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| **FrontDrive** (this repo, multi-seed submission) | **93.86** | **93.78** | **93.94** | **93.86 ± 0.08** |
| released Drive-JEPA perception-based v1 checkpoint, same evaluation code | 93.69 | | | |
| our from-scratch reproduction of the upstream recipe (e50 encoder, dev-selected epoch) | 93.50 | | | |

Paired per-scene differences vs the released checkpoint: +0.17 [−0.08, +0.43], +0.09 [−0.16, +0.34], +0.25 [−0.01, +0.51]
(scene-level bootstrap); vs our reproduction: +0.28…+0.44 with CIs above zero. Sub-scores move the same way in every seed:
ego progress +1.0, drivable-area compliance +0.35, no-at-fault collisions −0.4, TTC −0.8…−1.2, driving direction −1.0.

## Recipe

1. Start from the released Drive-JEPA perception-based v1 checkpoint (all agent weights).
2. Continue training for **5 epochs on all 1,192 navtrain logs** (train + val split lists, 103,288 samples; 1,613 steps/epoch at
   effective batch 64) with the upstream loss and online PDM scorer, Adam peak lr **3e-5** (backbone 0.1×), **cosine decay to 3e-6
   after 200 warm-up steps**, bf16 mixed precision, single H100.
3. Export fp16 agent weights after every epoch and **average the five exports** (uniform weight soup).
4. Three seeds; all three are submitted (the leaderboard reports max / mean / std). No seed or epoch is selected on navtest.

Weights: https://huggingface.co/HenryYHW/FrontDrive (three `*_soup_1_5.ckpt`, fp16, loadable by `DriveJEPAAgent.initialize`).

Why this recipe — the dev-set (navtrain official val, 18,157 scenes) story in `frontdrive/docs/evidence/`:
- the from-scratch upstream recipe reproduces the released weights only after dev selection (93.34 vs 93.46 dev); unselected late
  epochs sit 0.75 below (`2026-10-17_pb_baseline`);
- a uniform soup of the last five exports recovers +0.39 without selection; a per-step cosine lr adds +0.31 on the unselected late
  window but does not raise the ceiling (`2026-10-17_pb_baseline`);
- continued V-JEPA pretraining on 110 h / 4× of our own dash-cam video did not help the full model (−0.01 / −0.40 / −0.32 vs the
  public e50 encoder; `2026-10-20_pretrain_scale_test`), so the submission uses only public data and public weights;
- confidence-based ensembles (+0.08) and training-free re-selection among proposals (≤ 0) do not help (`2026-10-17_pb_baseline`
  §④(c), `2026-10-23_selection_rules`);
- continuing the released weights on the train split alone is neutral on dev (93.22–93.31 vs 93.46 dev-selected); adding the val
  logs is what the final models gain from (`2026-10-22_official_init_ft`).

## Repository layout

| path | content |
|---|---|
| `frontdrive/third_party_patches/` | `drive_jepa_548bb82.patch` against upstream Drive-JEPA commit `548bb82` (torch ≥ 2.6 `weights_only`, file-relative config, compact JPEG training cache, per-step lr schedule, continued-training init, all-logs split, submission-pickle checkpoint fix) and the generated split yamls under `untracked/` |
| `frontdrive/navsim_ext/` | compact JPEG training cache (`compact_cache*.py`), official train/val log split tooling, inference-time ensemble and selection-rule agents + Hydra configs |
| `frontdrive/scripts/h100/` | the H100 pipeline: `bootstrap_conda_cvpr.sh` (environment), `hf_fetch.sh` (public assets), `build_pb_cache.sh`, `train_pb.sh` (all training knobs), `phase5_pb_pipeline.sh` (train → exports → dev evals → soup), `pb_queue.sh`, `eval_pb.sh` (dev / navtest), `eval_pb_ensemble.sh`, `eval_pb_select.sh`, `navtest_finals_1023.sh`, `make_submission_1024.sh` |
| `frontdrive/eval/` | ledgers (`ledger.csv`, `navtest_evals.csv`), ingest, `soup_exports.py`, `pb_late_window.py`, `merge_submission.py` |
| `frontdrive/diagnostics/d5_error_buckets.py` | paired, log-clustered bootstrap over per-scene PDMS (all CIs in the evidence) |
| `frontdrive/pretrain/` | the V-JEPA continued-pretraining arms (negative results; not used by the submitted models) |
| `frontdrive/tests/` | unit tests for the cache, ensemble and selection code |
| `frontdrive/docs/evidence/` | dated evidence folders: numbers, paired reports, decisions |

## Reproducing the submitted models

```bash
# 1. upstream code at the pinned commit + our patch and generated split files
git clone https://github.com/linhanwang/Drive-JEPA third_party/Drive-JEPA && git -C third_party/Drive-JEPA checkout 548bb82
git -C third_party/Drive-JEPA apply ../../frontdrive/third_party_patches/drive_jepa_548bb82.patch
cp -r frontdrive/third_party_patches/untracked/navsim_v1 third_party/Drive-JEPA/
# 2. environment (conda env `cvpr`: torch 2.1.2+cu121, mmcv-full 1.7.2 built for sm_90, mmdet 2.28.2, navsim v1 editable)
bash frontdrive/scripts/h100/bootstrap_conda_cvpr.sh
# 3. public assets: NAVSIM navtrain/navtest logs + CAM_F0 frames, Drive-JEPA e50 encoder, v1 checkpoint, anchors, train_metric_cache
bash frontdrive/scripts/h100/hf_fetch.sh            # sha256-checked list in frontdrive/data/benchmarks/download_list_*.tsv
# 4. compact JPEG training cache (features of the two front frames + lidar2img + ego status)
bash frontdrive/scripts/h100/build_pb_cache.sh
# 5. one final seed (replace SEED); exports/soup_1_5.ckpt is the candidate
PB_INIT_CKPT=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt PB_TRAIN_LOGS=all \
PB_LR=3e-5 PB_LR_SCHED=cosine PB_LR_MIN_MULT=0.1 PB_LR_WARMUP=200 EPOCHS=5 EXPORT_EVERY=1 EXPORT_FROM=1 \
  bash frontdrive/scripts/h100/train_pb.sh pb_off_ft_all_sSEED SEED
python frontdrive/eval/soup_exports.py --out exports/soup_1_5.ckpt exports/epoch_{1,2,3,4,5}.ckpt
# 6. evaluate / build the submission pickle (navtest)
PB_SIDE_CALIB=1 bash frontdrive/scripts/h100/eval_pb.sh my_tag exports/soup_1_5.ckpt navtest 4
bash frontdrive/scripts/h100/make_submission_1024.sh && python frontdrive/eval/merge_submission.py ...
```
The scripts are written for one shared H100 (paths under `/home/henry/Desktop/Qwen_Drive`, tmux chains, a GPU-sharing policy); adapt
the root paths in `frontdrive/env/*.sh` for another machine.

## Rules compliance
Public data (NAVSIM navtrain) and public weights (Drive-JEPA, V-JEPA 2) only; navtest annotations are never used for training or
model selection; every navtest evaluation is logged in `frontdrive/eval/navtest_evals.csv` (8 in total for this project, one per
submitted seed).

## License and credits
Code: Apache License 2.0 (this repository builds on Drive-JEPA and NAVSIM, both Apache-2.0; the patch is a derivative work of
Drive-JEPA). Weights on Hugging Face: CC BY 4.0, derived from the released Drive-JEPA checkpoint.
Authors: Yuhang Wang, Jingxin Yang, Chuheng Wei, Suyang Chen, Hao Zhou.
Built on [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) (Wang et al.), [NAVSIM](https://github.com/autonomousvision/navsim)
and [V-JEPA 2](https://github.com/facebookresearch/vjepa2).
