#!/usr/bin/env bash
# E1 step 1 (plan v4 §5): e50 -> Drive-JEPA perception-free agent, official recipe, trained on the devkit's default navtrain
# train/val log split (978 / 214 logs) from the compact cache; periodic fp16 weight exports every 5 epochs for the learning curve.
#   usage: e1_train_perception_free.sh <tag> [seed] [extra hydra overrides...]
# Official recipe (navsim_v1/scripts/training/train_drive_jepa_perception_free.sh + metadata read from the released
# drive_jepa_perception_free_agent_vitl.ckpt on 2026-10-13): Adam 1e-4 (no wd, no scheduler), L1 length-normalised (alpha 5),
# encoder fine-tuned, 2 frames 512x256, 8-d ego status, torchrun over the visible GPUs with batch 32 PER GPU -- the released
# checkpoint has 1,329 optimizer steps per epoch = 85,109 / 64, i.e. TWO GPUs -> effective batch 64 ("we tested training on two
# L40S/A30/V100 GPUs"), precision bf16, 40 epochs, best val/loss_epoch 0.2271 at epoch 37 (global_step 49,173).
# Defaults here reproduce that on one GPU: BATCH=16 x ACCUM=4 = effective 64 (Adam, no BN, dropout 0 -> equivalent up to
# numerics), PRECISION=bf16, peak VRAM ~7 GB (fits the U-01 default rule next to Henry's SDD jobs).
# History: the first run e1_e50_pf_s0 (2026-10-12 15:10 -> 10-13 01:35, stopped during epoch 16) used 16-mixed (fp16) and
# BATCH=16 x ACCUM=2 = effective 32 (twice the official number of updates per epoch); its dev PDMS fell from 78.73 (epoch 5)
# to 75.09 (epoch 10) while val/loss improved -- see docs/evidence/2026-10-12_E1_learning_curve/. Kept as
# benchmarks/navsim/exp/e1_e50_pf_s0_fp16_eb32/; the corrected run is e1b_e50_pf_s0 (launched 2026-10-13 01:39).
# Resume: FRONTDRIVE_RESUME_CKPT=<path to last.ckpt> re-runs with the same tag.
# The perception-free module logs val/loss only (no online PDM score), so the 'best' checkpoint is selected by val/loss_epoch;
# model selection for the plan uses dev-set PDMS of the periodic fp16 exports (eval_perception_free.sh), not this checkpoint.
set -euo pipefail
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_env.sh
TAG=${1:?tag}; SEED=${2:-0}; shift $(( $# >= 2 ? 2 : $# ))
E50=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt
# Encoder initialisation: FRONTDRIVE_PRETRAIN_PT overrides the e50 file (e.g. an export of pretrain/export_target_encoder.py:
# same {target_encoder: module.backbone.*} layout) so the E1 protocol can compare encoders with everything else unchanged.
PRETRAIN_PT=${FRONTDRIVE_PRETRAIN_PT:-$E50}
# Per-epoch late exports: FRONTDRIVE_EXPORT_FROM=<1-based epoch> additionally exports every epoch from that epoch on
# (protocol 2026-10-14: evaluate the last five epochs on dev and compare late-export means, see C-29).
[ -n "${FRONTDRIVE_EXPORT_FROM:-}" ] && export FRONTDRIVE_EXPORT_FROM
CACHE=$NAVSIM_LOCAL/cache/navtrain_compact
OUT=$NAVSIM_EXP_ROOT/$TAG
mkdir -p "$OUT"
# Disk: full Lightning checkpoints are ~3.7 GB each, so periodic snapshots are fp16 agent-weight exports (~0.6 GB, loadable by
# agent.initialize / eval_perception_free.sh) every FRONTDRIVE_EXPORT_EVERY epochs; Lightning keeps best (val/score) + last.ckpt.
export WANDB_MODE=disabled FRONTDRIVE_NAVSIM_EXT=/home/henry/Desktop/Qwen_Drive/frontdrive FRONTDRIVE_EXPORT_EVERY=${FRONTDRIVE_EXPORT_EVERY:-5} FRONTDRIVE_SAVE_LAST=1
[ -n "${FRONTDRIVE_CKPT_EVERY:-}" ] && export FRONTDRIVE_CKPT_EVERY
export FRONTDRIVE_RESUME_CKPT=${FRONTDRIVE_RESUME_CKPT:-}
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu_before.txt"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
  agent=drive_jepa_perception_free_agent agent.pretrain_pt_path="$PRETRAIN_PT" agent.image_architecture=vit_large \
  agent.num_keyval=129 agent.front_only=true agent.tf_dropout=0.0 agent.freeze_encoder=false agent.double_image=true agent.lr=1e-4 \
  cache_path="$CACHE" use_cache_without_dataset=true experiment_name="$TAG" output_dir="$OUT" seed="$SEED" \
  dataloader.params.batch_size=${BATCH:-16} dataloader.params.num_workers=8 dataloader.params.prefetch_factor=4 \
  trainer.params.accumulate_grad_batches=${ACCUM:-4} trainer.params.max_epochs=${EPOCHS:-40} trainer.params.precision=${PRECISION:-bf16} \
  trainer.checkpoint.monitor=val/loss_epoch trainer.checkpoint.mode=min \
  trainer.params.strategy=auto "$@" 2>&1 | tee -a "$OUT/train.log"
