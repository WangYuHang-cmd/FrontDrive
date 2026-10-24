#!/usr/bin/env bash
# Perception-free filler jobs ON THE SERVER (tmux `fdpf`; plan v4.3 §7 "中断任务"): finish the two E1 runs that the local
# memory guard interrupted on 2026-10-16 and evaluate their late exports on dev, so the pf-side encoder comparison (C-38
# evidence) is complete: (a) e1_Cp_e40_pf_s0 epoch 40 (resume from last.ckpt = end of epoch 39) + dev evals 38/39/40,
# (b) e1b_e50_pf_s0 epochs 42-45 (resume from end of epoch 41) + dev evals 41-45. GPU ~7 GB per training, ~6 GB per eval;
# CPU: 8 loader workers / a single-process scorer -> light next to the pb trainers. Idempotent: skips finished exports/evals.
# Software: FRONTDRIVE_USE_VENV=1 keeps the uv venv (torch 2.8.0+cu128, Lightning 2.5.6 — the versions these runs were
# started with locally; resuming a 2.5.6 checkpoint under the conda env's Lightning 2.2.1 would add a version confound).
# Cross-hardware continuation (5090 -> H100) is disclosed in the ledger notes by eval/e1_ingest.sh.
# Eval lines "EVAL <prefix>_ep<N> EXIT <rc>" go to state/pf_fillers.log for eval/e1_autoingest.sh (registry chain_log).
# Launch: tmux new-session -d -s fdpf "bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/phase6_pf_fillers.sh >> /home/henry/Desktop/Qwen_Drive/state/pf_fillers.log 2>&1"
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp; PX=$QD/benchmarks/pretrain/exports
export FRONTDRIVE_USE_VENV=1
source "$F/scripts/h100/gpu_policy.sh"
log() { echo "[$(date -u +%FT%TZ)] [pf] $*"; }
ours_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac
  done | awk '{s+=$1} END{print s+0}'
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
wait_capacity() {  # <need MiB>
  while :; do
    if [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; then log "marker present -> waiting"; sleep 120; continue; fi
    local used ours l1; used=$(gpu_used); ours=$(ours_mib); l1=$(awk '{print int($1)}' /proc/loadavg)
    if free_for_us "$1" "$used" "$ours" && [ "$l1" -le "${LOAD_MAX:-40}" ]; then return; fi
    log "waiting for $1 MiB of room (gpu used $used MiB, ours $ours MiB)"; sleep 120
  done
}
printf '# pf fillers active since %s\n' "$(date -u +%FT%TZ)" > "$S/pf_fillers.enabled"
cd "$F"
# ---- (a) C' E1: epoch 40
RUN=$EXP/e1_Cp_e40_pf_s0
if [ ! -f "$RUN/exports/epoch_40.ckpt" ]; then
  test -f "$RUN/checkpoints/last.ckpt" || { log "no $RUN/checkpoints/last.ckpt"; exit 2; }
  wait_capacity 8000
  log "train e1_Cp_e40_pf_s0 -> epoch 40 (resume from last.ckpt, exports from 36)"
  FRONTDRIVE_RESUME_CKPT=$RUN/checkpoints/last.ckpt FRONTDRIVE_EXPORT_FROM=36 EPOCHS=40 \
    FRONTDRIVE_PRETRAIN_PT=$PX/e2_Cp_comma_plain_from_e50_e40_target_encoder.pt \
    bash scripts/e1_train_perception_free.sh e1_Cp_e40_pf_s0 0 > "$S/pf_train_e1_Cp.out" 2>&1; rc=$?
  log "train e1_Cp_e40_pf_s0 rc=$rc exports: $(ls "$RUN/exports" 2>/dev/null | tr '\n' ' ')"
fi
for N in 38 39 40; do
  TAG=e1_Cp_s0_ep$N; CK=$RUN/exports/epoch_$N.ckpt
  [ -f "$EXP/eval/$TAG/report.json" ] && { log "EVAL $TAG already done"; continue; }
  [ -f "$CK" ] || { log "EVAL $TAG skipped: no $CK"; continue; }
  wait_capacity 7000
  log "EVAL $TAG START"
  nice -n 5 bash scripts/eval_perception_free.sh "$TAG" "$CK" navtrain_val_official > "$S/eval_$TAG.log" 2>&1; rc=$?
  log "EVAL $TAG EXIT $rc at $(date -Is) $(tr -d '\n' < "$EXP/eval/$TAG/report.json" 2>/dev/null | cut -c1-200)"
done
# ---- (b) E1b s0: epochs 42-45
RUN=$EXP/e1b_e50_pf_s0
if [ ! -f "$RUN/exports/epoch_45.ckpt" ]; then
  test -f "$RUN/checkpoints/last.ckpt" || { log "no $RUN/checkpoints/last.ckpt"; exit 2; }
  wait_capacity 8000
  log "train e1b_e50_pf_s0 -> epoch 45 (resume from last.ckpt = end of 41, exports from 41)"
  FRONTDRIVE_RESUME_CKPT=$RUN/checkpoints/last.ckpt FRONTDRIVE_EXPORT_FROM=41 EPOCHS=45 \
    bash scripts/e1_train_perception_free.sh e1b_e50_pf_s0 0 > "$S/pf_train_e1b.out" 2>&1; rc=$?
  log "train e1b_e50_pf_s0 rc=$rc exports: $(ls "$RUN/exports" 2>/dev/null | tr '\n' ' ')"
fi
for N in 41 42 43 44 45; do
  TAG=e1b_s0_ep$N; CK=$RUN/exports/epoch_$N.ckpt
  [ -f "$EXP/eval/$TAG/report.json" ] && { log "EVAL $TAG already done"; continue; }
  [ -f "$CK" ] || { log "EVAL $TAG skipped: no $CK"; continue; }
  wait_capacity 7000
  log "EVAL $TAG START"
  nice -n 5 bash scripts/eval_perception_free.sh "$TAG" "$CK" navtrain_val_official > "$S/eval_$TAG.log" 2>&1; rc=$?
  log "EVAL $TAG EXIT $rc at $(date -Is) $(tr -d '\n' < "$EXP/eval/$TAG/report.json" 2>/dev/null | cut -c1-200)"
done
rm -f "$S/pf_fillers.enabled"
log "done"
