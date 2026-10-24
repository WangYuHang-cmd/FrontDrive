#!/usr/bin/env bash
# Dev-evaluate the Lightning best checkpoint (checkpoints/best_epoch=<N>.ckpt = best val/score_epoch, dev-selected exactly
# like the released Drive-JEPA weights) of finished pb runs, as tag <run>_best<N>. Complements the unselected late-window
# exports: the pair (late mean, best ckpt) is what we compare against e50 (92.71 / 93.34) and the official ckpt (93.46).
# Waits for each run's final export (exports/epoch_<EPOCHS>.ckpt) and for GPU/CPU room (gpu_policy.sh); skips existing reports.
# Usage (tmux fdevalbest2): bash scripts/h100/eval_best_ckpts.sh pb_Cp_s0 pb_Dp_s0 pb_e50_s1 [...]
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp
source "$F/scripts/h100/gpu_policy.sh"; EVAL_MIB=${EVAL_MIB:-13000}; EPOCHS=${PB_EPOCHS:-20}
log() { echo "[$(date -u +%FT%TZ)] [evalbest] $*"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
ours_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac
  done | awk '{s+=$1} END{print s+0}'
}
for RUN in "$@"; do
  while [ ! -f "$EXP/$RUN/exports/epoch_$EPOCHS.ckpt" ]; do log "$RUN: training not finished (no exports/epoch_$EPOCHS.ckpt) -> waiting"; sleep 600; done
  CK=$(ls "$EXP/$RUN"/checkpoints/best_epoch=*.ckpt 2>/dev/null | head -n 1)
  [ -n "$CK" ] || { log "$RUN: no checkpoints/best_epoch=*.ckpt -> skipped"; continue; }
  N=$(basename "$CK" .ckpt); N=${N#best_epoch=}; TAG=${RUN}_best$N
  if [ -f "$EXP/eval_pb/$TAG/report.json" ]; then log "$TAG: report exists -> skip"; continue; fi
  while :; do
    { [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; } && { log "marker present -> waiting"; sleep 120; continue; }
    used=$(gpu_used); ours=$(ours_mib); l=$(awk '{print int($1)}' /proc/loadavg)
    free_for_us "$EVAL_MIB" "$used" "$ours" && [ "$l" -le "$LOAD_MAX" ] && break
    log "$TAG: waiting for room (gpu used $used MiB, ours $ours MiB, load1 $l)"; sleep 120
  done
  log "$TAG: START ($CK)"
  PB_SIDE_CALIB=1 nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$TAG" "$CK" navtrain_val_official 4 > "$S/eval_$TAG.log" 2>&1; rc=$?
  log "$TAG: EXIT $rc $(tr -d '\n' < "$EXP/eval_pb/$TAG/report.json" 2>/dev/null | cut -c1-300)"
done
log "done"
