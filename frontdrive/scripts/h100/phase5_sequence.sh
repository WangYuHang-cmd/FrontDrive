#!/usr/bin/env bash
# Plan v4.3 ② -> ③ back to back on the server without a relay in between (Claude's session cannot ssh; every launch is a
# Henry round trip). Waits for the hand-launched e50 chain (tmux `fdpb`) to end, then runs scripts/h100/phase5_pb_pipeline.sh
# once per item of PB_SEQUENCE ("<tag>:<encoder .pt or empty = e50>"), default: finish pb_e50_s0 (idempotent: drains any
# pending evals) -> pb_Cp_s0 (C' export) -> pb_Dp_s0 (D' export). Every chain is idempotent/resumable, so a rerun of this
# script is safe; STOP/PAUSE markers block the next start (the running chain handles them itself).
# Launch (tmux):  tmux new-session -d -s fdpbseq "bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/phase5_sequence.sh >> /home/henry/Desktop/Qwen_Drive/state/phase5.log 2>&1"
# Reboot: boot.sh relaunches it while state/phase5_sequence.enabled exists (removed when the sequence is done).
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; PX=$QD/benchmarks/pretrain/exports
log() { echo "[$(date -u +%FT%TZ)] [sequence] $*"; }
SEQ=${PB_SEQUENCE:-"pb_e50_s0: pb_Cp_s0:$PX/e2_Cp_comma_plain_from_e50_e40_target_encoder.pt pb_Dp_s0:$PX/e2_Dp_comma_motion_from_e50_e40_target_encoder.pt"}
printf '# phase5 sequence active since %s\nPB_SEQUENCE=%s\n' "$(date -u +%FT%TZ)" "$SEQ" > "$S/phase5_sequence.enabled"
log "start: $SEQ"
if tmux has-session -t fdpb 2>/dev/null; then
  log "waiting for the running chain in tmux fdpb to end"
  while tmux has-session -t fdpb 2>/dev/null; do sleep 300; done
  log "tmux fdpb ended"
fi
for item in $SEQ; do
  tag=${item%%:*}; pt=${item#*:}
  if [ -n "$pt" ] && [ ! -f "$pt" ]; then log "$tag: encoder $pt missing -> skipped"; continue; fi
  while [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do sleep 60; done
  log "$tag: chain start (pretrain=${pt:-e50}, exports every ${PB_EXPORT_EVERY:-20} + epochs >= ${PB_EXPORT_FROM:-16})"
  # ③ runs export (and dev-evaluate) only the late window 16-20: the e50 learning-curve points 5/10/15 are taken once from ②
  PB_TAG=$tag PB_PRETRAIN_PT=$pt PB_EXPORT_EVERY="${PB_EXPORT_EVERY:-20}" PB_EXPORT_FROM="${PB_EXPORT_FROM:-16}" bash "$F/scripts/h100/phase5_pb_pipeline.sh"; rc=$?
  log "$tag: chain rc=$rc"
done
rm -f "$S/phase5_sequence.enabled"
log "done"
