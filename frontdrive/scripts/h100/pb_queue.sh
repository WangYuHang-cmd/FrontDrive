#!/usr/bin/env bash
# Minimal declarative queue for pb training chains ON THE SERVER (tmux `fdqueue`). Every POLL seconds it reads
# state/pb_queue.txt (one job per line: "<tag> <encoder .pt or -> <seed> [KEY=VALUE ...]"; '#' comments) and, while fewer than
# MAX_PB pb chains hold a run lock, launches the first job that is neither running (lock held) nor finished (final export
# present) in its own tmux session fdpb_<tag> through scripts/h100/phase5_pb_pipeline.sh (act-ckpt on, late-window exports).
# Capacity: ours + 13 GB <= CAP_MIB and >= 20 GB left for the other user; STOP/PAUSE markers block launches. The queue file can
# be edited any time (Claude pushes it with rsync; no ssh needed) — remove a line to un-queue a job that has not started.
# Launch: tmux new-session -d -s fdqueue "bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/pb_queue.sh >> /home/henry/Desktop/Qwen_Drive/state/pb_queue.log 2>&1"
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp
QF=$S/pb_queue.txt; POLL=${POLL:-300}; TRAIN_MIB=13000   # a pb chain with act-ckpt on needs ~12 GB
source "$F/scripts/h100/gpu_policy.sh"
log() { echo "[$(date -u +%FT%TZ)] [queue] $*"; }
ours_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac
  done | awk '{s+=$1} END{print s+0}'
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
lock_held() { [ -e "$1" ] || return 1; ( exec 9<"$1"; ! flock -n 9 ) 2>/dev/null; }
n_running() { local n=0 l; for l in "$EXP"/pb_*/.phase5.lock; do [ -e "$l" ] && lock_held "$l" && n=$((n + 1)); done; echo "$n"; }
printf '# pb queue active since %s\n' "$(date -u +%FT%TZ)" > "$S/pb_queue.enabled"
log "start: MAX_PB=$MAX_PB POLL=$POLL queue file $QF"
while :; do
  sleep "$POLL"
  source "$F/scripts/h100/gpu_policy.sh"   # re-read every poll so policy edits apply live
  [ -f "$QF" ] || continue
  if [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; then continue; fi
  nr=$(n_running); [ "$nr" -lt "$MAX_PB" ] || continue
  l1=$(awk '{print int($1)}' /proc/loadavg); [ "$l1" -le "${LOAD_MAX_LAUNCH:-30}" ] || continue   # CPU-bound host: no new trainer while it thrashes
  used=$(gpu_used); ours=$(ours_mib)
  free_for_us "$TRAIN_MIB" "$used" "$ours" || continue
  while read -r tag pt seed extra; do
    case "$tag" in ""|\#*) continue;; esac
    out=$EXP/$tag
    ep=20; for kv in $extra; do case "$kv" in PB_EPOCHS=*) ep=${kv#PB_EPOCHS=};; esac; done   # the job's own epoch count (⑤ runs: 5)
    { [ -f "$out/exports/epoch_$ep.ckpt" ] || [ -f "$out/.phase5.done" ]; } && continue   # finished
    lock_held "$out/.phase5.lock" && continue                  # running
    tmux has-session -t "fdpb_$tag" 2>/dev/null && continue    # launching
    [ "$pt" = "-" ] && pt=""
    if [ -n "$pt" ] && [ ! -f "$pt" ]; then log "$tag: encoder $pt missing -> skipped"; continue; fi
    log "$tag: launch (running $nr/$MAX_PB, gpu used $used MiB, ours $ours MiB) seed=$seed pretrain=${pt:-e50} extra=[$extra]"
    tmux new-session -d -s "fdpb_$tag" "env PB_TAG=$tag PB_SEED=${seed:-0} PB_PRETRAIN_PT=$pt PB_EXPORT_EVERY=20 $extra bash $F/scripts/h100/phase5_pb_pipeline.sh >> $S/phase5.log 2>&1"
    break   # one launch per poll
  done < "$QF"
done
