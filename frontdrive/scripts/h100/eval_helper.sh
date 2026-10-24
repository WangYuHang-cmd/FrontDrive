#!/usr/bin/env bash
# Extra dev-eval stream for a pb run whose chain drains its exports one at a time (the chain started 2026-10-17 with a
# single-eval dispatcher and a load gate of 16, which two concurrent trainers keep exceeding). Walks the given epochs from
# the LAST to the first so the two streams meet in the middle instead of colliding; a tag is skipped when its report.json
# exists or when its run.log was written in the last 10 min (= the chain is evaluating it right now). Before starting a tag
# it writes a placeholder report.json ({"placeholder": true}) so the chain's dispatcher skips the tag; eval_pb.sh overwrites
# the placeholder with the real report, and on failure the placeholder is removed again so the chain can retry.
# Capacity rules as in the chain: ours + EVAL_MIB <= CAP_MIB, >= 20 GB left for the other user, load1 <= LOAD_MAX.
# Usage (tmux fdevalhelp):  bash scripts/h100/eval_helper.sh <run_tag> <epoch> [epoch ...]      e.g.  pb_e50_s0 20 19 18 17 16 15 10
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp
RUN=${1:?run tag}; shift; EPOCHS=("$@"); [ ${#EPOCHS[@]} -gt 0 ] || { echo "no epochs given"; exit 2; }
source "$F/scripts/h100/gpu_policy.sh"; EVAL_MIB=${EVAL_MIB:-13000}
log() { echo "[$(date -u +%FT%TZ)] [evalhelper $RUN] $*"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
ours_mib() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac
  done | awk '{s+=$1} END{print s+0}'
}
load1() { awk '{print int($1)}' /proc/loadavg; }
log "start: epochs ${EPOCHS[*]} (last to first)"
for N in "${EPOCHS[@]}"; do
  TAG=${RUN}_ep$N; CKPT=$EXP/$RUN/exports/epoch_$N.ckpt; OUT=$EXP/eval_pb/$TAG
  while [ ! -f "$CKPT" ]; do log "$TAG: waiting for $CKPT"; sleep 300; done
  if [ -f "$OUT/report.json" ]; then log "$TAG: report exists -> skip"; continue; fi
  if [ -f "$OUT/run.log" ] && [ $(( $(date +%s) - $(stat -c %Y "$OUT/run.log") )) -lt 600 ]; then log "$TAG: being evaluated by the chain (run.log fresh) -> skip"; continue; fi
  while :; do
    [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ] && { log "marker present -> waiting"; sleep 120; continue; }
    used=$(gpu_used); ours=$(ours_mib); l=$(load1)
    if free_for_us "$EVAL_MIB" "$used" "$ours" && [ "$l" -le "$LOAD_MAX" ]; then break; fi
    log "$TAG: waiting for capacity (gpu used $used MiB, ours $ours MiB, load1 $l)"; sleep 120
  done
  mkdir -p "$OUT"; printf '{"tag": "%s", "placeholder": true, "since": "%s"}\n' "$TAG" "$(date -u +%FT%TZ)" > "$OUT/report.json"
  log "$TAG: START (gpu used $used MiB, ours $ours MiB, load1 $l)"
  PB_SIDE_CALIB=1 nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$TAG" "$CKPT" navtrain_val_official 4 > "$S/eval_$TAG.helper.log" 2>&1; rc=$?
  if grep -q '"pdms"' "$OUT/report.json" 2>/dev/null; then log "$TAG: EXIT $rc $(tr -d '\n' < "$OUT/report.json" | cut -c1-300)"
  else rm -f "$OUT/report.json"; log "$TAG: EXIT $rc without a report -> placeholder removed (state/eval_$TAG.helper.log)"; fi
done
log "done"
