#!/usr/bin/env bash
# Plan v4.3 ④(c), 2026-10-21: inference-time ensemble evaluations on dev (navtrain official val), sequential, one tmux session.
#   1. smoke: 16 scenes, mode "first" (must reproduce member 0 = pb_e50_cos_s0 soup per-scene scores; checked locally after pull)
#   2. pb_ens_cos01_argmax      argmax_conf   {cos_s0 soup, cos_s1 soup}
#   3. pb_ens_cos01_meanclose   mean_if_close {cos_s0 soup, cos_s1 soup}   (close_m 0.5 m)
#   4. pb_ens_cos01e50_argmax   argmax_conf   {cos_s0 soup, cos_s1 soup, e50_s0 soup}
# Launch (Henry):  tmux new-session -d -s fdens "bash <this file> >> <state>/phase5.log 2>&1"
# Idempotent: finished tags (report.json present) are skipped. Per-tag logs: exp/eval_pb/<tag>/{run.log,chain.out}.
set -uo pipefail
F=/home/exouser/henry/Qwen_Drive/frontdrive; E=/home/exouser/henry/Qwen_Drive/benchmarks/navsim/exp; ST=/home/exouser/henry/Qwen_Drive/state
log() { echo "[ENS][$(date -u +%FT%TZ)] $*"; }
C0=$E/pb_e50_cos_s0/exports/soup_16_20.ckpt; C1=$E/pb_e50_cos_s1/exports/soup_16_20.ckpt; B0=$E/pb_e50_s0/exports/soup_16_20.ckpt
for c in "$C0" "$C1" "$B0"; do test -f "$c" || { log "missing member checkpoint $c -> stop"; exit 1; }; done
WORKERS=${WORKERS:-4}
if [ ! -f "$E/eval_pb/ens_smoke_cos01_first/report.json" ]; then
  log "smoke start (16 scenes, mode first, 2 workers)"
  mkdir -p "$E/eval_pb/ens_smoke_cos01_first"
  SMOKE_SCENES=16 WORKERS=2 bash "$F/scripts/h100/eval_pb_ensemble.sh" ens_smoke_cos01_first first "$C0" "$C1" > "$E/eval_pb/ens_smoke_cos01_first/chain.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$E/eval_pb/ens_smoke_cos01_first/report.json" ]; then
    log "smoke FAILED rc=$rc -> stop; last lines:"; tail -n 15 "$E/eval_pb/ens_smoke_cos01_first/chain.out"; exit 2
  fi
  log "smoke ok: $(tail -n 1 "$E/eval_pb/ens_smoke_cos01_first/chain.out" | cut -c1-200)"
else
  log "smoke report exists -> skip"
fi
run() {
  local tag=$1 mode=$2; shift 2
  if [ -f "$E/eval_pb/$tag/report.json" ]; then log "$tag report exists -> skip"; return 0; fi
  mkdir -p "$E/eval_pb/$tag"
  log "EVAL $tag ($mode, $# members, $WORKERS workers) start"
  WORKERS=$WORKERS bash "$F/scripts/h100/eval_pb_ensemble.sh" "$tag" "$mode" "$@" > "$E/eval_pb/$tag/chain.out" 2>&1
  local r=$?
  local summary; summary=$(python3 -c "import json;r=json.load(open('$E/eval_pb/$tag/report.json'));print('PDMS',r['pdms'],'valid',r['n_valid'],'/',r['n_scenes'])" 2>/dev/null || echo "no report")
  log "EVAL $tag EXIT $r $summary"
}
run pb_ens_cos01_argmax argmax_conf "$C0" "$C1"
run pb_ens_cos01_meanclose mean_if_close "$C0" "$C1"
run pb_ens_cos01e50_argmax argmax_conf "$C0" "$C1" "$B0"
log "ensemble chain done"
