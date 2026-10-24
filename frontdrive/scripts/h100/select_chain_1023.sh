#!/usr/bin/env bash
# ④(c) training-free selection rules on the dev-clean probe soup (pb_off_ft_s0/exports/soup_1_5.ckpt, dev 93.22 under the model's own
# argmax). Smoke (16 scenes, mode argmax: must reproduce pb_off_ft_s0_soup1_5 per scene), then full dev evals, one at a time, each
# waiting for CPU room (load1 <= LOAD_MAX) and GPU room (gpu_policy.sh free_for_us). Launch by Henry:
#   tmux new-session -d -s fdsel "bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/select_chain_1023.sh >> /home/exouser/henry/Qwen_Drive/state/phase5.log 2>&1"
# Readout (pre-stated): a mode beats the same checkpoint's argmax per scene by >= +0.2 with CI low > 0, AND the "progressive" control
# does not also improve -> adopt the rule for the ⑤ submission (then one navtest eval of final s0 + rule = 8 of 8).
set -uo pipefail
F=/home/exouser/henry/Qwen_Drive/frontdrive; E=/home/exouser/henry/Qwen_Drive/benchmarks/navsim/exp; S=/home/exouser/henry/Qwen_Drive/state
source "$F/scripts/h100/gpu_policy.sh"; EVAL_MIB=${EVAL_MIB:-13000}
CK=$E/pb_off_ft_s0/exports/soup_1_5.ckpt
log() { echo "[SEL][$(date -u +%FT%TZ)] $*"; }
load1() { awk '{print int($1)}' /proc/loadavg; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
ours_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac; done | awk '{s+=$1} END{print s+0}'; }
wait_room() { while :; do [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ] && { log "STOP/PAUSE -> wait"; sleep 300; continue; }; [ "$(load1)" -le "$LOAD_MAX" ] || { log "load1 $(load1) > $LOAD_MAX -> wait"; sleep 300; continue; }; free_for_us "$EVAL_MIB" "$(gpu_used)" "$(ours_mib)" && return 0; log "no GPU room -> wait"; sleep 300; done; }
run() {  # <tag> <mode> <eps>
  local TAG=$1 MODE=$2 EPS=$3 rc
  if [ -f "$E/eval_pb/$TAG/report.json" ]; then log "$TAG: report exists -> skip"; return 0; fi
  wait_room; mkdir -p "$E/eval_pb/$TAG"
  log "EVAL $TAG ($MODE, eps $EPS, 4 workers, load1 $(load1)) start"
  WORKERS=4 nice -n 5 bash "$F/scripts/h100/eval_pb_select.sh" "$TAG" "$MODE" "$EPS" "$CK" > "$E/eval_pb/$TAG/chain.out" 2>&1; rc=$?
  if [ -f "$E/eval_pb/$TAG/report.json" ]; then log "EVAL $TAG EXIT $rc $(python3 -c "import json; d=json.load(open('$E/eval_pb/$TAG/report.json')); print('PDMS', d.get('pdms'), 'valid', d.get('n_valid'), '/', d.get('n_scenes'))")"
  else log "EVAL $TAG EXIT $rc without report -> $E/eval_pb/$TAG/chain.out"; fi
}
test -f "$CK" || { log "missing $CK -> stop"; exit 2; }
log "selection chain start (member $CK, sha8 $(sha256sum "$CK" | cut -c1-8))"
if [ ! -f "$E/eval_pb/sel_smoke_argmax/report.json" ]; then
  wait_room; mkdir -p "$E/eval_pb/sel_smoke_argmax"; log "smoke start (16 scenes, mode argmax, 2 workers)"
  SMOKE_SCENES=16 WORKERS=2 bash "$F/scripts/h100/eval_pb_select.sh" sel_smoke_argmax argmax 0 "$CK" > "$E/eval_pb/sel_smoke_argmax/chain.out" 2>&1; rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$E/eval_pb/sel_smoke_argmax/report.json" ]; then log "smoke FAILED rc=$rc -> stop"; tail -n 15 "$E/eval_pb/sel_smoke_argmax/chain.out"; exit 2; fi
  log "smoke ok: $(tr -d '\n' < "$E/eval_pb/sel_smoke_argmax/report.json" | cut -c1-200)"
fi
run pb_sel_offsoup_minheads minheads 0
run pb_sel_offsoup_cons002 conservative 0.02
run pb_sel_offsoup_cons005 conservative 0.05
run pb_sel_offsoup_prog002 progressive 0.02
log "selection chain done"
