#!/usr/bin/env bash
# ⑤ final candidates -> navtest, ONE evaluation per seed (plan v4.3 §5 budget: 4 used before -> 7 of 8 after), launched by Henry:
#   tmux new-session -d -s fdnt "bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/navtest_finals_1023.sh >> /home/exouser/henry/Qwen_Drive/state/phase5.log 2>&1"
# For each seed it waits for exp/pb_off_ft_all_s<N>/exports/soup_1_5.ckpt and the chain's .phase5.done marker, waits for CPU room
# (load1 <= LOAD_MAX, no STOP/PAUSE), then runs eval_pb.sh <tag> <soup> navtest 4 with NAVTEST_PURPOSE set (the budget ledger line).
# Nothing here selects a seed on navtest: all three are evaluated and reported (multi-seed leaderboard submission). The navtest
# reports are never ingested as dev rows (split navtest); pull.sh merges eval/navtest_evals_server.csv into eval/navtest_evals.csv.
# NT_INCLUDE_PROBE=1 additionally evaluates the dev-verifiable probe soup (pb_off_ft_s0, train logs only) -> 8 of 8: it isolates the
# "+val logs" effect (finals - probe) from the "continued training" effect (probe - official 93.69). Off by default (last budget slot).
set -uo pipefail
F=/home/exouser/henry/Qwen_Drive/frontdrive; E=/home/exouser/henry/Qwen_Drive/benchmarks/navsim/exp; S=/home/exouser/henry/Qwen_Drive/state
source "$F/scripts/h100/gpu_policy.sh"
log() { echo "[NT][$(date -u +%FT%TZ)] $*"; }
load1() { awk '{print int($1)}' /proc/loadavg; }
run_navtest() {  # <tag> <ckpt> <run dir> <purpose>
  local TAG=$1 CK=$2 RUN=$3 PURPOSE=$4 rc
  if [ -f "$E/eval_pb/$TAG/report.json" ]; then log "$TAG: report exists -> skip"; return 0; fi
  while [ ! -f "$CK" ] || [ ! -f "$RUN/.phase5.done" ]; do log "$TAG: waiting for $CK and $RUN/.phase5.done"; sleep 600; done
  while [ "$(load1)" -gt "$LOAD_MAX" ] || [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "$TAG: load1 $(load1) > $LOAD_MAX or STOP/PAUSE -> wait"; sleep 300; done
  mkdir -p "$E/eval_pb/$TAG"
  log "EVAL $TAG start (navtest, 4 workers, load1 $(load1), sha8 $(sha256sum "$CK" | cut -c1-8))"
  NAVTEST_PURPOSE="$PURPOSE" PB_SIDE_CALIB=1 nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$TAG" "$CK" navtest 4 > "$E/eval_pb/$TAG/chain.out" 2>&1
  rc=$?
  if [ -f "$E/eval_pb/$TAG/report.json" ]; then
    log "EVAL $TAG EXIT $rc $(python3 -c "import json; d=json.load(open('$E/eval_pb/$TAG/report.json')); print('PDMS', d.get('pdms'), 'valid', d.get('n_valid'), '/', d.get('n_scenes'), 'sha8', d.get('ckpt_sha8'))")"
  else log "EVAL $TAG EXIT $rc without report -> $E/eval_pb/$TAG/chain.out"; fi
}
log "navtest finals chain start (seeds 0 1 2, soup_1_5 each; LOAD_MAX=$LOAD_MAX; include probe: ${NT_INCLUDE_PROBE:-0})"
for i in 0 1 2; do
  run_navtest "pb_off_ft_all_s${i}_soup1_5_navtest" "$E/pb_off_ft_all_s$i/exports/soup_1_5.ckpt" "$E/pb_off_ft_all_s$i" \
    "plan v4.3 ⑤ final candidate seed $i: released pb v1 weights continued 5 ep (Adam 3e-5 cosine, warm-up 200) on all 1,192 navtrain logs, soup_1_5; 1 of 3 seeds, all reported, none selected on navtest (budget 4 -> 7 of 8)"
done
if [ "${NT_INCLUDE_PROBE:-0}" = 1 ]; then
  run_navtest "pb_off_ft_s0_soup1_5_navtest" "$E/pb_off_ft_s0/exports/soup_1_5.ckpt" "$E/pb_off_ft_s0" \
    "plan v4.3 ⑤ probe (train logs only, same recipe as the finals), soup_1_5: isolates the +val-logs effect of the finals (8 of 8)"
fi
log "navtest finals chain done"
