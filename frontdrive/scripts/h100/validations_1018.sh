#!/usr/bin/env bash
# Spare-GPU validations of 2026-10-18 (tmux `fdval`), each a navtest evaluation that the plan's budget already allocates
# (plan v4.3 §5: total <= 8; used so far: 1 = official pb ckpt upstream path):
#   (A) protocol check  — the OFFICIAL pb v1 ckpt on navtest with side cameras taken from calibration only (PB_SIDE_CALIB=1,
#       no CAM_L0/R0/B0 pixels) vs the recorded upstream-path 93.69: proves on 12,146 scenes that the "single front camera"
#       input protocol is equivalent (budget line "协议核查").
#   (B) RUN_BEST13_NAVTEST=1 — our self-trained e50 baseline's dev-selected checkpoint (pb_e50_s0/best_epoch=13.ckpt) on
#       navtest (budget line "② 冻结后的自训 e50 基线 1"): where a pure reproduction would stand on the leaderboard.
# Both runs append to exp/eval/navtest_evals_server.csv (merged into eval/navtest_evals.csv by pull.sh). Sequential, 4 workers.
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp; DJ=$EXP/Drive-JEPA-cache
log() { echo "[$(date -u +%FT%TZ)] [val1018] $*"; }
run() {  # <tag> <ckpt> <purpose>
  if [ -f "$EXP/eval_pb/$1/report.json" ]; then log "$1: report exists -> skip"; return; fi
  while [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "marker present -> waiting"; sleep 120; done
  log "$1: START"
  PB_SIDE_CALIB=1 NAVTEST_PURPOSE="$3" nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$1" "$2" navtest 4 > "$S/eval_$1.log" 2>&1; rc=$?
  log "$1: EXIT $rc $(tr -d '\n' < "$EXP/eval_pb/$1/report.json" 2>/dev/null | cut -c1-300)"
}
run pb_official_v1_navtest_sidecalib "$DJ/drive_jepa_perception_based_agent_vitl.ckpt" \
    "protocol check: official pb v1 ckpt with calibration-only side cams vs upstream path 93.69 (plan §5 reserve)"
if [ "${RUN_BEST13_NAVTEST:-0}" = 1 ]; then
  run pb_e50_s0_best13_navtest "$EXP/pb_e50_s0/checkpoints/best_epoch=13.ckpt" \
      "self-trained e50 baseline, dev-selected best ckpt (plan §5 item ②: frozen baseline, 1 eval)"
fi
log "done"
