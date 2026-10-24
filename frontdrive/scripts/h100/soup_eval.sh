#!/usr/bin/env bash
# Zero-training-cost candidate: average the late-window fp16 exports of a pb run (eval/soup_exports.py) and dev-evaluate the
# average as <run>_soup<first>_<last> (PB_SIDE_CALIB=1). GPU ~8 GB for the eval, CPU light (one 4-process eval).
# Usage (tmux fdsoup): bash scripts/h100/soup_eval.sh <run> <epoch> [epoch ...]      e.g. pb_e50_s0 16 17 18 19 20
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp
RUN=${1:?run}; shift; EPOCHS=("$@"); [ ${#EPOCHS[@]} -ge 2 ] || { echo "give >= 2 epochs"; exit 2; }
log() { echo "[$(date -u +%FT%TZ)] [soup] $*"; }
TAG=${RUN}_soup${EPOCHS[0]}_${EPOCHS[-1]}
OUT=$EXP/$RUN/exports/soup_${EPOCHS[0]}_${EPOCHS[-1]}.ckpt
if [ -f "$EXP/eval_pb/$TAG/report.json" ]; then log "$TAG: report exists -> nothing to do"; exit 0; fi
FILES=(); for e in "${EPOCHS[@]}"; do f=$EXP/$RUN/exports/epoch_$e.ckpt; while [ ! -f "$f" ]; do log "$TAG: waiting for $f"; sleep 600; done; FILES+=("$f"); done
if [ ! -f "$OUT" ]; then
  log "$TAG: averaging ${#FILES[@]} exports -> $OUT"
  ( source "$F/env/drivejepa_pb_env.sh" && python "$F/eval/soup_exports.py" --out "$OUT" "${FILES[@]}" ) || { log "soup failed"; exit 3; }
fi
while [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "marker present -> waiting"; sleep 120; done
log "$TAG: START eval"
PB_SIDE_CALIB=1 nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$TAG" "$OUT" navtrain_val_official 4 > "$S/eval_$TAG.log" 2>&1; rc=$?
log "$TAG: EXIT $rc $(tr -d '\n' < "$EXP/eval_pb/$TAG/report.json" 2>/dev/null | cut -c1-300)"
