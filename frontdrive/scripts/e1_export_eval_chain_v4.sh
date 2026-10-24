#!/usr/bin/env bash
# Committed record of scratchpad/e1_export_eval_chain_v4.sh (2026-10-16 session); the live chains run from the scratchpad copy.
# v4.1 (2026-10-16 11:15): + 90 s yield after each evaluation (see the end of the loop).
# v4 (2026-10-16): identical to v3 except the GPU-window wait is 72 h (864 x 5 min) instead of 24 h, because the post-E1 queue can pack 2 fine-tunes + 1 pretraining arm for up to ~20 h, during which no 8.5 GB window exists.
# Generic E1 export -> dev-eval chain, v3.  Usage: e1_export_eval_chain_v3.sh <run_dir> <tag_prefix> <epoch> [<epoch> ...]
# v3 (2026-10-15): the v1/v2 "no other eval running" test was `pgrep -f "[e]val_perception_free.sh"`, which matched the
# launcher shell of the v1 chain itself (its command line contained the chain script as a heredoc) -> both chains waited
# for ~10 h.  v3 serialises evaluations with an atomic mkdir lock (stale if the owner pid is dead) and anchors the trainer
# liveness test to the real python command line.  Launch it as `bash <this file> <args>` only (no heredoc in the launcher).
# For each epoch: wait for <run_dir>/exports/epoch_<N>.ckpt (give up if the trainer vanished), take the lock, wait for a GPU
# window (eval peak 5.4 GB -> U-01: free >= 8.5 GB on two consecutive 5-min checks), run scripts/eval_perception_free.sh
# <tag_prefix>_ep<N> on the official val logs, release the lock, print the report line.
cd /home/henry/Desktop/Qwen_Drive/frontdrive
R=$1; P=$2; shift 2
LOCK=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/eval/.eval_lock
TOTAL=32607
trainer_alive() { pgrep -f "^python .*run_training.py.*experiment_name=$(basename "$R")" > /dev/null; }
release() { [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; }
trap release EXIT
until [ -f "$R/train.log" ]; do sleep 600; done
for N in "$@"; do
  X=$R/exports/epoch_$N.ckpt
  while [ ! -f "$X" ]; do
    if ! trainer_alive; then
      sleep 120; [ -f "$X" ] && break
      echo "TRAINER GONE before epoch_$N export at $(date '+%m-%d %H:%M')"
      tr '\r' '\n' < "$R/train.log" | grep -iE "Traceback|Error" | tail -3 | cut -c1-200
      tr '\r' '\n' < "$R/train.log" | grep -E "^Epoch [0-9]+:" | tail -1 | cut -c1-140
      exit 0
    fi
    sleep 600
  done
  echo "epoch_$N export ready at $(date '+%m-%d %H:%M')"
  while ! mkdir "$LOCK" 2>/dev/null; do
    opid=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$opid" ] && ! kill -0 "$opid" 2>/dev/null; then echo "removing stale lock of pid $opid"; rm -rf "$LOCK"; continue; fi
    sleep 120
  done
  echo $$ > "$LOCK/pid"
  ok=0
  for i in $(seq 1 864); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); free=$((TOTAL-used))
    if [ "$free" -ge 8500 ]; then ok=$((ok+1)); else ok=0; fi
    if [ $ok -ge 2 ]; then break; fi
    sleep 300
  done
  if [ $ok -lt 2 ]; then release; echo "no 8.5 GB window within 72 h for epoch_$N eval -> skipped"; continue; fi
  echo "launching eval ${P}_ep$N at $(date '+%m-%d %H:%M') (free ${free} MiB)"
  bash scripts/eval_perception_free.sh ${P}_ep$N "$X" navtrain_val_official > /home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/eval/${P}_ep$N.driver.out 2>&1
  rc=$?
  release
  echo "EVAL ${P}_ep$N EXIT $rc at $(date '+%m-%d %H:%M')"
  cat /home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/eval/${P}_ep$N/report.json 2>/dev/null | tr -d '\n' | cut -c1-600; echo
  sleep 90   # v4.1 (2026-10-16 11:15): yield the lock for 90 s before re-taking it, so the GPU queue (polls every 20 s) and other chains can get in between back-to-back evaluations
done
echo "EXPORT-EVAL CHAIN $P DONE $(date '+%m-%d %H:%M')"
