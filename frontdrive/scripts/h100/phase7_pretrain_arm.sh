#!/usr/bin/env bash
# Phase 7 ON THE SERVER (tmux `fdpretrain`): one V-JEPA continued-pretraining arm at H100 scale -> EMA encoder export -> the
# pb training of that encoder queued automatically (plan v4.3 §7 "后移的研究臂（仅空闲窗口）" = the ④ scale test after ③ came
# out negative; Henry 2026-10-19: use the idle GPU). GPU-heavy, CPU-light (8 decord workers) -> runs beside the pb trainers.
#   usage: bash scripts/h100/phase7_pretrain_arm.sh <config.yaml> <export_name> [seed=0]
#          e.g. pretrain/configs/e2_Cp4_comma_plain_from_e50_b32.yaml Cp4
#   1. 60-step smoke in <folder>_smoke (s/step, peak VRAM; SKIP_SMOKE=1 to skip)      -> state/pretrain.log
#   2. full run, auto-resume from <folder>/latest.pt (<= MAX_RETRY), state/STOP -> SIGINT (train_motion saves + exits
#      'aborted'), resumes when STOP is removed; VRAM sampled to <folder>/gpu_mem.csv
#   3. pretrain/export_target_encoder.py <folder>/latest.pt -> benchmarks/pretrain/exports/<run>_e40_target_encoder.pt
#   4. appends "pb_<export_name>_s<seed> <export> <seed>" to state/pb_queue.txt (pb_queue.sh launches it when a slot is free)
# Software: the uv venv (torch 2.8 / decord / av, FRONTDRIVE_USE_VENV=1) the arms were developed with on the 5090.
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; PX=$QD/benchmarks/pretrain/exports
CFG=${1:?config yaml (path relative to frontdrive or absolute)}; NAME=${2:?export name, e.g. Cp4}; SEED=${3:-0}
MAX_RETRY=${MAX_RETRY:-2}; SAMPLE_S=${SAMPLE_S:-30}
export FRONTDRIVE_USE_VENV=1 TZ=America/New_York WANDB_MODE=disabled OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
source "$F/env/drivejepa_env.sh"
cd "$F"; export PYTHONPATH=$F:${PYTHONPATH:-}
log() { echo "[$(date -u +%FT%TZ)] [pretrain $NAME] $*"; }
[ -f "$CFG" ] || { log "no config $CFG"; exit 2; }
read -r FOLDER EPOCHS IPE < <(python - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); print(c["folder"], c["optimization"]["epochs"], c["optimization"]["ipe"])
PY
)
RUN=$(basename "$FOLDER"); EXPECT=$((EPOCHS * IPE)); EXPORT=$PX/${RUN}_e40_target_encoder.pt
mkdir -p "$FOLDER" "$PX" "$S/smoke_cfg"
printf '# pretrain arm active since %s\nCFG=%s\nNAME=%s\nSEED=%s\n' "$(date -u +%FT%TZ)" "$CFG" "$NAME" "$SEED" > "$S/pretrain.enabled.$NAME"
log "start: cfg=$CFG run=$RUN expect $EXPECT steps ($EPOCHS x $IPE), export -> $EXPORT, python=$(command -v python) torch=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run_arm() {  # <cfg> <folder>: runs train_motion, samples VRAM, honours STOP; returns the trainer's rc
  local cfg=$1 out=$2 pid ours
  mkdir -p "$out"
  python -m pretrain.train_motion --fname "$cfg" --devices cuda:0 >> "$out/train.out" 2>&1 &
  pid=$!; echo "$pid" > "$out/train.pid"
  set +e
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    ours=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$pid" '$1==p{print $2}' | sed -n 1p)
    echo "$(date +%s),${ours:-0}" >> "$out/gpu_mem.csv"
    if [ -f "$S/STOP" ]; then kill -INT "$pid" 2>/dev/null && log "STOP present: SIGINT sent to pid $pid"; fi
    if [ $((i % 20)) -eq 0 ]; then log "vram ${ours:-0} MiB | $(tail -n 1 "$out/log_r0.csv" 2>/dev/null | cut -d, -f1-3,8-10)"; fi
    i=$((i + 1)); sleep "$SAMPLE_S"
  done
  wait "$pid"; local rc=$?
  return $rc
}
status_of() { python - "$1" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "accounting.json")
try:
    a = json.load(open(p)); print(a.get("exit_status", "?"), int(a.get("steps", 0)), int(a.get("wall_s", 0)))
except Exception:
    print("none 0 0")
PY
}
peak_of() { awk -F, '$2+0>m{m=$2+0} END{print m+0}' "$1/gpu_mem.csv" 2>/dev/null; }

# ---- 1. smoke (60 optimizer steps) in <folder>_smoke
if [ "${SKIP_SMOKE:-0}" != 1 ] && [ ! -f "$FOLDER/latest.pt" ]; then
  SCFG=$S/smoke_cfg/$(basename "${CFG%.yaml}")_smoke.yaml
  sed -e "s|^folder: .*|folder: ${FOLDER}_smoke|" -e "s|^  max_steps: .*|  max_steps: 60|" "$CFG" > "$SCFG"
  rm -rf "${FOLDER:?}_smoke"
  log "smoke: 60 steps with $SCFG"
  run_arm "$SCFG" "${FOLDER}_smoke"; rc=$?
  read -r st steps wall < <(status_of "${FOLDER}_smoke")
  log "smoke: rc=$rc exit_status=$st steps=$steps wall=${wall}s -> $(python -c "print(f'{$wall/max($steps,1):.2f} s/step')") peak $(peak_of "${FOLDER}_smoke") MiB"
  [ "$st" = ok ] && [ "$steps" -ge 60 ] || { log "smoke failed -> abort (see ${FOLDER}_smoke/train.out)"; exit 3; }
fi

# ---- 2. full run with resume
tries=0
while :; do
  read -r st steps wall < <(status_of "$FOLDER")
  if [ "$st" = ok ] && [ "$steps" -ge "$EXPECT" ]; then log "run complete: $steps steps, wall $((wall / 3600)) h"; break; fi
  while [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "marker present -> waiting"; sleep 120; done
  log "train attempt $((tries + 1)) (resume from latest.pt if present: $([ -f "$FOLDER/latest.pt" ] && echo yes || echo no))"
  run_arm "$CFG" "$FOLDER"; rc=$?
  read -r st steps wall < <(status_of "$FOLDER")
  log "trainer exit rc=$rc exit_status=$st steps=$steps peak $(peak_of "$FOLDER") MiB"
  if [ "$st" = ok ] && [ "$steps" -ge "$EXPECT" ]; then continue; fi
  if [ -f "$S/STOP" ]; then log "stopped by STOP -> will resume"; sleep 60; continue; fi
  if [ "$st" = nan_loss ]; then log "nan loss -> giving up"; exit 4; fi
  tries=$((tries + 1)); [ $tries -le "$MAX_RETRY" ] || { log "giving up after $MAX_RETRY retries"; exit 4; }
  sleep 120
done

# ---- 3. export the EMA target encoder
if [ ! -f "$EXPORT" ]; then
  log "export: $FOLDER/latest.pt -> $EXPORT"
  python pretrain/export_target_encoder.py "$FOLDER/latest.pt" "$EXPORT" >> "$S/pretrain.log" 2>&1 || { log "export failed"; exit 5; }
fi
log "export ready: $(sha256sum "$EXPORT" | cut -c1-12) $(du -h "$EXPORT" | cut -f1)"

# ---- 4. queue the pb training of this encoder
TAG=pb_${NAME}_s${SEED}
if ! grep -qE "^$TAG " "$S/pb_queue.txt" 2>/dev/null; then
  printf '%s %s %s\n' "$TAG" "$EXPORT" "$SEED" >> "$S/pb_queue.txt"; log "queued $TAG (pb_queue.txt)"
else log "$TAG already queued"; fi
rm -f "$S/pretrain.enabled.$NAME"
log "done"
