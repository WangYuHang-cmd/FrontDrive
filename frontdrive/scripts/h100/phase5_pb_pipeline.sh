#!/usr/bin/env bash
# Phase 5 ON THE SERVER (tmux `fdpb`): perception-based pipeline of plan v4.3 §2 ②/③ in one resumable chain.
#   1. pb training cache  (scripts/h100/build_pb_cache.sh: derived from the JPEG compact cache + log pickles, CPU only)
#   2. training smoke     (scripts/h100/smoke_pb_train.sh 64 96 128: s/step, peak VRAM, OOM; picks the baseline knobs)
#   3. training           (scripts/h100/train_pb.sh $PB_TAG: effective batch 64 = 64 x 1, 20 epochs, ACT_CKPT from the smoke;
#                          PB_PRETRAIN_PT="" = upstream e50 init (②); a C'/D' export = ③)
#   4. dev evals          (scripts/h100/eval_pb.sh <tag> <ckpt> navtrain_val_official 4, PB_SIDE_CALIB=1) of every export and
#                          of the official pb v1 ckpt (its dev reference value), dispatched while training runs whenever
#                          the job fits under scripts/h100/gpu_policy.sh (no self-cap since 2026-10-18) and load1 <= LOAD_MAX; drained afterwards.
# Idempotent: each stage is skipped when its product exists; training resumes from checkpoints/last.ckpt (server-side
# auto-resume <= MAX_RETRY, plan H.10 ②). Markers (README_SHARED_GPU.md): state/STOP -> our trainer gets SIGTERM within
# ~30 s and nothing new starts; state/PAUSE -> nothing new starts. Removing them resumes automatically.
# Launch:  tmux new-session -d -s fdpb "bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/phase5_pb_pipeline.sh >> /home/henry/Desktop/Qwen_Drive/state/phase5.log 2>&1"
# Knobs:   PB_TAG=pb_e50_s0 PB_SEED=0 PB_PRETRAIN_PT= PB_EPOCHS=20 PB_WORKERS=6 CACHE_WORKERS=12 CAP_MIB=60000 EVAL_MIB=13000 MAX_RETRY=2 SKIP_SMOKE=0
#          PB_ACT_CKPT=1 (default: activation checkpointing on = 12 GB/run so several runs share the GPU; 0 = 41 GB; pick = smoke pick)
#          Several chains with different PB_TAGs may run at once (own tmux session each; a per-TAG lock refuses duplicates).
#          PB_EXPORT_EVERY=5 PB_EXPORT_FROM=16 (exports at every N-th epoch and at every epoch >= FROM; each export costs one
#          dev eval of ~1 GPU-hour on the CPU-bound scorer -> the ③ encoder swaps use PB_EXPORT_EVERY=20 = late window only)
#          ⑤ (2026-10-22): PB_LR / PB_INIT_CKPT / PB_TRAIN_LOGS / PB_LR_SCHED / PB_LR_MIN_MULT / PB_LR_WARMUP reach train_pb.sh through the
#          environment; PB_SKIP_DEV_EVAL=1 = final runs on train+val (dev in-sample): no dev evals, only the soup of the exports is built
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp; DJ=$EXP/Drive-JEPA-cache
TAG=${PB_TAG:-pb_e50_s0}; SEED=${PB_SEED:-0}; PRETRAIN_PT=${PB_PRETRAIN_PT:-}
EPOCHS=${PB_EPOCHS:-20}; WORKERS=${PB_WORKERS:-6}; CACHE_WORKERS=${CACHE_WORKERS:-12}
source "$F/scripts/h100/gpu_policy.sh"   # CAP_MIB / FRIEND_RESERVE_MIB / OOM_MARGIN_MIB / LOAD_MAX / free_for_us (Henry 2026-10-18: no self-cap)
EVAL_MIB=${EVAL_MIB:-13000}; MAX_RETRY=${MAX_RETRY:-2}
PBC=$EXP/train_drive_jepa_perception_based_cache; SM=$EXP/pb_smoke; OUT=$EXP/$TAG
log() { echo "[$(date -u +%FT%TZ)] [$TAG] $*"; }
# one chain per TAG: several chains may run side by side (e.g. ② in tmux fdpb and ③-C' in another session while the
# GPU has room), but never two for the same run dir
mkdir -p "$OUT"; exec 9>"$OUT/.phase5.lock"
if ! flock -n 9; then echo "[$(date -u +%FT%TZ)] [$TAG] another phase5 chain holds $OUT/.phase5.lock -> exiting (nothing to do)"; exit 0; fi
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
ours_mib() {  # MiB held by compute processes whose cwd is under our tree (everything else counts as the other user's)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid mem; do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in /home/exouser/henry*|/home/henry/Desktop/Qwen_Drive*) echo "$mem";; esac
  done | awk '{s+=$1} END{print s+0}'
}
load1() { awk '{print int($1)}' /proc/loadavg; }
markers() { ls "$S/STOP" "$S/PAUSE" 2>/dev/null | xargs -rn1 basename | tr '\n' ' '; }
wait_marker() { while [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "marker present ($(markers)) -> waiting"; sleep 60; done; }

log "phase5 start: TAG=$TAG SEED=$SEED EPOCHS=$EPOCHS WORKERS=$WORKERS pretrain=${PRETRAIN_PT:-e50} CAP_MIB=$CAP_MIB host=$(hostname)"
log "gpu used/total: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1); disk: $(df -h / | awk 'NR==2{print $4" free"}'); load1 $(load1)"
test -f "$F/env/.use_conda_cvpr" || { log "env/.use_conda_cvpr missing: run scripts/h100/bootstrap_conda_cvpr.sh first"; exit 2; }
if [ -n "$PRETRAIN_PT" ] && [ ! -f "$PRETRAIN_PT" ]; then log "PB_PRETRAIN_PT=$PRETRAIN_PT not found"; exit 2; fi
# reboot resilience: boot.sh (user crontab @reboot) relaunches every chain whose marker exists (one marker per TAG, knobs inside)
printf '# phase5 chain active since %s\nPB_TAG=%s\nPB_SEED=%s\nPB_PRETRAIN_PT=%s\nPB_EPOCHS=%s\nPB_WORKERS=%s\nCAP_MIB=%s\nPB_EXPORT_EVERY=%s\nPB_EXPORT_FROM=%s\nPB_ACT_CKPT=%s\n' \
  "$(date -u +%FT%TZ)" "$TAG" "$SEED" "$PRETRAIN_PT" "$EPOCHS" "$WORKERS" "$CAP_MIB" "${PB_EXPORT_EVERY:-5}" "${PB_EXPORT_FROM:-16}" "${PB_ACT_CKPT:-1}" > "$S/phase5.enabled.$TAG"
env | grep -E '^PB_(LR|LR_SCHED|LR_MIN_MULT|LR_WARMUP|INIT_CKPT|TRAIN_LOGS|SKIP_DEV_EVAL)=' >> "$S/phase5.enabled.$TAG" || true   # ④(b)/⑤ knobs survive a reboot (boot.sh relaunch)

# ---- 1. pb training cache (CPU; resumable; ~12 GB)
if [ -f "$PBC/.phase5_cache_ok" ]; then
  log "stage cache: done ($PBC)"
else
  for attempt in 1 2; do
    log "stage cache: build_pb_cache.sh $CACHE_WORKERS (attempt $attempt; log state/pb_cache_build.log)"
    bash "$F/scripts/h100/build_pb_cache.sh" "$CACHE_WORKERS" >> "$S/pb_cache_build.log" 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    log "cache build rc=$rc: $(tail -n 2 "$S/pb_cache_build.log" | tr '\n' ' ' | cut -c1-300)"
  done
  [ $rc -eq 0 ] || { log "cache build failed twice -> abort"; exit 3; }
  touch "$PBC/.phase5_cache_ok"; log "stage cache: DONE $(du -sh "$PBC" 2>/dev/null | cut -f1)"
fi

# ---- 2. training smoke (needs a mostly free GPU for the OOM boundary to mean anything)
if [ "${SKIP_SMOKE:-0}" = 1 ] && [ -f "$SM/pick.json" ]; then log "stage smoke: skipped (SKIP_SMOKE=1, pick.json exists)"; fi
if [ ! -f "$SM/pick.json" ]; then
  t0=$(date +%s)
  while [ "$(gpu_used)" -gt 4000 ] && [ $(( $(date +%s) - t0 )) -lt 3600 ]; do log "stage smoke: waits for the GPU (used $(gpu_used) MiB)"; sleep 120; done
  wait_marker
  log "stage smoke: smoke_pb_train.sh 64 96 128 (log state/pb_smoke.log)"
  bash "$F/scripts/h100/smoke_pb_train.sh" 64 96 128 >> "$S/pb_smoke.log" 2>&1; log "stage smoke: rc=$?"
  [ -f "$SM/pick.json" ] || { log "no $SM/pick.json -> abort (see state/pb_smoke.log)"; exit 4; }
fi
ACT_PICK=$(python3 -c "import json; d=json.load(open('$SM/pick.json')); b=d.get('baseline_recommendation'); print(b['ACT_CKPT'] if b else 'none')")
if [ "$ACT_PICK" = none ]; then log "smoke found no baseline fitting under $CAP_MIB MiB -> abort ($SM/summary.tsv)"; exit 4; fi
log "stage smoke: pick $(python3 -c "import json; d=json.load(open('$SM/pick.json')); print(json.dumps({k: d[k] for k in ('largest_micro_batch_fitting','largest_micro_batch_peak_mib','baseline_recommendation')}))")"
# Activation checkpointing: the smoke showed the step is CPU-bound (same 2.86 s/step with it on or off), so keeping the
# activations (41 GB) buys no speed; the memory is better spent on a SECOND training beside the first (Henry 2026-10-17:
# use the spare GPU). Default therefore 1 (12 GB per run, identical math); PB_ACT_CKPT=0 or =pick to override.
case "${PB_ACT_CKPT:-1}" in pick) ACT=$ACT_PICK;; *) ACT=${PB_ACT_CKPT:-1};; esac
log "activation checkpointing for this run: ACT_CKPT=$ACT (smoke pick was $ACT_PICK; PB_ACT_CKPT=${PB_ACT_CKPT:-1})"

# ---- 4 (dispatcher). dev evals: official pb v1 ckpt reference first, then every export, one at a time
EVAL_PID=""; EVAL_TAG=""; declare -A FAILS
eval_items() {
  # PB_SKIP_DEV_EVAL=1 (⑤ final runs on train+val: dev is in the training set) -> build the soup of the exports, evaluate nothing on dev
  local skip=${PB_SKIP_DEV_EVAL:-0}
  [ "$skip" = 1 ] || echo "pb_official_v1_dev $DJ/drive_jepa_perception_based_agent_vitl.ckpt"
  if [ "$skip" != 1 ]; then for c in $(ls "$OUT"/exports/epoch_*.ckpt 2>/dev/null | sort -V); do n=$(basename "$c" .ckpt); echo "${TAG}_ep${n#epoch_} $c"; done; fi
  # after training: the Lightning best val/score_epoch checkpoint (dev-selected like the released weights) as <TAG>_best<N>,
  # and the weight average of the late-window exports (④ finding 2026-10-19: +0.39 over the late average, no selection) as <TAG>_soup<a>_<b>
  if train_done; then
    if [ "$skip" != 1 ]; then for c in "$OUT"/checkpoints/best_epoch=*.ckpt; do [ -f "$c" ] || continue; n=$(basename "$c" .ckpt); echo "${TAG}_best${n#best_epoch=} $c"; done; fi
    local a=${PB_EXPORT_FROM:-16} b=$EPOCHS soup=$OUT/exports/soup_${PB_EXPORT_FROM:-16}_$EPOCHS.ckpt members=()
    for e in $(seq "$a" "$b"); do [ -f "$OUT/exports/epoch_$e.ckpt" ] && members+=("$OUT/exports/epoch_$e.ckpt"); done
    if [ ! -f "$soup" ] && [ ${#members[@]} -ge 2 ] && [ ! -f "$soup.failed" ]; then
      ( source "$F/env/drivejepa_pb_env.sh" && python "$F/eval/soup_exports.py" --out "$soup" "${members[@]}" ) > "$OUT/soup.log" 2>&1 || touch "$soup.failed"
    fi
    [ -f "$soup" ] && [ "$skip" != 1 ] && echo "${TAG}_soup${a}_${b} $soup"
  fi
  return 0
}
pending_evals() { eval_items | while read -r t c; do [ -f "$EXP/eval_pb/$t/report.json" ] || [ ! -f "$c" ] || [ "${FAILS[$t]:-0}" -ge 2 ] || echo "$t"; done | wc -l; }
dispatch_evals() {
  if [ -n "$EVAL_PID" ]; then
    kill -0 "$EVAL_PID" 2>/dev/null && return
    wait "$EVAL_PID" 2>/dev/null; rc=$?
    if [ -f "$EXP/eval_pb/$EVAL_TAG/report.json" ]; then log "EVAL $EVAL_TAG EXIT $rc $(tr -d '\n' < "$EXP/eval_pb/$EVAL_TAG/report.json" | cut -c1-400)"
    else FAILS[$EVAL_TAG]=$(( ${FAILS[$EVAL_TAG]:-0} + 1 )); log "EVAL $EVAL_TAG EXIT $rc without report (failure ${FAILS[$EVAL_TAG]}/2; state/eval_$EVAL_TAG.log)"; fi
    EVAL_PID=""; EVAL_TAG=""
  fi
  { [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; } && return
  local used ours; used=$(gpu_used); ours=$(ours_mib)
  free_for_us "$EVAL_MIB" "$used" "$ours" || return
  [ "$(load1)" -le "$LOAD_MAX" ] || return
  while read -r t c; do
    [ -f "$EXP/eval_pb/$t/report.json" ] && continue
    [ -f "$c" ] || continue
    [ "${FAILS[$t]:-0}" -ge 2 ] && continue
    log "EVAL $t START ckpt=$c (gpu used $used MiB, ours $ours MiB, load1 $(load1))"
    PB_SIDE_CALIB=1 nice -n 5 bash "$F/scripts/h100/eval_pb.sh" "$t" "$c" navtrain_val_official 4 > "$S/eval_$t.log" 2>&1 &
    EVAL_PID=$!; EVAL_TAG=$t; return
  done < <(eval_items)
}

# ---- 3. training (effective batch 64 = 64 x 1; VRAM knob = ACT_CKPT from the smoke); STOP -> SIGTERM to the trainer
train_done() { [ -f "$OUT/exports/epoch_$EPOCHS.ckpt" ]; }
run_train() {
  local resume=""; [ -f "$OUT/checkpoints/last.ckpt" ] && resume=$OUT/checkpoints/last.ckpt
  log "train $TAG seed=$SEED epochs=$EPOCHS BATCH=64 ACCUM=1 ACT_CKPT=$ACT WORKERS=$WORKERS pretrain=${PRETRAIN_PT:-e50} init_ckpt=${PB_INIT_CKPT:-none} lr=${PB_LR:-1e-4} sched=${PB_LR_SCHED:-none} train_logs=${PB_TRAIN_LOGS:-train} resume=${resume:-none} (monitor lines -> $OUT/monitor.log, full log $OUT/train.log)"
  # train_pb.sh tees the trainer into $OUT/train.log itself and writes its own monitor lines to $OUT/monitor.log; its stdout
  # (the \r progress flood) is dropped here (an earlier grep filter produced a 16 MB copy / lost lines glued to progress fragments)
  env BATCH=64 ACCUM=1 EPOCHS="$EPOCHS" WORKERS="$WORKERS" ACT_CKPT="$ACT" PRETRAIN_PT="$PRETRAIN_PT" RESUME="$resume" PB_VRAM_SAMPLE_S=30 \
      EXPORT_EVERY="${PB_EXPORT_EVERY:-5}" EXPORT_FROM="${PB_EXPORT_FROM:-16}" \
      bash "$F/scripts/h100/train_pb.sh" "$TAG" "$SEED" > /dev/null 2>> "$OUT/train_pb.stderr" &
  local wpid=$! tp rc
  while kill -0 "$wpid" 2>/dev/null; do
    if [ -f "$S/STOP" ]; then
      tp=$(cat "$OUT/train.pid" 2>/dev/null || true)
      if [ -n "$tp" ] && kill -0 "$tp" 2>/dev/null; then kill -TERM "$tp" 2>/dev/null && log "STOP present: SIGTERM sent to trainer pid $tp"; fi
    fi
    dispatch_evals
    sleep 30
  done
  wait "$wpid"; rc=$?
  # the wrapper may die while its trainer lives on (seen 2026-10-18: SIGPIPE in its monitor loop) -> never relaunch over a
  # live trainer: keep waiting for the trainer pid, keep dispatching evals and honouring STOP, then report by the exports
  tp=$(cat "$OUT/train.pid" 2>/dev/null || true)
  if [ -n "$tp" ] && kill -0 "$tp" 2>/dev/null; then
    log "wrapper exited rc=$rc but trainer pid $tp is alive -> waiting for it (no relaunch)"
    while kill -0 "$tp" 2>/dev/null; do
      [ -f "$S/STOP" ] && kill -TERM "$tp" 2>/dev/null && log "STOP present: SIGTERM sent to trainer pid $tp"
      dispatch_evals; sleep 30
    done
    train_done && rc=0
  fi
  return "$rc"
}
tries=0
until train_done; do
  wait_marker
  run_train; rc=$?
  if train_done; then log "train $TAG finished rc=$rc: $(tail -n 1 "$OUT/monitor.log" 2>/dev/null | cut -c1-300)"; break; fi
  if [ -f "$S/STOP" ]; then log "train $TAG stopped by STOP (rc=$rc); resumes from last.ckpt once STOP is removed"; sleep 30; continue; fi
  tries=$((tries + 1))
  if [ $tries -gt "$MAX_RETRY" ]; then log "train $TAG failed rc=$rc after $MAX_RETRY retries -> giving up ($OUT/monitor.log, $OUT/train_pb.stderr, $OUT/train.log)"; break; fi
  log "train $TAG rc=$rc -> retry $tries/$MAX_RETRY from last.ckpt in 120 s"; sleep 120
done
[ -f "$OUT/accounting.json" ] && log "accounting: $(python3 -c "import json; a=json.load(open('$OUT/accounting.json')); print(json.dumps({k: a.get(k) for k in ('effective_batch','encoder_activation_checkpointing','steps_per_epoch','peak_vram_mib_our_pid','s_per_step_last_progress','wall_s','rc','oom','exports')}))")"

# ---- drain the evals
while :; do
  dispatch_evals
  [ -z "$EVAL_PID" ] && [ "$(pending_evals)" -eq 0 ] && break
  sleep 60
done
rm -f "$S/phase5.enabled.$TAG"; touch "$OUT/.phase5.done"   # pb_queue.sh: finished (also when EPOCHS != 20)
log "phase5 done. dev reports:"
for r in "$EXP"/eval_pb/*/report.json; do
  [ -f "$r" ] || continue
  python3 -c "import json,sys; d=json.load(open('$r')); print('  ', d['tag'], d['split'], 'pdms', d.get('pdms'), 'n', d.get('n_scenes'), 'sha8', d.get('ckpt_sha8'))"
done
