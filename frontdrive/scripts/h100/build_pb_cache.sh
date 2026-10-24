#!/usr/bin/env bash
# Build the perception-based (pb) Drive-JEPA training cache WITHOUT side-camera images and without any image I/O:
# derive it from the perception-free compact JPEG cache (push step S1: identical CAM_F0 [2,3] crops, JPEG q95 512x256)
# and the navtrain log pickles (push step S8), see navsim_ext/compact_cache_pb.py. Val split first, then the train split,
# both into the upstream cache_path:
#   $NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache   (~12 GB, 103,288 tokens = 18,179 val + 85,109 train;
#   upstream <log>/<token>/{drive_jepa_feature,drive_jepa_target}.gz layout + the _COMPACT_JPEG marker that makes the
#   patched run_training.py decode the JPEGs through CompactCacheOnlyDataset)
# Idempotent / resumable (existing token dirs are skipped); needs no GPU; minutes with 8 workers (CPU + disk bound).
# Usage (server, inside tmux):  bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/build_pb_cache.sh [workers=8]
#   PB_CACHE_ENV=pf         use venv-pf instead of venv-pb (same output: numpy arrays + JPEG bytes; only the derivation
#                           engine differs: venv-pb runs the patched vendored builder code itself, venv-pf its numpy mirror)
#   PB_CACHE_OUT=<dir>      alternative output directory (then pass CACHE=<dir> to train_pb.sh)
#   PB_CACHE_ENGINE=auto    auto | builder | numpy
#   PB_CACHE_VERIFY=50      random tokens re-checked afterwards: builder mode (CompactCacheOnlyDataset vs the patched
#                           upstream builder on dev logs with CAM_F0 blobs, needs venv-pb) else pf mode; 0 = skip
set -euo pipefail
WORKERS=${1:-8}
F=/home/henry/Desktop/Qwen_Drive/frontdrive
if [ "${PB_CACHE_ENV:-pb}" = pf ]; then source "$F/env/drivejepa_env.sh"; else source "$F/env/drivejepa_pb_env.sh"; fi
COMPACT=${PB_COMPACT:-$NAVSIM_LOCAL/cache/navtrain_compact}
LOGS=$OPENSCENE_DATA_ROOT/navsim_logs/trainval
BLOBS=$OPENSCENE_DATA_ROOT/sensor_blobs/trainval
OUT=${PB_CACHE_OUT:-$NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache}
ENGINE=${PB_CACHE_ENGINE:-auto}
VERIFY_N=${PB_CACHE_VERIFY:-50}
EXPECT_LOGS=1192; EXPECT_FILES=206576; EXPECT_TOKENS=103288; EXPECT_VAL=18179; EXPECT_TRAIN=85109   # navtrain
mkdir -p "$(dirname "$OUT")"
BLOG=$OUT.build.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$BLOG"; }
if [ -f "$FRONTDRIVE/env/.use_conda_cvpr" ]; then VENV_PY=${CONDA_PREFIX:-/nonexistent}/bin/python
else VENV_PY=$FRONTDRIVE/env/drivejepa$([ "${PB_CACHE_ENV:-pb}" = pf ] || echo _pb)/bin/python; fi
test -x "$VENV_PY" || { log "missing interpreter $VENV_PY (scripts/h100/bootstrap_conda_cvpr.sh / bootstrap_pb_env.sh / bootstrap_env.sh)"; exit 2; }
[ "$(command -v python)" = "$VENV_PY" ] || { log "python on PATH is $(command -v python), expected $VENV_PY"; exit 2; }
python -c "import torch, navsim, yaml, cv2" 2>/dev/null || { log "venv not usable (torch / navsim / yaml / cv2) with $VENV_PY"; exit 2; }
test -d "$COMPACT/_COMPACT_JPEG" || { log "no compact cache at $COMPACT (push step S1 first)"; exit 2; }
test -d "$LOGS" || { log "no log pickles under $LOGS (push step S8 first)"; exit 2; }
n_logs=$(find "$COMPACT" -mindepth 1 -maxdepth 1 -type d ! -name '_*' | wc -l)
n_files=$(find "$COMPACT" -mindepth 3 -maxdepth 3 -type f -name '*.gz' | wc -l)
n_pkl=$(find "$LOGS" -maxdepth 1 -name '*.pkl' | wc -l)
log "compact cache: $n_logs logs / $n_files files (expect $EXPECT_LOGS / $EXPECT_FILES); log pickles: $n_pkl; python: $(command -v python); engine: $ENGINE"
[ "$n_files" -ge "$EXPECT_FILES" ] || { log "compact cache incomplete ($n_files < $EXPECT_FILES files)"; exit 2; }
missing=$(find "$COMPACT" -mindepth 1 -maxdepth 1 -type d ! -name '_*' -printf '%f\n' | while IFS= read -r l; do [ -f "$LOGS/$l.pkl" ] || echo "$l"; done | wc -l)
[ "$missing" -eq 0 ] || { log "$missing compact-cache logs have no pickle under $LOGS (push step S8 first)"; exit 2; }
mkdir -p "$OUT"
cd "$F"
for SPLIT in navtrain_val_official navtrain_train_official; do
  log "build split=$SPLIT -> $OUT ($WORKERS workers)"
  nice -n 5 python -m navsim_ext.compact_cache_pb build --split "$SPLIT" --compact "$COMPACT" --logs "$LOGS" --out "$OUT" \
    --workers "$WORKERS" --engine "$ENGINE" 2>&1 | tee -a "$BLOG" || { log "build of $SPLIT failed (resumable: re-run)"; exit 3; }
done
COUNT=$(python -m navsim_ext.compact_cache_pb count --out "$OUT")
log "pb cache: $COUNT (expect tokens_complete $EXPECT_TOKENS = $EXPECT_VAL val + $EXPECT_TRAIN train)"
n_tok=$(printf '%s' "$COUNT" | python -c "import json,sys; print(json.load(sys.stdin)['tokens_complete'])")
[ "$n_tok" -ge "$EXPECT_TOKENS" ] || { log "pb cache incomplete ($n_tok < $EXPECT_TOKENS tokens): re-run this script (resumable)"; exit 3; }
n_dirs=$(printf '%s' "$COUNT" | python -c "import json,sys; print(json.load(sys.stdin)['token_dirs'])")
[ "$n_dirs" -eq "$n_tok" ] || { log "$((n_dirs - n_tok)) token dir(s) lack a pickle (interrupted write; CacheOnlyDataset lists every token dir and would crash on them): re-run this script (resumable, incomplete tokens are rewritten)"; exit 3; }
if [ "$VERIFY_N" -gt 0 ]; then
  log "verify $VERIFY_N random tokens (mode ${PB_CACHE_VERIFY_MODE:-auto}: builder on dev logs with CAM_F0 blobs under $BLOBS when mmcv is importable, else pf)"
  python -m navsim_ext.compact_cache_pb verify --compact "$COMPACT" --logs "$LOGS" --out "$OUT" --n "$VERIFY_N" \
    --mode "${PB_CACHE_VERIFY_MODE:-auto}" --blobs "$BLOBS" 2>&1 | tee -a "$BLOG" || { log "verify FAILED"; exit 4; }
fi
MC=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache
if ls "$MC"/metadata/*.csv >/dev/null 2>&1; then
  log "token coverage vs $MC (informative here; train_pb.sh enforces it: compute_score KeyErrors on uncovered tokens)"
  python -m navsim_ext.compact_cache_pb coverage --out "$OUT" --metric-cache "$MC" --check-anchors 2>&1 | tee -a "$BLOG" \
    || log "coverage gaps (see above): train_pb.sh with PB_COVERAGE_MOVE_ASIDE=1 moves them aside"
else
  log "no $MC/metadata/*.csv yet (hf_fetch.sh train_metric_cache.tar): coverage check deferred to train_pb.sh"
fi
du -sh "$OUT" | tee -a "$BLOG"
log "DONE $OUT"
