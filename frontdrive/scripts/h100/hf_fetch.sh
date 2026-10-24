#!/usr/bin/env bash
# Download Drive-JEPA public weights / caches straight from HuggingFace ON THE SERVER, verify every file
# against its LFS pointer (sha256 + size) and against the shas recorded in this repo where known, extract
# the tars into $NAVSIM_EXP_ROOT/Drive-JEPA-cache (the layout the upstream scripts expect) and rewrite the
# metric caches' metadata CSV paths to this machine.
# Usage (server, inside tmux):  bash scripts/h100/hf_fetch.sh            # default v1 set
#                               bash scripts/h100/hf_fetch.sh metric_cache_v2.tar   # explicit files
set -euo pipefail
FD=/home/henry/Desktop/Qwen_Drive/frontdrive
D=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp/Drive-JEPA-cache   # == /home/exouser/henry/... via the symlink
REPO=https://huggingface.co/datasets/LinhanWang/Drive-JEPA
DEFAULT=(vitl_merge_3dataset_e50.pt drive_jepa_perception_free_agent_vitl.ckpt \
         drive_jepa_perception_based_agent_vitl.ckpt anchors_scores_index.tar metric_cache.tar \
         train_metric_cache.tar)
declare -A KNOWN=(   # sha256 recorded locally in data/benchmarks/download_list_*.tsv (2026-10-08 / 10-11)
  [vitl_merge_3dataset_e50.pt]=4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f
  [drive_jepa_perception_free_agent_vitl.ckpt]=982960aed9e82900584d20c676ba9a7498add912bed94a3e7fa5f4d22bff6237
  [drive_jepa_perception_based_agent_vitl.ckpt]=b3e1d4ab723715be07c8266a7dfbd58abffed6990d654afcf5fbcd8cd97eeb34
  [anchors_scores_index.tar]=503f18e11249184a9f201f4ab422491cde1974eb8adba77fef2548dc7a90240d
  [metric_cache.tar]=9107c2ce88e62e75c2e5cbdcf5d8417a6ebb4acdc4ff1c4caa850a27239d88d5
)
FILES=("${@:-${DEFAULT[@]}}")
umask 077; mkdir -p "$D"; cd "$D"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
declare -A EXP SIZE
log "resolving LFS pointers"
for f in "${FILES[@]}"; do
  ptr=$(curl -sfL --retry 5 "$REPO/raw/main/$f") || { log "cannot read LFS pointer for $f (wrong name?)"; exit 1; }
  EXP[$f]=$(printf '%s\n' "$ptr" | awk '/^oid sha256:/{sub("oid sha256:","");print;exit}')
  SIZE[$f]=$(printf '%s\n' "$ptr" | awk '/^size /{print $2;exit}')
  [ -n "${EXP[$f]}" ] && [ -n "${SIZE[$f]}" ] || { log "pointer fields missing for $f"; exit 1; }
  if [ -n "${KNOWN[$f]:-}" ] && [ "${KNOWN[$f]}" != "${EXP[$f]}" ]; then
    log "UPSTREAM CHANGED $f: pointer ${EXP[$f]} != recorded ${KNOWN[$f]} -> stop (push the local copy instead)"; exit 3
  fi
  log "  $f  size=${SIZE[$f]}  sha=${EXP[$f]:0:12}…$([ -n "${KNOWN[$f]:-}" ] && echo ' (matches recorded)')"
done
for f in "${FILES[@]}"; do
  exp=${EXP[$f]}; size=${SIZE[$f]}
  if [ -f "$f" ] && [ -f "$f.sha256ok" ] && [ "$(stat -c %s "$f")" = "$size" ]; then log "have $f (verified earlier)"; continue; fi
  if [ ! -f "$f.part" ] || [ "$(stat -c %s "$f.part")" != "$size" ]; then
    log "fetch $f ($size bytes)"
    curl -L --fail --retry 8 --retry-all-errors --retry-delay 15 -C - -o "$f.part" "$REPO/resolve/main/$f" \
      || { rc=$?; [ -f "$f.part" ] && [ "$(stat -c %s "$f.part")" = "$size" ] || { log "download failed rc=$rc"; exit $rc; }; }
  fi
  got=$(sha256sum "$f.part" | cut -d' ' -f1)
  if [ "$got" != "$exp" ]; then mv -f "$f.part" "$f.corrupt.$(date +%s)"; log "SHA MISMATCH $f: got $got expected $exp -> moved aside, re-run"; exit 2; fi
  mv -f "$f.part" "$f"; echo "$exp  $f" > "$f.sha256ok"; log "ok $f"
done
for t in "${FILES[@]}"; do
  case "$t" in *.tar)
    if [ ! -f "$t.extracted" ]; then log "extract $t (top: $(tar -tf "$t" | head -1))"; tar -xf "$t" && touch "$t.extracted"; fi
    case "$t" in *metric_cache*.tar)
      log "rewrite metadata paths of ${t%.tar}"; bash "$FD/scripts/fix_metric_cache_paths.sh" "$D/${t%.tar}" || { log "metadata fix FAILED for ${t%.tar}"; exit 4; };;
    esac;;
  esac
done
log "contents of $D:"; ls -la "$D" | grep -v '\.sha256ok$'
