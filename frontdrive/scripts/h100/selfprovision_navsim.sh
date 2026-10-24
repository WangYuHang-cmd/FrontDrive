#!/usr/bin/env bash
# Self-provision ALL public NAVSIM assets ON THE SERVER straight from HuggingFace / S3 (the local uplink is
# ~0.5 MB/s): maps, trainval + test logs, navtest camera blobs (CAM_B0/F0/L0/R0 for the unpatched pb agent),
# navtrain CAM_F0 frames, then rebuild the compact JPEG training cache and the dev metric cache exactly as
# they were built locally. Resumable: per-step / per-shard markers in $QD/state/prov.*; safe to re-run.
# Run ON THE SERVER inside tmux:  bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/selfprovision_navsim.sh
set -u
QD=/home/henry/Desktop/Qwen_Drive            # symlink -> /home/exouser/henry/Qwen_Drive
F=$QD/frontdrive; BM=$QD/benchmarks/navsim; DS=$BM/dataset; EXP=$BM/exp; S=$QD/state
L=$F/scripts/h100/lists; DL=$BM/_dl; TMP=$BM/_extract
PAR=${PAR:-3}                                  # concurrent shard downloads
export QD F BM DS EXP S L DL TMP
umask 077; mkdir -p "$DL" "$TMP" "$S" "$DS" "$EXP" "$BM/cache"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }; export -f log
free_gb() { df -BG --output=avail "$BM" | tail -1 | tr -dc 0-9; }; export -f free_gb
done_mark() { [ -f "$S/prov.$1.done" ]; }
mark() { touch "$S/prov.$1.done"; log "DONE $1"; }

get() {  # get <url> <size> <sha|-> <name>  -> $DL/<name> verified (resumable)
  local url=$1 size=$2 sha=$3 name=$4 out="$DL/$4"
  if [ -f "$out.ok" ]; then return 0; fi
  if [ ! -f "$out" ] || [ "$(stat -c %s "$out")" != "$size" ]; then
    curl -sL --fail --retry 8 --retry-all-errors --retry-delay 15 -C - -o "$out" "$url" \
      || { [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ] || { log "download failed: $name"; return 1; }; }
  fi
  [ "$size" = 0 ] || [ "$(stat -c %s "$out")" = "$size" ] || { log "size mismatch $name: $(stat -c %s "$out") != $size"; rm -f "${out:?}"; return 1; }
  if [ "$sha" != "-" ]; then
    local got; got=$(sha256sum "$out" | cut -d' ' -f1)
    [ "$got" = "$sha" ] || { log "sha mismatch $name"; mv -f "$out" "$out.corrupt"; return 1; }
  fi
  touch "$out.ok"
}; export -f get
row() { awk -F'\t' -v g="$1" '$1==g{print $2"\t"$3"\t"$4"\t"$5}' "$L"/download_list_*.tsv; }
cleanup_shard() {  # cleanup_shard <tmpdir> <name>: remove the extraction dir and the downloaded shard
  local t=$1 name=$2
  rm -rf "${t:?}"; rm -f "${DL:?}/${name:?}" "${DL:?}/${name:?}.ok"
}; export -f cleanup_shard

shard_test() {  # one navtest camera shard: download, extract CAM_B0/F0/L0/R0, merge, delete
  local url=$1 size=$2 sha=$3 name=$4
  [ -f "$S/prov.test_cam.$name.done" ] && return 0
  [ "$(free_gb)" -ge 100 ] || { log "disk < 100 GB, skipping $name"; return 1; }
  get "$url" "$size" "$sha" "$name" || return 1
  local t="$TMP/$name.d"; rm -rf "${t:?}"; mkdir -p "$t"
  tar -xzf "$DL/$name" -C "$t" --wildcards '*/CAM_F0/*' '*/CAM_L0/*' '*/CAM_R0/*' '*/CAM_B0/*' || { log "extract failed $name"; return 1; }
  local src; src=$(find "$t" -maxdepth 4 -type d -name test | head -1); [ -n "$src" ] || src=$(find "$t" -mindepth 1 -maxdepth 1 -type d | head -1)
  mkdir -p "$DS/test_sensor_blobs/test"
  rsync -a --remove-source-files "$src/" "$DS/test_sensor_blobs/test/" && cleanup_shard "$t" "$name" \
    && touch "$S/prov.test_cam.$name.done" && log "test shard ok $name (free $(free_gb) GB)"
}; export -f shard_test
shard_train() {  # one navtrain shard: download, extract CAM_F0 only, merge, delete
  local url=$1 size=$2 sha=$3 name=$4
  [ -f "$S/prov.navtrain.$name.done" ] && return 0
  [ "$(free_gb)" -ge 100 ] || { log "disk < 100 GB, skipping $name"; return 1; }
  get "$url" "$size" "$sha" "$name" || return 1
  local t="$TMP/$name.d"; rm -rf "${t:?}"; mkdir -p "$t"
  tar -xzf "$DL/$name" -C "$t" --wildcards '*/CAM_F0/*' || { log "extract failed $name"; return 1; }
  mkdir -p "$DS/trainval_sensor_blobs/trainval"
  local d; for d in "$t"/*/; do rsync -a --remove-source-files "$d" "$DS/trainval_sensor_blobs/trainval/" || return 1; done
  cleanup_shard "$t" "$name" && touch "$S/prov.navtrain.$name.done" && log "navtrain shard ok $name (free $(free_gb) GB)"
}; export -f shard_train

log "selfprovision start; free $(free_gb) GB; PAR=$PAR"
# 1 maps
if ! done_mark maps; then
  read -r url size sha name < <(row maps); get "$url" "$size" "$sha" "$name" && (cd "$TMP" && unzip -qo "$DL/$name") \
    && { [ -d "$DS/maps" ] || mv "$TMP/nuplan-maps-v1.0" "$DS/maps"; } && ls "$DS/maps" | head -3 && mark maps
fi
# 2 logs (trainval + test) and the relative symlink dirs
if ! done_mark meta_trainval; then
  read -r url size sha name < <(row navsim_meta_trainval); get "$url" "$size" "$sha" "$name" && rm -rf "${TMP:?}/mt" && mkdir -p "$TMP/mt" \
    && tar -xzf "$DL/$name" -C "$TMP/mt" && mkdir -p "$DS/trainval_navsim_logs" && { [ -d "$DS/trainval_navsim_logs/trainval" ] || mv "$TMP/mt/openscene-v1.1/meta_datas/trainval" "$DS/trainval_navsim_logs/trainval"; } \
    && log "trainval logs: $(ls "$DS/trainval_navsim_logs/trainval" | wc -l) pkl" && rm -rf "${TMP:?}/mt" && rm -f "${DL:?}/${name:?}" && mark meta_trainval
fi
if ! done_mark meta_test; then
  read -r url size sha name < <(row navsim_meta); get "$url" "$size" "$sha" "$name" && rm -rf "${TMP:?}/mtest" && mkdir -p "$TMP/mtest" \
    && tar -xzf "$DL/$name" -C "$TMP/mtest" && mkdir -p "$DS/test_navsim_logs" && { [ -d "$DS/test_navsim_logs/test" ] || mv "$TMP/mtest/openscene-v1.1/meta_datas/test" "$DS/test_navsim_logs/test"; } \
    && log "test logs: $(ls "$DS/test_navsim_logs/test" | wc -l) pkl" && rm -rf "${TMP:?}/mtest" && rm -f "${DL:?}/${name:?}" && mark meta_test
fi
mkdir -p "$DS/navsim_logs" "$DS/sensor_blobs" "$DS/trainval_sensor_blobs/trainval" "$DS/test_sensor_blobs/test"
ln -sfn ../trainval_navsim_logs/trainval "$DS/navsim_logs/trainval"; ln -sfn ../test_navsim_logs/test "$DS/navsim_logs/test"
ln -sfn ../trainval_sensor_blobs/trainval "$DS/sensor_blobs/trainval"; ln -sfn ../test_sensor_blobs/test "$DS/sensor_blobs/test"
# 3 navtest camera shards (4 cameras), 4 navtrain shards (CAM_F0), PAR at a time
if ! done_mark test_cam; then
  row navsim_cam | xargs -P "$PAR" -L 1 bash -c 'shard_test "$0" "$1" "$2" "$3"'
  n=$(ls "$S"/prov.test_cam.*.done 2>/dev/null | wc -l); log "test shards done: $n / $(row navsim_cam | wc -l)"
  [ "$n" -eq "$(row navsim_cam | wc -l)" ] && mark test_cam
fi
if ! done_mark navtrain; then
  row navtrain | xargs -P "$PAR" -L 1 bash -c 'shard_train "$0" "$1" "$2" "$3"'
  n=$(ls "$S"/prov.navtrain.*.done 2>/dev/null | wc -l); log "navtrain shards done: $n / $(row navtrain | wc -l)"
  if [ "$n" -eq "$(row navtrain | wc -l)" ]; then
    miss=$(while read -r p; do [ -f "$DS/trainval_sensor_blobs/trainval/$p" ] || echo "$p"; done < "$L/navtrain_camf0_23.files.txt" | wc -l)
    log "navtrain CAM_F0 frames [2,3] missing: $miss / $(wc -l < "$L/navtrain_camf0_23.files.txt")"
    [ "$miss" -eq 0 ] && mark navtrain
  fi
fi
# 5 compact JPEG training cache (needs venv-pf) and 6 dev metric cache
if done_mark navtrain && ! done_mark compact; then
  until ( source "$F/env/drivejepa_env.sh" && python -c "import navsim, nuplan, cv2" ) >/dev/null 2>&1; do log "waiting for the python env (conda cvpr / venv-pf)"; sleep 120; done
  source "$F/env/drivejepa_env.sh"; cd "$F"
  PYTHONWARNINGS=ignore python -m navsim_ext.compact_cache build --split navtrain --cache "$BM/cache/navtrain_compact" --workers 16 --quality 95
  n=$(find "$BM/cache/navtrain_compact" -name '*.gz' | wc -l); log "compact cache files: $n (expect 206576)"
  [ "$n" -ge 206576 ] && PYTHONWARNINGS=ignore python -m navsim_ext.compact_cache verify --split navtrain --cache "$BM/cache/navtrain_compact" --limit 60 && mark compact
fi
if done_mark meta_trainval && done_mark maps && ! done_mark devmc; then
  until ( source "$F/env/drivejepa_env.sh" && python -c "import navsim, nuplan" ) >/dev/null 2>&1; do log "waiting for the python env (conda cvpr / venv-pf)"; sleep 120; done
  source "$F/env/drivejepa_env.sh"; cd "$DRIVEJEPA_ROOT"
  python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_val_official \
    cache.cache_path="$EXP/metric_cache_navtrain_val_official" worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=16
  n=$(find "$EXP/metric_cache_navtrain_val_official" -name 'metric_cache.pkl' | wc -l); log "dev metric cache entries: $n (local had 18157)"
  [ "$n" -ge 18100 ] && mark devmc
fi
log "selfprovision end; markers: $(ls "$S"/prov.*.done 2>/dev/null | grep -vE 'test_cam\.|navtrain\.' | xargs -n1 basename | tr '\n' ' '); free $(free_gb) GB"
