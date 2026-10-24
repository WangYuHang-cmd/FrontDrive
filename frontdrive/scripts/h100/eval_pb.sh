#!/usr/bin/env bash
# Evaluate a Drive-JEPA perception-based checkpoint with the upstream run_pdm_score pipeline (venv-pb).
# Mirrors third_party/Drive-JEPA/navsim_v1/scripts/evaluation/eval_drive_jepa_perception_based.sh.
# Usage:  bash scripts/h100/eval_pb.sh <tag> <checkpoint.ckpt> [navtest|navtrain_val_official] [workers]
#   navtest               -> $NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache (12,146 scenes; COUNTS against the navtest budget;
#                            set NAVTEST_PURPOSE="..." so the ledger line explains why)
#   navtrain_val_official -> $NAVSIM_EXP_ROOT/metric_cache_navtrain_val_official (dev, 18,157 scenes)
# Output: $NAVSIM_EXP_ROOT/eval_pb/<tag>/<timestamp>.csv + report.json; navtest runs are appended to
#         $NAVSIM_EXP_ROOT/eval/navtest_evals_server.csv (pulled and merged into eval/navtest_evals.csv locally).
set -euo pipefail
TAG=${1:?tag}; CKPT=${2:?checkpoint}; SPLIT=${3:-navtrain_val_official}; WORKERS=${4:-4}
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_pb_env.sh
test -f "$CKPT" || { echo "no checkpoint $CKPT"; exit 1; }
case "$SPLIT" in
  navtest) MC=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache; BLOBS=$OPENSCENE_DATA_ROOT/sensor_blobs/test;;
  navtrain_val_official) MC=$NAVSIM_EXP_ROOT/metric_cache_navtrain_val_official; BLOBS=$OPENSCENE_DATA_ROOT/sensor_blobs/trainval;;
  *) echo "unknown split $SPLIT"; exit 1;;
esac
test -d "$MC" || { echo "no metric cache $MC"; exit 1; }
first=$(sed -n 2p "$MC"/metadata/*.csv | head -1 | tr ',' '\n' | grep -m1 '^/' || true)
[ -n "$first" ] && [ -f "$first" ] || { echo "metric cache metadata points to a missing file ($first); run scripts/fix_metric_cache_paths.sh $MC"; exit 3; }
# PB_SIDE_CALIB=1 -> +agent.config.side_cams_from_calibration=true (lidar2img from the pickles' calibration, no side JPEGs;
# identical features by construction — use it for our own checkpoints trained with SIDE_CALIB=1). Default 0 = upstream path.
SIDE_CALIB=${PB_SIDE_CALIB:-0}; EXTRA=()
if [ "$SIDE_CALIB" = 1 ]; then EXTRA+=("+agent.config.side_cams_from_calibration=true")
elif [ "${PB_REQUIRE_SIDE_CAMS:-1}" = 1 ]; then
  ls -d "$BLOBS"/*/CAM_L0 >/dev/null 2>&1 || { echo "no CAM_L0 under $BLOBS: the upstream pb agent loads L0/R0/B0 frame 3 (set PB_SIDE_CALIB=1 to use calibration-only side cameras)"; exit 3; }
fi
SHA8=$(sha256sum "$CKPT" | cut -c1-8)
OUT=$NAVSIM_EXP_ROOT/eval_pb/$TAG; mkdir -p "$OUT" "$NAVSIM_EXP_ROOT/eval"
# Hydra's override grammar rejects '=' , ',' and spaces inside a bare value (Lightning's best_epoch=13.ckpt broke
# agent.checkpoint_path=... with "mismatched input '='", 2026-10-18) -> hand Hydra a symlink with a plain name instead
CKPT_ARG=$CKPT
case "$CKPT" in *=*|*,*|*" "*|*"'"*) ln -sfn "$CKPT" "$OUT/checkpoint_for_hydra.ckpt"; CKPT_ARG=$OUT/checkpoint_for_hydra.ckpt;; esac
cd "$DRIVEJEPA_ROOT"
echo "[$(date -u +%FT%TZ)] eval_pb $TAG split=$SPLIT workers=$WORKERS ckpt=$CKPT sha8=$SHA8 side_calib=$SIDE_CALIB"
python -u navsim/planning/script/run_pdm_score.py \
  train_test_split="$SPLIT" \
  agent=drive_jepa_perception_based_agent \
  agent.config.latent=False \
  agent.checkpoint_path="$CKPT_ARG" \
  worker=single_machine_thread_pool \
  worker.max_workers="$WORKERS" \
  worker.use_process_pool=true \
  metric_cache_path="$MC" \
  experiment_name="eval_pb/$TAG" \
  output_dir="$OUT" "${EXTRA[@]}" 2>&1 | tee -a "$OUT/run.log"
CSV=$(find "$OUT" -maxdepth 1 -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python - "$CSV" "$OUT/report.json" "$TAG" "$SPLIT" "$CKPT" "$SHA8" <<'PY'
import json, sys
import pandas as pd
csv, out, tag, split, ckpt, sha8 = sys.argv[1:7]
df = pd.read_csv(csv, index_col=0)
rows = df[df["token"].astype(str) != "average"]            # run_pdm_score appends an 'average' row
valid = rows[rows["valid"] == True] if "valid" in rows.columns else rows
keys = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
sub = {k: round(float(valid[k].mean()) * 100, 2) for k in keys if k in valid.columns}
rep = {"tag": tag, "split": split, "ckpt": ckpt, "ckpt_sha8": sha8, "n_scenes": int(len(rows)),
       "n_valid": int(len(valid)), "per_scene_csv": csv, "pdms": sub.get("score"), **sub}
json.dump(rep, open(out, "w"), indent=1, allow_nan=False)
print(json.dumps(rep))
if len(valid) < len(rows):
    print(f"ONLY {len(valid)}/{len(rows)} scenes valid"); sys.exit(4)
PY
if [ "$SPLIT" = navtest ]; then
  printf '%s,%s,%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$TAG" "$SHA8" "$(python -c "import json;print(json.load(open('$OUT/report.json'))['pdms'])")" \
    "${NAVTEST_PURPOSE:-unspecified}" "h100" >> "$NAVSIM_EXP_ROOT/eval/navtest_evals_server.csv"
fi
