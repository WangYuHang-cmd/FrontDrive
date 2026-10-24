#!/usr/bin/env bash
# Evaluate a Drive-JEPA perception-free checkpoint (official or ours) on a NAVSIM split with the local metric cache.
#   usage: eval_perception_free.sh <tag> <lightning_ckpt> <split> [metric_cache_dir]
#   split: navtest (budget: <= 2 evaluations per variant, every call is appended to eval/navtest_evals.csv)
#          navtrain_val_official (dev set = devkit val logs within navtrain; metric cache built by run_metric_caching)
# Step 1 = GPU inference -> submission.pkl (sequential worker), step 2 = CPU PDM scoring from the submission.
set -euo pipefail
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_env.sh
TAG=${1:?tag}; CKPT=${2:?checkpoint}; SPLIT=${3:?split}
DJ=$NAVSIM_EXP_ROOT/Drive-JEPA-cache
case "$SPLIT" in
  navtest) MC=${4:-$DJ/metric_cache} ;;
  navtrain_val_official) MC=${4:-$NAVSIM_EXP_ROOT/metric_cache_navtrain_val_official} ;;
  *) MC=${4:?metric cache dir required for split $SPLIT} ;;
esac
E50=${E50:-$DJ/vitl_merge_3dataset_e50.pt}
OUT=$NAVSIM_EXP_ROOT/eval/$TAG
mkdir -p "$OUT"
sha8=$(sha256sum "$CKPT" | cut -c1-8)
if [ "$SPLIT" = navtest ]; then
  LOGF=/home/henry/Desktop/Qwen_Drive/frontdrive/eval/navtest_evals.csv
  [ -f "$LOGF" ] || echo "date,tag,ckpt_sha8,ckpt" > "$LOGF"
  echo "$(date -Is),$TAG,$sha8,$(basename "$CKPT")" >> "$LOGF"
  echo "[eval] navtest evaluation #$(( $(wc -l < "$LOGF") - 1 )) logged to $LOGF"
fi
cd "$DRIVEJEPA_ROOT"
echo "[eval] step 1: submission pickle ($SPLIT) with $CKPT (sha8 $sha8)"
python navsim/planning/script/run_create_submission_pickle.py \
  train_test_split="$SPLIT" worker=sequential \
  agent=drive_jepa_perception_free_agent \
  agent.pretrain_pt_path="$E50" agent.image_architecture=vit_large agent.num_keyval=129 agent.front_only=true \
  agent.tf_dropout=0.0 agent.freeze_encoder=false agent.double_image=true agent.checkpoint_path="$CKPT" \
  experiment_name="$TAG" output_dir="$OUT" \
  team_name=frontdrive authors=frontdrive email=none institution=none country=none 2>&1 | tee "$OUT/step1.log" | grep -v "^\s*$" | tail -3
echo "[eval] step 2: PDM score from submission with $MC"
python navsim/planning/script/run_pdm_score_from_submission.py \
  metric_cache_path="$MC" submission_file_path="$OUT/submission.pkl" output_dir="$OUT/score" 2>&1 | tee "$OUT/step2.log" | tail -3
python - "$OUT" "$TAG" "$SPLIT" "$sha8" "$CKPT" <<'PY'
import glob, sys, json, pandas as pd
out, tag, split, sha8, ckpt = sys.argv[1:6]
f = sorted(glob.glob(out + "/score/**/*.csv", recursive=True))[-1]
df = pd.read_csv(f)
df = df[df["token"].astype(str) != "average"].reset_index(drop=True)  # navsim appends an aggregate row
valid = df["valid"] if "valid" in df else pd.Series([True] * len(df))
cols = [c for c in ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance") if c in df.columns]
rep = {"tag": tag, "split": split, "ckpt": ckpt, "ckpt_sha8": sha8, "n_scenes": int(len(df)), "n_valid": int(valid.sum()), "per_scene_csv": f}
rep.update({c: round(float(df.loc[valid, c].mean() * 100), 2) for c in cols})
json.dump(rep, open(out + "/report.json", "w"), indent=1); print(json.dumps(rep))
PY
