#!/usr/bin/env bash
# E0 (plan v4 §5): reproduce Drive-JEPA perception-free navtest PDMS (expected 89.0 ± 0.5) with the official checkpoint
# in our drivejepa environment, on the local sparse CAM_F0 mirror. Two steps: (1) GPU inference -> submission.pkl,
# (2) CPU scoring from the submission with the v1 metric cache.
#   bash scripts/e0_eval_perception_free.sh [TAG]        (TAG default e0_pf_official)
set -euo pipefail
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_env.sh
TAG=${1:-e0_pf_official}
CACHE=$NAVSIM_EXP_ROOT/Drive-JEPA-cache
CKPT=${CKPT:-$CACHE/drive_jepa_perception_free_agent_vitl.ckpt}
E50=${E50:-$CACHE/vitl_merge_3dataset_e50.pt}
OUT=$NAVSIM_EXP_ROOT/$TAG
mkdir -p "$OUT"
# devkit default layout via symlinks (navsim_logs/<split>, sensor_blobs/<split>)
mkdir -p "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
[ -e "$OPENSCENE_DATA_ROOT/navsim_logs/test" ] || ln -s ../test_navsim_logs/test "$OPENSCENE_DATA_ROOT/navsim_logs/test"
[ -e "$OPENSCENE_DATA_ROOT/sensor_blobs/test" ] || ln -s ../test_sensor_blobs/test "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
cd "$DRIVEJEPA_ROOT"
echo "[E0] step 1: submission pickle with $CKPT"
python navsim/planning/script/run_create_submission_pickle.py \
  train_test_split=navtest worker=sequential \
  agent=drive_jepa_perception_free_agent \
  agent.pretrain_pt_path="$E50" agent.image_architecture=vit_large agent.num_keyval=129 agent.front_only=true \
  agent.tf_dropout=0.0 agent.freeze_encoder=false agent.double_image=true agent.checkpoint_path="$CKPT" \
  experiment_name="$TAG" output_dir="$OUT" \
  team_name=frontdrive authors=frontdrive email=none institution=none country=none 2>&1 | tee "$OUT/step1.log" | grep -v "^\s*$" | tail -5
ls -la "$OUT/submission.pkl"
echo "[E0] step 2: PDM score from submission"
python navsim/planning/script/run_pdm_score_from_submission.py \
  metric_cache_path="$CACHE/metric_cache" submission_file_path="$OUT/submission.pkl" output_dir="$OUT/score" 2>&1 | tee "$OUT/step2.log" | tail -5
python - "$OUT/score" <<'PY'
import glob, sys, pandas as pd
d = sys.argv[1]; f = sorted(glob.glob(d + "/**/*.csv", recursive=True))
df = pd.read_csv(f[-1]); print("csv:", f[-1], "rows", len(df), "valid", int(df.get("valid", pd.Series([True]*len(df))).sum()))
cols = [c for c in df.columns if c in ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort")]
print((df[cols].mean() * 100).round(2).to_string())
PY
