#!/usr/bin/env bash
# After the navtrain compact cache is built AND verified: delete the raw CAM_F0 frames of the devkit TRAIN logs
# (the compact cache holds everything training needs) but KEEP the val logs' frames, which run_create_submission_pickle
# reads for dev-set evaluation. Frees ~15 GB of the ~19 GB of local raw frames.
set -euo pipefail
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_env.sh
RAW=$OPENSCENE_DATA_ROOT/trainval_sensor_blobs/trainval
CACHE=$NAVSIM_LOCAL/cache/navtrain_compact
[ -d "$CACHE/_COMPACT_JPEG" ] || { echo "compact cache marker missing: $CACHE"; exit 2; }
python - "$RAW" <<'PY'
import sys, os, shutil, yaml
raw = sys.argv[1]
cfg = os.environ["NAVSIM_DEVKIT_ROOT"] + "/navsim/planning/script/config/common/train_test_split/scene_filter/"
val = set(yaml.safe_load(open(cfg + "navtrain_val_official.yaml"))["log_names"])
train = set(yaml.safe_load(open(cfg + "navtrain_train_official.yaml"))["log_names"])
n_del = n_keep = 0
for log in sorted(os.listdir(raw)):
    p = os.path.join(raw, log)
    if log in train and log not in val:
        shutil.rmtree(p); n_del += 1
    else:
        n_keep += 1
print(f"deleted {n_del} train-log frame dirs, kept {n_keep} (val logs {len(val)})")
PY
du -sh "$RAW" | cut -f1
