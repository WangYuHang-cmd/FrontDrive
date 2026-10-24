#!/usr/bin/env bash
# Dev (or navtest) evaluation of ONE pb checkpoint under a training-free proposal-selection rule (navsim_ext.pb_select_agent).
#   usage: eval_pb_select.sh <tag> <mode> <eps> <ckpt>        mode: argmax | minheads | conservative | progressive
#   env:   SPLIT=navtrain_val_official (default) | navtest (counts against the budget; set NAVTEST_PURPOSE), WORKERS=4,
#          SMOKE_SCENES=<n> (first n scenes only; use a tag sel_smoke_<...> so that pull.sh never ingests it)
# Output like eval_pb.sh: $NAVSIM_EXP_ROOT/eval_pb/<tag>/{<stamp>.csv,report.json,run.log}; report.json carries mode / eps / member.
set -euo pipefail
TAG=${1:?tag}; MODE=${2:?mode}; EPS=${3:?eps}; CKPT=${4:?ckpt}
SPLIT=${SPLIT:-navtrain_val_official}; WORKERS=${WORKERS:-4}
EXTRA=(); [ -n "${SMOKE_SCENES:-}" ] && EXTRA+=("train_test_split.scene_filter.max_scenes=$SMOKE_SCENES")
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_pb_env.sh
export PYTHONPATH=$FRONTDRIVE:${PYTHONPATH:-}
case "$SPLIT" in
  navtest) MC=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache;;
  navtrain_val_official) MC=$NAVSIM_EXP_ROOT/metric_cache_navtrain_val_official;;
  *) echo "unknown split $SPLIT"; exit 1;;
esac
test -f "$CKPT" || { echo "no checkpoint $CKPT"; exit 1; }
OUT=$NAVSIM_EXP_ROOT/eval_pb/$TAG; mkdir -p "$OUT" "$NAVSIM_EXP_ROOT/eval"
ln -sfn "$CKPT" "$OUT/member.ckpt"; SHA=$(sha256sum "$CKPT" | cut -c1-8)   # Hydra rejects '=' inside values -> plain-named symlink
cd "$DRIVEJEPA_ROOT"
echo "[$(date -u +%FT%TZ)] eval_pb_select $TAG split=$SPLIT mode=$MODE eps=$EPS member=$CKPT sha8=$SHA"
python -u navsim/planning/script/run_pdm_score.py \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$FRONTDRIVE/navsim_ext/configs]" \
  train_test_split="$SPLIT" \
  agent=drive_jepa_pb_select_agent \
  agent.checkpoint_path="$OUT/member.ckpt" agent.mode="$MODE" agent.eps="$EPS" \
  worker=single_machine_thread_pool worker.max_workers="$WORKERS" worker.use_process_pool=true \
  metric_cache_path="$MC" experiment_name="eval_pb/$TAG" output_dir="$OUT" "${EXTRA[@]}" 2>&1 | tee -a "$OUT/run.log"
CSV=$(find "$OUT" -maxdepth 1 -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python - "$CSV" "$OUT/report.json" "$TAG" "$SPLIT" "$MODE" "$EPS" "$CKPT" "$SHA" <<'PY'
import json, sys
import pandas as pd
csv, out, tag, split, mode, eps, member, sha = sys.argv[1:9]
df = pd.read_csv(csv, index_col=0)
rows = df[df["token"].astype(str) != "average"]
valid = rows[rows["valid"] == True] if "valid" in rows.columns else rows
keys = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
sub = {k: round(float(valid[k].mean()) * 100, 2) for k in keys if k in valid.columns}
rep = {"tag": tag, "split": split, "ckpt": f"select[{mode},eps={eps}]", "ckpt_sha8": sha, "select_mode": mode, "eps": float(eps), "member": member,
       "n_scenes": int(len(rows)), "n_valid": int(len(valid)), "per_scene_csv": csv, "pdms": sub.get("score"), **sub}
json.dump(rep, open(out, "w"), indent=1, allow_nan=False); print(json.dumps(rep))
if len(valid) < len(rows): print(f"ONLY {len(valid)}/{len(rows)} scenes valid"); sys.exit(4)
PY
if [ "$SPLIT" = navtest ] && [ -z "${SMOKE_SCENES:-}" ]; then
  printf '%s,%s,%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$TAG" "$SHA" "$(python -c "import json;print(json.load(open('$OUT/report.json'))['pdms'])")" "${NAVTEST_PURPOSE:-unspecified}" "h100" >> "$NAVSIM_EXP_ROOT/eval/navtest_evals_server.csv"
fi
