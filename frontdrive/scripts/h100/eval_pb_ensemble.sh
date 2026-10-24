#!/usr/bin/env bash
# Dev (or navtest) evaluation of an inference-time ensemble of pb checkpoints (navsim_ext.pb_ensemble_agent).
#   usage: eval_pb_ensemble.sh <tag> <mode> <ckpt1> <ckpt2> [ckpt3 ...]        mode: argmax_conf | mean_if_close | first
#   env:   SPLIT=navtrain_val_official (default) | navtest (counts against the budget; set NAVTEST_PURPOSE), WORKERS=4, CLOSE_M=0.5
# Output like eval_pb.sh: $NAVSIM_EXP_ROOT/eval_pb/<tag>/{<stamp>.csv,report.json,run.log}; report.json carries members + mode.
set -euo pipefail
TAG=${1:?tag}; MODE=${2:?mode}; shift 2; MEMBERS=("$@"); [ ${#MEMBERS[@]} -ge 2 ] || { echo "need >= 2 checkpoints"; exit 2; }
SPLIT=${SPLIT:-navtrain_val_official}; WORKERS=${WORKERS:-4}; CLOSE_M=${CLOSE_M:-0.5}
# SMOKE_SCENES=<n> -> only the first n scenes (train_test_split.scene_filter.max_scenes); use a tag ens_smoke_<...> so that
# pull.sh never ingests it. Mode "first" on a smoke must reproduce member 0's single-model per-scene scores exactly.
EXTRA=(); [ -n "${SMOKE_SCENES:-}" ] && EXTRA+=("train_test_split.scene_filter.max_scenes=$SMOKE_SCENES")
source /home/henry/Desktop/Qwen_Drive/frontdrive/env/drivejepa_pb_env.sh
export PYTHONPATH=$FRONTDRIVE:${PYTHONPATH:-}
case "$SPLIT" in
  navtest) MC=$NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache;;
  navtrain_val_official) MC=$NAVSIM_EXP_ROOT/metric_cache_navtrain_val_official;;
  *) echo "unknown split $SPLIT"; exit 1;;
esac
OUT=$NAVSIM_EXP_ROOT/eval_pb/$TAG; mkdir -p "$OUT" "$NAVSIM_EXP_ROOT/eval"
# Hydra rejects '=' etc. inside list items -> plain-named symlinks
LIST=(); SHAS=()
for i in "${!MEMBERS[@]}"; do
  test -f "${MEMBERS[$i]}" || { echo "no checkpoint ${MEMBERS[$i]}"; exit 1; }
  ln -sfn "${MEMBERS[$i]}" "$OUT/member_$i.ckpt"; LIST+=("$OUT/member_$i.ckpt"); SHAS+=("$(sha256sum "${MEMBERS[$i]}" | cut -c1-8)")
done
CKLIST=$(IFS=,; echo "[${LIST[*]}]")
cd "$DRIVEJEPA_ROOT"
echo "[$(date -u +%FT%TZ)] eval_pb_ensemble $TAG split=$SPLIT mode=$MODE close_m=$CLOSE_M members=${MEMBERS[*]} sha8=${SHAS[*]}"
python -u navsim/planning/script/run_pdm_score.py \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$FRONTDRIVE/navsim_ext/configs]" \
  train_test_split="$SPLIT" \
  agent=drive_jepa_pb_ensemble_agent \
  "agent.checkpoint_paths=$CKLIST" agent.mode="$MODE" agent.close_m="$CLOSE_M" \
  worker=single_machine_thread_pool worker.max_workers="$WORKERS" worker.use_process_pool=true \
  metric_cache_path="$MC" experiment_name="eval_pb/$TAG" output_dir="$OUT" "${EXTRA[@]}" 2>&1 | tee -a "$OUT/run.log"
CSV=$(find "$OUT" -maxdepth 1 -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python - "$CSV" "$OUT/report.json" "$TAG" "$SPLIT" "$MODE" "$CLOSE_M" "${MEMBERS[*]}" "${SHAS[*]}" <<'PY'
import json, sys
import pandas as pd
csv, out, tag, split, mode, close_m, members, shas = sys.argv[1:9]
df = pd.read_csv(csv, index_col=0)
rows = df[df["token"].astype(str) != "average"]
valid = rows[rows["valid"] == True] if "valid" in rows.columns else rows
keys = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
sub = {k: round(float(valid[k].mean()) * 100, 2) for k in keys if k in valid.columns}
rep = {"tag": tag, "split": split, "ckpt": f"ensemble[{mode}]", "ckpt_sha8": "+".join(shas.split()), "ensemble_mode": mode, "close_m": float(close_m),
       "members": members.split(), "n_scenes": int(len(rows)), "n_valid": int(len(valid)), "per_scene_csv": csv, "pdms": sub.get("score"), **sub}
json.dump(rep, open(out, "w"), indent=1, allow_nan=False); print(json.dumps(rep))
if len(valid) < len(rows): print(f"ONLY {len(valid)}/{len(rows)} scenes valid"); sys.exit(4)
PY
if [ "$SPLIT" = navtest ] && [ -z "${SMOKE_SCENES:-}" ]; then
  printf '%s,%s,%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$TAG" "$(IFS=+; echo "${SHAS[*]}")" "$(python -c "import json;print(json.load(open('$OUT/report.json'))['pdms'])")" "${NAVTEST_PURPOSE:-unspecified}" "h100" >> "$NAVSIM_EXP_ROOT/eval/navtest_evals_server.csv"
fi
