#!/usr/bin/env bash
# ⑤ submission pickles for the three final seeds (plan v4.3 ⑤ / G-D): run_create_submission_pickle.py on navtest with each seed's
# soup_1_5.ckpt — the same checkpoints and single-front protocol as the budgeted navtest evaluations; inference only, no scoring
# (no new budget use). Team metadata fields are placeholders here: the local merge (eval/merge_submission.py) writes the real team
# fields and combines the three seeds into ONE multi-seed submission.pkl (predictions = [seed0, seed1, seed2]). Launch by Henry:
#   tmux new-session -d -s fdsub "bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/make_submission_1024.sh >> /home/exouser/henry/Qwen_Drive/state/phase5.log 2>&1"
set -uo pipefail
F=/home/exouser/henry/Qwen_Drive/frontdrive; E=/home/exouser/henry/Qwen_Drive/benchmarks/navsim/exp; S=/home/exouser/henry/Qwen_Drive/state
source "$F/scripts/h100/gpu_policy.sh"
log() { echo "[SUB][$(date -u +%FT%TZ)] $*"; }
load1() { awk '{print int($1)}' /proc/loadavg; }
source "$F/env/drivejepa_pb_env.sh"
cd "$DRIVEJEPA_ROOT"
log "submission chain start (seeds 0 1 2, soup_1_5 each, navtest inference only)"
for i in 0 1 2; do
  CK=$E/pb_off_ft_all_s$i/exports/soup_1_5.ckpt; OUT=$E/submission/pb_off_ft_all_s$i; mkdir -p "$OUT"
  if [ -f "$OUT/submission.pkl" ]; then log "seed $i: submission.pkl exists -> skip"; continue; fi
  test -f "$CK" || { log "seed $i: missing $CK -> stop"; exit 2; }
  while [ "$(load1)" -gt "$LOAD_MAX" ] || [ -f "$S/STOP" ] || [ -f "$S/PAUSE" ]; do log "seed $i: load1 $(load1) > $LOAD_MAX or STOP/PAUSE -> wait"; sleep 300; done
  ln -sfn "$CK" "$OUT/member.ckpt"   # Hydra rejects '=' inside values -> plain-named symlink
  log "seed $i: create submission pickle start (sha8 $(sha256sum "$CK" | cut -c1-8), load1 $(load1))"
  python -u navsim/planning/script/run_create_submission_pickle.py \
    agent=drive_jepa_perception_based_agent agent.config.latent=False agent.checkpoint_path="$OUT/member.ckpt" \
    +agent.config.side_cams_from_calibration=true \
    experiment_name="submission/pb_off_ft_all_s$i" output_dir="$OUT" \
    team_name=FrontDrive authors=tbd email=tbd institution=tbd country=tbd > "$OUT/create.out" 2>&1; rc=$?
  if [ -f "$OUT/submission.pkl" ]; then
    n=$(python -c "import pickle; d=pickle.load(open('$OUT/submission.pkl','rb')); print(len(d['predictions'][0]))" 2>/dev/null || echo "?")
    log "seed $i: EXIT $rc submission.pkl $(stat -c %s "$OUT/submission.pkl") bytes, $n tokens"
  else log "seed $i: EXIT $rc without submission.pkl -> $OUT/create.out"; tail -n 5 "$OUT/create.out"; fi
done
log "submission pickles done"
