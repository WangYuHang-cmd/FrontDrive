#!/usr/bin/env bash
# Phase 4 ON THE SERVER (tmux): run the two verification evaluations as soon as their inputs exist.
#  (ii)  official perception-free ckpt on dev  -> expect PDMS 87.10 ± 0.05 on 18,157 scenes   (venv-pf)
#  (iii) official perception-based v1 ckpt on navtest -> expect 93.3 ± 0.3 on 12,146 scenes  (venv-pb; counts as navtest eval #1)
# Idempotent: skips an eval whose report.json exists; polls every 5 min for prerequisites; exits when both are done.
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; EXP=$QD/benchmarks/navsim/exp; DJ=$EXP/Drive-JEPA-cache
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
gpu_free() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2-$1}'; }
PF_TAG=e0_official_pf_dev_h100; PB_TAG=pb_official_v1_navtest
while :; do
  done_pf=0; done_pb=0
  [ -f "$EXP/eval/$PF_TAG/report.json" ] && done_pf=1
  [ -f "$EXP/eval_pb/$PB_TAG/report.json" ] && done_pb=1
  [ "$done_pf" = 1 ] && [ "$done_pb" = 1 ] && { log "both verification evals done"; break; }
  if [ "$done_pf" = 0 ] && [ -f "$S/prov.devmc.done" ] && [ -f "$S/prov.navtrain.done" ] && [ "$(gpu_free)" -ge 9000 ]; then
    log "(ii) starting pf official dev eval"
    ( source "$F/env/drivejepa_env.sh"; cd "$F"; bash scripts/eval_perception_free.sh "$PF_TAG" "$DJ/drive_jepa_perception_free_agent_vitl.ckpt" navtrain_val_official ) > "$S/eval_pf_official_dev.log" 2>&1
    rc=$?; log "(ii) exit $rc"; [ -f "$EXP/eval/$PF_TAG/report.json" ] && { log "(ii) report: $(cat "$EXP/eval/$PF_TAG/report.json" | tr -d '\n' | cut -c1-400)"; echo "expect PDMS 87.10 ± 0.05, n_scenes 18157"; }
  fi
  if [ "$done_pb" = 0 ] && [ -f "$S/prov.test_cam.done" ] && [ -f "$DJ/metric_cache.tar.extracted" ] && [ "$(gpu_free)" -ge 16000 ]; then
    if ( source "$F/env/drivejepa_pb_env.sh" && env -u LD_LIBRARY_PATH -u CPATH python "$F/scripts/h100/pb_env_check.py" ) > "$S/pb_env_check_last.log" 2>&1; then
      log "(iii) starting pb official v1 navtest eval (navtest budget #1)"
      NAVTEST_PURPOSE="official pb v1 ckpt reproduction (G-A part 2)" bash "$F/scripts/h100/eval_pb.sh" "$PB_TAG" "$DJ/drive_jepa_perception_based_agent_vitl.ckpt" navtest 4 > "$S/eval_pb_official_navtest.log" 2>&1
      rc=$?; log "(iii) exit $rc"; [ -f "$EXP/eval_pb/$PB_TAG/report.json" ] && { log "(iii) report: $(cat "$EXP/eval_pb/$PB_TAG/report.json" | tr -d '\n' | cut -c1-400)"; echo "expect PDMS 93.3 ± 0.3, n_scenes 12146"; }
    elif grep -q "phase3_fix end" "$S/phase3_fix.log" 2>/dev/null; then
      log "(iii) sm_90 kernel unavailable after phase3 -> running with the exact PyTorch deformable-attention fallback (MSDA_PYTORCH=1, slower)"
      MSDA_PYTORCH=1 NAVTEST_PURPOSE="official pb v1 ckpt reproduction (G-A part 2; MSDA pytorch fallback)" bash "$F/scripts/h100/eval_pb.sh" "$PB_TAG" "$DJ/drive_jepa_perception_based_agent_vitl.ckpt" navtest 4 > "$S/eval_pb_official_navtest.log" 2>&1
      rc=$?; log "(iii) exit $rc"; [ -f "$EXP/eval_pb/$PB_TAG/report.json" ] && { log "(iii) report: $(cat "$EXP/eval_pb/$PB_TAG/report.json" | tr -d '\n' | cut -c1-400)"; echo "expect PDMS 93.3 ± 0.3, n_scenes 12146"; }
    else
      log "(iii) waiting: venv-pb CUDA kernel check not passing yet ($(tail -n 1 "$S/pb_env_check_last.log"))"
    fi
  fi
  sleep 300
done
