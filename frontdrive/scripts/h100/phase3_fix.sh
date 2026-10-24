#!/usr/bin/env bash
# Phase-3 fixes ON THE SERVER (run in tmux):
#  1 repair the nested log dirs produced by the first provisioning run (meta_datas/<split> moved one level too deep)
#  2 rewrite the metric caches' metadata CSV paths (the first hf_fetch run predates that step)
#  3 rebuild mmcv-full 1.7.2 from source for sm_90 in venv-pb (the prebuilt cu121/torch2.1 wheel has no H100 kernels:
#    "no kernel image is available for execution on the device"), using nvcc from the lmod nvhpc module
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; DS=$QD/benchmarks/navsim/dataset; S=$QD/state
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "phase3_fix start"
# --- 1 nested log dirs
for split in trainval test; do
  d=$DS/${split}_navsim_logs/$split
  if [ -d "$d/$split" ]; then
    log "repairing nested $d/$split"
    find "$d/$split" -mindepth 1 -maxdepth 1 -exec mv -t "$d" {} +
    rmdir "$d/$split" && log "repaired"
  fi
  log "$split logs: $(ls "$d" 2>/dev/null | wc -l) entries (expect trainval 1310 / test 147)"
done
# --- 2 metadata path rewrite (downloads are skipped, only extraction/rewrite runs)
bash "$F/scripts/h100/hf_fetch.sh" anchors_scores_index.tar metric_cache.tar train_metric_cache.tar || log "hf_fetch rewrite step returned $?"
# --- 3 mmcv for sm_90
VENV=$F/env/drivejepa_pb; PY=$VENV/bin/python
[ -x "$PY" ] || { log "no venv-pb at $VENV"; exit 1; }
source /etc/profile.d/lmod.sh 2>/dev/null || source /opt/lmod/lmod/init/bash 2>/dev/null || true
module load nvhpc/25.7/nvhpc 2>/dev/null || module load nvhpc 2>/dev/null || true
# torch 2.1.0+cu121 needs a CUDA 12.x nvcc (major version must match); the nvhpc module's default nvcc is 11.8
NVCC=$(ls -d /software/u24/nvhpc/*/Linux_x86_64/*/cuda/12.*/bin/nvcc /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/12.*/bin/nvcc /usr/local/cuda-12*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
[ -n "$NVCC" ] || NVCC=$(command -v nvcc || true)
[ -n "$NVCC" ] || { log "no nvcc found -> cannot build mmcv; plan B (MSDA_PYTORCH fallback) applies"; log "phase3_fix end"; exit 2; }
if ! "$NVCC" --version | grep -q "release 12\."; then log "nvcc $NVCC is not CUDA 12.x -> cannot build mmcv for torch cu121; plan B applies"; log "phase3_fix end"; exit 2; fi
export CUDA_HOME=$(dirname "$(dirname "$NVCC")"); export PATH=$CUDA_HOME/bin:$PATH
log "nvcc: $NVCC ($("$NVCC" --version | tail -1)); CUDA_HOME=$CUDA_HOME"
export CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST="9.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=16 PIP_CACHE_DIR=/home/exouser/henry/tools/pip-cache
"$PY" -m pip install -q ninja psutil
"$PY" -m pip uninstall -y -q mmcv-full mmcv_full 2>/dev/null || true
log "building mmcv-full 1.7.2 from source for sm_90 (20-40 min)"
"$PY" -m pip install --no-binary mmcv-full --no-build-isolation -v mmcv-full==1.7.2 > "$S/mmcv_build.log" 2>&1 \
  && log "mmcv build ok" || { log "mmcv build FAILED (see $S/mmcv_build.log); plan B (MSDA_PYTORCH fallback) applies"; grep -nE "error|Error" "$S/mmcv_build.log" | grep -vE "Werror|-Wno" | tail -n 15; log "phase3_fix end"; exit 3; }
env -u LD_LIBRARY_PATH -u CPATH "$PY" "$F/scripts/h100/pb_env_check.py" && log "PB ENV CHECK OK (sm_90 kernels)" || log "pb env check FAILED rc=$?"
log "phase3_fix end"
