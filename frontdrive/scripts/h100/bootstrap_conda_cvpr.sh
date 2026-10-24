#!/usr/bin/env bash
# Build the conda environment `cvpr` ON THE SERVER (Henry 2026-10-17: use conda, env name cvpr, for all code).
# One stack for everything (upstream Drive-JEPA pins): CPython 3.10, torch 2.1.0+cu121 (sm_90 ok), the upstream
# requirements (hydra 1.2.0, numpy 1.24, PL 2.2.1, mmdet 2.28.2, …), mmcv-full 1.7.2 BUILT FROM SOURCE for sm_90
# (the prebuilt wheel has no H100 kernels), plus our pretraining deps (decord, av). Miniforge lives under
# /home/exouser/henry/tools. Idempotent. Run inside tmux:
#   bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/bootstrap_conda_cvpr.sh
set -uo pipefail
H=/home/exouser/henry; QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; TOOLS=$H/tools; S=$QD/state
BASE=$TOOLS/miniforge3; DJ=$F/third_party/Drive-JEPA
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
export PIP_CACHE_DIR=$TOOLS/pip-cache
mkdir -p "$TOOLS" "$S" "$PIP_CACHE_DIR"
exec > >(tee -a "$S/bootstrap_conda_cvpr.log") 2>&1
# 1 Miniforge (conda-forge; no Anaconda defaults channel / ToS)
if [ ! -x "$BASE/bin/conda" ]; then
  log "installing Miniforge3 to $BASE"
  # resolve the versioned asset names of the latest release (the unversioned .sha256 alias does not exist -> 404)
  REL=$(curl -fsSL --retry 5 https://api.github.com/repos/conda-forge/miniforge/releases/latest) || { log "GitHub release lookup failed"; exit 1; }
  URL=$(printf '%s' "$REL" | grep -oE 'https://[^"]*/Miniforge3-[0-9.]+-[0-9]+-Linux-x86_64\.sh' | head -1)
  SHAURL=$(printf '%s' "$REL" | grep -oE 'https://[^"]*/Miniforge3-[0-9.]+-[0-9]+-Linux-x86_64\.sh\.sha256' | head -1)
  [ -n "$URL" ] || { log "could not find the Miniforge installer asset in the release metadata"; exit 1; }
  log "installer: $URL"
  curl -fsSL --retry 5 -o "$TOOLS/Miniforge3.sh" "$URL" || { log "Miniforge download failed"; exit 1; }
  if [ -n "$SHAURL" ] && curl -fsSL --retry 5 -o "$TOOLS/Miniforge3.sh.sha256" "$SHAURL"; then
    (cd "$TOOLS" && awk '{print $1"  Miniforge3.sh"}' Miniforge3.sh.sha256 | sha256sum -c -) || { log "Miniforge installer sha256 mismatch"; exit 1; }
  else
    log "WARNING: no .sha256 asset for the installer; proceeding on HTTPS integrity only"
  fi
  bash "$TOOLS/Miniforge3.sh" -b -u -p "$BASE" || { log "Miniforge install failed"; exit 1; }
fi
# shellcheck disable=SC1091
source "$BASE/etc/profile.d/conda.sh"
log "conda $(conda --version)"
# 2 env cvpr
if [ ! -x "$BASE/envs/cvpr/bin/python" ]; then
  log "creating env cvpr (python 3.10)"
  conda create -y -q -n cvpr python=3.10 pip setuptools=65.5.1 wheel ninja psutil || { log "conda create failed"; exit 1; }
fi
conda activate cvpr
PY=$(command -v python); log "python: $PY ($($PY --version 2>&1))"
# 3 torch 2.1.2 + cu121 (upstream pins 2.1.0; 2.1.2 is the bugfix release of the same series, same cu121 ABI -> same mmcv/PL compatibility)
$PY -c "import torch; assert torch.__version__.startswith('2.1.') and torch.version.cuda=='12.1'" 2>/dev/null || {
  log "installing torch 2.1.2+cu121"
  $PY -m pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 || { log "torch install failed"; exit 1; }
}
# 4 upstream requirements minus the prebuilt mmcv (built from source below) + our extra deps
REQ=$F/env/.requirements_cvpr.txt
grep -vE '^\s*-f |mmcv_full|^\s*#|^\s*$' "$DJ/requirements.txt" | grep -vE '^(torch|torchvision)==' > "$REQ"   # torch pins are satisfied by the +cu121 2.1.2 wheels
printf '%s\n' 'omegaconf<2.4' 'decord==0.6.0' 'av>=12,<18' 'einops' 'timm' 'pyyaml' 'tqdm' >> "$REQ"   # omegaconf>=2.4 rejects the devkit split yamls (>10k nodes)
log "installing upstream requirements (+decord/av)"
$PY -m pip install -r "$REQ" > "$S/pip_req_cvpr.log" 2>&1 || { log "requirements install failed (see $S/pip_req_cvpr.log)"; tail -n 30 "$S/pip_req_cvpr.log"; exit 1; }
# navsim (vendored, patched) editable — before the mmcv gate (pb_env_check imports navsim; no mmcv dependency)
$PY -m pip install -q --no-deps -e "$DJ/navsim_v1" || { log "navsim editable install failed"; exit 1; }
# 5 mmcv-full 1.7.2 from source for sm_90 with a CUDA 12.x nvcc (nvhpc module default is 11.8 -> mismatch)
$PY "$F/scripts/h100/pb_env_check.py" >/dev/null 2>&1; PB_RC0=$?   # 0 ok | 2 no CUDA visible (not a build problem) | 3 no sm_90 kernel | 4 import failure
if ! $PY -c "import mmcv, sys; sys.exit(0 if mmcv.__version__=='1.7.2' else 1)" 2>/dev/null || [ "$PB_RC0" = 3 ] || [ "$PB_RC0" = 4 ]; then
  NVCC=$(ls -d /software/u24/nvhpc/*/Linux_x86_64/*/cuda/12.*/bin/nvcc /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/12.*/bin/nvcc /usr/local/cuda-12*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$NVCC" ] || { log "no CUDA 12.x nvcc found -> cannot build mmcv (MSDA_PYTORCH=1 fallback remains)"; NVCC=""; }
  if [ -n "$NVCC" ]; then
    export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH="$(dirname "$NVCC"):$PATH"
    # HPC-SDK layout: cublas_v2.h / cusparse.h / cusolverDn.h (pulled in by ATen/cuda/CUDAContext.h) live in math_libs/<ver>/include
    MATHINC=$(dirname "$(dirname "$CUDA_HOME")")/math_libs/$(basename "$CUDA_HOME")/include
    if [ ! -f "$CUDA_HOME/include/cublas_v2.h" ]; then
      if [ -f "$MATHINC/cublas_v2.h" ]; then export CPATH="$MATHINC${CPATH:+:$CPATH}"
      else SP=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])'); export CPATH="$SP/nvidia/cublas/include:$SP/nvidia/cusparse/include:$SP/nvidia/cusolver/include${CPATH:+:$CPATH}"; fi
    fi
    for h in cublas_v2.h cusparse.h cusolverDn.h; do ok=0; for d in "$CUDA_HOME/include" ${CPATH//:/ }; do [ -f "$d/$h" ] && ok=1; done; [ $ok = 1 ] || log "WARNING: $h not found in CUDA_HOME/include or CPATH -> mmcv build will fail"; done
    export CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST="9.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=16
    log "building mmcv-full 1.7.2 for sm_90 with $NVCC ($("$NVCC" --version | tail -1)) — 20-40 min (a cached wheel from the venv-pb build is reused if present and valid)"
    $PY -m pip uninstall -y -q mmcv-full mmcv_full mmcv 2>/dev/null || true
    $PY -m pip install --no-binary mmcv-full --no-build-isolation -v mmcv-full==1.7.2 > "$S/mmcv_build_cvpr.log" 2>&1 \
      && log "mmcv install ok" || { log "mmcv build FAILED (see $S/mmcv_build_cvpr.log)"; grep -nE "error|Error" "$S/mmcv_build_cvpr.log" | grep -vE "Werror|-Wno" | tail -n 12; }
    if ! env -u LD_LIBRARY_PATH -u CPATH $PY "$F/scripts/h100/pb_env_check.py" >/dev/null 2>&1; then
      log "installed mmcv fails the sm_90 kernel check -> rebuilding without the pip wheel cache"
      $PY -m pip uninstall -y -q mmcv-full 2>/dev/null || true; $PY -m pip cache remove 'mmcv*' >/dev/null 2>&1 || true
      $PY -m pip install --no-cache-dir --no-binary mmcv-full --no-build-isolation -v mmcv-full==1.7.2 > "$S/mmcv_build_cvpr2.log" 2>&1 \
        && log "mmcv rebuild ok" || { log "mmcv rebuild FAILED (see $S/mmcv_build_cvpr2.log)"; grep -nE "error|Error" "$S/mmcv_build_cvpr2.log" | grep -vE "Werror|-Wno" | tail -n 12; }
    fi
  fi
fi
# 7 checks (clean env: no module LD_LIBRARY_PATH/CPATH leakage)
log "--- pb_env_check (mmcv sm_90 kernel vs pytorch reference)"; env -u LD_LIBRARY_PATH -u CPATH $PY "$F/scripts/h100/pb_env_check.py"; PB_RC=$?
log "--- pretraining / perception-free imports"
$PY - <<'PYC'
import importlib, sys
mods = ["torch","torchvision","decord","av","cv2","hydra","pytorch_lightning","numpy","pandas","scipy","sklearn","navsim","nuplan","timm","einops","shapely","geopandas","yaml","omegaconf","mmdet","mmcv"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, repr(e)[:120]))
import torch, numpy, hydra, pytorch_lightning
print("torch", torch.__version__, torch.version.cuda, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("numpy", numpy.__version__, "hydra", hydra.__version__, "PL", pytorch_lightning.__version__)
print("IMPORT FAILURES:", bad) if bad else print("IMPORTS OK")
sys.exit(1 if bad else 0)
PYC
IMP_RC=$?
if [ "$IMP_RC" = 0 ] && { [ "$PB_RC" = 0 ] || [ "${ALLOW_MSDA_FALLBACK:-0}" = 1 ]; }; then
  touch "$F/env/.use_conda_cvpr"; log "env/.use_conda_cvpr set -> drivejepa_env.sh / drivejepa_pb_env.sh now activate conda env cvpr"
  $PY -m pip list 2>/dev/null > "$F/env/cvpr_pip_list_$(date -u +%F).txt"
else
  log "imports failed or no sm_90 mmcv kernel (pb_env_check rc=$PB_RC; set ALLOW_MSDA_FALLBACK=1 to accept the PyTorch fallback) -> NOT switching the env files to cvpr"
fi
log "DONE bootstrap_conda_cvpr (pb_env_check rc=$PB_RC, imports rc=$IMP_RC)"
