#!/usr/bin/env bash
# venv-pb: the upstream Drive-JEPA environment for the perception-based agent (needs the mmcv
# ms_deform_attn CUDA op). Installed exactly as third_party/Drive-JEPA/README.md:89-91 and
# requirements.txt pin it (torch 2.1.0+cu121, mmcv_full 1.7.2 prebuilt wheel, mmdet 2.28.2, numpy 1.24,
# hydra 1.2.0, PL 2.2.1). The H100 (sm_90) is supported by the cu121 wheels. Kept separate from venv-pf.
# Run ON THE SERVER inside tmux:  bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/bootstrap_pb_env.sh
set -euo pipefail
H=/home/exouser/henry; QD=$H/Qwen_Drive; F=$QD/frontdrive; TOOLS=$H/tools
export UV_INSTALL_DIR=$TOOLS/bin UV_NO_MODIFY_PATH=1 UV_PYTHON_INSTALL_DIR=$TOOLS/uv/python UV_CACHE_DIR=$TOOLS/uv/cache
export PIP_CACHE_DIR=$TOOLS/pip-cache
export PATH=$TOOLS/bin:$PATH
umask 077
mkdir -p "$TOOLS/bin" "$TOOLS/uv" "$PIP_CACHE_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
uv python install 3.10 >/dev/null 2>&1 || true
VENV=$F/env/drivejepa_pb; PY=$VENV/bin/python; DJ=$F/third_party/Drive-JEPA
test -f "$DJ/requirements.txt" || { log "missing $DJ/requirements.txt"; exit 1; }
[ -x "$PY" ] || { log "creating $VENV (seeded with pip)"; uv venv --python 3.10 --seed "$VENV"; }
log "pip/setuptools/wheel"
"$PY" -m pip install -q --upgrade pip wheel 'setuptools==65.5.1'
log "torch 2.1.0 + cu121 (upstream pin)"
"$PY" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
log "upstream requirements.txt (incl. mmcv_full 1.7.2 from the openmmlab cu121/torch2.1 index, mmdet 2.28.2)"
"$PY" -m pip install -r "$DJ/requirements.txt" 'omegaconf<2.4'   # omegaconf>=2.4 rejects the devkit split yamls (>10k nodes)
log "navsim_v1 editable"
(cd "$DJ/navsim_v1" && "$PY" -m pip install -e .)
log "smoke: torch / mmcv ops / mmdet / navsim import"
"$PY" "$F/scripts/h100/pb_env_check.py"
log "DONE venv-pb at $VENV"
