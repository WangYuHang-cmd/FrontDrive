#!/usr/bin/env bash
# Build the perception-free / pretraining venv (venv-pf) on the H100 server.
# Run ON THE SERVER inside tmux:  bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/bootstrap_env.sh
# Everything (uv binary, managed Python, caches, venv) lives under /home/exouser/henry.
# Mirrors the local venv exactly: CPython 3.10.22, torch 2.8.0+cu128, the frozen package set,
# nuplan-devkit at the recorded commit, navsim editable from the patched vendored tree.
set -euo pipefail
H=/home/exouser/henry
QD=$H/Qwen_Drive
F=$QD/frontdrive
TOOLS=$H/tools
export UV_INSTALL_DIR=$TOOLS/bin UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR=$TOOLS/uv/python UV_CACHE_DIR=$TOOLS/uv/cache
export PATH=$TOOLS/bin:$PATH
umask 077
install -d -m 700 "$H" "$QD" "$TOOLS" "$TOOLS/bin" "$TOOLS/uv"
command -v git >/dev/null 2>&1 || { echo "git is required (sudo apt-get install -y git)"; exit 1; }

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv 0.12.23 into $TOOLS/bin"
  curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
fi
log "uv $(uv --version)"

log "installing CPython 3.10.22 (uv-managed, under $UV_PYTHON_INSTALL_DIR)"
uv python install cpython-3.10.22-linux-x86_64-gnu || uv python install 3.10

VENV=$F/env/drivejepa
PY=$VENV/bin/python
FR=$F/env/drivejepa_freeze_2026-10-16.txt
test -f "$FR" || { log "missing freeze file $FR"; exit 1; }
test -d "$F/third_party/Drive-JEPA/navsim_v1" || { log "missing vendored Drive-JEPA tree"; exit 1; }

if [ ! -x "$PY" ]; then
  log "creating venv $VENV"
  uv venv --python 3.10.22 "$VENV" || uv venv --python 3.10 "$VENV"
fi

log "phase 1/4: torch stack from the PyTorch cu128 index (sm_90 is included in cu128 wheels)"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128

log "phase 2/4: the rest of the frozen set, exactly as pinned, no resolver"
REST=$F/env/.freeze_rest_h100.txt
grep -vE '^(torch==|torchvision==|triton==|nvidia-)|^-e |^nuplan-devkit|^navsim' "$FR" > "$REST"
if ! uv pip install --python "$PY" --no-deps -r "$REST"; then
  log "bulk install failed; retrying package by package and recording deviations"
  : > "$F/env/drivejepa_freeze_h100_deviations.txt"
  while read -r line; do
    [ -z "$line" ] && continue
    uv pip install --python "$PY" --no-deps "$line" >/dev/null 2>&1 </dev/null \
      || { echo "FAILED_PIN $line" | tee -a "$F/env/drivejepa_freeze_h100_deviations.txt"; \
           uv pip install --python "$PY" --no-deps "${line%%==*}" >/dev/null 2>&1 </dev/null \
             && echo "INSTALLED_UNPINNED ${line%%==*}" | tee -a "$F/env/drivejepa_freeze_h100_deviations.txt" || true; }
  done < "$REST"
fi

log "phase 3/4: nuplan-devkit at the recorded commit (--no-deps; its own pins conflict)"
uv pip install --python "$PY" --no-deps \
  'nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@ce3c323af01c0d7ec5672f7832ef53f9c679aab0'

log "phase 4/4: patched vendored navsim (editable, --no-deps)"
uv pip install --python "$PY" --no-deps -e "$F/third_party/Drive-JEPA/navsim_v1"

log "pip check (report only)"
uv pip check --python "$PY" || true

log "system libs needed by cv2 (non-headless build)"
ldconfig -p | grep -E 'libGL\.so\.1|libgthread-2\.0' | head -2 || log "WARNING: libGL/libgthread not found"

log "env check"
"$PY" "$F/scripts/h100/env_check.py"
log "DONE venv-pf at $VENV"
