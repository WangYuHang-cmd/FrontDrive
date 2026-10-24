#!/usr/bin/env bash
# Post-step for the conda env cvpr ON THE SERVER: wait for the bootstrap (tmux fdconda) to finish, pin
# omegaconf<2.4 (devkit yaml node limit), re-run the environment checks, report the switch marker.
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
while tmux has-session -t fdconda 2>/dev/null; do sleep 60; done
log "fdconda finished; bootstrap log tail:"; tail -n 4 "$S/bootstrap_conda_cvpr.log" 2>/dev/null
BASE=/home/exouser/henry/tools/miniforge3
[ -f "$BASE/etc/profile.d/conda.sh" ] || { log "no miniforge at $BASE"; exit 1; }
# shellcheck disable=SC1091
source "$BASE/etc/profile.d/conda.sh" && conda activate cvpr || { log "conda activate cvpr failed"; exit 1; }
python -m pip install -q 'omegaconf<2.4' && log "omegaconf pinned: $(python -c 'import omegaconf;print(omegaconf.__version__)')"
log "--- pb_env_check"; env -u LD_LIBRARY_PATH -u CPATH python "$F/scripts/h100/pb_env_check.py"; rc=$?
log "--- env marker: $([ -f "$F/env/.use_conda_cvpr" ] && echo set || echo NOT set) (pb_env_check rc=$rc)"
log "--- python via env file:"; ( source "$F/env/drivejepa_env.sh" && command -v python && python -c "import torch, navsim; print('torch', torch.__version__, 'cuda', torch.version.cuda)" )
log "DONE cvpr_postfix"
