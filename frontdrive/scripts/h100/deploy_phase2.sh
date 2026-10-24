#!/usr/bin/env bash
# Phase 2 (local): push the current code (incl. the public download lists) and start the server-side
# self-provisioning of all public NAVSIM assets in tmux `fdprov`. Idempotent.
set -euo pipefail
F=/home/henry/Desktop/Qwen_Drive/frontdrive
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "1/2 push code (S0)"; bash "$F/scripts/h100/push.sh" S0
log "2/2 start fdprov on the server"
ssh -o BatchMode=yes h100 'RQ=/home/exouser/henry/Qwen_Drive; chmod +x $RQ/frontdrive/scripts/h100/*.sh; tmux has-session -t fdprov 2>/dev/null || tmux new-session -d -s fdprov "bash $RQ/frontdrive/scripts/h100/selfprovision_navsim.sh >> $RQ/state/provision.log 2>&1"; echo "tmux sessions:"; tmux ls; echo "--- bootstrap_env.log:"; tail -n 3 $RQ/state/bootstrap_env.log 2>/dev/null; echo "--- hf_fetch.log:"; tail -n 3 $RQ/state/hf_fetch.log 2>/dev/null; df -h / | tail -1'
log "DONE. Follow with: ssh h100 'tail -n 5 /home/exouser/henry/Qwen_Drive/state/provision.log'"
