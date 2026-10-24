#!/usr/bin/env bash
# Phase-1 bootstrap of the shared H100 server, run on the LOCAL machine (needs `ssh h100` to work):
#   1 directory layout under /home/exouser/henry   2 sudo symlink /home/henry/Desktop/Qwen_Drive
#   3 push code + vendored Drive-JEPA (S0, S0b)      4 start server tmux jobs: fdenv (venv-pf then venv-pb), fdhf (HF downloads)
# Idempotent: safe to re-run.
set -euo pipefail
F=/home/henry/Desktop/Qwen_Drive/frontdrive
S=/home/henry/Desktop/Qwen_Drive/state
mkdir -p "$S"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "1/4 layout"
ssh -o BatchMode=yes h100 'umask 077; H=/home/exouser/henry; for d in Qwen_Drive/frontdrive Qwen_Drive/benchmarks/navsim/dataset Qwen_Drive/benchmarks/navsim/cache Qwen_Drive/benchmarks/navsim/exp/eval Qwen_Drive/benchmarks/pretrain/clips Qwen_Drive/benchmarks/pretrain/lists Qwen_Drive/benchmarks/pretrain/runs Qwen_Drive/benchmarks/pretrain/exports Qwen_Drive/benchmarks/pretrain/smoke Qwen_Drive/state Qwen_Drive/archive tools; do mkdir -p "$H/$d"; done; chmod 700 "$H"; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; ls -ld "$H"'

log "2/4 path-compatibility symlink (sudo)"
ssh -o BatchMode=yes h100 'if [ ! -L /home/henry/Desktop/Qwen_Drive ]; then sudo mkdir -p /home/henry/Desktop && sudo ln -sfn /home/exouser/henry/Qwen_Drive /home/henry/Desktop/Qwen_Drive && sudo chmod 755 /home/henry /home/henry/Desktop; fi; echo "symlink -> $(readlink -f /home/henry/Desktop/Qwen_Drive)"'

log "3/4 push code (S0) and the vendored Drive-JEPA tree (S0b)"
bash "$F/scripts/h100/push.sh" S0 S0b

log "4/4 server jobs in tmux"
ssh -o BatchMode=yes h100 'RQ=/home/exouser/henry/Qwen_Drive; H=/home/exouser/henry; chmod +x $RQ/frontdrive/scripts/h100/*.sh; tmux has-session -t fdenv 2>/dev/null || tmux new-session -d -s fdenv "bash $RQ/frontdrive/scripts/h100/bootstrap_env.sh > $RQ/state/bootstrap_env.log 2>&1 && bash $RQ/frontdrive/scripts/h100/bootstrap_pb_env.sh > $RQ/state/bootstrap_pb_env.log 2>&1"; tmux has-session -t fdhf 2>/dev/null || tmux new-session -d -s fdhf "bash $RQ/frontdrive/scripts/h100/hf_fetch.sh > $RQ/state/hf_fetch.log 2>&1"; cp -n $RQ/frontdrive/scripts/h100/README_SHARED_GPU.md $H/README_SHARED_GPU.md 2>/dev/null || true; ln -sfn $H/README_SHARED_GPU.md ~/README_FROM_HENRY.md; echo "tmux sessions:"; tmux ls'
log "DONE. Follow progress with:  ssh h100 'tail -n 3 /home/exouser/henry/Qwen_Drive/state/bootstrap_env.log /home/exouser/henry/Qwen_Drive/state/bootstrap_pb_env.log /home/exouser/henry/Qwen_Drive/state/hf_fetch.log'"
