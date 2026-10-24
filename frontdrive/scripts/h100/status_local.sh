#!/usr/bin/env bash
# Local wrapper: print the server status (foreground, <= 30 s) and append it to the local state log.
set -u
S=/home/henry/Desktop/Qwen_Drive/state; mkdir -p "$S"
H=h100; ssh -o BatchMode=yes -o ConnectTimeout=15 $H true 2>/dev/null || H=h100-ts
ssh -o BatchMode=yes -o ConnectTimeout=20 $H 'bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/status.sh' 2>&1 | tee -a "$S/status_local.log"
