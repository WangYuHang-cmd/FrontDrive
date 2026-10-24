#!/usr/bin/env bash
# Wait until the local GPU is quiet (util <= 5 %, used <= 1500 MiB) for two consecutive 60 s checks, then exec the command.
MAXWAIT=${MAXWAIT:-21600}; t=0; ok=0
while [ "$t" -lt "$MAXWAIT" ]; do
  read -r used util <<<"$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')"
  if [ "${util:-100}" -le 5 ] && [ "${used:-99999}" -le 1500 ]; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 2 ] && break
  sleep 60; t=$((t+60))
done
if [ "$ok" -lt 2 ]; then echo "GPU never quiet within $MAXWAIT s (last used=$used util=$util)"; exit 3; fi
echo "GPU quiet at $(date -Is): used=${used}MiB util=${util}%"
exec "$@"
