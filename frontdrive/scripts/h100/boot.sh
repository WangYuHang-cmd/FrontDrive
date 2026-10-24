#!/usr/bin/env bash
# Reboot resilience on the shared H100 server. Installed as a user crontab line:
#   @reboot sleep 120 && /bin/bash /home/exouser/henry/Qwen_Drive/frontdrive/scripts/h100/boot.sh >> /home/exouser/henry/Qwen_Drive/state/boot.log 2>&1
# Waits for CUDA, honours the STOP marker and a disk floor, then re-creates the tmux session that runs
# the (idempotent) queue. Manual relaunch is the same tmux line.
set -u
H=/home/exouser/henry; QD=$H/Qwen_Drive; S=$QD/state; F=$QD/frontdrive
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "boot.sh start boot_id=$(cat /proc/sys/kernel/random/boot_id)"
for _ in $(seq 1 90); do nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { log "NO CUDA after 15 min -> not launching (driver/module problem?)"; exit 1; }
[ -e "$S/STOP" ] && { log "STOP marker present -> not launching"; exit 0; }
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 60 ] || { log "disk ${free_gb} GB < 60 GB -> not launching"; exit 1; }
for m in "$S"/phase5.enabled "$S"/phase5.enabled.*; do   # one marker per chain (knobs as PB_*=... lines), written by phase5_pb_pipeline.sh, removed when done
  [ -e "$m" ] || continue
  TAGM=$(grep -E '^PB_TAG=' "$m" | head -1 | cut -d= -f2-); [ -n "$TAGM" ] || TAGM=pb_e50_s0
  SESS=fdpb_$TAGM; [ "$m" = "$S/phase5.enabled" ] && SESS=fdpb   # the first (pre-per-tag) marker used session fdpb
  if tmux has-session -t "$SESS" 2>/dev/null; then log "tmux $SESS already exists"
  else
    KN=$(grep -E '^PB_[A-Z_]+=[^ ]*$' "$m" | tr '\n' ' ')
    tmux new-session -d -s "$SESS" -n pb "env $KN bash $F/scripts/h100/phase5_pb_pipeline.sh >> $S/phase5.log 2>&1"
    log "phase5 pb chain relaunched in tmux $SESS ($KN)"
  fi
done
if [ -e "$S/pb_queue.enabled" ] && ! tmux has-session -t fdqueue 2>/dev/null; then   # declarative pb queue (pb_queue.sh)
  tmux new-session -d -s fdqueue -n queue "bash $F/scripts/h100/pb_queue.sh >> $S/pb_queue.log 2>&1"; log "pb queue relaunched in tmux fdqueue"
fi
if [ -e "$S/pf_fillers.enabled" ] && ! tmux has-session -t fdpf 2>/dev/null; then   # perception-free fillers (phase6_pf_fillers.sh)
  tmux new-session -d -s fdpf -n pf "bash $F/scripts/h100/phase6_pf_fillers.sh >> $S/pf_fillers.log 2>&1"; log "pf fillers relaunched in tmux fdpf"
fi
for m in "$S"/pretrain.enabled.*; do   # phase7_pretrain_arm.sh markers (CFG/NAME/SEED inside); resumes from latest.pt
  [ -e "$m" ] || continue
  NM=$(grep -E '^NAME=' "$m" | cut -d= -f2-); CF=$(grep -E '^CFG=' "$m" | cut -d= -f2-); SD=$(grep -E '^SEED=' "$m" | cut -d= -f2-)
  if tmux has-session -t "fdpretrain_$NM" 2>/dev/null; then log "tmux fdpretrain_$NM already exists"
  else tmux new-session -d -s "fdpretrain_$NM" -n pt "cd $F && SKIP_SMOKE=1 bash $F/scripts/h100/phase7_pretrain_arm.sh $CF $NM ${SD:-0} >> $S/pretrain.log 2>&1"; log "pretrain arm $NM relaunched in tmux fdpretrain_$NM"; fi
done
if [ -e "$S/www.enabled" ] && ! tmux has-session -t fdweb 2>/dev/null; then   # live HTML progress page (progress_web.sh)
  tmux new-session -d -s fdweb -n web "bash $F/scripts/h100/progress_web.sh >> $S/www.log 2>&1"; log "progress web relaunched in tmux fdweb"
fi
if [ -e "$S/phase5_sequence.enabled" ]; then   # phase5_sequence.sh (② -> ③ back to back) waits for fdpb, then continues its list
  if tmux has-session -t fdpbseq 2>/dev/null; then log "tmux fdpbseq already exists"
  else
    SQ=$(grep -E '^PB_SEQUENCE=' "$S/phase5_sequence.enabled" | head -1 | cut -d= -f2-)
    tmux new-session -d -s fdpbseq -n seq "env PB_SEQUENCE='$SQ' bash $F/scripts/h100/phase5_sequence.sh >> $S/phase5.log 2>&1"
    log "phase5 sequence relaunched in tmux fdpbseq"
  fi
fi
if tmux has-session -t fdq 2>/dev/null; then log "tmux fdq already exists"; exit 0; fi
if [ -x "$F/scripts/h100/queue_h100.sh" ]; then
  tmux new-session -d -s fdq -n queue "bash $F/scripts/h100/queue_h100.sh >> $S/queue.log 2>&1"
  log "queue launched in tmux fdq"
else
  log "no queue script yet (manual phase) -> nothing launched"
fi
