#!/usr/bin/env bash
# Live HTML progress page ON THE SERVER (tmux `fdweb`): regenerates state/www/index.html every INTERVAL seconds with
# scripts/h100/progress_page.py and serves state/www with Python's http.server bound to the server's Tailscale address only
# (nothing is exposed on the public interface; `tailscale ip -4` is read at runtime so no address is written anywhere).
# Open  http://<server tailscale ip>:8787/  from any device in the tailnet. Fallback without tailscale: binds 127.0.0.1
# (then use an ssh tunnel: ssh -L 8787:127.0.0.1:8787 h100). boot.sh relaunches this while state/www.enabled exists.
# Launch: tmux new-session -d -s fdweb "bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/progress_web.sh >> /home/henry/Desktop/Qwen_Drive/state/www.log 2>&1"
set -u
QD=/home/henry/Desktop/Qwen_Drive; F=$QD/frontdrive; S=$QD/state; WWW=$S/www
PORT=${PORT:-8787}; INTERVAL=${INTERVAL:-60}
log() { echo "[$(date -u +%FT%TZ)] [web] $*"; }
mkdir -p "$WWW"
BIND=$(tailscale ip -4 2>/dev/null | head -n 1)
[ -n "$BIND" ] || { BIND=127.0.0.1; log "tailscale ip unavailable -> binding $BIND (use an ssh tunnel)"; }
printf '# progress web active since %s\nPORT=%s\n' "$(date -u +%FT%TZ)" "$PORT" > "$S/www.enabled"
python3 "$F/scripts/h100/progress_page.py" --out "$WWW/index.html" --refresh "$INTERVAL" || log "first page generation failed"
python3 -m http.server --bind "$BIND" --directory "$WWW" "$PORT" >> "$S/www.http.log" 2>&1 &
HTTP_PID=$!
log "serving $WWW on http://$BIND:$PORT/ (pid $HTTP_PID); regenerating every $INTERVAL s"
trap 'kill $HTTP_PID 2>/dev/null; rm -f "$S/www.enabled"; log "stopped"; exit 0' INT TERM
while :; do
  sleep "$INTERVAL"
  python3 "$F/scripts/h100/progress_page.py" --out "$WWW/index.html" --refresh "$INTERVAL" 2>> "$S/www.log" || log "page generation failed (see above)"
  kill -0 "$HTTP_PID" 2>/dev/null || { log "http.server died -> restarting"; python3 -m http.server --bind "$BIND" --directory "$WWW" "$PORT" >> "$S/www.http.log" 2>&1 & HTTP_PID=$!; }
done
