#!/usr/bin/env bash
# Read-only status of our work on the shared H100 server. Run ON THE SERVER (also used by cron every
# 10 min to refresh state/status/latest.txt, and by the local wrapper scripts/h100/status_local.sh).
# Prints no addresses of other hosts and no command lines of other users' processes beyond the name.
H=/home/exouser/henry; QD=$H/Qwen_Drive; S=$QD/state; B=$QD/benchmarks/navsim/exp; P=$QD/benchmarks/pretrain
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC) up $(( $(cut -d. -f1 /proc/uptime) / 3600 ))h boot_id $(cat /proc/sys/kernel/random/boot_id)"
echo "== markers PAUSE=$([ -e $S/PAUSE ] && echo yes || echo no) STOP=$([ -e $S/STOP ] && echo yes || echo no) NOTE: $(head -c 200 $S/NOTE 2>/dev/null | tr '\n' ' ')"
echo "== gpu: $(nvidia-smi --query-gpu=driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1)"
echo "== gpu procs (pid MiB ours|FOREIGN name):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r p m; do
  [ -n "$p" ] || continue
  o=FOREIGN; case "$(readlink /proc/$p/cwd 2>/dev/null)" in $H*) o=ours;; esac
  echo "  $p ${m}MiB $o $(cat /proc/$p/comm 2>/dev/null)"
done
echo "== tmux: $(tmux ls 2>&1 | tr '\n' ';')"
echo "== disk free $(df -h / | awk 'NR==2{print $4}') | ram avail $(free -g | awk 'NR==2{print $7}')G | load $(cut -d' ' -f1-3 /proc/loadavg) | ntp=$(timedatectl show -p NTPSynchronized --value 2>/dev/null)"
echo "== our tree: $(du -sh $H 2>/dev/null | cut -f1) under $H"
for f in $B/*/train.log; do [ -f "$f" ] && echo "== E1 $(basename "$(dirname "$f")") mtime $(date -u -r "$f" +%m-%dT%H:%MZ) $(tr '\r' '\n' < "$f" | grep -E '^Epoch [0-9]+:' | tail -1 | cut -c1-70)"; done
for f in $P/runs/*/log_r0.csv; do [ -f "$f" ] && echo "== ARM $(basename "$(dirname "$f")") mtime $(date -u -r "$f" +%m-%dT%H:%MZ) last=$(tail -n 1 "$f" | cut -d, -f1-3)"; done
echo "== evals with report.json: pf $(ls $B/eval/*/report.json 2>/dev/null | wc -l) | pb $(ls $B/eval_pb/*/report.json 2>/dev/null | wc -l)"
[ -f "$S/queue.log" ] && { echo "== queue tail:"; tail -n 5 "$S/queue.log" | cut -c1-160; }
[ -f "$S/phase5.log" ] && { echo "== phase5 (pb chain) tail:"; tail -n 4 "$S/phase5.log" | cut -c1-200; }
[ -f "$B/pb_smoke/summary.tsv" ] && { echo "== pb_smoke rows: $(($(wc -l < "$B/pb_smoke/summary.tsv") - 1)) pick: $(python3 -c "import json; d=json.load(open('$B/pb_smoke/pick.json')); print(d.get('baseline_recommendation'))" 2>/dev/null)"; }
echo "== recent logins (users only): $(last -n 10 2>/dev/null | awk '$1!="reboot" && $1!="wtmp" && NF>3{print $1}' | sort | uniq -c | tr '\n' ' ')"
echo "== oom-killer events (7d): $(journalctl -k -q --since -7d 2>/dev/null | grep -ci 'out of memory') | our crontab lines: $(crontab -l 2>/dev/null | grep -c scripts/h100)"
