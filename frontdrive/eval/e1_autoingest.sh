#!/usr/bin/env bash
# Watch the E1 eval chains listed in a registry and ingest every finished evaluation with eval/e1_ingest.sh, then refresh
# the README results block.  v3 (2026-10-16): registry-driven (replaces the scratchpad watchers v1/v2).
# Usage: eval/e1_autoingest.sh <registry.tsv> <done_list> [max_iterations (5-min polls; default 2016 = 7 days)]
#   registry columns (tab): prefix  run_dir  init_label  export_pt  objective  accounting(-)  chain_log
#   The chain log is scanned for lines "EVAL <prefix>_ep<N> EXIT <rc>" written by scratchpad/e1_export_eval_chain_v3.sh.
set -uo pipefail
cd /home/henry/Desktop/Qwen_Drive/frontdrive
REG=$1; DONE=$2; MAX=${3:-2016}; SLEEP=${SLEEP:-300}; touch "$DONE"
EV=docs/evidence/2026-10-14_E1_encoder_comparison
LOGDIR=/tmp/claude-1001/-home-henry-Desktop-Qwen-Drive/5408a5ef-9d7f-4718-9d96-6350ca86081b/scratchpad/e1cmp; mkdir -p "$LOGDIR"
for i in $(seq 1 "$MAX"); do
  changed=0
  while IFS=$'\t' read -r prefix run init ept obj acct clog; do
    [ "$prefix" = prefix ] && continue; [ -z "$prefix" ] && continue; [ -f "$clog" ] || continue
    while read -r _ tag _ rc; do
      [ -n "$tag" ] || continue
      grep -qx "$tag" "$DONE" && continue
      if [ "$rc" != 0 ]; then echo "[$(date '+%m-%d %H:%M')] $tag failed rc=$rc -> not ingested"; echo "$tag" >> "$DONE"; continue; fi
      N=${tag##*_ep}
      case "$prefix" in e1b_s0) rep=e1b; seed=0; maxep=45;; e1b_s1) rep=e1b_s1; seed=1; maxep=40;; *) rep=$prefix; seed=0; maxep=40;; esac
      echo "[$(date '+%m-%d %H:%M')] ingesting $tag"
      if SEED=$seed MAXEP=$maxep EV=$EV bash eval/e1_ingest.sh "$prefix" "$N" "$run" "$init" "$ept" "$obj" "$acct" "$rep" > "$LOGDIR/ingest_$tag.log" 2>&1; then tail -n 1 "$LOGDIR/ingest_$tag.log"; changed=1; else echo "INGEST FAILED $tag"; tail -n 5 "$LOGDIR/ingest_$tag.log"; fi
      echo "$tag" >> "$DONE"
    done < <(grep -oE "EVAL ${prefix}_ep[0-9]+ EXIT [0-9]+" "$clog" 2>/dev/null)
  done < "$REG"
  if [ "$changed" = 1 ]; then source env/drivejepa_env.sh; python eval/e1cmp_summary.py "$EV" --readme 2>&1 | tail -n 1; fi
  [ "$i" -lt "$MAX" ] && sleep "$SLEEP"
done
echo "AUTOINGEST EXIT $(date -Is)"
