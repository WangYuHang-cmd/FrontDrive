#!/usr/bin/env bash
# Local control plane: pull small results from the H100 server and ingest them locally so that
# docs/ledger/README stay single-sourced on this machine (plan v4.3 appendix H.7). <= ~60 s per run.
# Usage: bash scripts/h100/pull.sh    (cron: */10 * * * * flock -n <state>/.pull.lock bash .../pull.sh >> <state>/pull.log 2>&1)
set -uo pipefail
LQ=/home/henry/Desktop/Qwen_Drive; F=$LQ/frontdrive; S=$LQ/state; B=$LQ/benchmarks/navsim/exp; P=$LQ/benchmarks/pretrain
RQ=/home/exouser/henry/Qwen_Drive
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
avail_gb=$(free -g | awk 'NR==2{print $7}')
[ "${avail_gb:-0}" -ge 8 ] || { log "skip: local RAM available ${avail_gb} GB < 8 GB"; exit 0; }
# reachability is probed with rsync only: the org policy of Henry's Claude Code denies a bare `ssh` command while
# rsync (ssh transport, BatchMode/ConnectTimeout from ~/.ssh/config) is allowed; never call ssh directly here.
reach() { rsync --timeout=20 --list-only "$1:$RQ/state/" >/dev/null 2>&1; }
H=h100; reach "$H" || H=h100-ts
reach "$H" || { log "server unreachable on both routes"; exit 1; }
BK="$S/.pull_backup/$(date +%F)"
RS=(rsync -a --timeout=300 --backup --backup-dir="$BK")
mkdir -p "$S" "$B/eval" "$B/eval_pb" "$P/runs" "$P/exports"
# server state (registry, chain/queue logs, status); local-only control files are protected
"${RS[@]}" --exclude='q_*.train.out' --exclude='pb_train_*.out' --exclude='www/' --exclude='smoke_cfg/' --exclude='e1cmp_ingested.txt' --exclude='push.*' \
  --exclude='pull*.log' --exclude='status_local.log' --exclude='.pull_backup/' --exclude='.pull.lock' \
  "$H:$RQ/state/" "$S/"                                                                              || log "state pull failed"
"${RS[@]}" -m --include='*/' --include='report.json' --include='*.csv' --include='step?.log' --include='*.driver.out' --exclude='*' \
  "$H:$RQ/benchmarks/navsim/exp/eval/" "$B/eval/"                                                    || log "eval pull failed"
"${RS[@]}" -m --include='/*/' --include='/*/report.json' --include='/*/*.csv' --include='/*/run.log' --exclude='*' \
  "$H:$RQ/benchmarks/navsim/exp/eval_pb/" "$B/eval_pb/"                                              || log "eval_pb pull failed"
"${RS[@]}" -m --include='/*/' --include='/*/accounting.json' --include='/*/log_r0.csv' --include='/*/params-pretrain.yaml' --include='/*/gpu_before*.txt' --exclude='*' \
  "$H:$RQ/benchmarks/pretrain/runs/" "$P/runs/"                                                      || log "runs pull failed"
"${RS[@]}" --timeout=900 "$H:$RQ/benchmarks/pretrain/exports/" "$P/exports/"                        || log "exports pull failed"
# per-run small artefacts (pf: run_training.log; pb train_pb.sh: params/accounting/gpu_mem) + the whole pb_smoke table (no ckpts)
"${RS[@]}" -m --exclude='*.ckpt' --exclude='*.pkl' --include='/*/' --include='/*/run_training.log' --include='/*/params.json' --include='/*/accounting.json' \
  --include='/*/gpu_mem.csv' --include='/*/gpu_before.txt' --include='/*/monitor.log' --include='/*/train_pb.stderr' --include='/pb_smoke/**' --exclude='*' "$H:$RQ/benchmarks/navsim/exp/" "$B/" || log "run artefacts pull failed"
# navtest budget ledger: merge server-side navtest runs (pf + pb) into eval/navtest_evals.csv, dedup on (date,tag,sha8)
python3 - "$B/eval/navtest_evals_server.csv" "$F/eval/navtest_evals.csv" <<'PY' || log "navtest merge failed"
import os, sys
src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(src):
    have = set(open(dst).read().splitlines()) if os.path.exists(dst) else set()
    new = [l for l in open(src).read().splitlines() if l.strip() and l not in have]
    if new:
        with open(dst, "a") as fh: fh.write("\n".join(new) + "\n")
        print(f"navtest ledger: {len(new)} new server rows merged")
PY
# pb dev evals (eval_pb/<tag>/report.json): ingest every new one -> evidence copy, paired D5 reports vs the official pb dev
# reference and the e50 baseline export of the same epoch, ledger row (eval/pb_ingest.sh; rule table by tag prefix)
for rep in "$B"/eval_pb/*/report.json; do
  [ -f "$rep" ] || continue
  tag=$(basename "$(dirname "$rep")")
  grep -q '"split": "navtrain_val_official"' "$rep" || continue
  [ -f "$F/docs/evidence/2026-10-17_pb_baseline/${tag}_report.json" ] && continue
  case "$tag" in
    pb_official_v1_dev) args=(official e50);;
    pb_e50_s0_ep*|pb_e50_s0_best*|pb_e50_s0_soup*) args=(e50 e50 pb_e50_s0);;
    pb_e50_s1_ep*|pb_e50_s1_best*|pb_e50_s1_soup*) args=(e50 e50 pb_e50_s1);;
    pb_e50_s2_ep*|pb_e50_s2_best*|pb_e50_s2_soup*) args=(e50 e50 pb_e50_s2);;
    pb_e50_cos_s[0-9]_*) run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(e50 e50 "$run");;   # ④(b) cosine-lr variant of the baseline
    ens_smoke_*|sel_smoke_*|*_smoke) continue;;   # few-scene smoke tests (eval_pb_ensemble.sh / eval_pb_select.sh SMOKE_SCENES) are never ingested
    pb_ens_*) args=(ensemble e50 -);;   # inference-time ensembles (navsim_ext.pb_ensemble_agent; members/mode in the report)
    pb_sel_*) args=(select e50 -);;     # training-free selection rules on one ckpt (navsim_ext.pb_select_agent; mode/eps/member in the report)
    pb_off_ft_all_s[0-9]_*) continue;;   # ⑤ final models trained on train+val: dev is in-sample, never ingested (navtest ledger only)
    pb_off_ft_s[0-9]_*) run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(offinit b3e1d4ab723715be07c8266a7dfbd58abffed6990d654afcf5fbcd8cd97eeb34 "$run");;   # ⑤ probe: released pb v1 ckpt init, train logs only (sha256 of the ckpt from data/benchmarks/download_list_drivejepa_v1.tsv)
    pb_Cp4_s[0-9]_*) run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(Cp4_e40 "$P/exports/e2_Cp4_comma_plain_from_e50_b32_e40_target_encoder.pt" "$run");;   # H100 4x scale test (comma)
    pb_A4_s[0-9]_*)  run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(A4_e40 "$P/exports/e2_A4_public_plain_from_e50_b32_e40_target_encoder.pt" "$run");;    # H100 4x scale test (public)
    pb_Cp_s[0-9]_ep*|pb_Cp_s[0-9]_best*|pb_Cp_s[0-9]_soup*) run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(Cp_e40 "$P/exports/e2_Cp_comma_plain_from_e50_e40_target_encoder.pt" "$run");;
    pb_Dp_s[0-9]_ep*|pb_Dp_s[0-9]_best*|pb_Dp_s[0-9]_soup*) run=${tag%%_ep*}; run=${run%%_best*}; run=${run%%_soup*}; args=(Dp_e40 "$P/exports/e2_Dp_comma_motion_from_e50_e40_target_encoder.pt" "$run");;
    *) log "pb eval $tag: no ingest rule -> run eval/pb_ingest.sh by hand"; continue;;
  esac
  log "pb ingest $tag"
  bash "$F/eval/pb_ingest.sh" "$tag" "${args[@]}" >> "$S/pull_ingest.log" 2>&1 && tail -n 1 "$S/pull_ingest.log" || log "pb ingest $tag failed (state/pull_ingest.log)"
done
# one ingest pass over whatever is new (pulled registry + chain logs; appends ledger rows, refreshes the README)
if [ -f "$S/e1_registry.tsv" ]; then
  SLEEP=0 bash "$F/eval/e1_autoingest.sh" "$S/e1_registry.tsv" "$S/e1cmp_ingested.txt" 1 >> "$S/pull_ingest.log" 2>&1 || log "ingest pass returned $?"
fi
log "pull done via $H"
