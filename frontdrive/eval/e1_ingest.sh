#!/usr/bin/env bash
# Ingest one E1 dev evaluation (perception-free protocol, official val logs): copy report.json into the evidence dir, run D5
# paired reports (vs the official checkpoint; vs E1b seed 0 at the same epoch when that export was evaluated), append a
# ledger row.  Generalises scratchpad/e1cmp_ingest.sh (2026-10-15) to the E1b seed-1 run, the E1b seed-0 extension
# (epochs 41-45) and the E1 fine-tunes of later E2 arms.
# Usage: eval/e1_ingest.sh <tag_prefix> <epoch> <run_dir> <init_label> <encoder_pt> <pretrain_objective> <accounting_json|-> [report_name_prefix]
#   tag = <tag_prefix>_ep<epoch> must have benchmarks/navsim/exp/eval/<tag>/report.json (written by scripts/eval_perception_free.sh).
#   <encoder_pt> = the encoder file the run was initialised from (sha256 -> ledger); <accounting_json> = the E2 arm's
#   accounting.json (tokens / FLOPs) or '-' for the official e50 encoder; [report_name_prefix] names the copied report
#   (<prefix>_ep<N>_report.json; default = tag_prefix; the E1b seed-0 run uses 'e1b' to match the learning-curve files).
# Env: DRY=1 -> print the ledger row instead of appending; EV=<dir> evidence dir (default docs/evidence/2026-10-14_E1_encoder_comparison);
#      SEED (default 0); MAXEP (default 40; 45 for the extended E1b run); PSUBSET pretrain-subset label (default derived from init_label).
set -euo pipefail
cd /home/henry/Desktop/Qwen_Drive/frontdrive
source env/drivejepa_env.sh
P=$1; N=$2; RUN=$3; INIT=$4; EPT=$5; OBJ=$6; ACCT=$7; REPNAME=${8:-$P}
TAG=${P}_ep$N
EV=${EV:-docs/evidence/2026-10-14_E1_encoder_comparison}; mkdir -p "$EV"
SEED=${SEED:-0}; MAXEP=${MAXEP:-40}
EXP=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp
ATTRS=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/diagnostics/d5/attrs_navtrain_val_official.csv
OFF=$EXP/eval/e0_official_pf_dev/score/2026.10.12.20.12.50.csv
REP=$EXP/eval/$TAG/report.json; [ -f "$REP" ] || { echo "no report for $TAG"; exit 1; }
CSV=$(ls "$EXP/eval/$TAG"/score/*.csv | tail -n 1)
cp "$REP" "$EV/${REPNAME}_ep${N}_report.json"
ESHA=$(sha256sum "$EPT" | cut -c1-8)
if [ "$ACCT" != "-" ] && [ -f "$ACCT" ]; then
  read -r PTOK PFLOP PSTEPS < <(python3 -c "import json;d=json.load(open('$ACCT'));print('%.3e %.3e %d'%(d['tokens'],d['flops_est'],d['steps']))")
  DRAWS=$(python3 -c "import json;d=json.load(open('$ACCT'));print(d['steps']*d['batch_size']*d.get('world_size',1))")
else PTOK=official; PFLOP=official; PSTEPS=official; DRAWS=""; fi
if [ -z "${PSUBSET:-}" ]; then
  case "$INIT" in e50) PSUBSET="public3(official e50)";; A_*|B_*) PSUBSET=public_v0;; C_*|D_*|Dshuf_*) PSUBSET="public_v0+comma_v0";; *) PSUBSET=comma_v0;; esac
fi
S_TMP=/tmp/claude-1001/-home-henry-Desktop-Qwen-Drive/5408a5ef-9d7f-4718-9d96-6350ca86081b/scratchpad/e1cmp; mkdir -p "$S_TMP"
d5() { # a_csv a_tag b_csv b_tag
  local out=$S_TMP/d5_${2}_vs_${4}; mkdir -p "$out"
  python diagnostics/d5_error_buckets.py report --attrs "$ATTRS" --scores "$1" --scores-b "$3" --tag-a "$2" --tag-b "$4" --out "$out" > "$out/stdout.txt" 2>&1 || { echo "D5 failed for $2 vs $4"; tail -n 5 "$out/stdout.txt"; return 1; }
  cp "$out/report.md" "$EV/report_${2}_vs_${4}.md"; cp "$out/report.json" "$EV/report_${2}_vs_${4}.json"
}
d5 "$OFF" e0_official "$CSV" "$TAG"
E1B=""
if [ "$P" != e1b_s0 ]; then E1B=$(ls "$EXP/eval/e1b_s0_ep$N"/score/*.csv 2>/dev/null | tail -n 1 || true); fi
if [ -n "$E1B" ]; then d5 "$E1B" "e1b_ep$N" "$CSV" "$TAG"; fi
python - "$EV" "$TAG" "$N" "$RUN" "$INIT" "$ESHA" "$OBJ" "$PTOK" "$PFLOP" "$CSV" "${E1B:-}" "${DRY:-0}" "$SEED" "$MAXEP" "$PSUBSET" "$PSTEPS" "$DRAWS" "$REPNAME" <<'PY'
import json, sys, os, re
ev, tag, n, run, init, esha, obj, ptok, pflop, csv, e1b, dry, seed, maxep, psub, psteps, draws, repname = sys.argv[1:19]
rep = json.load(open(f"{ev}/{repname}_ep{n}_report.json"))
def delta(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); row = [r for r in d["diff"] if r["bucket"] == "all"][0]
    return row["delta_PDMS"], row["ci_low"], row["ci_high"], row.get("delta_EP"), row.get("delta_DAC")
dv_off = delta(f"{ev}/report_e0_official_vs_{tag}.json")
dv_e1b = delta(f"{ev}/report_e1b_ep{n}_vs_{tag}.json")
sub = " / ".join(f"{k} {rep[c]:.2f}" for k, c in [("NC","no_at_fault_collisions"),("DAC","drivable_area_compliance"),("EP","ego_progress"),("TTC","time_to_collision_within_bound"),("C","comfort"),("DDC","driving_direction_compliance")] if c in rep)
m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.\d{2}\.csv$", csv)
stamp = f"{m[1]}{m[2]}{m[3]}-{m[4]}{m[5]}"; date = f"{m[1]}-{m[2]}-{m[3]}"
fmt = lambda d: f"{d[0]:+.2f} [{d[1]:.2f},{d[2]:.2f}]"
if init == "e50":
    notes = f"E1b official recipe (e50 encoder, sha256 {esha}...), seed {seed}, epoch {n} of {maxep}" + (" (post-recipe extension at constant lr for the late-epoch mean)" if int(n) > 40 else "") + f"; dev PDMS {rep['score']:.2f}: {sub}; paired vs official 87.10: {fmt(dv_off)}"
else:
    notes = f"E1 encoder comparison, encoder init {init} (E2 arm export, sha256 {esha}...), official recipe seed {seed}, epoch {n} of {maxep}; dev PDMS {rep['score']:.2f}: {sub}; paired vs official 87.10: {fmt(dv_off)}"
if dv_e1b: notes += f"; paired vs E1b seed 0 epoch {n}: {fmt(dv_e1b)} (EP {dv_e1b[3]:+.2f}; DAC {dv_e1b[4]:+.2f})"
notes += f"; export sha8 {rep['ckpt_sha8']}"
steps = int(n) * 5319 // 4
row = [f"{stamp}_{tag}_dev", date, "E1", "", "", "", "", "", "", f"{os.path.basename(run)}/exports/epoch_{n}.ckpt", "navtrain_train_official(978 logs)", "", "", str(steps), "", "", str(seed), "", "", "7.0", "navtrain_val_official", "N", "1", "", "", "", "", f"{rep['score']:.2f}", f"{rep['ego_progress']:.2f}", "", "", f"{ev}/{repname}_ep{n}_report.json", '"' + notes.replace('"', "'") + '"', esha, init, "N", "pf", psub, obj, psteps, ptok, pflop, draws, str(rep["n_scenes"]), f"{dv_off[1]:.2f}", f"{dv_off[2]:.2f}", "0.42-0.49", "dev"]
assert len(row) == 48, len(row)
line = ",".join(row)
if dry == "1": print("DRY ledger row:", line)
else:
    open("eval/ledger.csv", "a").write(line + "\n")
print(f"INGESTED {tag}: PDMS {rep['score']:.2f} | vs official {fmt(dv_off)}" + (f" | vs E1b s0 ep{n} {fmt(dv_e1b)}" if dv_e1b else " | no E1b seed-0 export at this epoch"))
PY
