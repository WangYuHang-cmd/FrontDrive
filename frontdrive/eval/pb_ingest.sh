#!/usr/bin/env bash
# Ingest one perception-based (pb) dev evaluation written on the H100 by scripts/h100/eval_pb.sh and pulled by
# scripts/h100/pull.sh (benchmarks/navsim/exp/eval_pb/<tag>/{report.json,<stamp>.csv}): copy the report into the evidence
# dir, run paired log-clustered bootstrap reports (diagnostics/d5_error_buckets.py) vs the official pb v1 ckpt's dev
# reference (eval_pb/pb_official_v1_dev) and, for encoder-swap runs, vs the e50 baseline export of the same epoch
# (eval_pb/pb_e50_s0_ep<N>), then append a ledger row (protocol pb, dev_or_test dev).
# Usage: eval/pb_ingest.sh <tag> <init_label> <encoder_pt|e50> [run_tag]
#   <tag>         eval_pb/<tag>/report.json, e.g. pb_e50_s0_ep16, pb_Cp_s0_ep18, pb_official_v1_dev
#   <init_label>  e50 | Cp_e40 | Dp_e40 | official | offinit (⑤ continued training from the released pb v1 ckpt)   (ledger encoder_init)
#   <encoder_pt>  the encoder file the run was initialised from (sha256 -> ledger), 'e50' for the upstream default file, or a 64-hex
#                 sha256 when the file is not on this machine (offinit: the released pb ckpt lives only on the server)
#   [run_tag]     training run dir under benchmarks/navsim/exp (default: <tag> without _ep<N>) -> accounting.json knobs
# Env: DRY=1 print the row instead of appending; EV=<evidence dir> (default docs/evidence/2026-10-17_pb_baseline);
#      NBOOT=10000 bootstrap draws; REF=pb_official_v1_dev reference eval tag; MAXEP=20
set -euo pipefail
cd /home/henry/Desktop/Qwen_Drive/frontdrive
source env/drivejepa_env.sh
TAG=$1; INIT=$2; EPT=$3; RUN=${4:-${1%_ep*}}
EV=${EV:-docs/evidence/2026-10-17_pb_baseline}; mkdir -p "$EV"
NBOOT=${NBOOT:-10000}; REF=${REF:-pb_official_v1_dev}; MAXEP=${MAXEP:-20}
EXP=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/exp
DJ=$EXP/Drive-JEPA-cache
ATTRS=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim/diagnostics/d5/attrs_navtrain_val_official.csv
REP=$EXP/eval_pb/$TAG/report.json; [ -f "$REP" ] || { echo "no report for $TAG ($REP)"; exit 1; }
CSV=$(ls -t "$EXP/eval_pb/$TAG"/*.csv | head -n 1)
cp "$REP" "$EV/${TAG}_report.json"
[ "$EPT" = e50 ] && EPT=$DJ/vitl_merge_3dataset_e50.pt
if [[ "$EPT" =~ ^[0-9a-f]{64}$ ]]; then ESHA=${EPT:0:8}
else [ -f "$EPT" ] || { echo "encoder file $EPT not found"; exit 1; }; ESHA=$(sha256sum "$EPT" | cut -c1-8); fi
S_TMP=/tmp/claude-1001/-home-henry-Desktop-Qwen-Drive/5408a5ef-9d7f-4718-9d96-6350ca86081b/scratchpad/pbcmp; mkdir -p "$S_TMP"
d5() {  # a_csv a_tag b_csv b_tag -> $EV/report_<a>_vs_<b>.{md,json}
  local out=$S_TMP/d5_${2}_vs_${4}; mkdir -p "$out"
  python diagnostics/d5_error_buckets.py report --attrs "$ATTRS" --scores "$1" --scores-b "$3" --tag-a "$2" --tag-b "$4" --n-boot "$NBOOT" --out "$out" > "$out/stdout.txt" 2>&1 \
    || { echo "D5 failed for $2 vs $4"; tail -n 5 "$out/stdout.txt"; return 1; }
  cp "$out/report.md" "$EV/report_${2}_vs_${4}.md"; cp "$out/report.json" "$EV/report_${2}_vs_${4}.json"
}
REFCSV=""; [ "$TAG" != "$REF" ] && REFCSV=$(ls -t "$EXP/eval_pb/$REF"/*.csv 2>/dev/null | head -n 1 || true)
[ -n "$REFCSV" ] && d5 "$REFCSV" "$REF" "$CSV" "$TAG"
EP=""; BASECSV=""; BEST=0
if [[ "$TAG" == *_ep* ]]; then
  EP=${TAG##*_ep}
  if [[ "$TAG" != pb_e50_s0_ep* ]] && [ "$INIT" != offinit ]; then BASECSV=$(ls -t "$EXP/eval_pb/pb_e50_s0_ep$EP"/*.csv 2>/dev/null | head -n 1 || true); fi
  [ -n "$BASECSV" ] && d5 "$BASECSV" "pb_e50_s0_ep$EP" "$CSV" "$TAG"
elif [[ "$TAG" == *_best* ]]; then   # Lightning best_epoch=<N>.ckpt (0-based) = best val/score_epoch after epoch N+1, dev-selected like the released ckpt
  EP=$(( ${TAG##*_best} + 1 )); BEST=1
elif [[ "$TAG" == *_soup* ]]; then   # weight average of exports <first>_<last> (eval/soup_exports.py); EP = last epoch, BEST=2 marks the soup
  EP=${TAG##*_}; BEST=2
elif [[ "$TAG" == pb_ens_* ]]; then  # inference-time ensemble (navsim_ext.pb_ensemble_agent); members and mode are in the report
  BEST=3
elif [[ "$TAG" == pb_sel_* ]]; then  # training-free selection rule on one checkpoint (navsim_ext.pb_select_agent); mode/eps/member in the report
  BEST=4
fi
ACCT=$EXP/$RUN/accounting.json; PARAMS=$EXP/$RUN/params.json
python - "$EV" "$TAG" "$INIT" "$ESHA" "$CSV" "$REF" "${REFCSV:+1}" "$EP" "${BASECSV:+1}" "$ACCT" "$PARAMS" "$RUN" "$MAXEP" "${DRY:-0}" "$EXP" "$BEST" <<'PY'
import json, os, re, sys
ev, tag, init, esha, csv, ref, has_ref, ep, has_base, acct_p, params_p, run, maxep, dry, exp, best = sys.argv[1:17]
rep = json.load(open(f"{ev}/{tag}_report.json"))
def delta(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); row = [r for r in d["diff"] if r["bucket"] == "all"][0]
    return row["delta_PDMS"], row["ci_low"], row["ci_high"], row.get("delta_EP"), row.get("delta_DAC")
fmt = lambda d: f"{d[0]:+.2f} [{d[1]:.2f},{d[2]:.2f}]"
dv_ref = delta(f"{ev}/report_{ref}_vs_{tag}.json") if has_ref else None
dv_base = delta(f"{ev}/report_pb_e50_s0_ep{ep}_vs_{tag}.json") if (has_base and ep) else None
sub = " / ".join(f"{k} {rep[c]:.2f}" for k, c in [("NC","no_at_fault_collisions"),("DAC","drivable_area_compliance"),("EP","ego_progress"),
                                                  ("TTC","time_to_collision_within_bound"),("C","comfort"),("DDC","driving_direction_compliance")] if c in rep)
m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.\d{2}\.csv$", csv)
stamp = f"{m[1]}{m[2]}{m[3]}-{m[4]}{m[5]}"; date = f"{m[1]}-{m[2]}-{m[3]}"
acct = json.load(open(acct_p)) if os.path.exists(acct_p) else {}
params = json.load(open(params_p)) if os.path.exists(params_p) else {}
knobs = params or acct
seed = str(knobs.get("seed", "")) if ep else ""
steps_ep = acct.get("steps_per_epoch")
opt_steps = str(steps_ep * int(ep)) if (steps_ep and ep) else ""
peak = acct.get("peak_vram_mib_our_pid"); peak_gb = f"{peak/1024:.1f}" if peak else ""
sps = acct.get("s_per_step_last_progress"); sps_s = f"{sps:.2f}" if sps else ""
act = knobs.get("encoder_activation_checkpointing"); act_s = "on" if act is None or act else "off"
ckpt = rep["ckpt"].replace(exp + "/", "")
if ep:
    lr_s = str(knobs.get("lr", "1e-4")); sched = str(knobs.get("lr_schedule", "none") or "none")
    warm = str(knobs.get("lr_warmup_steps", "0") or "0")
    sched_s = "" if sched == "none" else f", {sched} lr decay to {knobs.get('lr_min_mult', '0.1')}x" + (f" after {warm} warm-up steps" if warm != "0" else "")
    init_s = (f"init = released pb v1 ckpt, all agent weights (sha256 {esha}...), continued training on the {knobs.get('train_logs', 'train')} logs"
              if init == "offinit" else f"encoder init {init} (sha256 {esha}...)")
    recipe = (f"pb Drive-JEPA perception-based, upstream recipe on 1 H100 (effective batch {knobs.get('effective_batch', 64)} = "
              f"{knobs.get('batch', 64)}x{knobs.get('accum', 1)}, {knobs.get('epochs', maxep)} epochs, Adam {lr_s} backbone 0.1x{sched_s}, 16-mixed, "
              f"side cams from calibration, act-ckpt {act_s}, conda cvpr torch {knobs.get('torch', '2.1.2')}), {init_s}, "
              f"seed {seed}, epoch {ep} of {knobs.get('epochs', maxep)}")
    if best == "1":
        recipe += " = Lightning best val/score_epoch checkpoint (selected on the dev logs exactly like the released official ckpt; NOT an unselected late-window export)"
    elif best == "2":
        recipe += f" = uniform weight average of the fp16 exports {tag.split('_soup')[-1].replace('_', '-')} (eval/soup_exports.py; zero training cost, no dev selection involved)"
    data_subset = "navtrain_train_official(978 logs)"
elif best == "3":
    recipe = (f"inference-time ensemble ({rep.get('ensemble_mode')}, close_m {rep.get('close_m')}) of {len(rep.get('members', []))} pb checkpoints: "
              f"{', '.join(str(m).split('/exp/')[-1] for m in rep.get('members', []))} (navsim_ext.pb_ensemble_agent; single trajectory output, no training, no dev selection)")
    data_subset = "navtrain_train_official(978 logs)"
elif best == "4":
    recipe = (f"training-free proposal selection '{rep.get('select_mode')}' (eps {rep.get('eps')}) on {str(rep.get('member', '')).split('/exp/')[-1]} "
              f"(navsim_ext.pb_select_agent; the model's own proposals and score heads, single trajectory output, no training; eps tuned on dev)")
    data_subset = "navtrain_train_official(978 logs)"
else:
    recipe = f"official Drive-JEPA perception-based v1 ckpt on the dev split (dev reference for plan v4.3 ②/③; side cams from calibration = identical features by construction), encoder {init} (sha256 {esha}...)"
    data_subset = ""
notes = f"{recipe}; dev PDMS {rep['score']:.2f}: {sub}"
if dv_ref: notes += f"; paired vs official pb v1 dev ref: {fmt(dv_ref)} (EP {dv_ref[3]:+.2f}; DAC {dv_ref[4]:+.2f})"
if dv_base: notes += f"; paired vs e50 baseline ep{ep}: {fmt(dv_base)} (EP {dv_base[3]:+.2f}; DAC {dv_base[4]:+.2f})"
if acct.get("wall_s") and acct.get("rc") == 0: notes += f"; run wall {acct['wall_s']/3600:.1f} h, peak VRAM {peak} MiB, {sps_s} s/step"
notes += f"; eval sha8 {rep['ckpt_sha8']}; hardware h100"
psub = {"e50": "public3(official e50)", "official": "public3(official e50)", "offinit": "public3(official e50)"}.get(init, "public_v0+comma_v0" if init[:1] in "ABCD" and "_" in init and init[0] in "CD" else "comma_v0")
if init.startswith(("Cp", "Dp")): psub = "comma_v0"
obj = "official" if init in ("e50", "official") else ("motion" if init.startswith("D") else "plain")
if init == "offinit": obj = "official_ft"   # ⑤: the released weights continued with our recipe
if best == "3": psub, obj = "public3(official e50)", "ensemble"
if best == "4": psub, obj = "public3(official e50)", "select"
row = [f"{stamp}_{tag}_dev", date, "PB", "", "", "", "", "", "", ckpt, data_subset, "", "", opt_steps, "", "", seed, "", "", peak_gb,
       "navtrain_val_official", "N", "1", "", "", "", "", f"{rep['score']:.2f}", f"{rep['ego_progress']:.2f}", "", "",
       f"{ev}/{tag}_report.json", '"' + notes.replace('"', "'") + '"', esha, init, "N", "pb", psub, obj, "", "", "", "",
       str(rep["n_scenes"]), f"{dv_ref[1]:.2f}" if dv_ref else "", f"{dv_ref[2]:.2f}" if dv_ref else "", sps_s, "dev"]
assert len(row) == 48, len(row)
line = ",".join(row)
if dry == "1": print("DRY ledger row:", line)
else:
    have = open("eval/ledger.csv").read()
    if row[0] + "," in have: print(f"ledger already has {row[0]} -> not appended")
    else: open("eval/ledger.csv", "a").write(line + "\n")
print(f"INGESTED {tag}: PDMS {rep['score']:.2f}" + (f" | vs official pb dev ref {fmt(dv_ref)}" if dv_ref else "") + (f" | vs e50 ep{ep} {fmt(dv_base)}" if dv_base else ""))
PY
