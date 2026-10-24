#!/usr/bin/env bash
# Training smoke of the pb agent on the H100 (plan v4.3 appendix H.8 level (iv)): micro-batch sweep through
# scripts/h100/train_pb.sh with trainer.params.limit_train_batches=STEPS, limit_val_batches=VAL_BATCHES, max_epochs=1
# (no exports), recording rc / CUDA OOM, s/step (PL progress bar), peak VRAM of our PID (nvidia-smi --query-compute-apps,
# PB_VRAM_SAMPLE_S=2) and whether last.ckpt was written, into $NAVSIM_EXP_ROOT/pb_smoke/<name>/ and the table
# $NAVSIM_EXP_ROOT/pb_smoke/summary.tsv. The sweep stops at the first OOM (larger sizes skipped).
# Rows: <b> for each size (ACT_CKPT=1 = upstream activation checkpointing) and, for the first size, <b>_noackpt (ACT_CKPT=0:
# the ViT keeps its activations -> more VRAM, faster, identical math; the way to push VRAM up WITHOUT changing the recipe).
# Pick (pick.json): the largest micro-batch with rc 0, no OOM and peak <= CAP_MIB (default 60000, H.10 ① self-limit on the
# shared 80 GB), plus the recommended BASELINE knobs: effective batch stays 64 = BATCH 64 x ACCUM 1 (Henry 2026-10-17: no
# gradient accumulation; larger micro-batches are throughput measurements only -> this script passes
# ALLOW_NONSTANDARD_BATCH=1 for them; a bigger effective batch is a §2-④ knowledge point with the lr scaling recorded).
# The baseline size is then resumed from its last.ckpt for a second epoch (RESUME_CHECK=1) to prove the resume path.
#   usage: bash /home/henry/Desktop/Qwen_Drive/frontdrive/scripts/h100/smoke_pb_train.sh [batch sizes...]   (default: 64 96 128)
#   env:   STEPS=30 VAL_BATCHES=5 WORKERS=6 CAP_MIB=60000 ACT_CKPT_SWEEP=1 RESUME_CHECK=1 PRETRAIN_PT=<encoder .pt> CACHE=<pb cache>
set -uo pipefail
F=/home/henry/Desktop/Qwen_Drive/frontdrive
source "$F/env/drivejepa_pb_env.sh"
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(64 96 128)
STEPS=${STEPS:-30}; VAL=${VAL_BATCHES:-5}; WORKERS=${WORKERS:-6}; CAP_MIB=${CAP_MIB:-60000}
ACT_CKPT_SWEEP=${ACT_CKPT_SWEEP:-1}; RESUME_CHECK=${RESUME_CHECK:-1}; SAMPLE_S=${PB_VRAM_SAMPLE_S:-2}   # 2 s: a 30-step run lasts ~1 min
ROOT=$NAVSIM_EXP_ROOT/pb_smoke; mkdir -p "$ROOT"; SUM=$ROOT/summary.tsv
log() { echo "[$(date -u +%FT%TZ)] $*"; }
[ -f "$SUM" ] || printf 'utc\tname\tbatch\taccum\teff_batch\tact_ckpt\tsteps\tmode\trc\toom\ts_per_step\tpeak_vram_mib\tfits_cap\tlast_ckpt\twall_s\tout\n' > "$SUM"
row() {  # row <name> <mode> <out>  -> appends a summary line; exit 2 on OOM, else the run's rc
  python - "$1" "$2" "$3" "$STEPS" "$CAP_MIB" "$SUM" <<'PY'
import json, os, sys, time
name, mode, out, steps, cap, summ = sys.argv[1:7]
try:
    a = json.load(open(f"{out}/accounting.json"))
except Exception:
    a = {}
peak = a.get("peak_vram_mib_our_pid")
act = a.get("encoder_activation_checkpointing")
fits = int(a.get("rc") == 0 and not a.get("oom") and peak is not None and peak <= int(cap))
line = "\t".join(str(x) for x in [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), name, a.get("batch"), a.get("accum"), a.get("effective_batch"),
                                  "on" if act is None or act else "off", steps, mode, a.get("rc"), int(bool(a.get("oom"))),
                                  a.get("s_per_step_last_progress"), peak, fits, int(os.path.exists(f"{out}/checkpoints/last.ckpt")), a.get("wall_s"), out])
open(summ, "a").write(line + "\n"); print(line)
sys.exit(2 if a.get("oom") else (a.get("rc") if a.get("rc") is not None else 1))
PY
}
run_one() {  # run_one <name> <batch> <act_ckpt 0|1> <first|resume> [env assignments...]
  local name=$1 b=$2 ac=$3 mode=$4; shift 4
  local out=$ROOT/$name allow=0 epochs=1
  [ "$b" -eq 64 ] || allow=1          # micro-batch != 64 => effective batch != 64: throughput measurement only
  [ "$mode" = resume ] && epochs=2
  if [ "$mode" = first ]; then rm -rf "${out:?}"; fi
  log "smoke $name: batch=$b accum=1 act_ckpt=$ac steps=$STEPS val_batches=$VAL mode=$mode -> $out"
  env BATCH="$b" ACCUM=1 EPOCHS="$epochs" WORKERS="$WORKERS" ACT_CKPT="$ac" EXPORT_EVERY=1000 EXPORT_FROM=0 SAVE_LAST=1 \
      PB_VRAM_SAMPLE_S="$SAMPLE_S" ALLOW_NONSTANDARD_BATCH="$allow" "$@" \
      bash "$F/scripts/h100/train_pb.sh" "pb_smoke/$name" 0 trainer.params.limit_train_batches="$STEPS" trainer.params.limit_val_batches="$VAL" || true
  row "$name" "$mode" "$out"
}
RUNS=()
first_size=${SIZES[0]}
for B in "${SIZES[@]}"; do
  run_one "$B" "$B" 1 first; rc=$?
  RUNS+=("$ROOT/$B")
  if [ "$B" = "$first_size" ] && [ "$ACT_CKPT_SWEEP" = 1 ]; then
    run_one "${B}_noackpt" "$B" 0 first || log "batch $B without activation checkpointing: rc/OOM recorded (does not stop the sweep)"
    RUNS+=("$ROOT/${B}_noackpt")
  fi
  if [ "$rc" -eq 2 ]; then log "batch $B: CUDA OOM -> larger sizes skipped"; break; fi
  if [ "$rc" -ne 0 ]; then log "batch $B: failed rc=$rc (see $ROOT/$B/train.log) -> stop"; break; fi
done
python - "$ROOT/pick.json" "$CAP_MIB" "${RUNS[@]}" <<'PY'
import json, os, sys
pick_path, cap = sys.argv[1], int(sys.argv[2])
rows = []
for out in sys.argv[3:]:
    try:
        a = json.load(open(f"{out}/accounting.json"))
    except Exception:
        continue
    peak = a.get("peak_vram_mib_our_pid")
    a["fits"] = bool(a.get("rc") == 0 and not a.get("oom") and peak is not None and peak <= cap)
    a["name"] = os.path.basename(out)
    rows.append(a)
fit = [r for r in rows if r["fits"]]
largest = max((r for r in fit if r.get("encoder_activation_checkpointing") is None), key=lambda r: r["batch"], default=None)
base_on = next((r for r in fit if r["batch"] == 64 and r["accum"] == 1 and r.get("encoder_activation_checkpointing") is None), None)
base_off = next((r for r in fit if r["batch"] == 64 and r["accum"] == 1 and r.get("encoder_activation_checkpointing") is False), None)
baseline = base_off or base_on
pick = {"cap_mib": cap,
        "largest_micro_batch_fitting": largest["batch"] if largest else None,
        "largest_micro_batch_peak_mib": largest["peak_vram_mib_our_pid"] if largest else None,
        "baseline_recommendation": None if baseline is None else {
            "BATCH": 64, "ACCUM": 1, "ACT_CKPT": 0 if baseline is base_off else 1, "peak_vram_mib": baseline["peak_vram_mib_our_pid"],
            "s_per_step": baseline["s_per_step_last_progress"], "note": "effective batch 64 = official recipe; ACT_CKPT=0 = same math, more VRAM, faster"},
        "rows": [{k: r.get(k) for k in ("name", "batch", "accum", "effective_batch", "encoder_activation_checkpointing", "rc", "oom",
                                         "peak_vram_mib_our_pid", "s_per_step_last_progress", "wall_s", "fits")} for r in rows]}
json.dump(pick, open(pick_path, "w"), indent=1)
print(json.dumps(pick, indent=1))
if largest is None:
    print("NO micro-batch fits under the cap (or every run failed) -- inspect pb_smoke/*/train.log")
else:
    print(f"PICK largest micro-batch under {cap} MiB: {largest['batch']} (peak {largest['peak_vram_mib_our_pid']} MiB); "
          f"batches > 64 need ALLOW_NONSTANDARD_BATCH=1 and a recorded lr scaling (plan §2 ④)")
if baseline is not None:
    b = pick["baseline_recommendation"]
    print(f"RECOMMEND baseline (effective batch 64): BATCH=64 ACCUM=1 ACT_CKPT={b['ACT_CKPT']}  (peak {b['peak_vram_mib']} MiB, {b['s_per_step']} s/step)")
PY
if [ "$RESUME_CHECK" = 1 ] && [ -f "$ROOT/$first_size/checkpoints/last.ckpt" ]; then
  log "resume check: batch $first_size from $ROOT/$first_size/checkpoints/last.ckpt for a second epoch"
  run_one "$first_size" "$first_size" 1 resume RESUME="$ROOT/$first_size/checkpoints/last.ckpt" PB_SKIP_COVERAGE_CHECK=1 || log "resume check failed"
elif [ "$RESUME_CHECK" = 1 ]; then
  log "resume check skipped: no last.ckpt under $ROOT/$first_size/checkpoints"
fi
log "summary ($SUM):"
column -t -s $'\t' "$SUM"
