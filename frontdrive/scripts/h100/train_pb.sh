#!/usr/bin/env bash
# Train the Drive-JEPA perception-based (pb) agent on ONE GPU with the upstream recipe (venv-pb; plan v4.3 §2 ②/③).
# Upstream (third_party/Drive-JEPA/navsim_v1/scripts/training/train_drive_jepa_perception_based.sh, README "two
# L40S/A30/V100"): torchrun over 2 GPUs x batch 32 = effective batch 64, 20 epochs, Adam 1e-4 (backbone 0.1x),
# precision 16-mixed, no scheduler / weight decay / grad clipping, online PDM scorer (16 processes) over
# $NAVSIM_EXP_ROOT/Drive-JEPA-cache/{train_metric_cache,anchors_scores_index}, validation on the 214 val logs each epoch.
# Here: one process (strategy=auto), BATCH x ACCUM = effective batch; default 64 x 1 = the official effective batch
# WITHOUT gradient accumulation (Henry 2026-10-17: largest micro-batch that fits, no accumulation unless needed; VRAM as
# high as possible -> ACT_CKPT=0 disables the ViT's activation checkpointing = more VRAM, faster, identical math).
# The script REFUSES effective batches != 64 unless ALLOW_NONSTANDARD_BATCH=1 (plan §2 ④: lr scaling recorded separately).
# Training cache = the derived compact pb cache (scripts/h100/build_pb_cache.sh; CompactCacheOnlyDataset decodes the JPEGs);
# encoder init swappable (PRETRAIN_PT -> +agent.config.pretrain_pt_path, same layout as e50); fp16 weight exports of the
# late epochs (exports/epoch_N.ckpt, loadable by eval_pb.sh), last.ckpt for resume; peak VRAM sampled from nvidia-smi for
# our PID into $OUT/gpu_mem.csv; $OUT/params.json (all knobs + git describe) at start, $OUT/accounting.json at the end.
#   usage: [TAG=<tag> SEED=<seed>] train_pb.sh <tag> [seed=0] [extra hydra overrides...]   -> $NAVSIM_EXP_ROOT/<tag>/
#   knobs: BATCH=64 ACCUM=1 EPOCHS=20 WORKERS=6 PRECISION=16-mixed SIDE_CALIB=1 ACT_CKPT=1
#          CACHE=$NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache
#          PRETRAIN_PT=<encoder .pt>  ("" = upstream e50 file; e.g. the C'/D' exports of pretrain/export_target_encoder.py)
#          EXPORT_EVERY=5 EXPORT_FROM=16 SAVE_LAST=1 RESUME=<.../checkpoints/last.ckpt> CKPT_EVERY=   (-> FRONTDRIVE_* of run_training.py;
#          the FRONTDRIVE_* names are accepted as well)
#          ALLOW_NONSTANDARD_BATCH=1  run with BATCH*ACCUM != 64 (throughput smoke / §2-④ knowledge points only)
#          PB_SKIP_COVERAGE_CHECK=1   skip the train_metric_cache token-coverage check
#          PB_COVERAGE_MOVE_ASIDE=1   move uncovered cache tokens to ${CACHE}_uncovered instead of refusing to start
#          PB_VRAM_SAMPLE_S=30        nvidia-smi sampling period (s)
#          PB_LR=1e-4                 Adam lr (agent.lr; the backbone gets 0.1x of it) — ⑤ continued training uses a lower peak lr
#          PB_INIT_CKPT=<agent .ckpt> ⑤: initialise ALL agent weights from a full checkpoint (e.g. the released pb v1 ckpt) before training
#          PB_TRAIN_LOGS=train|all    ⑤: 'all' trains on the train AND val logs of navtrain (103,288 tokens; dev becomes in-sample)
set -euo pipefail
F=/home/henry/Desktop/Qwen_Drive/frontdrive
# positional: <tag> [seed] [hydra overrides...]; TAG/SEED from the environment leave the positionals to the overrides, and a
# non-numeric 2nd argument is an override, not a seed (review 2026-10-17: `train_pb.sh mytag trainer.params.max_epochs=1` used to
# pass the override as seed=...)
if [ -z "${TAG:-}" ]; then TAG=${1:-}; [ $# -ge 1 ] && shift; fi
[ -n "${TAG:-}" ] || { echo "usage: train_pb.sh <tag> [seed] [hydra overrides...]  (or TAG=<tag> in the environment)"; exit 2; }
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then SEED=${SEED:-$1}; shift; else SEED=${SEED:-0}; fi
source "$F/env/drivejepa_pb_env.sh"
DJ=$NAVSIM_EXP_ROOT/Drive-JEPA-cache
CACHE=${CACHE:-$NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache}
BATCH=${BATCH:-64}; ACCUM=${ACCUM:-1}; EPOCHS=${EPOCHS:-20}; WORKERS=${WORKERS:-6}; PRECISION=${PRECISION:-16-mixed}
SIDE_CALIB=${SIDE_CALIB:-1}; ACT_CKPT=${ACT_CKPT:-1}
PRETRAIN_PT=${PRETRAIN_PT:-${FRONTDRIVE_PRETRAIN_PT:-}}
EXPORT_EVERY=${EXPORT_EVERY:-${FRONTDRIVE_EXPORT_EVERY:-5}}; EXPORT_FROM=${EXPORT_FROM:-${FRONTDRIVE_EXPORT_FROM:-16}}
SAVE_LAST=${SAVE_LAST:-${FRONTDRIVE_SAVE_LAST:-1}}; RESUME=${RESUME:-${FRONTDRIVE_RESUME_CKPT:-}}; CKPT_EVERY=${CKPT_EVERY:-${FRONTDRIVE_CKPT_EVERY:-}}
SAMPLE_S=${PB_VRAM_SAMPLE_S:-30}
EFF=$((BATCH * ACCUM))
OUT=$NAVSIM_EXP_ROOT/$TAG
log() { echo "[$(date -u +%FT%TZ)] $*"; }

# ---- recipe guard: the official-recipe baseline is effective batch 64
if [ "$EFF" -ne 64 ] && [ "${ALLOW_NONSTANDARD_BATCH:-0}" != 1 ]; then
  log "REFUSED: effective batch BATCH*ACCUM = $BATCH*$ACCUM = $EFF != 64 (official recipe). Set ALLOW_NONSTANDARD_BATCH=1 for throughput smokes / §2-④ experiments (record the lr scaling)."
  exit 2
fi
# ---- pre-checks: fail before the 5 GB encoder load
if [ -f "$FRONTDRIVE/env/.use_conda_cvpr" ]; then
  [ "${CONDA_DEFAULT_ENV:-}" = cvpr ] || { log "conda env cvpr not active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}); run scripts/h100/bootstrap_conda_cvpr.sh"; exit 2; }
  VENV_PY=$CONDA_PREFIX/bin/python
else
  VENV_PY=$FRONTDRIVE/env/drivejepa_pb/bin/python
fi
[ -x "$VENV_PY" ] && [ "$(command -v python)" = "$VENV_PY" ] || { log "interpreter $VENV_PY missing or not first on PATH (python=$(command -v python)); run scripts/h100/bootstrap_conda_cvpr.sh"; exit 2; }
python -c "import mmcv.ops, navsim, pytorch_lightning" 2>/dev/null || { log "venv-pb not usable (mmcv.ops / navsim / pytorch_lightning import failed in $VENV_PY); run scripts/h100/bootstrap_pb_env.sh"; exit 2; }
E50=$DJ/vitl_merge_3dataset_e50.pt
test -f "$E50" || { log "missing $E50 (the upstream default init; hf_fetch.sh)"; exit 2; }
if [ -n "$PRETRAIN_PT" ]; then test -f "$PRETRAIN_PT" || { log "missing PRETRAIN_PT=$PRETRAIN_PT"; exit 2; }; fi
if [ -n "${PB_INIT_CKPT:-}" ]; then test -f "$PB_INIT_CKPT" || { log "missing PB_INIT_CKPT=$PB_INIT_CKPT"; exit 2; }; fi
test -d "$CACHE/_COMPACT_JPEG" || { log "no compact pb cache at $CACHE (scripts/h100/build_pb_cache.sh)"; exit 2; }
test -f "$DRIVEJEPA_ROOT/data/8192.npy" || { log "missing anchors $DRIVEJEPA_ROOT/data/8192.npy"; exit 2; }
if [ -n "$RESUME" ]; then test -f "$RESUME" || { log "missing RESUME=$RESUME"; exit 2; }; fi
MC=$DJ/train_metric_cache
ls "$MC"/metadata/*.csv >/dev/null 2>&1 || { log "no $MC/metadata/*.csv (hf_fetch.sh train_metric_cache.tar)"; exit 2; }
n_csv=$(ls "$MC"/metadata/*.csv | wc -l)
[ "$n_csv" -eq 1 ] || { log "$n_csv metadata csv files under $MC/metadata: MetricCacheLoader reads only ONE of them (arbitrary iterdir order) -> merge + rewrite first: bash $F/scripts/fix_metric_cache_paths.sh $MC"; exit 2; }
first=$(sed -n 2p "$(ls "$MC"/metadata/*.csv | head -1)" | tr ',' '\n' | grep -m1 '^/' || true)
[ -n "$first" ] && [ -f "$first" ] || { log "train_metric_cache metadata points to a missing file ($first): bash $F/scripts/fix_metric_cache_paths.sh $MC"; exit 2; }
test -d "$DJ/anchors_scores_index" || { log "no $DJ/anchors_scores_index (hf_fetch.sh anchors_scores_index.tar)"; exit 2; }
anchor_example="${first/train_metric_cache/anchors_scores_index}.npy"
test -f "$anchor_example" || { log "anchors index missing for the first metric-cache row: $anchor_example"; exit 2; }
if [ "${PB_SKIP_COVERAGE_CHECK:-0}" != 1 ]; then
  log "token coverage check: every cache token must be in train_metric_cache (+ anchors index), else compute_score KeyErrors"
  cov_args=(--out "$CACHE" --metric-cache "$MC" --check-anchors)
  if [ "${PB_COVERAGE_MOVE_ASIDE:-0}" = 1 ]; then cov_args+=(--move-aside "${CACHE}_uncovered"); fi
  (cd "$F" && python -m navsim_ext.compact_cache_pb coverage "${cov_args[@]}") \
    || { log "coverage check failed: PB_COVERAGE_MOVE_ASIDE=1 moves the uncovered tokens aside, PB_SKIP_COVERAGE_CHECK=1 skips the check"; exit 2; }
fi

mkdir -p "$OUT"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/monitor.log"; }   # from here on our own lines also go to $OUT/monitor.log (the stdout stream mixes with the trainer's \r progress bar and is useless to filter)
export WANDB_MODE=disabled FRONTDRIVE_NAVSIM_EXT=$F TZ=America/New_York
# omegaconf >= 2.4 (what an unpinned `pip install -r requirements.txt` resolves) refuses YAML files with > 10,000 nodes:
# default_train_val_test_log_split.yaml (14,561 logs) and scene_filter/navtrain*.yaml (103k tokens) -> raise the limit
# (ignored by omegaconf 2.3; the real fix is `pip install 'omegaconf<2.4'` in venv-pb, see docs/evidence/2026-10-17_pb_recipe)
export OMEGACONF_MAX_YAML_EXPANDED_NODES=${OMEGACONF_MAX_YAML_EXPANDED_NODES:-10000000}
export FRONTDRIVE_EXPORT_EVERY=$EXPORT_EVERY FRONTDRIVE_EXPORT_FROM=$EXPORT_FROM
if [ "$SAVE_LAST" = 1 ]; then export FRONTDRIVE_SAVE_LAST=1; else unset FRONTDRIVE_SAVE_LAST || true; fi
if [ -n "$CKPT_EVERY" ]; then export FRONTDRIVE_CKPT_EVERY=$CKPT_EVERY; else unset FRONTDRIVE_CKPT_EVERY || true; fi
export FRONTDRIVE_RESUME_CKPT=$RESUME
if [ -n "${PB_INIT_CKPT:-}" ]; then export FRONTDRIVE_INIT_CKPT=$PB_INIT_CKPT; else unset FRONTDRIVE_INIT_CKPT || true; fi   # ⑤ full-agent init
if [ "${PB_TRAIN_LOGS:-train}" = all ]; then export FRONTDRIVE_TRAIN_LOGS=all; else unset FRONTDRIVE_TRAIN_LOGS || true; fi   # ⑤ train+val logs
OVR=(agent=drive_jepa_perception_based_agent experiment_name="$TAG" output_dir="$OUT" seed="$SEED" train_test_split=navtrain
     cache_path="$CACHE/" use_cache_without_dataset=True force_cache_computation=False
     dataloader.params.batch_size="$BATCH" dataloader.params.num_workers="$WORKERS" dataloader.params.prefetch_factor=4
     trainer.params.accumulate_grad_batches="$ACCUM" trainer.params.max_epochs="$EPOCHS" trainer.params.precision="$PRECISION"
     trainer.params.strategy=auto)
if [ "$SIDE_CALIB" = 1 ]; then OVR+=(+agent.config.side_cams_from_calibration=true); fi   # no CAM_L0/R0/B0 JPEGs needed anywhere
if [ -n "$PRETRAIN_PT" ]; then OVR+=(+agent.config.pretrain_pt_path="$PRETRAIN_PT"); fi
if [ -n "${PB_LR:-}" ]; then OVR+=(agent.lr="$PB_LR"); fi   # ⑤: lower peak lr for continued training (upstream constant 1e-4)
if [ "$ACT_CKPT" = 0 ]; then OVR+=(+agent.config.encoder_activation_checkpointing=false); fi   # VRAM for speed, same math
# plan v4.3 ④(b): PB_LR_SCHED=cosine|step adds a per-step lr schedule (upstream = constant); the step count is the navtrain
# train split (85,109 samples, drop_last) / BATCH per epoch x EPOCHS — identical to the released ckpt's 1,329 steps/epoch at 64
LR_SCHED=${PB_LR_SCHED:-none}; LR_MIN_MULT=${PB_LR_MIN_MULT:-0.1}; LR_WARMUP=${PB_LR_WARMUP:-0}
if [ "$LR_SCHED" != none ]; then
  N_TRAIN=85109; [ "${PB_TRAIN_LOGS:-train}" = all ] && N_TRAIN=103288   # all 1,192 navtrain logs (train + val) = every cache token
  STEPS_TOTAL=$(( N_TRAIN / BATCH * EPOCHS ))
  OVR+=(+agent.config.lr_schedule="$LR_SCHED" +agent.config.lr_total_steps="$STEPS_TOTAL" +agent.config.lr_min_mult="$LR_MIN_MULT" +agent.config.lr_warmup_steps="$LR_WARMUP")
fi
GIT_DESC=$(git -C "$F" describe --always --dirty 2>/dev/null || echo "no-git")   # the server copy ships without .git
START=$(date +%s)
python - "$OUT" "$F" "$GIT_DESC" "$START" "$TAG" "$SEED" "$BATCH" "$ACCUM" "$EPOCHS" "$WORKERS" "$PRECISION" "$SIDE_CALIB" "$ACT_CKPT" \
         "${PRETRAIN_PT:-$E50}" "$CACHE" "$EXPORT_EVERY" "$EXPORT_FROM" "$SAVE_LAST" "$RESUME" "$CKPT_EVERY" "${ALLOW_NONSTANDARD_BATCH:-0}" "$*" <<'PY'
import hashlib, json, os, platform, socket, sys
out, F, git_desc, start = sys.argv[1:5]
tag, seed, batch, accum, epochs, workers, precision, side_calib, act_ckpt, pretrain, cache = sys.argv[5:16]
export_every, export_from, save_last, resume, ckpt_every, allow_ns, extra = sys.argv[16:23]
def sha12(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12] if os.path.exists(p) else None
params = {"tag": tag, "seed": int(seed), "batch": int(batch), "accum": int(accum), "effective_batch": int(batch) * int(accum),
          "epochs": int(epochs), "dataloader_workers": int(workers), "precision": precision, "strategy": "auto",
          "side_cams_from_calibration": side_calib == "1", "encoder_activation_checkpointing": None if act_ckpt == "1" else False,
          "pretrain_pt": pretrain, "cache": cache, "export_every": int(export_every), "export_from": int(export_from),
          "save_last": save_last == "1", "resume_ckpt": resume or None, "ckpt_every": int(ckpt_every) if ckpt_every else None,
          "allow_nonstandard_batch": allow_ns == "1", "extra_overrides": extra, "lr": os.environ.get("PB_LR", "1e-4"),
          "optimizer": f"Adam lr {os.environ.get('PB_LR', '1e-4')}, backbone 0.1x" + ("" if os.environ.get("PB_LR") else " (upstream)"),
          "init_ckpt": os.environ.get("PB_INIT_CKPT") or None, "train_logs": os.environ.get("PB_TRAIN_LOGS", "train"),
          "lr_schedule": os.environ.get("PB_LR_SCHED", "none"), "lr_min_mult": os.environ.get("PB_LR_MIN_MULT", "0.1"), "lr_warmup_steps": os.environ.get("PB_LR_WARMUP", "0"),
          "frontdrive_git_describe": git_desc, "patch_sha12": sha12(f"{F}/third_party_patches/drive_jepa_548bb82.patch"),
          "run_training_sha12": sha12(f"{F}/third_party/Drive-JEPA/navsim_v1/navsim/planning/script/run_training.py"),
          "host": socket.gethostname(), "python": platform.python_version(), "start_utc": int(start)}
try:
    import torch, pytorch_lightning as pl
    params.update({"torch": torch.__version__, "pytorch_lightning": pl.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
except Exception as e:  # noqa: BLE001
    params["torch_probe_error"] = str(e)[:120]
json.dump(params, open(f"{out}/params.json", "w"), indent=1)
PY
cd "$DRIVEJEPA_ROOT"   # ImgEncoder resolves ./vjepa2/configs/eval/vitl/in1k.yaml from the CWD (patched: file-relative fallback)
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/gpu_before.txt" 2>/dev/null || true
log "train_pb $TAG seed=$SEED batch=$BATCH accum=$ACCUM (effective $EFF) epochs=$EPOCHS precision=$PRECISION workers=$WORKERS act_ckpt=$ACT_CKPT side_calib=$SIDE_CALIB cache=$CACHE pretrain=${PRETRAIN_PT:-e50} resume=${RESUME:-none} git=$GIT_DESC init_ckpt=${PB_INIT_CKPT:-none} train_logs=${PB_TRAIN_LOGS:-train} lr=${PB_LR:-1e-4} extra: $*" | tee -a "$OUT/train.log"
python -u navsim/planning/script/run_training.py "${OVR[@]}" "$@" > >(tee -a "$OUT/train.log") 2>&1 &
PID=$!
echo "$PID" > "$OUT/train.pid"
: > "$OUT/gpu_mem.csv"   # unix_time,our_pid_mib,gpu_total_used_mib,gpu_util_pct
i=0
set +e   # monitor loop: never let a failed sample kill this wrapper (2026-10-18: with two trainers on the GPU the early-exit
         # awk below made nvidia-smi die of SIGPIPE -> pipefail status 141 -> errexit ended the wrapper 4 h into the run; the
         # orphaned trainer finished fine but accounting/VRAM sampling were lost and a chain could have relaunched a duplicate)
while kill -0 "$PID" 2>/dev/null; do
  ours=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$PID" '$1==p{print $2}' | sed -n 1p)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n 1p | tr -d ' ')
  echo "$(date +%s),${ours:-0},${gpu:-0,0}" >> "$OUT/gpu_mem.csv"
  if [ $((i % 10)) -eq 0 ]; then
    prog=$(tail -c 20000 "$OUT/train.log" 2>/dev/null | tr '\r' '\n' | grep -E 'Epoch [0-9]+:' | tail -n 1 | cut -c1-100 || true)
    log "vram ours=${ours:-0}MiB gpu(used_mib,util)=${gpu:-?} | ${prog:-starting}"
  fi
  i=$((i + 1)); sleep "$SAMPLE_S"
done
set -e
RC=0; wait "$PID" || RC=$?
sleep 2   # let tee flush
END=$(date +%s)
python - "$OUT" "$START" "$END" "$RC" <<'PY'
import glob, json, os, re, sys
out, start, end, rc = sys.argv[1:5]
params = json.load(open(f"{out}/params.json"))
ours, total = [], []
for line in open(f"{out}/gpu_mem.csv"):
    p = line.strip().split(",")
    if len(p) >= 2 and p[1].isdigit() and int(p[1]) > 0:  # per-PID MiB; 0 = our PID not on the GPU yet, '[N/A]' = no per-process accounting
        ours.append(int(p[1]))
    if len(p) >= 3 and p[2].isdigit():
        total.append(int(p[2]))
before = None  # GPU memory in use before our launch (gpu_before.txt: "used MiB, total MiB, util %")
try:
    before = int(re.match(r"\s*(\d+)", open(f"{out}/gpu_before.txt").read()).group(1))
except Exception:  # noqa: BLE001
    pass
if ours:
    peak, peak_src = max(ours), "nvidia-smi --query-compute-apps used_memory of our PID"
elif total and before is not None:  # review 2026-10-17: fall back when per-process accounting is unavailable instead of reporting None (= 'does not fit')
    peak, peak_src = max(total) - before, "gpu memory.used minus gpu_before.txt (per-PID accounting unavailable; other tenants' growth included)"
else:
    peak, peak_src = None, "unavailable"
log = open(f"{out}/train.log", errors="replace").read().replace("\r", "\n")
m = re.search(r"Num training samples: (\d+)", log); n_train = int(m.group(1)) if m else None
m = re.search(r"Num validation samples: (\d+)", log); n_val = int(m.group(1)) if m else None
prog = [l for l in log.split("\n") if re.search(r"Epoch \d+:", l) and re.search(r"\d(it/s|s/it)", l)]  # PL progress bar lines (with or without v_num)
last = prog[-1][:160] if prog else ""
m = re.search(r"([\d.]+)(it/s|s/it)", last); s_per_step = (1 / float(m.group(1)) if m.group(2) == "it/s" else float(m.group(1))) if m else None
m = re.search(r"\[ImgEncoder\] loaded encoder weights from (\S+)", log)
batch, accum = params["batch"], params["accum"]
acc = {**params, "pretrain_pt_loaded": m.group(1) if m else None, "n_train_samples": n_train, "n_val_samples": n_val,
       "steps_per_epoch": (n_train // batch) if n_train else None, "optimizer_steps_per_epoch": (n_train // batch // accum) if n_train else None,
       "peak_vram_mib_our_pid": peak, "peak_vram_source": peak_src, "gpu_used_mib_before_launch": before,
       "peak_gpu_total_used_mib": max(total) if total else None, "vram_samples": len(total), "vram_samples_our_pid": len(ours),
       "s_per_step_last_progress": s_per_step, "last_progress": last, "wall_s": int(end) - int(start), "rc": int(rc),
       "oom": ("CUDA out of memory" in log) or ("OutOfMemoryError" in log),
       "exports": sorted(os.path.basename(p) for p in glob.glob(f"{out}/exports/*.ckpt")),
       "checkpoints": sorted(os.path.basename(p) for p in glob.glob(f"{out}/checkpoints/*.ckpt")), "end_utc": int(end)}
json.dump(acc, open(f"{out}/accounting.json", "w"), indent=1)
print(json.dumps({k: acc[k] for k in ("tag", "effective_batch", "encoder_activation_checkpointing", "steps_per_epoch", "peak_vram_mib_our_pid",
                                      "s_per_step_last_progress", "wall_s", "rc", "oom")}))
PY
log "train_pb $TAG finished rc=$RC (params: $OUT/params.json, accounting: $OUT/accounting.json)"
exit "$RC"
