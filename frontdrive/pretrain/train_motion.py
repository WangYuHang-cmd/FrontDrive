# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# [motion] frontdrive fork of the (patched) vendored V-JEPA 2 trainer
#          third_party/Drive-JEPA/navsim_v1/vjepa2/app/vjepa/train.py for the plan v4.1 §3.1 research arm:
#          CAN-constrained cross-camera motion representation. Everything that is not marked "# [motion]" is the
#          vendored file verbatim; with motion.enabled=false the training computation is identical to the vendored
#          trainer (same data, masks, losses and RNG consumption). When enabled, three terms are added to the JEPA loss:
#            L_pred    = L1( P_phi(z_s, m_{s:s+d}, K, dt), sg(zbar_{s+d}) )   motion-conditioned future-slot prediction
#            L_readout = Huber( R_psi(z_s, sg(zbar_{s+d}), K, dt), m )        motion read-out from real token pairs
#            L_cons    = MSE( e_R(view1 | common region), e_R(view2) )        imaging-vs-motion consistency (nested view)
#          Gradients of all terms reach the ONLINE encoder through z_s; target-encoder tokens are always stop-grad.
#          Review 2026-10-12 (shortcuts / geometry / robustness): L_pred is restricted to target positions the online
#          encoder did NOT see (motion.pred_masked_only), R_psi's target branch is dropped per item with probability
#          motion.readout_tgt_drop so the online branch is necessary, items with a thin context slot are excluded
#          (motion.min_ctx_tokens), the CAN-mismatch control permutes within (speed bin x slot gap) strata and masks
#          fixed points, and every step logs shortcut diagnostics (loss_readout_noctx, grad_ratio_ctx, per-target Huber
#          terms, e_std collapse monitor). Checkpoints are written atomically with a disk-space check and a fallback
#          to latest.prev.pt on resume; accounting.json records an exit_status.
#          Launch: python -m pretrain.train_motion --fname <yaml> --devices cuda:0

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import atexit  # [motion] accounting.json is also written when the process exits abnormally
import copy
import gc
import hashlib  # [motion] config hash for accounting.json
import json  # [motion] accounting.json
import random
import shutil  # [motion] disk-space check before writing a checkpoint
import socket  # [motion] free port for the single-process group
import time

import numpy as np
import torch
import torch.distributed as dist  # [motion] process-group bootstrap when not launched through app/main.py
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# [motion] make_transforms / init_data are replaced by the motion-aware dataset below (identical clips/masks when
#          motion is disabled); everything else is imported from the vendored package unchanged
from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# [motion] frontdrive modules (binding interfaces, see pretrain/motion_data.py and pretrain/motion_heads.py)
from pretrain.motion_data import (
    COMMA_ONLY_FEATURES,
    DEFAULT_TARGET_SCALES,
    MOTION_SEQ_FEATURES,
    MOTION_TARGETS,
    TARGET_SCALE_PER_SLOT,
    MotionTransform,
    make_motion_videodataset,
    motion_seq_len,
)
from pretrain.motion_heads import (
    FlopMeter,
    MotionConditionedPredictor,
    MotionReadout,
    consistency_loss,
    embedding_spread,
    expand_target_scales,
    gather_slot_tokens,
    online_gradient_ratio,
    rays_for_tokens,
    readout_loss,
    readout_loss_terms,
    shuffle_within_speed_strata,
)

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


# [motion] ------------------------------------------------------------------------------------------------------------ #
# Defaults for args["motion"] (plan v4.1 §3.1). Any key may be overridden in the yaml. Arm B/D = enabled with
# use_pred + use_readout; controls: cond_motion=false (plain future-prediction head), use_readout=false,
# shuffle_motion=true (CAN mismatch within speed strata), cons_weight=0 (no consistency term).
MOTION_DEFAULTS = dict(
    enabled=False,
    weight_pred=1.0,  # weight of L_pred
    weight_readout=1.0,  # weight of L_readout
    cons_weight=0.0,  # weight of L_cons; 0 disables the second (nested) view entirely
    cons_every=2,  # run the consistency pass every N optimizer steps (global_step % cons_every == 0)
    cond_motion=True,  # P_phi sees the ego-motion sequence (False -> plain future prediction control)
    use_pred=True,  # build / train P_phi
    use_readout=True,  # train R_psi with L_readout (R_psi is still built when cons_weight > 0)
    shuffle_motion=False,  # CAN-mismatch control: permute motion inputs AND targets within speed strata
    use_comma_only_fields=False,  # add steer_deg / intervention flags to the motion sequence (comma-only ablation)
    pred_depth=6,  # P_phi transformer depth
    readout_depth=3,  # R_psi transformer depth
    head_dim=384,  # width of both heads
    head_heads=6,  # attention heads of both heads
    target_scales=dict(DEFAULT_TARGET_SCALES),  # read-out target scales (m, m, rad, m/s, rad/s, m/s, rad/s); dx/dy/dyaw/dv x d
    mask_config=0,  # which encoder mask (index into args["mask"]) feeds the heads; 0 = 8 blocks @ 15 %
    view2_scale=[0.6, 0.85],  # nested second view: side scale range relative to view 1 (L_cons only)
    readout_tgt_drop=0.5,  # per-item probability of removing R_psi's target-token branch (online branch becomes the only input)
    pred_masked_only=True,  # L_pred only on target positions the online encoder did not see (no copy path through z_s)
    min_ctx_tokens=16,  # items whose context slot has fewer visible tokens are excluded from L_pred / L_readout
    rays_road_frame=True,  # rotate per-token rays into the road-aligned camera frame with the clip's cal_pitch / cal_yaw
    require_can=False,  # drop clips without a CAN sidecar from the sample list (default: keep them, like the plain arms)
)
# [motion] CSV diagnostics appended after the vendored-style columns (one value per optimizer step)
MOTION_DIAG_COLUMNS = (
    "loss_pred_vis",  # L1 of P_phi on target positions the online encoder DID see (copy path; not trained on when pred_masked_only)
    "loss_readout_noctx",  # L_readout recomputed with the online context removed (no grad): gap to loss_readout = online contribution
    "grad_ratio_ctx",  # ||dL_readout/dz_ctx|| / ||dL_readout/dz_tgt||: share of the readout signal reaching the online encoder
    "e_std",  # batch-wise std of the normalised R_psi embedding (collapse monitor for L_cons)
    "frac_ctx_ok",  # fraction of items with >= min_ctx_tokens visible context tokens (used by the heads)
    "ctx_vis_mean",  # mean number of visible online tokens in the context slot
    "frac_mismatch",  # shuffle_motion: fraction of items that actually received another item's motion
) + tuple(f"huber_{k}" for k in MOTION_TARGETS)  # per-target Huber terms of L_readout (NaN: no valid entry in the batch)


def _ensure_process_group(device):
    """[motion] The vendored trainer wraps the models in DistributedDataParallel and relies on app/main.py having
    initialised a (possibly single-process) process group. When main() is called directly (tests, the launcher
    below after a failed default-port init) create a world-size-1 group on a free local port: gloo on cpu, nccl on
    cuda. No-op when a group already exists."""
    if dist.is_available() and dist.is_initialized():
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    backend = "gloo" if device.type == "cpu" else "nccl"
    dist.init_process_group(backend=backend, init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    logger.info(f"[motion] initialised single-process {backend} group on port {port}")


def _config_hash(args):
    """[motion] 12-hex sha256 of the sorted JSON dump of the run config (accounting.json / checkpoints)."""
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _unwrap(module):
    """[motion] state_dict / attribute access through an optional DistributedDataParallel wrapper."""
    return module.module if isinstance(module, DistributedDataParallel) else module


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _slot_full(h, slot, n_spatial):
    """[motion] All n_spatial tokens of slot ``slot[b]`` from a full (unmasked) token tensor h (B, N, D).

    Token flattening in the vendored PatchEmbed3D is time-major (flat = slot * n_spatial + row * W_p + col), so the
    tokens of one slot are a contiguous block. The visible-token counterpart (per-sample gather with padding and keep
    masks, since the number of visible tokens per slot differs across samples) is motion_heads.gather_slot_tokens.
    """
    base = slot[:, None] * n_spatial + torch.arange(n_spatial, device=h.device)[None, :]  # (B, n_spatial)
    return torch.gather(h, 1, base.unsqueeze(-1).expand(-1, -1, h.size(-1)))


def _fm_get(d, key):
    """[motion] tolerant lookup in FlopMeter dicts ("tokens" / "flops" or "total_*")."""
    return d.get(key, d.get(f"total_{key}", 0))


# [motion] ------------------------------------------------------------------------------------------------------------ #


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    if isinstance(crop_size, (list, tuple)):  # frontdrive: YAML lists must become tuples for the mask generator
        crop_size = tuple(int(c) for c in crop_size)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)
    random_horizontal_flip = cfgs_data_aug.get("random_horizontal_flip", True)  # frontdrive patch: driving + CAN heads need no flips

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # -- [motion] research-arm config (defaults above) + meta.device / meta.max_steps
    cfgs_motion = dict(MOTION_DEFAULTS)
    cfgs_motion.update(args.get("motion") or {})
    motion_enabled = bool(cfgs_motion["enabled"])
    weight_pred = float(cfgs_motion["weight_pred"])
    weight_readout = float(cfgs_motion["weight_readout"])
    cons_weight = float(cfgs_motion["cons_weight"])
    cons_every = max(1, int(cfgs_motion["cons_every"]))
    cond_motion = bool(cfgs_motion["cond_motion"])
    use_pred = bool(cfgs_motion["use_pred"])
    use_readout = bool(cfgs_motion["use_readout"])
    shuffle_motion = bool(cfgs_motion["shuffle_motion"])
    use_comma_only_fields = bool(cfgs_motion["use_comma_only_fields"])
    motion_mask_config = int(cfgs_motion["mask_config"])
    use_view2 = motion_enabled and cons_weight > 0
    view2_scale = tuple(cfgs_motion["view2_scale"]) if use_view2 else None
    readout_tgt_drop = float(cfgs_motion["readout_tgt_drop"])
    pred_masked_only = bool(cfgs_motion["pred_masked_only"])
    min_ctx_tokens = int(cfgs_motion["min_ctx_tokens"])
    rays_road_frame = bool(cfgs_motion["rays_road_frame"])
    require_can = bool(cfgs_motion["require_can"])
    device_name = str(cfgs_meta.get("device", "cuda:0"))
    max_steps = cfgs_meta.get("max_steps", None)  # stop after this many optimizer steps (tests / smoke runs)
    if motion_enabled:
        assert 0 <= motion_mask_config < len(cfgs_mask), "motion.mask_config must index args['mask']"
        # several clip lists (arm C/D mixture: public + comma with datasets_weights) are fine as long as they share one
        # fpc: MaskCollator then builds a single fpc group and z_enc[0] / masks_enc[0] hold the whole batch
        assert len(set(dataset_fpcs)) == 1, "[motion] motion heads need a single fpc group (all dataset_fpcs equal)"
        assert float(cfgs_mask[motion_mask_config]["spatial_scale"][1]) < 0.5, (
            "[motion] mask_config must select the light context mask (spatial_scale < 0.5): the vendored collator "
            "truncates kept indices from the end, so a heavy mask leaves the later context slots without tokens"
        )
        assert cons_weight == 0 or use_readout, (
            "[motion] cons_weight > 0 requires use_readout: L_cons alone has a constant solution (R_psi collapses)"
        )
        assert 0.0 <= readout_tgt_drop < 1.0, "[motion] readout_tgt_drop must be in [0, 1)"
        assert len(cfgs_motion["target_scales"]) == len(MOTION_TARGETS) and all(k in cfgs_motion["target_scales"] for k in MOTION_TARGETS), (
            f"[motion] target_scales must have one entry per target {MOTION_TARGETS}"
        )
        logger.info(f"[motion] config: { {k: v for k, v in cfgs_motion.items() if k != 'target_scales'} }")
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device ([motion] meta.device, default cuda:0 as in the vendored file; "cpu" only when asked for explicitly:
    #    a cuda device that is not available (typo in --devices, clobbered CUDA_VISIBLE_DEVICES) is an error, never a
    #    silent fall-back to hours of CPU training)
    if device_name.startswith("cpu"):
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[motion] meta.device={device_name!r} requested but CUDA is unavailable in this process "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); set meta.device: cpu to train on CPU"
            )
        device = torch.device(device_name)
        torch.cuda.set_device(device)
    logger.info(f"[motion] device {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    _ensure_process_group(device)  # [motion] DDP below needs a group when main() is not launched via app/main.py

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger ([motion] extra loss columns + per-step tokens / FLOPs estimate)
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_jepa"),
        ("%.5f", "loss_pred"),
        ("%.5f", "loss_readout"),
        ("%.5f", "loss_cons"),
        ("%d", "iter_ms"),
        ("%d", "gpu_ms"),
        ("%d", "data_ms"),
        ("%d", "tokens_step"),
        ("%.6e", "flops_step"),
        *[("%.5f", name) for name in MOTION_DIAG_COLUMNS],  # [motion] shortcut / balance diagnostics (0 when disabled)
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- [motion] heads P_phi / R_psi. Built AFTER the backbones and under a forked CPU RNG (own seed) so that the
    #    backbone initialisation AND the data / mask RNG streams are identical across arms (matched seeds, plan §6:
    #    arm C and arm D see the same clips, crops and masks). The heads are always fresh (never read from init_from).
    _crop_hw = crop_size if isinstance(crop_size, tuple) else (int(crop_size), int(crop_size))
    n_slots = max_num_frames // tubelet_size
    n_spatial = (_crop_hw[0] // patch_size) * (_crop_hw[1] // patch_size)
    n_patches_per_clip = n_slots * n_spatial
    motion_pred = motion_readout = None
    target_scales = target_per_slot = None
    if motion_enabled:
        n_feat = len(MOTION_SEQ_FEATURES) + (len(COMMA_ONLY_FEATURES) if use_comma_only_fields else 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 7919)
            if use_pred:
                motion_pred = MotionConditionedPredictor(
                    enc_dim=encoder.backbone.embed_dim,
                    dim=int(cfgs_motion["head_dim"]),
                    depth=int(cfgs_motion["pred_depth"]),
                    heads=int(cfgs_motion["head_heads"]),
                    n_spatial=n_spatial,
                    n_slots=n_slots,
                    n_feat=n_feat,
                    cond_motion=cond_motion,
                    seq_len=motion_seq_len(max_num_frames, tubelet_size),  # rows of the motion sequence for this fpc
                ).to(device)
            if use_readout or cons_weight > 0:
                motion_readout = MotionReadout(
                    enc_dim=encoder.backbone.embed_dim,
                    dim=int(cfgs_motion["head_dim"]),
                    depth=int(cfgs_motion["readout_depth"]),
                    heads=int(cfgs_motion["head_heads"]),
                    n_spatial=n_spatial,
                    n_slots=n_slots,
                    n_targets=len(MOTION_TARGETS),
                ).to(device)
        # per-horizon normalisation: entries in TARGET_SCALE_PER_SLOT are multiplied by the slot gap d per sample
        target_scales = torch.tensor(
            [float(cfgs_motion["target_scales"][k]) for k in MOTION_TARGETS], dtype=torch.float32, device=device
        )
        target_per_slot = [k in TARGET_SCALE_PER_SLOT for k in MOTION_TARGETS]
        logger.info(
            f"[motion] P_phi params: {_count_params(motion_pred) if motion_pred else 0}, "
            f"R_psi params: {_count_params(motion_readout) if motion_readout else 0}"
        )
    motion_modules = [m for m in (motion_pred, motion_readout) if m is not None]
    # [motion] per-token FLOPs accounting: 2*params*tokens per forward, x3 with backward (see FlopMeter)
    flops = FlopMeter()
    n_params_encoder = _count_params(encoder)
    n_params_predictor = _count_params(predictor)
    n_params_heads = {m: _count_params(m) for m in motion_modules}
    # [motion] CPU generators for the CAN-mismatch control (shuffle within strata) and the per-item target-branch
    #          dropout of R_psi; both seeded from meta.seed so arms with matched seeds draw identical sequences
    shuffle_gen = torch.Generator().manual_seed(int(seed) + 1)
    drop_gen = torch.Generator().manual_seed(int(seed) + 2)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    # -- [motion] MotionTransform reproduces the vendored VideoTransform for view 1 (same ops and RNG order, flip
    #    off) and additionally returns the crop box / intrinsics (+ a nested second view when L_cons is on).
    #    Augmentations the vendored transform supports but MotionTransform does not must be off.
    assert not random_horizontal_flip, "[motion] random_horizontal_flip must be false (yaw / steering signs)"
    assert reprob == 0 and not use_aa and not motion_shift, "[motion] reprob / auto_augment / motion_shift unsupported"
    transform = MotionTransform(
        crop_size=crop_size,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        view2_scale=view2_scale,
    )

    # -- init data-loaders/samplers ([motion] motion-aware dataset; sidecars are read but unused when disabled)
    assert dataset_type.lower() == "videodataset", "[motion] only VideoDataset is supported"
    _dw_kwargs = {} if datasets_weights is None else {"datasets_weights": datasets_weights}
    (unsupervised_dataset, unsupervised_loader, unsupervised_sampler) = make_motion_videodataset(
        data_paths=dataset_paths,
        batch_size=batch_size,
        fps=fps,
        dataset_fpcs=dataset_fpcs,
        transform=transform,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        use_comma_only=use_comma_only_fields,
        seed=0,  # private (s, d) / view-2 RNG seed material; the DistributedSampler keeps its default seed 0
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        rays_road_frame=rays_road_frame,
        require_can=require_can and motion_enabled,
        **_dw_kwargs,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    # -- [motion] the heads follow the predictor's lr / wd schedule: two extra AdamW param groups with the vendored
    #    decay rule (weights with wd; biases and 1-d params with weight_decay 0 and WD_exclude). WarmupCosineSchedule
    #    / CosineWDSchedule iterate over optimizer.param_groups, so the added groups are scheduled in place.
    if motion_modules:
        _named = [(n, p) for m in motion_modules for n, p in m.named_parameters()]
        optimizer.add_param_group({"params": [p for n, p in _named if ("bias" not in n) and (len(p.shape) != 1)]})
        optimizer.add_param_group(
            {
                "params": [p for n, p in _named if ("bias" in n) or (len(p.shape) == 1)],
                "WD_exclude": True,
                "weight_decay": 0,
            }
        )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    # -- [motion] heads are left unwrapped on a single process; with world_size > 1 they need gradient all-reduce.
    #    R_psi runs up to three forwards per step (pair, view-1 common region, view 2), which DDP only tolerates
    #    without unused-parameter search -> plain DDP, all head parameters are used in every forward.
    if world_size > 1 and motion_modules:
        if motion_pred is not None:
            motion_pred = DistributedDataParallel(motion_pred)
        if motion_readout is not None:
            motion_readout = DistributedDataParallel(motion_readout)
        logger.warning("[motion] multi-process head training is untested (plan: single 5090 first)")

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    _acc_prev = None  # [motion] accounting snapshot restored from a resumed checkpoint
    latest_prev_path = os.path.join(folder, "latest.prev.pt")  # [motion] previous latest.pt (atomic save keeps it)
    resumed = False  # [motion] True once a training checkpoint was restored (init_from is skipped then)
    if load_path is None and os.path.exists(latest_path):
        # [motion] vendored quirk: with load_checkpoint=false an existing <folder>/latest.pt enters load_checkpoint
        #          with r_path=None (torch.load(None) -> crash after 3 retries); resume from latest.pt instead.
        load_path = latest_path
        logger.info(f"[motion] found {latest_path}; resuming from it")
    if load_model or os.path.exists(latest_path):
        # [motion] a truncated latest.pt (SIGKILL / ENOSPC during the write) must not make every later launch crash:
        #          try latest.pt, then latest.prev.pt; a checkpoint that was asked for explicitly must load, an
        #          auto-resume that finds only broken files starts fresh (loudly) and keeps the broken file aside.
        _candidates = [load_path]
        if load_path == latest_path and os.path.exists(latest_prev_path):
            _candidates.append(latest_prev_path)
        _loaded = None
        for _cand in _candidates:
            try:
                _loaded = load_checkpoint(
                    r_path=_cand,
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    opt=optimizer,
                    scaler=scaler,
                )
                load_path = _cand
                break
            except Exception as e:  # noqa: BLE001 (unreadable / incomplete checkpoint file)
                logger.error(f"[motion] could not resume from {_cand}: {e}")
        if _loaded is None:
            if cfgs_meta.get("load_checkpoint") or resume_preempt:
                raise RuntimeError(f"[motion] none of the checkpoints {_candidates} could be loaded")
            _broken = latest_path + ".broken"
            logger.error(f"[motion] no usable checkpoint in {folder}; starting fresh (unreadable latest.pt kept as {_broken})")
            os.replace(latest_path, _broken)
        else:
            resumed = True
            (
                encoder,
                predictor,
                target_encoder,
                optimizer,
                scaler,
                start_epoch,
            ) = _loaded
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()
            # -- [motion] restore the heads (motion arms) and the cumulative accounting (EVERY resumed run) saved by
            #    save_checkpoint below.  2026-10-16: the accounting restore used to sit inside the motion-only branch, so
            #    a resumed PLAIN arm (A / C, motion disabled) finished with exit_status ok but steps = the last segment
            #    only (< 100k), which the post-E1 queue would have read as an unclean finish (review finding, fixed here).
            _ck = torch.load(load_path, map_location="cpu", weights_only=False, mmap=True)  # 2nd read: mmap
            if motion_modules or motion_enabled:
                _heads = _ck.get("motion_heads") or {}
                for _name, _mod in (("pred", motion_pred), ("readout", motion_readout)):
                    if _mod is not None and _heads.get(_name) is not None:
                        _r = _unwrap(_mod).load_state_dict(_heads[_name], strict=True)
                        logger.info(f"[motion] loaded {_name} head from epoch {start_epoch}: {_r}")
                    elif _mod is not None:
                        logger.warning(f"[motion] checkpoint has no '{_name}' head; keeping the fresh initialisation")
            if _ck.get("accounting"):
                _acc_prev = dict(_ck["accounting"])
                logger.info(f"[motion] restored cumulative accounting from the checkpoint: steps={_acc_prev.get('steps')} "
                            f"wall_s={_acc_prev.get('wall_s')}")
            del _ck
    # -- frontdrive patch: weights-only initialisation (encoder / predictor / target_encoder) from a released checkpoint,
    #    fresh optimizer and schedules (continued pretraining from e50 or vitl.pt). [motion] runs AFTER the resume
    #    attempt so that a fresh start after an unreadable latest.pt still gets the e50 weights.
    init_from = cfgs_meta.get("init_from", None)
    if init_from and not resumed:
        logger.info(f"[frontdrive] init_from weights-only: {init_from}")
        _ck = torch.load(init_from, map_location="cpu", weights_only=False)
        def _strip(sd):  # DDP prefix only; MultiSeqWrapper / PredictorMultiSeqWrapper keep their own "backbone." prefix
            return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        # [motion] load into the modules INSIDE the DDP wrappers (their keys are "backbone.*"); the vendored patch
        #          loads into the DDP wrapper ("module.backbone.*") after stripping "module." and therefore matches
        #          nothing (missing == unexpected == all keys, seen 2026-10-11 with e50). A weights-only init that
        #          leaves encoder / target-encoder keys missing is an error here, not a warning.
        def _load(name, module, sd, strict_keys):
            _r = _unwrap(module).load_state_dict(sd, strict=False)
            logger.info(f"[frontdrive] {name} missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}")
            if strict_keys and (_r.missing_keys or _r.unexpected_keys):
                raise RuntimeError(f"[motion] init_from: {name} keys do not match: missing={_r.missing_keys[:5]} "
                                   f"unexpected={_r.unexpected_keys[:5]}")
            if _r.missing_keys:
                logger.warning(f"[motion] init_from: {name} missing keys (kept fresh): {_r.missing_keys[:8]}")
        _enc_sd = _strip(_ck.get("encoder", _ck.get("target_encoder")))
        _load("encoder", encoder, _enc_sd, strict_keys=True)
        if "predictor" in _ck:
            _load("predictor", predictor, _strip(_ck["predictor"]), strict_keys=False)  # mask-token count may differ
        _tgt_sd = _strip(_ck.get("target_encoder", _ck.get("encoder")))
        _load("target_encoder", target_encoder, _tgt_sd, strict_keys=True)
        del _ck

    # -- [motion] cumulative run accounting (steps / tokens / FLOPs estimate / wall-clock); heads make
    #    "same steps != same compute", so every run writes <folder>/accounting.json (per epoch and at exit)
    config_hash = _config_hash(args)
    acc = {
        "steps": int(_acc_prev["steps"]) if _acc_prev else 0,
        "wall_s": float(_acc_prev["wall_s"]) if _acc_prev else 0.0,
        "wall_s_steps": float(_acc_prev.get("wall_s_steps", 0.0)) if _acc_prev else 0.0,
        "tokens": int(_acc_prev["tokens"]) if _acc_prev else 0,
        "flops_est": float(_acc_prev["flops_est"]) if _acc_prev else 0.0,
    }
    run_t0 = time.time()
    run_state = {"exit_status": "running", "accounting_done": False}  # [motion] exit status for accounting.json

    def write_accounting(exit_status=None):
        if rank != 0:
            return
        if exit_status is not None:
            run_state["exit_status"] = exit_status
        tot = flops.totals()
        acc_out = {
            "steps": acc["steps"],
            "batch_size": batch_size,
            "world_size": world_size,
            "n_patches_per_clip": n_patches_per_clip,
            "clip_tokens": acc["steps"] * batch_size * world_size * n_patches_per_clip,  # input patch tokens
            "tokens": acc["tokens"] + int(_fm_get(tot, "tokens")),  # tokens processed by all modules (FlopMeter)
            "flops_est": acc["flops_est"] + float(_fm_get(tot, "flops")),  # 2*params*tokens (x3 with backward)
            "wall_s": acc["wall_s"] + (time.time() - run_t0),  # wall-clock of this process incl. setup
            "wall_s_steps": acc["wall_s_steps"],  # sum of per-step iteration times
            "exit_status": run_state["exit_status"],  # running | ok | nan_loss | data_error | aborted (exception / signal)
            "config_hash": config_hash,
            "motion": {k: v for k, v in cfgs_motion.items()},
            "flop_meter": tot,  # per-module tokens / flops of this process (FlopMeter.totals())
            "params": {
                "encoder": n_params_encoder,
                "predictor": n_params_predictor,
                "motion_pred": n_params_heads.get(_unwrap(motion_pred), 0) if motion_pred is not None else 0,
                "motion_readout": n_params_heads.get(_unwrap(motion_readout), 0) if motion_readout is not None else 0,
            },
        }
        try:
            with open(os.path.join(folder, "accounting.json"), "w") as f:
                json.dump(acc_out, f, indent=2)
        except Exception as e:
            logger.error(f"[motion] could not write accounting.json: {e}")
        return acc_out

    def _accounting_at_exit():
        # [motion] runs at interpreter exit (uncaught exception, SIGINT/SIGTERM -> KeyboardInterrupt, normal end): an
        #          interrupted partial epoch is then visible in the ledger instead of a stale per-epoch snapshot
        if not run_state["accounting_done"]:
            write_accounting("aborted" if run_state["exit_status"] == "running" else None)

    atexit.register(_accounting_at_exit)

    def _checkpoint_bytes(weights_only):
        """[motion] rough size of a checkpoint: fp32 weights of encoder + target encoder + predictor + heads, plus the two
        AdamW moments of every trained parameter unless weights_only (ViT-L: ~5.1 GB full, ~2.5 GB weights-only)."""
        n_heads = sum(n_params_heads.values())
        n_weights = 2 * n_params_encoder + n_params_predictor + n_heads
        n_opt = 0 if weights_only else 2 * (n_params_encoder + n_params_predictor + n_heads)
        return 4 * (n_weights + n_opt) + (64 << 20)

    def save_checkpoint(epoch, path, weights_only=False):
        """[motion] Atomic checkpoint write: <path>.tmp -> os.replace onto <path>; the previous latest.pt survives as
        latest.prev.pt (resume fallback). Refuses to write when the free space is below twice the estimated size and
        fails loudly (logger.error + raise) instead of training on for hours without a valid checkpoint.
        weights_only=True (periodic e{N}.pt) stores encoder / target_encoder / predictor / heads without the optimizer."""
        if rank != 0:
            return
        need = 2 * _checkpoint_bytes(weights_only)
        free = shutil.disk_usage(folder).free
        if free < need:
            msg = f"[motion] refusing to write {path}: {free / 2**30:.1f} GB free < 2 x estimated checkpoint size {need / 2 / 2**30:.1f} GB"
            logger.error(msg)
            raise RuntimeError(msg)
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "weights_only": weights_only,
            # [motion] heads (plain state dicts, no DDP prefix) + config + accounting snapshot
            "motion_heads": {
                "pred": None if motion_pred is None else _unwrap(motion_pred).state_dict(),
                "readout": None if motion_readout is None else _unwrap(motion_readout).state_dict(),
            },
            "motion_cfg": cfgs_motion,
            "config_hash": config_hash,
            "accounting": write_accounting(),
        }
        if not weights_only:
            save_dict["opt"] = optimizer.state_dict()
            save_dict["scaler"] = None if scaler is None else scaler.state_dict()
        tmp_path = path + ".tmp"
        try:
            torch.save(save_dict, tmp_path)
            if path == latest_path and os.path.exists(latest_path):
                os.replace(latest_path, latest_prev_path)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"[motion] checkpoint save failed for {path}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    global_step = start_epoch * ipe  # [motion] optimizer steps so far (cons_every cadence, max_steps)
    stop_training = False  # [motion] set when meta.max_steps is reached
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()
        # [motion] per-epoch meters of the extra losses and of the two shortcut diagnostics
        jepa_meter, pred_meter, readout_meter, cons_meter = (AverageMeter() for _ in range(4))
        noctx_meter, gratio_meter = AverageMeter(), AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        run_state["exit_status"] = "data_error"  # [motion] recorded by the exit-time accounting
                        raise e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                all_extra = []  # [motion] udata = (buffer_list, label, extra, clip_indices); extra used when enabled
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                    all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                    if motion_enabled:
                        all_extra += [{k: v.to(device, non_blocking=True) for k, v in udata[2].items()}]
                return all_clips, all_masks_enc, all_masks_pred, all_extra

            clips, masks_enc, masks_pred, extra = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            # [motion] consistency pass on this step? (needs the nested view and R_psi)
            do_cons = use_view2 and motion_readout is not None and (global_step % cons_every == 0)

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                        return h

                # [motion] the vendored forward_context(c) = predictor(encoder(c, masks_enc), masks_enc, masks_pred)
                #          is split so that the online tokens z_enc can also feed the motion heads. On consistency
                #          steps the nested second view is appended as an extra "dataset" entry of the SAME encoder
                #          call (one DDP forward per step; MultiSeqWrapper processes the entries independently, so
                #          z_enc[0] is identical to the vendored output).
                def forward_encoder(c, m):
                    return encoder(c, m)

                def forward_predictor(z_enc):
                    return predictor(z_enc, masks_enc, masks_pred)

                def loss_fn(z, h):
                    # Assumption: predictor will have returned only masked tokens for z
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    loss /= n
                    return loss

                def motion_losses(z_enc, h, ex):
                    """[motion] L_pred / L_readout / L_cons for fpc-dataset 0 and mask config motion_mask_config.

                    Conventions: slot s = ex["s"], target slot s + d; the context tokens are the ONLINE encoder's
                    visible tokens of slot s (gradient flows to the deployed encoder), the target tokens are the
                    layer-normed TARGET-encoder tokens of slot s + d (all n_spatial positions, stop-grad, same
                    normalisation as the JEPA loss). Rays are the per-patch unit rays of the view's intrinsics
                    (camera frame x right, y down, z forward; road-aligned when motion.rays_road_frame). Motion
                    targets/sequences follow motion_data.py (vehicle frame of kept frame c_{2s}: dx forward m, dy
                    left m, dyaw CCW rad, v_mean m/s, yaw_rate_mean rad/s, dv m/s, dyaw_rate rad/s).
                    Shortcut counter-measures (review 2026-10-12):
                      * L_pred is averaged over the target positions the online encoder did NOT see (the tube masks
                        keep the same spatial positions in every slot, so visible target tokens are literally in
                        z_s); the L1 on the visible positions is logged separately (loss_pred_vis).
                      * R_psi's target branch is dropped per item with probability readout_tgt_drop (tgt_keep all
                        False), so on those items the online tokens are the only input and the deployed encoder
                        necessarily receives the readout gradient; loss_readout_noctx (context removed, no grad)
                        and grad_ratio_ctx (||dL/dz_ctx|| / ||dL/dz_tgt|| via a detached leaf probe) are logged.
                      * Items whose context slot has < min_ctx_tokens visible tokens (collator truncation, finding:
                        slot 2 empty in 6 % of samples) are excluded from L_pred / L_readout.
                      * shuffle_motion permutes within (speed bin x slot gap d) strata among items with CAN; items
                        left matched (singletons, no sidecar) get all targets / sequence rows masked.
                    Returns (loss_pred, loss_readout, loss_cons, diag) with diag = {MOTION_DIAG_COLUMNS: tensor}.
                    """
                    mc = motion_mask_config
                    z0, idx0 = z_enc[0][mc], masks_enc[0][mc]  # (B, Nk, D), (B, Nk) flat indices
                    s, d = ex["s"].long(), ex["d"].long()
                    B = z0.size(0)
                    dev0 = z0.device
                    zero = torch.zeros((), device=dev0)
                    diag = {k: zero for k in MOTION_DIAG_COLUMNS}
                    z_ctx, ctx_idx, ctx_keep = gather_slot_tokens(z0, idx0, s, n_spatial)  # padded, keep mask
                    rays = ex["rays"].reshape(B, n_spatial, 3)  # (B, Hp, Wp, 3) -> per-token unit rays
                    rays_ctx = rays_for_tokens(ex["rays"], ctx_idx)
                    h_tgt = _slot_full(h[0], s + d, n_spatial).detach()  # (B, n_spatial, D) layer-normed, sg
                    tgt_idx = torch.arange(n_spatial, device=dev0).unsqueeze(0).expand(B, -1)
                    # target positions visible to the online encoder (slot s+d members of the kept indices)
                    member_t = (idx0 // n_spatial) == (s + d)[:, None]
                    vis_tgt = torch.zeros(B, n_spatial, dtype=torch.long, device=dev0).scatter_add_(
                        1, torch.where(member_t, idx0 % n_spatial, torch.zeros_like(idx0)), member_t.long()
                    ) > 0
                    n_vis = ctx_keep.sum(1)
                    ctx_ok = n_vis >= min_ctx_tokens  # (B,) items with a usable context slot
                    diag["frac_ctx_ok"], diag["ctx_vis_mean"] = ctx_ok.float().mean(), n_vis.float().mean()
                    seq, seq_mask, seq_valid = ex["seq"], ex["seq_mask"], ex["seq_valid"]
                    target, target_valid, dt, speed = ex["target"], ex["target_valid"], ex["dt"], ex["speed"]
                    if shuffle_motion:  # CAN-mismatch control: motion inputs AND targets permuted within strata
                        perm = shuffle_within_speed_strata(
                            speed.detach().cpu(), generator=shuffle_gen, strata=d.cpu(), valid=ex["has_motion"].cpu()
                        ).to(dev0)
                        mismatched = perm != torch.arange(B, device=dev0)
                        seq, seq_mask, seq_valid, target, target_valid, dt, speed = (
                            t[perm] for t in (seq, seq_mask, seq_valid, target, target_valid, dt, speed)
                        )
                        # an item that kept its own motion is NOT a mismatch control: mask it out of the motion terms
                        target_valid = target_valid & mismatched[:, None]
                        seq_mask = seq_mask & mismatched[:, None]
                        diag["frac_mismatch"] = mismatched.float().mean()
                    target_valid = target_valid & ctx_ok[:, None]
                    scales_b = expand_target_scales(target_scales, d, target_per_slot)  # (B, T) per-horizon scales
                    l_pred = l_readout = l_cons = 0
                    nc = z_ctx.size(1)
                    if motion_pred is not None:
                        z_hat = motion_pred(
                            z_ctx, ctx_idx, rays_ctx, seq, seq_mask, dt, tgt_idx, rays, s, d, ctx_keep, seq_valid
                        )  # (B, n_spatial, D)
                        err = torch.abs(z_hat.float() - h_tgt.float()).mean(-1)  # (B, n_spatial) L1 like the JEPA loss
                        w_msk = (~vis_tgt) & ctx_ok[:, None]  # positions the online encoder could not copy
                        w_vis = vis_tgt & ctx_ok[:, None]
                        w_all = w_msk if pred_masked_only else ctx_ok[:, None].expand_as(err)
                        l_pred = (err * w_all).sum() / w_all.sum().clamp(min=1)
                        if bool(w_vis.any()):
                            diag["loss_pred_vis"] = (err.detach() * w_vis).sum() / w_vis.sum()
                        flops.add("motion_pred", B * (nc + seq.size(1) + n_spatial), n_params_heads[_unwrap(motion_pred)])
                    if motion_readout is not None and use_readout:
                        # per-item target-branch dropout (CPU generator -> device) and a detached LEAF probe of the
                        # target tokens for the gradient-ratio diagnostic (not connected to the target encoder)
                        drop = (torch.rand(B, generator=drop_gen) < readout_tgt_drop).to(dev0)
                        tgt_keep = (~drop)[:, None].expand(-1, n_spatial)
                        h_probe = h_tgt.detach().requires_grad_(True)
                        m_hat, e_r = motion_readout(
                            z_ctx, ctx_idx, rays_ctx, h_probe, tgt_idx, rays, dt, s, d, ctx_keep, tgt_keep
                        )
                        l_readout = readout_loss(m_hat.float(), target, target_valid, scales_b)
                        terms = readout_loss_terms(m_hat, target, target_valid, scales_b)
                        for _k, _name in enumerate(MOTION_TARGETS):
                            diag[f"huber_{_name}"] = terms[_k]
                        diag["e_std"] = embedding_spread(e_r)
                        diag["grad_ratio_ctx"], _, _ = online_gradient_ratio(l_readout, z_ctx, h_probe)
                        with torch.no_grad():  # same inputs with the online context removed
                            m_noctx, _ = motion_readout(
                                z_ctx.detach(), ctx_idx, rays_ctx, h_tgt, tgt_idx, rays, dt, s, d,
                                torch.zeros_like(ctx_keep), tgt_keep,
                            )
                            diag["loss_readout_noctx"] = readout_loss(m_noctx.float(), target, target_valid, scales_b)
                        n_tok = B * (nc + n_spatial + 1)
                        flops.add("motion_readout", n_tok, n_params_heads[_unwrap(motion_readout)])
                        # diagnostics: one no-grad forward + one extra backward through R_psi (~2 forwards)
                        flops.add("motion_readout_diag", 3 * n_tok, n_params_heads[_unwrap(motion_readout)], backward=False)
                    if do_cons:
                        # view 1 restricted to the common visible region (ctx: visible AND common; tgt: common)
                        common = ex["common_mask"].bool()  # (B, n_spatial) over view-1 tokens
                        keep1 = ctx_keep & torch.gather(common, 1, ctx_idx)
                        _, e1 = motion_readout(
                            z_ctx, ctx_idx, rays_ctx, h_tgt, tgt_idx, rays, dt, s, d, keep1, common
                        )
                        # view 2 (nested zoom crop, own intrinsics): encoder output appended as entry 1
                        z2, idx2 = z_enc[1][0], masks_enc[0][mc]
                        z_ctx2, ctx_idx2, ctx_keep2 = gather_slot_tokens(z2, idx2, s, n_spatial)
                        rays2 = ex["rays2"].reshape(B, n_spatial, 3)
                        rays_ctx2 = rays_for_tokens(ex["rays2"], ctx_idx2)
                        h_tgt2 = _slot_full(h[1], s + d, n_spatial).detach()
                        _, e2 = motion_readout(
                            z_ctx2, ctx_idx2, rays_ctx2, h_tgt2, tgt_idx, rays2, dt, s, d, ctx_keep2, None
                        )
                        l_cons = consistency_loss(e1.float(), e2.float())
                        if not use_readout:
                            diag["e_std"] = embedding_spread(e1)
                        flops.add(
                            "motion_readout",
                            B * (2 * n_spatial + nc + z_ctx2.size(1) + 2),
                            n_params_heads[_unwrap(motion_readout)],
                        )
                    return l_pred, l_readout, l_cons, diag

                # Step 1. Forward
                flops.reset_step()  # [motion] per-step token / FLOPs accounting
                diag = {}  # [motion]
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    enc_inputs, enc_masks = clips, masks_enc
                    if do_cons:  # [motion] nested view through the same encoder / target-encoder calls
                        enc_inputs = clips + [extra[0]["view2"]]
                        enc_masks = masks_enc + [[masks_enc[0][motion_mask_config]]]
                    h = forward_target(enc_inputs)
                    z_enc = forward_encoder(enc_inputs, enc_masks)
                    z = forward_predictor(z_enc[: len(clips)])
                    loss_jepa = loss_fn(z, h[: len(clips)])  # jepa prediction loss (vendored)
                    loss = loss_jepa
                    loss_pred = loss_readout = loss_cons = 0
                    if motion_enabled:  # [motion]
                        loss_pred, loss_readout, loss_cons, diag = motion_losses(z_enc, h, extra[0])
                        loss = loss + weight_pred * loss_pred + weight_readout * loss_readout + cons_weight * loss_cons
                # [motion] accounting of the vendored modules (tokens actually processed this step)
                for hi in h:
                    flops.add("target_encoder", hi.size(0) * hi.size(1), n_params_encoder, backward=False)
                for zi in z_enc:
                    for zij in zi:
                        flops.add("encoder", zij.size(0) * zij.size(1), n_params_encoder)
                for mxi, myi in zip(masks_enc, masks_pred):
                    for mxij, myij in zip(mxi, myi):
                        flops.add("predictor", mxij.size(0) * (mxij.size(1) + myij.size(1)), n_params_predictor)

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                step_tot = flops.step_totals()  # [motion]
                _f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)  # noqa: E731  [motion]
                return (
                    float(loss),
                    _new_lr,
                    _new_wd,
                    # [motion] extra terms (0.0 when disabled) and this step's token / FLOPs estimate
                    (_f(loss_jepa), _f(loss_pred), _f(loss_readout), _f(loss_cons)),
                    (int(_fm_get(step_tot, "tokens")), float(_fm_get(step_tot, "flops"))),
                    tuple(_f(diag.get(k, 0.0)) for k in MOTION_DIAG_COLUMNS),  # [motion] diagnostics row
                )

            (
                loss,
                _new_lr,
                _new_wd,
                (loss_jepa, loss_pred, loss_readout, loss_cons),
                (tokens_step, flops_step),
                diag_row,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)
            # [motion]
            jepa_meter.update(loss_jepa)
            pred_meter.update(loss_pred)
            readout_meter.update(loss_readout)
            cons_meter.update(loss_cons)
            noctx_meter.update(diag_row[MOTION_DIAG_COLUMNS.index("loss_readout_noctx")])
            gratio_meter.update(diag_row[MOTION_DIAG_COLUMNS.index("grad_ratio_ctx")])
            global_step += 1
            acc["steps"] += 1
            acc["wall_s_steps"] += iter_elapsed_time_ms / 1000.0

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_pred,
                    loss_readout,
                    loss_cons,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                    tokens_step,
                    flops_step,
                    *diag_row,  # [motion]
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "(jepa %.3f pred %.3f readout %.3f cons %.3f) "  # [motion]
                        "[readout noctx %.3f | ctx-grad ratio %.2f] "  # [motion] shortcut diagnostics (epoch averages)
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms] "
                        "[tokens/step: %d] [TFLOP/step: %.2f]"  # [motion]
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jepa_meter.avg,
                            pred_meter.avg,
                            readout_meter.avg,
                            cons_meter.avg,
                            noctx_meter.avg,
                            gratio_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            (torch.cuda.max_memory_allocated() / 1024.0**2) if device.type == "cuda" else 0.0,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                            tokens_step,
                            flops_step / 1e12,
                        )
                    )

            log_stats()
            if np.isnan(loss):
                run_state["exit_status"] = "nan_loss"  # [motion] recorded by the exit-time accounting
            assert not np.isnan(loss), "loss is nan"

            # [motion] optional early stop after meta.max_steps optimizer steps (tests / smoke runs)
            if max_steps is not None and global_step >= int(max_steps):
                logger.info(f"[motion] reached meta.max_steps={max_steps}; stopping after this step")
                stop_training = True
                break

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path, weights_only=True)  # [motion] no optimizer state (~half size)
        write_accounting()  # [motion] <folder>/accounting.json (also embedded in every checkpoint)
        if stop_training:  # [motion]
            break
    write_accounting("ok")  # [motion] final totals
    run_state["accounting_done"] = True


# [motion] ------------------------------------------------------------------------------------------------------------ #
# Minimal launcher mirroring app/main.py (one process per device, spawn start method):
#   python -m pretrain.train_motion --fname pretrain/configs/comma_vitl_256x512_8f_motion.yaml --devices cuda:0
# --devices selects the physical GPU(s) through CUDA_VISIBLE_DEVICES, so meta.device stays "cuda:0" inside a process.
def process_main(rank, fname, world_size, devices):
    import logging
    import pprint
    from pathlib import Path

    import yaml

    _logger = get_logger(force=True)
    _logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    # --devices cuda:N selects the physical GPU through CUDA_VISIBLE_DEVICES; a pre-set CUDA_VISIBLE_DEVICES is kept
    # (N then indexes the already visible devices) instead of being clobbered. An index that does not exist makes
    # main() raise (CUDA unavailable) rather than silently train on the CPU.
    _dev_index = str(devices[rank]).split(":")[-1]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        _logger.info(f"keeping CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}; --devices {devices[rank]} indexes the visible devices")
    elif _dev_index.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = _dev_index
    _logger.info(f"called-params {fname} (device {devices[rank]})")
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        Path(params["folder"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(params["folder"], "params-pretrain.yaml"), "w") as f:
            yaml.dump(params, f)
    # default-port group like app/main.py; if that fails (port busy) main() falls back to a free local port
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    _logger.info(f"Running... (rank: {rank}/{world_size})")
    main(args=params)


if __name__ == "__main__":
    import argparse
    import multiprocessing as pymp

    parser = argparse.ArgumentParser(description="frontdrive motion-arm V-JEPA trainer (fork of app/vjepa/train.py)")
    parser.add_argument("--fname", type=str, required=True, help="yaml config (see pretrain/configs)")
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"], help="local devices, one process each")
    cli = parser.parse_args()
    if len(cli.devices) == 1:
        process_main(rank=0, fname=cli.fname, world_size=1, devices=cli.devices)
    else:
        pymp.set_start_method("spawn")
        for _rank in range(len(cli.devices)):
            pymp.Process(target=process_main, args=(_rank, cli.fname, len(cli.devices), cli.devices)).start()
