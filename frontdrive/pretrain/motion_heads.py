"""Motion-constrained heads for the v4.1 research arm (plan section 3.1).

Added on top of the unchanged V-JEPA masked-prediction loss, for one slot pair (s, s+d) per sample:

    z_hat_{s+d} = P_phi(z_s, m_{s:s+d}, K, dt)        MotionConditionedPredictor  (L1 vs layer-normed target tokens)
    m_hat       = R_psi(z_s, sg(zbar_{s+d}), K, dt)   MotionReadout               (Huber on normalised motion targets)
    L_cons      = || n(e_view1) - n(e_view2) ||^2      consistency_loss on R_psi's pooled embedding of two nested views

Conventions (binding for the trainer and the data side)
--------------------------------------------------------
* Tokens.  The ViT-L/16 encoder flattens an 8-frame 256x512 clip time-major: flat index = slot * n_spatial + row * 32
  + col, with n_spatial = 16 * 32 = 512 positions per 1-s temporal slot (tubelet 2 frames @ 2 Hz) and 4 slots.  The
  heads work on one slot at a time: they take SPATIAL indices (0..n_spatial-1) plus the slot ids separately (`s` =
  context slot, `s + d` = target slot, 0 <= s, s + d < n_slots).  `gather_slot_tokens` turns the encoder's kept flat
  indices into padded per-slot tensors with keep masks (the number of visible tokens per slot differs across samples).
* Rays.  Per-token unit viewing rays in the camera frame (x right, y down, z forward) from the clip intrinsics after
  the crop (motion_data.patch_center_rays); they carry K into both heads as Fourier features.  `rays_for_tokens`
  gathers them per token.
* dt.  Seconds between the start-of-frame of the first frame of slot s and of slot s+d (nominal 1.0 * d).  Embedded as
  Fourier features of dt / DT_SCALE (DT_SCALE = 4 s puts the 0.5..3.5 s range inside [0, 1]).
* Motion sequence.  seq (B, L, F): rows = kept frames c_{2s}..c_{2(s+d)} (L <= max_len = motion_data.motion_seq_len(fpc),
  padded), features in the order motion_data.MOTION_SEQ_FEATURES [+ COMMA_ONLY_FEATURES]; seq_mask (B, L) marks rows
  with at least one valid feature and seq_valid (B, L, F) the per-feature validity (a sidecar without a gyro has a
  False yaw_rate column).  The data side passes RAW physical units (m/s, rad/s, m/s^2, m, m, rad, flags); the features
  are divided by fixed physical scales (_SEQ_FEATURE_SCALES, the only place this normalisation happens), invalid
  entries are zeroed and the validity flags are appended as F extra input channels (so "0 because unknown" and "0
  because standing still" stay distinguishable); padded rows are excluded from attention.
* Targets.  m = motion_data.MOTION_TARGETS = [dx, dy, dyaw, v_mean, yaw_rate_mean, dv, dyaw_rate] in the vehicle frame
  of frame c_{2s}: x forward (m), y left (m), yaw CCW (rad), window means in m/s and rad/s, and the pair-specific
  differences (target-slot tubelet mean minus context-slot tubelet mean) in m/s and rad/s.  R_psi predicts
  target / scales; the scales are per horizon (expand_target_scales: entries of TARGET_SCALE_PER_SLOT grow with the
  slot gap d) so every normalised target has std ~0.3-1; readout_loss applies them and the per-target validity mask,
  readout_loss_terms reports the per-target Huber means for logging.
* e_motion.  The LayerNorm-ed final state of R_psi's learned MOTION query (B, dim), before the output MLP.  It is not
  unit-norm: consistency_loss L2-normalises both sides itself (loss = 2 - 2 cos, in [0, 4]).  embedding_spread
  (batch-wise std of the normalised e) is the collapse monitor for L_cons.
* Gradients.  z_ctx (online-encoder tokens) keeps its graph in both heads; z_tgt must be detached from the target
  encoder's graph (asserted: `z_tgt.grad_fn is None`), so the target encoder never receives gradients and the readout
  never sees P_phi's prediction (no copy shortcut).  A detached LEAF probe `h.detach().requires_grad_(True)` is allowed:
  online_gradient_ratio uses it to measure ||dL/dz_ctx|| / ||dL/dz_tgt|| -- how much of the readout signal reaches
  the deployed encoder rather than the gradient-free target branch.
* Readout shortcut (review 2026-10-12).  The target-encoder tokens are a whole-clip representation (full attention over
  all slots) and the one-slot targets (dx ~ v*dt, v_mean) are readable from them alone, so R_psi can minimise L_readout
  without using z_ctx.  Counter-measures: pair-specific targets (dv, dyaw_rate), per-horizon scales, and in the trainer
  a per-item TARGET-BRANCH DROPOUT (tgt_keep all False with probability motion.readout_tgt_drop) that makes the online
  branch the only input on those items, plus the per-step diagnostics (loss with the context removed, gradient ratio).
* Mismatch control.  shuffle_within_speed_strata returns a fixed-point-free permutation inside each stratum (speed bin
  x optional extra key such as the slot gap d, so a shuffled item receives a dt consistent with its slot gap; random
  cycle per stratum; identity for singleton strata and for items flagged invalid, which the caller must then mask
  out), so the shuffled motion is desynchronised from the video while the speed marginal is preserved.
* FLOPs.  FlopMeter.estimate = 2 * params * tokens per forward pass, x3 when a backward pass follows (backward ~ 2x
  forward); attention's token^2 terms (~10-15 % for 2048-token ViT-L) are ignored.  Pure Python ints, so it is cheap.

No CUDA-only imports: everything here runs on CPU for the unit tests.  Python 3.10.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DT_SCALE = 4.0  # seconds; dt / DT_SCALE is what the Fourier embedding sees

# Fixed input scales (divide by) for the motion-sequence features, in the canonical order
# motion_data.MOTION_SEQ_FEATURES + motion_data.COMMA_ONLY_FEATURES (duplicated here: no dependency on the data module).
_SEQ_FEATURE_SCALES = {
    "v_ego": 30.0,        # m/s
    "yaw_rate": 0.5,      # rad/s
    "a_ego": 5.0,         # m/s^2
    "d_forward": 20.0,    # m per 0.5 s kept-frame step
    "d_left": 2.0,        # m
    "d_yaw": 0.5,         # rad
    "standstill": 1.0,    # bool
    "steer_deg": 90.0,    # deg (comma-only)
    "cc_enabled": 1.0,    # bool (comma-only)
    "cc_lat_active": 1.0,
    "cc_long_active": 1.0,
}


# ----------------------------------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------------------------------
def param_count(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters of `module`."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def gather_slot_tokens(z: torch.Tensor, idx: torch.Tensor, slot: torch.Tensor, n_spatial: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select, per sample, the tokens that belong to temporal slot `slot[b]` and pad to a common length.

    z (B, N, D): tokens whose time-major flat indices are idx (B, N) (flat = slot * n_spatial + spatial); slot (B,)
    long.
    Returns (z_slot (B, M, D), spatial_idx (B, M) long, keep (B, M) bool) with M = max member count over the batch.
    Padded rows are zero / index 0 / keep False.  Gradient flows to the selected rows of `z`.
    """
    member = (idx // n_spatial) == slot[:, None]  # (B, N)
    m_max = int(member.sum(1).max().item()) if member.numel() else 0
    # stable descending sort of the membership flags puts the members first, preserving their original order
    order = torch.argsort(member.to(torch.int8), dim=1, descending=True, stable=True)[:, :m_max]
    keep = torch.gather(member, 1, order)
    z_slot = torch.gather(z, 1, order.unsqueeze(-1).expand(-1, -1, z.shape[-1])) * keep.unsqueeze(-1).to(z.dtype)
    spatial = torch.where(keep, torch.gather(idx, 1, order) % n_spatial, torch.zeros_like(order))
    return z_slot, spatial, keep


def rays_for_tokens(rays: torch.Tensor, spatial_idx: torch.Tensor) -> torch.Tensor:
    """rays (B, H_p, W_p, 3) per-patch unit rays (row-major grid) -> (B, M, 3) rays of the tokens spatial_idx (B, M)."""
    flat = rays.flatten(1, 2)  # (B, H_p*W_p, 3), index = row * W_p + col
    return torch.gather(flat, 1, spatial_idx.unsqueeze(-1).expand(-1, -1, 3))


# ----------------------------------------------------------------------------------------------------------------------
# embeddings and transformer blocks (pre-norm, GELU, SDPA; same style as the vendored predictor, no CUDA dependency)
# ----------------------------------------------------------------------------------------------------------------------
class FourierEmbed(nn.Module):
    """x[..., in_dim] -> [..., out_dim]: Linear over [x, sin(pi 2^k x), cos(pi 2^k x)]_{k < n_freq} (fixed NeRF-style
    frequencies).  Inputs are expected in roughly [-1, 1] (unit rays) or [0, 1] (dt / DT_SCALE)."""

    def __init__(self, in_dim: int, n_freq: int, out_dim: int):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32), persistent=False)
        self.proj = nn.Linear(in_dim * (1 + 2 * n_freq), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        ang = (x.unsqueeze(-1) * self.freqs).flatten(-2)  # [..., in_dim * n_freq]
        return self.proj(torch.cat([x, ang.sin(), ang.cos()], dim=-1))


class MotionSequenceEncoder(nn.Module):
    """Ego-motion sequence -> (B, L, dim) tokens: [scaled features, per-feature validity flags] -> Linear(2F -> dim),
    + learned row-position embedding (max_len rows), + Fourier embedding of dt (broadcast over rows), LayerNorm; masked
    rows are zeroed.  `feat_valid` (B, L, F) bool marks the usable entries (None = all entries of valid rows); invalid
    or NaN entries count as 0 and are flagged 0 in the validity channels."""

    def __init__(self, n_feat: int, dim: int, max_len: int = 7, n_freq: int = 6):
        super().__init__()
        names = list(_SEQ_FEATURE_SCALES)
        scales = [_SEQ_FEATURE_SCALES[n] for n in names[:n_feat]] if n_feat in (7, 11) else [1.0] * n_feat
        self.register_buffer("feat_scale", torch.tensor(scales, dtype=torch.float32))
        self.n_feat, self.max_len = n_feat, max_len
        self.proj = nn.Linear(2 * n_feat, dim)
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        self.dt_embed = FourierEmbed(1, n_freq, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor,
                feat_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert seq.shape[1] <= self.max_len, f"sequence length {seq.shape[1]} exceeds max_len {self.max_len}"
        keep = mask.bool().unsqueeze(-1)  # (B, L, 1)
        valid = keep.expand_as(seq) if feat_valid is None else (feat_valid.bool() & keep)
        valid = valid & torch.isfinite(seq)
        x = torch.where(valid, torch.nan_to_num(seq.float(), nan=0.0), torch.zeros((), dtype=torch.float32, device=seq.device))
        x = torch.cat([x / self.feat_scale, valid.float()], dim=-1)
        x = self.proj(x) + self.pos[: seq.shape[1]] + self.dt_embed((dt.float() / DT_SCALE).unsqueeze(-1)).unsqueeze(1)
        return self.norm(x) * keep


class _Attention(nn.Module):
    """Multi-head self-attention with an optional key-padding mask (True = usable key), via SDPA."""

    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attn_mask = None if key_keep is None else key_keep[:, None, None, :]  # broadcast over heads and queries
        _tv = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
        if attn_mask is not None and q.is_cuda and _tv < (2, 2):
            # torch 2.1.x: masked SDPA on CUDA dispatches to the mem-efficient kernel whose arbitrary-mask support was new
            # in 2.1; these heads are small (dim 384), so take the exact math path there.
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), _Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), _MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, key_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_keep)
        return x + self.mlp(self.norm2(x))


def _init_transformer(module: nn.Module, blocks: nn.ModuleList, std: float = 0.02) -> None:
    """Vendored-predictor initialisation: trunc-normal Linear weights, zero biases, unit LayerNorm, and residual
    branches of block i rescaled by 1/sqrt(2 i)."""

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    module.apply(init)
    for i, blk in enumerate(blocks, start=1):
        blk.attn.proj.weight.data.div_(math.sqrt(2.0 * i))
        blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * i))


def _bool_keep(keep: Optional[torch.Tensor], B: int, N: int, device: torch.device) -> torch.Tensor:
    return torch.ones(B, N, dtype=torch.bool, device=device) if keep is None else keep.bool()


# ----------------------------------------------------------------------------------------------------------------------
# P_phi: motion-conditioned prediction of the target-encoder tokens of slot s+d
# ----------------------------------------------------------------------------------------------------------------------
class MotionConditionedPredictor(nn.Module):
    """P_phi.  Context = online-encoder tokens of slot s (projected to `dim`, + learned spatial pos-emb + slot emb(s)
    + ray Fourier emb); motion tokens from MotionSequenceEncoder (omitted when cond_motion=False -> plain
    future-prediction head); queries = learned mask token + pos-emb(tgt) + slot emb(s+d) + ray emb + dt emb.  A pre-norm
    transformer runs over [ctx; motion; queries] with full attention except that padded ctx rows and padded motion rows
    are masked out as keys; output = Linear(dim -> enc_dim) of the query positions, (B, Nt, enc_dim).
    init_std (vendored-predictor convention, 0.02 for dim >= 384) sets the trunc-normal scale of all Linear weights and
    learned embeddings; tiny test-sized heads (dim ~32) need ~0.1 to keep the context path at a trainable scale."""

    def __init__(self, enc_dim: int = 1024, dim: int = 384, depth: int = 6, heads: int = 6, n_spatial: int = 512,
                 n_slots: int = 4, n_feat: int = 7, cond_motion: bool = True, mlp_ratio: float = 4.0, n_freq: int = 6,
                 init_std: float = 0.02, seq_len: int = 7):
        super().__init__()
        self.enc_dim, self.dim, self.n_spatial, self.n_slots = enc_dim, dim, n_spatial, n_slots
        self.cond_motion = cond_motion
        self.embed = nn.Linear(enc_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(n_spatial, dim))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.ray_embed = FourierEmbed(3, n_freq, dim)
        self.dt_embed = FourierEmbed(1, n_freq, dim)
        self.motion_enc = MotionSequenceEncoder(n_feat, dim, max_len=seq_len, n_freq=n_freq) if cond_motion else None
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, enc_dim)
        _init_transformer(self, self.blocks, std=init_std)
        nn.init.trunc_normal_(self.pos_emb, std=init_std)
        nn.init.trunc_normal_(self.slot_emb, std=init_std)
        # mask token stays zero-initialised, as in the vendored predictor (zero_init_mask_tokens)

    def forward(self, z_ctx: torch.Tensor, ctx_idx: torch.Tensor, rays_ctx: torch.Tensor, seq: torch.Tensor,
                seq_mask: torch.Tensor, dt: torch.Tensor, tgt_idx: torch.Tensor, rays_tgt: torch.Tensor,
                s: torch.Tensor, d: torch.Tensor, ctx_keep: Optional[torch.Tensor] = None,
                seq_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nc, _ = z_ctx.shape
        Nt = tgt_idx.shape[1]
        assert bool(((s + d) < self.n_slots).all()) and bool((d >= 1).all()), "need 1 <= d and s + d < n_slots"
        ctx_keep = _bool_keep(ctx_keep, B, Nc, z_ctx.device)
        ctx = self.embed(z_ctx) + self.pos_emb[ctx_idx] + self.slot_emb[s].unsqueeze(1) + self.ray_embed(rays_ctx)
        ctx = ctx * ctx_keep.unsqueeze(-1)  # zero padded rows (they are also removed from the keys below)
        dt_e = self.dt_embed((dt.float() / DT_SCALE).unsqueeze(-1))  # (B, dim)
        queries = (self.mask_token + self.pos_emb[tgt_idx] + self.slot_emb[s + d].unsqueeze(1)
                   + self.ray_embed(rays_tgt) + dt_e.unsqueeze(1))
        parts, keeps = [ctx], [ctx_keep]
        if self.motion_enc is not None:
            parts.append(self.motion_enc(seq, seq_mask, dt, seq_valid))
            keeps.append(seq_mask.bool())
        parts.append(queries)
        keeps.append(torch.ones(B, Nt, dtype=torch.bool, device=z_ctx.device))
        x, key_keep = torch.cat(parts, dim=1), torch.cat(keeps, dim=1)
        for blk in self.blocks:
            x = blk(x, key_keep)
        return self.proj(self.norm(x[:, -Nt:]))


# ----------------------------------------------------------------------------------------------------------------------
# R_psi: motion readout from online slot-s tokens and stop-grad target slot-(s+d) tokens
# ----------------------------------------------------------------------------------------------------------------------
class MotionReadout(nn.Module):
    """R_psi.  A learned MOTION query (+ dt emb) is prepended to [ctx tokens + slot emb(s); tgt tokens + slot emb(s+d)]
    (separate input projections for online and target tokens, + pos-emb + ray emb) and `depth` pre-norm blocks run over
    the whole sequence with key-padding masks from ctx_keep / tgt_keep (the query is never masked, so no row is empty).
    e_motion = LayerNorm(final query state) (B, dim); m_hat = MLP(e_motion) (B, n_targets) predicts target / scales."""

    def __init__(self, enc_dim: int = 1024, dim: int = 384, depth: int = 3, heads: int = 6, n_spatial: int = 512,
                 n_slots: int = 4, n_targets: int = 7, mlp_ratio: float = 4.0, n_freq: int = 6, init_std: float = 0.02):
        super().__init__()
        self.enc_dim, self.dim, self.n_spatial, self.n_slots = enc_dim, dim, n_spatial, n_slots
        self.n_targets = n_targets
        self.embed_ctx = nn.Linear(enc_dim, dim)
        self.embed_tgt = nn.Linear(enc_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(n_spatial, dim))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.ray_embed = FourierEmbed(3, n_freq, dim)
        self.dt_embed = FourierEmbed(1, n_freq, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_targets))
        _init_transformer(self, self.blocks, std=init_std)
        for p in (self.pos_emb, self.slot_emb, self.query):
            nn.init.trunc_normal_(p, std=init_std)

    def forward(self, z_ctx: torch.Tensor, ctx_idx: torch.Tensor, rays_ctx: torch.Tensor, z_tgt: torch.Tensor,
                tgt_idx: torch.Tensor, rays_tgt: torch.Tensor, dt: torch.Tensor, s: torch.Tensor, d: torch.Tensor,
                ctx_keep: Optional[torch.Tensor] = None, tgt_keep: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # stop-grad invariant: z_tgt must not be connected to the target encoder's graph (a detached leaf probe with
        # requires_grad=True is fine -- see online_gradient_ratio; a graph output is refused)
        assert z_tgt.grad_fn is None, "z_tgt must be detached target-encoder tokens (call .detach() before R_psi)"
        B, Nc, _ = z_ctx.shape
        Nt = z_tgt.shape[1]
        assert bool(((s + d) < self.n_slots).all()) and bool((d >= 1).all()), "need 1 <= d and s + d < n_slots"
        ctx_keep, tgt_keep = _bool_keep(ctx_keep, B, Nc, z_ctx.device), _bool_keep(tgt_keep, B, Nt, z_ctx.device)
        ctx = self.embed_ctx(z_ctx) + self.pos_emb[ctx_idx] + self.slot_emb[s].unsqueeze(1) + self.ray_embed(rays_ctx)
        tgt = (self.embed_tgt(z_tgt) + self.pos_emb[tgt_idx] + self.slot_emb[s + d].unsqueeze(1)
               + self.ray_embed(rays_tgt))
        query = self.query.expand(B, 1, self.dim) + self.dt_embed((dt.float() / DT_SCALE).unsqueeze(-1)).unsqueeze(1)
        x = torch.cat([query, ctx * ctx_keep.unsqueeze(-1), tgt * tgt_keep.unsqueeze(-1)], dim=1)
        key_keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), ctx_keep, tgt_keep], dim=1)
        for blk in self.blocks:
            x = blk(x, key_keep)
        e_motion = self.norm(x[:, 0])
        return self.head(e_motion), e_motion


# ----------------------------------------------------------------------------------------------------------------------
# losses and controls
# ----------------------------------------------------------------------------------------------------------------------
def expand_target_scales(scales: torch.Tensor, d: torch.Tensor, per_slot: Sequence[bool]) -> torch.Tensor:
    """Per-sample normalisation scales (B, T): scales (T,) with the entries flagged in `per_slot` (T bools, e.g.
    motion_data.TARGET_SCALE_PER_SLOT membership) multiplied by the slot gap d (B,) long -- displacement-like targets
    grow linearly with the horizon (dx ~ v * d), so the per-horizon scale keeps their normalised spread ~constant."""
    ps = torch.as_tensor(list(per_slot), dtype=torch.bool, device=scales.device)
    growth = torch.where(ps[None, :], d.to(scales.device).float().clamp(min=1)[:, None], torch.ones((), device=scales.device))
    return scales.float()[None, :] * growth


def _huber_entries(m_hat: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Elementwise smooth-L1 (beta 1) between m_hat and target / scales, 0 at invalid entries; scales (T,) or (B, T)."""
    t = torch.where(valid, torch.nan_to_num(target.float(), nan=0.0), torch.zeros_like(m_hat, dtype=torch.float32))
    t = t / scales.to(m_hat.device).float()
    per_entry = F.smooth_l1_loss(m_hat.float(), t, reduction="none", beta=1.0)
    return torch.where(valid, per_entry, torch.zeros_like(per_entry))


def readout_loss(m_hat: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 (Huber, beta = 1) between m_hat (B, T) and target / scales over the entries where valid (B, T) is True.
    `scales` is (T,) or per-sample (B, T) (expand_target_scales).  Invalid entries may hold NaN (they are replaced
    before the loss, so no NaN leaks into the gradient).  Returns a graph-connected 0 when nothing is valid."""
    valid = valid.bool()
    if not bool(valid.any()):
        return (m_hat * 0.0).sum()
    return _huber_entries(m_hat, target, valid, scales).sum() / valid.sum()


def readout_loss_terms(m_hat: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Per-target mean Huber (T,) over the valid entries of each column (NaN where a column has no valid entry), for
    logging the balance between targets.  Detached."""
    valid = valid.bool()
    with torch.no_grad():
        per_entry = _huber_entries(m_hat.detach(), target, valid, scales)
        n = valid.sum(0).float()
        return torch.where(n > 0, per_entry.sum(0) / n.clamp(min=1), torch.full_like(n, float("nan")))


def consistency_loss(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of the squared L2 distance between the L2-normalised embeddings (= 2 - 2 cos, in [0, 4]).
    Both sides keep their gradient."""
    return (F.normalize(e1.float(), dim=-1) - F.normalize(e2.float(), dim=-1)).pow(2).sum(-1).mean()


def embedding_spread(e: torch.Tensor) -> torch.Tensor:
    """Collapse monitor for L_cons (plan 9.5): mean over dimensions of the batch-wise std of the L2-normalised
    embeddings (B, dim).  A constant embedding gives 0; i.i.d. unit vectors give ~1/sqrt(dim).  Detached."""
    with torch.no_grad():
        if e.shape[0] < 2:
            return torch.zeros((), device=e.device)
        return F.normalize(e.detach().float(), dim=-1).std(0).mean()


def online_gradient_ratio(loss: torch.Tensor, z_ctx: torch.Tensor, z_tgt_probe: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Readout-shortcut diagnostic: (||dloss/dz_ctx|| / (||dloss/dz_tgt|| + eps), ||dloss/dz_ctx||, ||dloss/dz_tgt||).

    `z_ctx` is the online context tensor fed to R_psi (graph-connected to the encoder; the gradient is taken w.r.t.
    this intermediate, so nothing is propagated into the encoder here) and `z_tgt_probe` the detached LEAF
    `h.detach().requires_grad_(True)` that was passed as z_tgt.  Uses autograd.grad with retain_graph=True (the
    caller's backward follows); the cost is one backward through R_psi only.  A ratio near 0 means L_readout is being
    minimised from the gradient-free target branch alone.  All outputs detached; zeros when `loss` has no graph."""
    if not torch.is_tensor(loss) or not loss.requires_grad:
        z = torch.zeros((), device=z_ctx.device)
        return z, z, z
    g_ctx, g_tgt = torch.autograd.grad(loss, [z_ctx, z_tgt_probe], retain_graph=True, allow_unused=True)
    n_ctx = torch.zeros((), device=z_ctx.device) if g_ctx is None else g_ctx.detach().float().norm()
    n_tgt = torch.zeros((), device=z_ctx.device) if g_tgt is None else g_tgt.detach().float().norm()
    return n_ctx / (n_tgt + 1e-12), n_ctx, n_tgt


def shuffle_within_speed_strata(speed: torch.Tensor, bins: Sequence[float] = (5.0, 12.0, 20.0),
                                generator: Optional[torch.Generator] = None, strata: Optional[torch.Tensor] = None,
                                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """CAN-mismatch control: perm (B,) long with perm[i] in the same stratum as i, and perm[i] != i whenever the stratum
    holds >= 2 items (a random cycle through its members; singleton strata map to themselves).  A stratum is the speed
    bin (edges `bins`, m/s) combined with the optional integer key `strata` (B,) (e.g. the slot gap d, so that the
    received dt / targets match the tokens' slot gap).  Items with `valid[i]` False (no CAN) are never permuted and
    never receive another item's motion; callers mask fixed points (perm == arange) out of the motion terms so that
    "shuffled" never silently means "matched".  Apply as seq[perm], target[perm], ... so item i receives the motion
    of item perm[i].  Drawn on the CPU from `generator` (or the global RNG), returned on speed.device."""
    speed_cpu = torch.nan_to_num(speed.detach().float().cpu(), nan=0.0)
    key = torch.bucketize(speed_cpu, torch.tensor(list(bins), dtype=torch.float32))
    if strata is not None:
        key = key * (int(strata.max().item()) + 2) + strata.detach().long().cpu()
    ok = torch.ones_like(key, dtype=torch.bool) if valid is None else valid.detach().bool().cpu()
    perm = torch.arange(speed_cpu.numel())
    for b in key[ok].unique():
        members = ((key == b) & ok).nonzero().flatten()
        if members.numel() < 2:
            continue
        order = members[torch.randperm(members.numel(), generator=generator)]
        perm[order] = order.roll(-1)  # order[j] -> order[j + 1]: one random cycle, no fixed points
    return perm.to(speed.device)


class FlopMeter:
    """Cheap token / FLOP accounting (pure Python ints).

    estimate(tokens, params, backward) = 2 * params * tokens * (3 if backward else 1): a dense forward pass costs one
    multiply-add per parameter per token (2 FLOPs), the backward pass roughly twice that (gradients w.r.t. activations
    and weights), so forward+backward = 3x forward.  Attention's token^2 terms and element-wise ops are ignored (the
    estimate compares arms under equal budgets; it is not absolute throughput).  Use backward=False for the EMA target
    encoder.
    `add` accumulates into the current step and the run totals; `reset_step` closes a step; `totals` is cumulative.
    """

    def __init__(self):
        self._step: Dict[str, list] = {}
        self._total: Dict[str, list] = {}
        self._params: Dict[str, int] = {}
        self.steps = 0

    @staticmethod
    def estimate(tokens: int, params: int, backward: bool = True) -> int:
        return 2 * int(params) * int(tokens) * (3 if backward else 1)

    def add(self, name: str, tokens: int, params: int, backward: bool = True) -> int:
        flops = self.estimate(tokens, params, backward)
        for acc in (self._step, self._total):
            entry = acc.setdefault(name, [0, 0])
            entry[0] += int(tokens)
            entry[1] += flops
        return flops

    def add_module(self, name: str, module: nn.Module, tokens: int, backward: bool = True) -> int:
        """Like `add`, with params counted from `module` once (cached by name)."""
        if name not in self._params:
            self._params[name] = param_count(module)
        return self.add(name, tokens, self._params[name], backward)

    @staticmethod
    def _summarise(acc: Dict[str, list]) -> dict:
        return {
            "tokens": sum(v[0] for v in acc.values()),
            "flops": sum(v[1] for v in acc.values()),
            "by_name": {k: {"tokens": v[0], "flops": v[1]} for k, v in acc.items()},
        }

    def step_totals(self) -> dict:
        return self._summarise(self._step)

    def reset_step(self) -> None:
        self._step = {}
        self.steps += 1

    def totals(self) -> dict:
        out = self._summarise(self._total)
        out["steps"] = self.steps
        return out


__all__ = [
    "DT_SCALE", "FourierEmbed", "MotionSequenceEncoder", "MotionConditionedPredictor", "MotionReadout",
    "readout_loss", "readout_loss_terms", "expand_target_scales", "consistency_loss", "embedding_spread",
    "online_gradient_ratio", "shuffle_within_speed_strata", "FlopMeter",
    "param_count", "gather_slot_tokens", "rays_for_tokens",
]
