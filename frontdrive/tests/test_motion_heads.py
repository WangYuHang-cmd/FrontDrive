"""Unit tests for pretrain/motion_heads.py (CPU only, deterministic, single-threaded).

Run:  source env/drivejepa_env.sh && python -m pytest tests/test_motion_heads.py -q
The two SYNTHETIC LEARNING TESTS train tiny heads (dim 32, 16 spatial tokens) for <= 300 Adam steps on data where the
slot-(s+d) tokens are a known linear function of the slot-s tokens and the motion.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # make `pretrain.*` importable without PYTHONPATH
    sys.path.insert(0, str(_ROOT))

from pretrain.motion_heads import (  # noqa: E402
    DT_SCALE,
    FlopMeter,
    FourierEmbed,
    MotionConditionedPredictor,
    MotionReadout,
    MotionSequenceEncoder,
    consistency_loss,
    embedding_spread,
    expand_target_scales,
    gather_slot_tokens,
    online_gradient_ratio,
    param_count,
    rays_for_tokens,
    readout_loss,
    readout_loss_terms,
    shuffle_within_speed_strata,
)

ENC, DIM, NSP, NT, NFEAT = 32, 32, 16, 5, 7  # tiny sizes for the CPU tests


@pytest.fixture(autouse=True)
def _deterministic_single_thread():
    """Tiny tensors run ~100x faster on one thread on a loaded box; also fixes the global seed per test."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(n)


def _unit_rays(*shape, g=None):
    return F.normalize(torch.randn(*shape, 3, generator=g), dim=-1)


def _batch(B=3, Nc=10, Nt=NSP, L=7, g=None):
    """Random head inputs (spatial indices, rays, motion sequence, dt, slot pair)."""
    g = g or torch.Generator().manual_seed(1)
    seq_mask = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        seq_mask[b, : 3 + 2 * (b % 2)] = True  # 3 or 5 valid rows
    return dict(
        z_ctx=torch.randn(B, Nc, ENC, generator=g),
        ctx_idx=torch.stack([torch.randperm(NSP, generator=g)[:Nc] for _ in range(B)]),
        rays_ctx=_unit_rays(B, Nc, g=g),
        seq=torch.randn(B, L, NFEAT, generator=g),
        seq_mask=seq_mask,
        dt=torch.tensor([1.0, 2.05, 0.98][:B]),
        tgt_idx=torch.arange(Nt).expand(B, -1).clone(),
        rays_tgt=_unit_rays(B, Nt, g=g),
        s=torch.tensor([0, 1, 2][:B]),
        d=torch.tensor([1, 2, 1][:B]),
    )


# ----------------------------------------------------------------------------------------------------------- embeddings
def test_fourier_embed_shapes_and_sensitivity():
    emb = FourierEmbed(3, 4, DIM)
    rays = _unit_rays(2, 7)
    out = emb(rays)
    assert out.shape == (2, 7, DIM) and torch.isfinite(out).all()
    assert FourierEmbed(1, 4, DIM)(torch.tensor([[0.25], [0.5]])).shape == (2, DIM)  # dt-style (B, 1) input
    assert torch.equal(emb(rays), out)  # deterministic
    assert not torch.allclose(emb(rays), emb(-rays))  # distinct inputs -> distinct embeddings


def test_sequence_encoder_masks_rows_and_tolerates_nan():
    enc = MotionSequenceEncoder(NFEAT, DIM, max_len=7)
    seq = torch.randn(2, 7, NFEAT)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    seq[:, 4:] = float("nan")  # padded rows may hold garbage
    seq[0, 1, 2] = float("nan")  # a NaN inside a valid row counts as 0
    out = enc(seq, mask, torch.tensor([1.0, 2.0]))
    assert out.shape == (2, 7, DIM) and torch.isfinite(out).all()
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))
    assert (out[mask].abs().sum(-1) > 0).all()
    assert not torch.allclose(out[0, :3], enc(seq, mask, torch.tensor([3.0, 2.0]))[0, :3])  # dt changes the tokens
    assert enc.feat_scale.tolist() == [30.0, 0.5, 5.0, 20.0, 2.0, 0.5, 1.0]  # canonical 7-feature scales


def test_sequence_encoder_per_feature_validity():
    """A missing feature (e.g. no gyro in the sidecar) only blanks its column: garbage in invalid entries is ignored,
    and "unknown" is distinguishable from a genuine 0 through the validity channels."""
    enc = MotionSequenceEncoder(NFEAT, DIM, max_len=7)
    seq = torch.randn(2, 7, NFEAT)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    valid = torch.ones(2, 7, NFEAT, dtype=torch.bool)
    valid[:, :, 1] = False  # yaw_rate column unknown
    dt = torch.tensor([1.0, 2.0])
    out = enc(seq, mask, dt, valid)
    garbage = seq.clone()
    garbage[:, :, 1] = 123.0
    assert torch.allclose(enc(garbage, mask, dt, valid), out, atol=1e-6)
    zeros = seq.clone()
    zeros[:, :, 1] = 0.0
    assert not torch.allclose(enc(zeros, mask, dt, None), out)  # known 0 != unknown
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))
    with pytest.raises(AssertionError):
        enc(torch.randn(2, 8, NFEAT), torch.ones(2, 8, dtype=torch.bool), dt)  # longer than max_len is an error
    assert MotionSequenceEncoder(NFEAT, DIM, max_len=15)(torch.randn(2, 15, NFEAT), torch.ones(2, 15, dtype=torch.bool), dt).shape == (2, 15, DIM)
    # P_phi forwards seq_valid to its sequence encoder
    b = _batch()
    B = b["z_ctx"].shape[0]
    P = MotionConditionedPredictor(ENC, DIM, depth=1, heads=4, n_spatial=NSP, n_slots=4, n_feat=NFEAT)
    P.eval()
    sv = torch.ones(B, 7, NFEAT, dtype=torch.bool)
    sv[:, :, 0] = False
    out_a = P(**b, seq_valid=sv)
    b2 = dict(b)
    b2["seq"] = b["seq"].clone()
    b2["seq"][:, :, 0] += 50.0  # invalid column: ignored
    assert torch.allclose(P(**b2, seq_valid=sv), out_a, atol=1e-5) and not torch.allclose(P(**b2), out_a)


# -------------------------------------------------------------------------------------------------------------- P_phi
def test_predictor_shapes_padding_and_cond_flag():
    b = _batch()
    B, Nc = b["z_ctx"].shape[:2]
    P = MotionConditionedPredictor(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_feat=NFEAT)
    P.eval()
    keep = torch.ones(B, Nc, dtype=torch.bool)
    keep[1, 6:] = False
    keep[2, :] = False  # a sample with no visible context tokens at all
    out = P(**b, ctx_keep=keep)
    assert out.shape == (B, NSP, ENC) and torch.isfinite(out).all()
    # padded context rows and padded motion rows must not influence the prediction
    b2 = dict(b)
    b2["z_ctx"] = torch.where(keep.unsqueeze(-1), b["z_ctx"], 100.0 * torch.randn_like(b["z_ctx"]))
    b2["seq"] = torch.where(b["seq_mask"].unsqueeze(-1), b["seq"], 100.0 * torch.randn_like(b["seq"]))
    assert torch.allclose(P(**b2, ctx_keep=keep), out, atol=1e-5)
    # ctx_keep=None means everything visible; a different motion sequence changes the output
    assert P(**b).shape == (B, NSP, ENC)
    b3 = dict(b)
    b3["seq"] = b["seq"] + 1.0
    assert not torch.allclose(P(**b3, ctx_keep=keep), out)
    # empty context (Nc = 0) still works
    b4 = dict(b, z_ctx=b["z_ctx"][:, :0], ctx_idx=b["ctx_idx"][:, :0], rays_ctx=b["rays_ctx"][:, :0])
    assert P(**b4).shape == (B, NSP, ENC)
    # cond_motion=False: plain future-prediction head that ignores the motion sequence
    P0 = MotionConditionedPredictor(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_feat=NFEAT,
                                    cond_motion=False)
    P0.eval()
    assert P0.motion_enc is None
    assert torch.allclose(P0(**b), P0(**b3), atol=1e-6)
    with pytest.raises(AssertionError):
        P(**dict(b, d=torch.tensor([4, 1, 1])))  # s + d >= n_slots
    with pytest.raises(AssertionError):
        P(**dict(b, d=torch.tensor([0, 1, 1])))  # d must be >= 1


def test_predictor_gradient_reaches_visible_context_only():
    b = _batch()
    B, Nc = b["z_ctx"].shape[:2]
    P = MotionConditionedPredictor(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_feat=NFEAT)
    z = b["z_ctx"].clone().requires_grad_(True)
    keep = torch.ones(B, Nc, dtype=torch.bool)
    keep[0, 5:] = False
    P(**dict(b, z_ctx=z), ctx_keep=keep).pow(2).mean().backward()
    assert z.grad is not None
    assert (z.grad[keep].abs().sum(-1) > 0).all()
    assert torch.equal(z.grad[~keep], torch.zeros_like(z.grad[~keep]))


# -------------------------------------------------------------------------------------------------------------- R_psi
def test_readout_shapes_stopgrad_and_padding():
    b = _batch()
    B, Nc = b["z_ctx"].shape[:2]
    R = MotionReadout(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_targets=NT)
    R.eval()
    z_ctx = b["z_ctx"].clone().requires_grad_(True)
    z_tgt_src = torch.randn(B, NSP, ENC, requires_grad=True)  # e.g. target-encoder output
    z_tgt = z_tgt_src.detach()
    ctx_keep = torch.ones(B, Nc, dtype=torch.bool)
    ctx_keep[0, 4:] = False
    tgt_keep = torch.ones(B, NSP, dtype=torch.bool)
    tgt_keep[1, 8:] = False
    args = (b["ctx_idx"], b["rays_ctx"], z_tgt, b["tgt_idx"], b["rays_tgt"], b["dt"], b["s"], b["d"])
    m_hat, e = R(z_ctx, *args, ctx_keep=ctx_keep, tgt_keep=tgt_keep)
    assert m_hat.shape == (B, NT) and e.shape == (B, DIM)
    assert torch.isfinite(m_hat).all() and torch.isfinite(e).all()
    # gradient reaches z_ctx (visible rows only) and never the target-encoder tokens
    (m_hat.sum() + e.sum()).backward()
    assert z_ctx.grad is not None and (z_ctx.grad[ctx_keep].abs().sum(-1) > 0).all()
    assert torch.equal(z_ctx.grad[~ctx_keep], torch.zeros_like(z_ctx.grad[~ctx_keep]))
    assert z_tgt_src.grad is None
    with pytest.raises(AssertionError):
        R(z_ctx, b["ctx_idx"], b["rays_ctx"], z_tgt_src * 1.0, *args[3:])  # graph-connected z_tgt is refused
    # ... but a detached LEAF probe (requires_grad, no grad_fn) is accepted: it is not connected to any encoder
    probe = z_tgt.detach().requires_grad_(True)
    m_probe, _ = R(z_ctx.detach(), b["ctx_idx"], b["rays_ctx"], probe, *args[3:], ctx_keep=ctx_keep, tgt_keep=tgt_keep)
    assert torch.allclose(m_probe, m_hat.detach(), atol=1e-6) and z_tgt_src.grad is None
    # padded rows (either side) do not influence the outputs
    z_ctx2 = torch.where(ctx_keep.unsqueeze(-1), z_ctx.detach(), 100.0 * torch.randn(B, Nc, ENC))
    z_tgt2 = torch.where(tgt_keep.unsqueeze(-1), z_tgt, 100.0 * torch.randn(B, NSP, ENC))
    m2, e2 = R(z_ctx2, b["ctx_idx"], b["rays_ctx"], z_tgt2, *args[3:], ctx_keep=ctx_keep, tgt_keep=tgt_keep)
    assert torch.allclose(m2, m_hat.detach(), atol=1e-5) and torch.allclose(e2, e.detach(), atol=1e-5)
    # ... but the content of visible target tokens does
    m3, _ = R(z_ctx.detach(), b["ctx_idx"], b["rays_ctx"], z_tgt + 1.0, *args[3:], ctx_keep=ctx_keep, tgt_keep=tgt_keep)
    assert not torch.allclose(m3, m_hat.detach())


# ------------------------------------------------------------------------------------------------------------- losses
def test_readout_loss_scaling_validity_and_empty():
    scales = torch.tensor([50.0, 5.0, 1.0, 30.0, 0.5])
    target = torch.tensor([[25.0, -1.0, 0.1, 15.0, 0.05], [float("nan"), 2.5, float("nan"), 3.0, -0.25]])
    valid = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 0, 1, 1]], dtype=torch.bool)
    m_hat = (torch.nan_to_num(target) / scales).clone().requires_grad_(True)
    assert readout_loss(m_hat, target, valid, scales).item() == pytest.approx(0.0)
    off = m_hat.detach().clone()
    off[0, 0] += 0.5  # inside the quadratic zone: 0.5 * 0.5^2 = 0.125
    off[1, 1] -= 3.0  # linear zone: 3 - 0.5 = 2.5
    expected = (0.125 + 2.5) / valid.sum().item()
    loss = readout_loss(off.requires_grad_(True), target, valid, scales)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert torch.isfinite(off.grad).all()  # NaNs in invalid entries never leak
    assert off.grad[1, 0] == 0 and off.grad[1, 2] == 0
    empty = readout_loss(m_hat, target, torch.zeros_like(valid), scales)
    assert empty.item() == 0.0 and empty.requires_grad


def test_expand_target_scales_and_per_target_terms():
    """Per-horizon scales (displacement-like targets grow with the slot gap d), (B, T) scales in readout_loss and the
    per-target Huber terms used for logging the balance between targets."""
    scales = torch.tensor([10.0, 1.0, 0.15, 10.0, 0.1, 1.0, 0.07])
    per_slot = [True, True, True, False, False, True, False]
    d = torch.tensor([1, 2, 3])
    S = expand_target_scales(scales, d, per_slot)
    assert S.shape == (3, 7) and S[0].tolist() == pytest.approx(scales.tolist())
    assert S[1, 0] == 20.0 and S[2, 2] == pytest.approx(0.45) and S[2, 3] == 10.0 and S[2, 5] == 3.0 and S[2, 6] == pytest.approx(0.07)
    m_hat = torch.zeros(3, 7)
    target = S * 0.5  # every normalised target is 0.5 -> Huber 0.125
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[:, 4] = False
    assert readout_loss(m_hat, target, valid, S).item() == pytest.approx(0.125)
    terms = readout_loss_terms(m_hat, target, valid, S)
    assert terms.shape == (7,) and torch.isnan(terms[4]) and torch.allclose(terms[[0, 1, 2, 3, 5, 6]], torch.full((6,), 0.125))
    target[1, 0] = 3.0 * S[1, 0]  # linear zone for one entry: 3 - 0.5 = 2.5
    terms = readout_loss_terms(m_hat, target, valid, S)
    assert terms[0].item() == pytest.approx((0.125 * 2 + 2.5) / 3)
    assert readout_loss(m_hat, target, valid, S).item() == pytest.approx((0.125 * 17 + 2.5) / 18)


def test_embedding_spread_collapse_monitor():
    e = torch.randn(64, DIM)
    spread = embedding_spread(e).item()
    assert 0.5 / DIM**0.5 < spread < 2.0 / DIM**0.5  # i.i.d. unit vectors: ~1/sqrt(dim)
    assert embedding_spread(torch.ones(8, DIM) * 3.0).item() == pytest.approx(0.0, abs=1e-7)  # collapsed
    assert embedding_spread(e[:1]).item() == 0.0 and not embedding_spread(e).requires_grad


def test_consistency_loss_is_cosine_distance():
    e1 = torch.randn(4, DIM)
    assert consistency_loss(e1, e1).item() == pytest.approx(0.0, abs=1e-6)
    assert consistency_loss(e1, 3.0 * e1).item() == pytest.approx(0.0, abs=1e-6)  # scale invariant
    assert consistency_loss(e1, -e1).item() == pytest.approx(4.0, abs=1e-5)
    e2 = torch.randn(4, DIM)
    cos = F.cosine_similarity(e1, e2, dim=-1)
    assert consistency_loss(e1, e2).item() == pytest.approx((2 - 2 * cos).mean().item(), abs=1e-5)
    a, b = e1.clone().requires_grad_(True), e2.clone().requires_grad_(True)
    consistency_loss(a, b).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0  # both sides keep their gradient


# ----------------------------------------------------------------------------------------------------------- controls
def test_shuffle_within_speed_strata_is_a_within_bin_derangement():
    bins = (5.0, 12.0, 20.0)
    speed = torch.tensor([0.0, 3.0, 4.9, 5.0, 11.9, 6.0, 12.0, 19.0, 15.0, 25.0, float("nan")])  # NaN -> 0 -> bin 0
    bin_id = torch.bucketize(torch.nan_to_num(speed), torch.tensor(bins))
    g = torch.Generator().manual_seed(7)
    perm = shuffle_within_speed_strata(speed, bins=bins, generator=g)
    assert perm.dtype == torch.long and perm.shape == speed.shape
    assert torch.equal(perm.sort().values, torch.arange(len(speed)))  # a permutation
    assert torch.equal(bin_id[perm], bin_id)  # never crosses a bin
    assert (perm == torch.arange(len(speed))).sum() == 1  # only the singleton bin (speed 25) is a fixed point
    assert perm[9] == 9
    assert torch.equal(perm, shuffle_within_speed_strata(speed, bins=bins, generator=torch.Generator().manual_seed(7)))
    # every bin has one item -> identity; two items -> swapped
    assert torch.equal(shuffle_within_speed_strata(torch.tensor([1.0, 10.0, 30.0])), torch.arange(3))
    assert torch.equal(shuffle_within_speed_strata(torch.tensor([1.0, 2.0])), torch.tensor([1, 0]))


def test_shuffle_strata_respect_slot_gap_and_validity():
    """(speed bin x slot gap) strata: a shuffled item receives a dt consistent with its slot gap; items without CAN
    are neither moved nor used as donors (the caller masks the remaining fixed points)."""
    bins = (5.0, 12.0, 20.0)
    g = torch.Generator().manual_seed(3)
    speed = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 15.0, 16.0, 0.0, 0.0, 2.5])
    d = torch.tensor([1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 2, 1])
    valid = torch.ones(12, dtype=torch.bool)
    valid[9:11] = False  # two items without a sidecar (speed 0 -> bin 0)
    bin_id = torch.bucketize(speed, torch.tensor(bins))
    for _ in range(5):
        perm = shuffle_within_speed_strata(speed, bins=bins, generator=g, strata=d, valid=valid)
        assert torch.equal(perm.sort().values, torch.arange(12))
        assert torch.equal(bin_id[perm], bin_id) and torch.equal(d[perm], d)
        assert perm[9] == 9 and perm[10] == 10  # invalid items stay put ...
        assert valid[perm[valid]].all()  # ... and never donate their (all-invalid) motion to a valid item
        fixed = perm == torch.arange(12)
        # strata among valid items: bin0/d1 {0,1,11}, bin0/d2 {2,3}, bin1/d1 {4,5,6}, bin2/d3 {7,8} -> all derangeable
        assert fixed.tolist() == [False] * 9 + [True, True, False]
    # without strata the slot gap can change (the old behaviour) -> stratifying matters
    perm = shuffle_within_speed_strata(speed, bins=bins, generator=torch.Generator().manual_seed(0), valid=valid)
    assert not torch.equal(d[perm], d)


def test_flop_meter_arithmetic():
    fm = FlopMeter()
    assert FlopMeter.estimate(tokens=10, params=7, backward=False) == 2 * 7 * 10
    assert FlopMeter.estimate(tokens=10, params=7, backward=True) == 3 * 2 * 7 * 10
    fm.add("encoder", tokens=1000, params=300, backward=True)
    fm.add("target_encoder", tokens=1000, params=300, backward=False)
    fm.add("encoder", tokens=500, params=300, backward=True)
    step = fm.step_totals()
    assert step["tokens"] == 2500 and step["flops"] == 3 * 2 * 300 * 1500 + 2 * 300 * 1000
    assert step["by_name"]["target_encoder"] == {"tokens": 1000, "flops": 600000}
    fm.reset_step()
    assert fm.step_totals()["tokens"] == 0 and fm.totals()["tokens"] == 2500 and fm.totals()["steps"] == 1
    fm.add("encoder", tokens=1, params=1)
    assert fm.totals()["flops"] == step["flops"] + 6 and fm.step_totals()["flops"] == 6
    lin = torch.nn.Linear(4, 3)  # 15 params
    assert fm.add_module("lin", lin, tokens=10, backward=False) == 2 * 15 * 10
    assert param_count(lin) == 15


def test_gather_slot_tokens_and_rays():
    B, N, n_spatial = 2, 40, NSP
    g = torch.Generator().manual_seed(3)
    idx = torch.stack([torch.randperm(4 * n_spatial, generator=g)[:N].sort().values for _ in range(B)])
    z = torch.randn(B, N, 6, generator=g, requires_grad=True)
    slot = torch.tensor([0, 3])
    z_slot, spatial, keep = gather_slot_tokens(z, idx, slot, n_spatial)
    counts = [(idx[b] // n_spatial == slot[b]).sum().item() for b in range(B)]
    assert z_slot.shape == (B, max(counts), 6) and keep.sum(1).tolist() == counts
    for b in range(B):
        members = idx[b][idx[b] // n_spatial == slot[b]]
        assert torch.equal(spatial[b, : counts[b]], members % n_spatial)
        assert torch.equal(z_slot[b, : counts[b]], z[b][idx[b] // n_spatial == slot[b]])
    assert torch.equal(z_slot[~keep], torch.zeros_like(z_slot[~keep])) and (spatial[~keep] == 0).all()
    z_slot.sum().backward()
    assert z.grad.abs().sum(-1).gt(0).sum() == sum(counts)  # gradient only to the selected rows
    rays = _unit_rays(B, 2, 8, g=g)  # (B, H_p=2, W_p=8, 3) -> 16 spatial positions
    rt = rays_for_tokens(rays, spatial)
    assert rt.shape == (B, max(counts), 3)
    assert torch.equal(rt[0, 0], rays[0, spatial[0, 0] // 8, spatial[0, 0] % 8])


def test_default_sizes_construct_and_run():
    P = MotionConditionedPredictor()
    R = MotionReadout()
    # 7 motion features (+7 validity channels) in the sequence encoder; 7 read-out targets (motion_data.MOTION_TARGETS)
    assert param_count(P) == 11_669_248 and param_count(MotionConditionedPredictor(cond_motion=False)) == 11_654_656
    assert param_count(R) == 6_481_159
    B, Nc = 1, 64
    z_ctx = torch.randn(B, Nc, 1024)
    ctx_idx = torch.randperm(512)[:Nc].unsqueeze(0)
    tgt_idx = torch.arange(512).unsqueeze(0)
    seq, seq_mask = torch.randn(B, 7, 7), torch.ones(B, 7, dtype=torch.bool)
    dt, s, d = torch.tensor([2.0]), torch.tensor([1]), torch.tensor([2])
    with torch.no_grad():
        out = P(z_ctx, ctx_idx, _unit_rays(B, Nc), seq, seq_mask, dt, tgt_idx, _unit_rays(B, 512), s, d)
        m_hat, e = R(z_ctx, ctx_idx, _unit_rays(B, Nc), torch.randn(B, 512, 1024), tgt_idx, _unit_rays(B, 512),
                     dt, s, d)
    assert out.shape == (B, 512, 1024) and m_hat.shape == (B, 7) and e.shape == (B, 384)
    assert torch.isfinite(out).all() and torch.isfinite(m_hat).all()
    assert DT_SCALE == 4.0


# ------------------------------------------------------------------------------------------- synthetic learning tests
def _synthetic_pair(g, B, Bm, A=None, token_noise=1.0):
    """Slot-s tokens z_s ~ N(0,1) (optionally a shared per-sample vector plus token_noise jitter), motion m ~ N(0,1)^5,
    slot-(s+d) tokens z_t = z_s @ A + m @ Bm (a known linear function of z_s and m)."""
    u = torch.randn(B, 1, ENC, generator=g)
    jitter = torch.randn(B, NSP, ENC, generator=g)
    z_s = u + token_noise * jitter if token_noise < 1.0 else jitter
    m = torch.randn(B, NT, generator=g)
    z_t = (z_s if A is None else z_s @ A) + (m @ Bm).unsqueeze(1)
    idx = torch.arange(NSP).expand(B, -1)
    rays = _unit_rays(B, NSP, g=g)
    seq = torch.zeros(B, 3, NFEAT)
    # the motion sequence carries m in its first 5 features on every row, at physical magnitudes (the encoder divides by
    # the canonical feature scales, so the heads see m itself)
    seq[:, :, :NT] = (m * torch.tensor([30.0, 0.5, 5.0, 20.0, 2.0])).unsqueeze(1)
    seq_mask = torch.ones(B, 3, dtype=torch.bool)
    common = dict(dt=torch.ones(B), s=torch.zeros(B, dtype=torch.long), d=torch.ones(B, dtype=torch.long))
    return z_s, z_t, m, idx, rays, seq, seq_mask, common


def test_readout_learns_motion_and_detects_can_mismatch():
    """R_psi trained on matched (z_s, z_{s+d}) pairs reads the motion out; with the target tokens of another sample in
    the same speed stratum (shuffle_within_speed_strata) the readout error must be much larger (plan section 9.7)."""
    g = torch.Generator().manual_seed(0)
    A = torch.linalg.qr(torch.randn(ENC, ENC, generator=g))[0]  # a rotation, so z_t is not simply z_s + shift
    Bm = F.normalize(torch.randn(NT, ENC, generator=g), dim=-1) * 2.0
    R = MotionReadout(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_targets=NT)
    opt = torch.optim.Adam(R.parameters(), lr=3e-3)
    scales = torch.ones(NT)
    for _ in range(250):
        z_s, z_t, m, idx, rays, _, _, c = _synthetic_pair(g, 32, Bm, A)
        m_hat, _ = R(z_s, idx, rays, z_t, idx, rays, **c)
        loss = readout_loss(m_hat, m, torch.ones_like(m, dtype=torch.bool), scales)
        opt.zero_grad()
        loss.backward()
        opt.step()
    R.eval()
    with torch.no_grad():
        z_s, z_t, m, idx, rays, _, _, c = _synthetic_pair(g, 256, Bm, A)
        valid = torch.ones_like(m, dtype=torch.bool)
        matched = readout_loss(R(z_s, idx, rays, z_t, idx, rays, **c)[0], m, valid, scales).item()
        speed = torch.rand(256, generator=g) * 30.0
        perm = shuffle_within_speed_strata(speed, generator=g)
        shuffled = readout_loss(R(z_s, idx, rays, z_t[perm], idx, rays, **c)[0], m, valid, scales).item()
    assert matched < 0.3 * shuffled, (matched, shuffled)
    assert shuffled > 0.3  # the shuffled pairs carry no information about m (Huber of a unit-normal target ~ 0.6)


def _shortcut_data(g, B, Bt, Bs):
    """Real structure of the readout problem: the target tokens carry the motion cleanly on every position (a whole-clip
    representation that is complete), the online context carries it weakly, under noise, and only half of its tokens
    are visible (JEPA mask)."""
    m = torch.randn(B, NT, generator=g)
    z_t = (m @ Bt).unsqueeze(1) + 0.3 * torch.randn(B, NSP, ENC, generator=g)
    z_s = 0.5 * (m @ Bs).unsqueeze(1) + torch.randn(B, NSP, ENC, generator=g)
    keep = torch.rand(B, NSP, generator=g) < 0.5
    idx = torch.arange(NSP).expand(B, -1)
    rays = _unit_rays(B, NSP, g=g)
    c = dict(dt=torch.ones(B), s=torch.zeros(B, dtype=torch.long), d=torch.ones(B, dtype=torch.long))
    return z_s, z_t, m, idx, rays, keep, c


def _train_readout_with_tgt_drop(p_drop, steps=200, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Bt = F.normalize(torch.randn(NT, ENC, generator=g), dim=-1) * 2.0
    Bs = F.normalize(torch.randn(NT, ENC, generator=g), dim=-1) * 2.0
    R = MotionReadout(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_targets=NT)
    opt = torch.optim.Adam(R.parameters(), lr=3e-3)
    scales = torch.ones(NT)
    ones = lambda t: torch.ones_like(t, dtype=torch.bool)  # noqa: E731
    for _ in range(steps):
        z_s, z_t, m, idx, rays, keep, c = _shortcut_data(g, 32, Bt, Bs)
        drop = torch.rand(32, generator=g) < p_drop  # per-item target-branch dropout (as in the trainer)
        m_hat, _ = R(z_s, idx, rays, z_t, idx, rays, **c, ctx_keep=keep, tgt_keep=(~drop)[:, None].expand(-1, NSP))
        loss = readout_loss(m_hat, m, ones(m), scales)
        opt.zero_grad()
        loss.backward()
        opt.step()
    R.eval()
    z_s, z_t, m, idx, rays, keep, c = _shortcut_data(g, 256, Bt, Bs)
    z_s = z_s.requires_grad_(True)
    probe = z_t.detach().requires_grad_(True)  # detached leaf: accepted by R_psi, not connected to any encoder
    m_hat, _ = R(z_s, idx, rays, probe, idx, rays, **c, ctx_keep=keep)
    loss = readout_loss(m_hat, m, ones(m), scales)
    ratio, g_ctx, g_tgt = online_gradient_ratio(loss, z_s, probe)
    assert g_ctx > 0 and g_tgt > 0 and ratio.item() == pytest.approx(g_ctx.item() / (g_tgt.item() + 1e-12))
    with torch.no_grad():
        no_ctx = readout_loss(R(z_s, idx, rays, z_t, idx, rays, **c, ctx_keep=torch.zeros(256, NSP, dtype=torch.bool))[0], m, ones(m), scales)
        no_tgt = readout_loss(R(z_s, idx, rays, z_t, idx, rays, **c, ctx_keep=keep, tgt_keep=torch.zeros(256, NSP, dtype=torch.bool))[0], m, ones(m), scales)
    return dict(loss=loss.item(), ratio=ratio.item(), no_ctx=no_ctx.item(), no_tgt=no_tgt.item())


def test_readout_shortcut_is_detected_and_target_dropout_restores_online_gradient():
    """Review 2026-10-12: when the target tokens alone explain the motion, R_psi trained on (z_s, z_tgt) minimises
    L_readout from the gradient-free target branch and the online encoder gets little signal.  The gradient-ratio
    diagnostic must expose this (ratio << 1, context-only readout useless), and the trainer's target-branch dropout
    (readout_tgt_drop) must restore a substantial online-side gradient and a working context-only readout."""
    plain = _train_readout_with_tgt_drop(0.0)
    dropped = _train_readout_with_tgt_drop(0.5)
    assert plain["ratio"] < 0.5, plain  # shortcut: the readout signal reaches mostly the target branch
    assert plain["no_tgt"] > 3 * plain["loss"], plain  # without the target branch the trained head is lost
    assert dropped["ratio"] > 0.6 and dropped["ratio"] > 1.5 * plain["ratio"], (plain, dropped)
    assert dropped["no_tgt"] < 0.9 * plain["no_tgt"], (plain, dropped)  # the online path now carries motion
    # no graph -> zeros instead of an exception (e.g. nothing valid in a batch)
    z = torch.zeros((), requires_grad=False)
    assert all(v.item() == 0.0 for v in online_gradient_ratio(z, torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)))


def test_predictor_conditioning_on_motion_lowers_loss():
    """P_phi with the motion sequence predicts z_{s+d} = z_s + m @ Bm better than the unconditioned future-prediction
    head, which can only regress to the mean over m."""
    results = {}
    for cond in (True, False):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        Bm = F.normalize(torch.randn(NT, ENC, generator=g), dim=-1) * 3.0
        # init_std 0.1: at dim 32 the vendored 0.02 init leaves the context path ~100x weaker than the query embeddings
        P = MotionConditionedPredictor(ENC, DIM, depth=2, heads=4, n_spatial=NSP, n_slots=4, n_feat=NFEAT,
                                       cond_motion=cond, init_std=0.1)
        opt = torch.optim.Adam(P.parameters(), lr=3e-3)
        for _ in range(300):
            z_s, z_t, _, idx, rays, seq, seq_mask, c = _synthetic_pair(g, 32, Bm, token_noise=0.3)
            keep = torch.rand(32, NSP, generator=g) < 0.5  # half of the context tokens visible, as in JEPA
            out = P(z_s, idx, rays, seq, seq_mask, c["dt"], idx, rays, c["s"], c["d"], keep)
            loss = F.l1_loss(out, z_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        P.eval()
        with torch.no_grad():
            z_s, z_t, _, idx, rays, seq, seq_mask, c = _synthetic_pair(g, 256, Bm, token_noise=0.3)
            keep = torch.rand(256, NSP, generator=g) < 0.5
            out = P(z_s, idx, rays, seq, seq_mask, c["dt"], idx, rays, c["s"], c["d"], keep)
            results[cond] = F.l1_loss(out, z_t).item()
    assert results[True] < 0.7 * results[False], results
