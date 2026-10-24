"""Unit tests for the training-free selection rules (navsim_ext.pb_select_agent); the real model is exercised by the dev smoke."""
import math

import pytest
import torch

from navsim_ext.pb_select_agent import DriveJEPASelectAgent, head_scores, select_proposal


def _props():
    # B=1, N=4 proposals, T=2 poses: final-pose displacements 1, 2, 3, 4 metres
    p = torch.zeros(1, 4, 2, 3)
    for n in range(4):
        p[0, n, -1, 0] = float(n + 1)
    return p


def _logits(final):  # [N] final-score logits -> [1, N, 3] with two junk sub-score columns
    l = torch.zeros(1, len(final), 3)
    l[0, :, -1] = torch.tensor(final)
    return l


def test_argmax_matches_model_rule():
    # heads: A agree (1.5/1.5 -> mean .818, min .818); B disagree (4.0/0.8 -> mean .836, min .690); C (2.0/1.0 -> .806/.731); D low
    p = _props(); l1 = _logits([1.5, 4.0, 2.0, -1.0]); l2 = _logits([1.5, 0.8, 1.0, -1.0])
    mean_s, min_s = head_scores(l1, l2)
    assert mean_s.argmax().item() == 1 and min_s.argmax().item() == 0
    traj, idx, n = select_proposal(p, l1, l2, "argmax")
    assert idx.tolist() == [1] and torch.equal(traj[0], p[0, 1]) and n.tolist() == [1]


def test_minheads_prefers_agreement():
    p = _props(); l1 = _logits([1.5, 4.0, 2.0, -1.0]); l2 = _logits([1.5, 0.8, 1.0, -1.0])
    _, idx, _ = select_proposal(p, l1, l2, "minheads")
    assert idx.tolist() == [0]            # proposal A: the heads agree (worst-head score .818 beats B's .690)


def test_conservative_and_progressive_within_eps():
    p = _props()
    # mean scores (sigmoid): proposals 1,2,3 near the best (logits 2.0, 1.9, 1.95), proposal 0 far below
    l = _logits([-2.0, 2.0, 1.9, 1.95])
    traj, idx, n = select_proposal(p, l, None, "conservative", eps=0.05)
    assert idx.tolist() == [1] and n.tolist() == [3]           # smallest displacement among {1,2,3} is proposal 1 (2 m)
    traj, idx, n = select_proposal(p, l, None, "progressive", eps=0.05)
    assert idx.tolist() == [3] and torch.equal(traj[0], p[0, 3])
    # eps = 0 -> only the argmax is a candidate -> identical to argmax
    _, idx0, n0 = select_proposal(p, l, None, "conservative", eps=0.0)
    assert idx0.tolist() == [1] and n0.tolist() == [1]


def test_bad_mode():
    with pytest.raises(ValueError):
        select_proposal(_props(), _logits([0.0, 1.0, 0.0, 0.0]), None, "bogus")


def test_agent_forward_with_stub(monkeypatch):
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    import navsim_ext.pb_select_agent as mod

    class Stub(torch.nn.Module):
        def __init__(self, config, lr, checkpoint_path):
            super().__init__(); self.p = checkpoint_path
        def initialize(self): self.init_called = True
        def get_sensor_config(self): return "sensor"
        def get_feature_builders(self): return ["fb"]
        def get_target_builders(self): return ["tb"]
        def forward(self, features):
            props = _props().repeat(2, 1, 1, 1)
            l1 = _logits([-2.0, 2.0, 1.9, 1.95]).repeat(2, 1, 1); l2 = l1.clone()
            score = torch.sigmoid(l1[..., -1])
            return {"proposals": props, "pred_logit": l1, "pred_logit2": l2, "pdm_score": score,
                    "trajectory": props[torch.arange(2), score.argmax(1)]}
    monkeypatch.setattr(mod, "DriveJEPAAgent", Stub)
    agent = DriveJEPASelectAgent(DriveJEPAConfig(), "ckpt", mode="conservative", eps=0.05)
    agent.initialize(); assert agent.member.init_called
    out = agent.forward({"ego_status": torch.zeros(2, 4, 11)})
    assert out["trajectory"].shape == (2, 2, 3) and out["select_index"].tolist() == [1, 1] and out["select_candidates"].tolist() == [3, 3]
    assert torch.equal(out["model_trajectory"], out["trajectory"])      # here the argmax proposal is also the most conservative candidate
    assert agent.name().startswith("DriveJEPASelectAgent[conservative")
