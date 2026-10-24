"""Unit tests for the inference-time ensemble (navsim_ext.pb_ensemble_agent): the selection rule and the agent's forward with
mocked members (the full ViT-L members are exercised by the dev evaluation on the server)."""
import math

import pytest
import torch

from navsim_ext.pb_ensemble_agent import DriveJEPAEnsembleAgent, select_trajectory


def _trajs():
    # B=2, M=2, T=3 poses (x, y, heading)
    a = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                      [[1.0, 0.1, 0.1], [2.0, 0.2, 0.1], [3.0, 0.3, 0.1]]])          # scene 0: members 0.3 m apart at most
    b = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                      [[1.0, 2.0, 0.5], [2.0, 4.0, 0.5], [3.0, 6.0, 0.5]]])          # scene 1: members disagree by metres
    return torch.stack([a, b])  # [2, 2, 3, 3]


def test_argmax_conf_picks_highest_confidence():
    trajs = _trajs()
    conf = torch.tensor([[0.90, 0.95], [0.80, 0.70]])
    out, choice, close = select_trajectory(trajs, conf, "argmax_conf")
    assert choice.tolist() == [1, 0]
    assert torch.equal(out[0], trajs[0, 1]) and torch.equal(out[1], trajs[1, 0])
    assert not close.any()


def test_mean_if_close_averages_only_close_scenes():
    trajs = _trajs()
    conf = torch.tensor([[0.90, 0.95], [0.80, 0.70]])
    out, choice, close = select_trajectory(trajs, conf, "mean_if_close", close_m=0.5)
    assert close.tolist() == [True, False]
    assert choice.tolist() == [-1, 0]
    assert torch.allclose(out[0, :, :2], trajs[0, :, :, :2].mean(dim=0))
    assert out[0, 0, 2] == pytest.approx(0.05, abs=1e-6)                 # circular mean of 0 and 0.1
    assert torch.equal(out[1], trajs[1, 0])                                # fell back to argmax_conf


def test_circular_mean_wraps():
    t = torch.tensor([[[[0.0, 0.0, math.pi - 0.1]], [[0.0, 0.0, -math.pi + 0.1]]]])  # headings near +-pi
    out, _, close = select_trajectory(t, torch.tensor([[0.5, 0.5]]), "mean_if_close", close_m=1.0)
    assert close.item() and abs(abs(out[0, 0, 2].item()) - math.pi) < 1e-5


def test_first_and_errors():
    trajs = _trajs(); conf = torch.tensor([[0.1, 0.9], [0.1, 0.9]])
    out, choice, _ = select_trajectory(trajs, conf, "first")
    assert choice.tolist() == [0, 0] and torch.equal(out, trajs[:, 0])
    with pytest.raises(ValueError):
        select_trajectory(trajs, conf, "bogus")


def test_agent_forward_with_mocked_members(monkeypatch):
    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
    # avoid building two ViT-L models: replace the member class by a tiny stub with the same interface
    import navsim_ext.pb_ensemble_agent as mod

    class Stub(torch.nn.Module):
        def __init__(self, config, lr, checkpoint_path):
            super().__init__(); self.p = checkpoint_path; self.k = int(checkpoint_path[-1])
        def initialize(self): self.init_called = True
        def get_sensor_config(self): return "sensor"
        def get_feature_builders(self): return ["fb"]
        def get_target_builders(self): return ["tb"]
        def forward(self, features):
            assert "lidar2img" in features
            features["lidar2img"] = features["lidar2img"][:, 1:2]       # mimic the in-place reassignment of the real model
            B = features["ego_status"].shape[0]
            traj = torch.full((B, 8, 3), float(self.k))
            score = torch.full((B, 32), 0.5); score[:, 0] = 0.6 + 0.1 * self.k
            return {"trajectory": traj, "pdm_score": score}
    monkeypatch.setattr(mod, "DriveJEPAAgent", Stub)
    agent = DriveJEPAEnsembleAgent(DriveJEPAConfig(), ["m1", "m2"], mode="argmax_conf")
    agent.initialize(); assert all(getattr(m, "init_called", False) for m in agent.members)
    feats = {"lidar2img": torch.zeros(2, 2, 4, 4), "ego_status": torch.zeros(2, 4, 11)}
    out = agent.forward(feats)
    assert out["trajectory"].shape == (2, 8, 3) and out["ensemble_choice"].tolist() == [1, 1]   # member 2 has the higher confidence
    assert torch.all(out["trajectory"] == 2.0)
    assert feats["lidar2img"].shape == (2, 2, 4, 4)                                           # caller's dict untouched (shallow copies)
    assert agent.get_feature_builders() == ["fb"] and agent.get_sensor_config() == "sensor"
