"""Training-free proposal-selection rules for ONE Drive-JEPA perception-based checkpoint (plan v4.3 ④(c), 2026-10-23).

The model scores its N trajectory proposals with two heads (`pred_logit`, `pred_logit2`; the last column of each is the
predicted final PDM score) and emits the proposal with the highest MEAN sigmoid. Every run we trained sits at EP +1 /
NC −0.4 / TTC −1 relative to the released weights (more progress, less safety), and the sub-score heads are untrained
(`sub_score_weight = 0` upstream), so the only training-free levers are the two final-score heads and the proposal geometry:
  "argmax"        mean of the two heads, argmax                           (= the model itself; sanity check)
  "minheads"      min of the two heads instead of the mean                (both heads must agree that a proposal is good)
  "conservative"  among proposals within `eps` of the best mean score, the SMALLEST progress (xy displacement of the last pose)
  "progressive"   the same candidate set, the LARGEST progress            (control: the opposite tilt)
One trajectory per scene, no oracle, no training; `eps` is a single scalar tuned on dev. Zero changes to the vendored tree.
"""
from typing import Dict, Optional, Tuple

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

MODES = ("argmax", "minheads", "conservative", "progressive")


def head_scores(logit1: torch.Tensor, logit2: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B, N, K] logits (last column = final score) -> (mean score [B, N], min score [B, N]) over the available heads."""
    s1 = torch.sigmoid(logit1[..., -1])
    if logit2 is None:
        return s1, s1
    s2 = torch.sigmoid(logit2[..., -1])
    return (s1 + s2) / 2, torch.minimum(s1, s2)


def select_proposal(proposals: torch.Tensor, logit1: torch.Tensor, logit2: Optional[torch.Tensor], mode: str, eps: float = 0.02):
    """proposals [B, N, T, 3] (x, y, heading), logits [B, N, K] -> (trajectory [B, T, 3], index [B], n_candidates [B])."""
    if mode not in MODES:
        raise ValueError(f"unknown selection mode {mode!r}; choose one of {MODES}")
    mean_s, min_s = head_scores(logit1, logit2)
    B, N = mean_s.shape
    ar = torch.arange(B, device=proposals.device)
    if mode == "argmax":
        idx = mean_s.argmax(dim=1)
        return proposals[ar, idx], idx, torch.ones(B, dtype=torch.long, device=proposals.device)
    if mode == "minheads":
        idx = min_s.argmax(dim=1)
        return proposals[ar, idx], idx, torch.ones(B, dtype=torch.long, device=proposals.device)
    best = mean_s.amax(dim=1, keepdim=True)
    cand = mean_s >= best - float(eps)                                   # [B, N] near-best proposals (always contains the argmax)
    progress = proposals[..., -1, :2].norm(dim=-1)                       # [B, N] displacement of the final pose (metres)
    if mode == "conservative":
        key = torch.where(cand, progress, torch.full_like(progress, float("inf")))
        idx = key.argmin(dim=1)
    else:  # progressive
        key = torch.where(cand, progress, torch.full_like(progress, float("-inf")))
        idx = key.argmax(dim=1)
    return proposals[ar, idx], idx, cand.sum(dim=1)


class DriveJEPASelectAgent(AbstractAgent):
    """A plain DriveJEPAAgent whose trajectory is chosen by `select_proposal` instead of the model's own argmax."""

    def __init__(self, config: DriveJEPAConfig, checkpoint_path: str, mode: str = "argmax", eps: float = 0.02, lr: float = 1e-4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown selection mode {mode!r}")
        self._config, self._mode, self._eps = config, mode, float(eps)
        self.checkpoint_path = checkpoint_path
        self.member = DriveJEPAAgent(config, lr=lr, checkpoint_path=checkpoint_path)

    def name(self) -> str:
        return f"DriveJEPASelectAgent[{self._mode},eps={self._eps}]"

    def initialize(self) -> None:
        self.member.initialize()

    def get_sensor_config(self):
        return self.member.get_sensor_config()

    def get_feature_builders(self):
        return self.member.get_feature_builders()

    def get_target_builders(self):
        return self.member.get_target_builders()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.member.forward(features)
        for k in ("proposals", "pred_logit"):
            if k not in out:
                raise KeyError(f"model output lacks {k!r}; DriveJEPASelectAgent needs the raw proposals and score logits")
        trajectory, idx, ncand = select_proposal(out["proposals"], out["pred_logit"], out.get("pred_logit2"), self._mode, self._eps)
        return {"trajectory": trajectory, "pdm_score": out["pdm_score"], "select_index": idx, "select_candidates": ncand,
                "model_trajectory": out["trajectory"]}
