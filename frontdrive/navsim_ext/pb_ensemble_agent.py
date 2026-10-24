"""Inference-time ensemble of Drive-JEPA perception-based agents (plan v4.3 ④(c), 2026-10-21).

Each member is a complete `DriveJEPAAgent` (own checkpoint). For one scene every member produces its own trajectory and the
predicted PDM score of that trajectory (max of its per-proposal `pdm_score`, a sigmoid in [0, 1]). The ensemble emits ONE
trajectory (leaderboard protocol unchanged):
  mode "argmax_conf":  the member whose predicted score for its own trajectory is highest wins;
  mode "mean_if_close": when all member trajectories lie within `close_m` metres of each other at every pose, their xy are
                        averaged and the heading is the circular mean; otherwise falls back to argmax_conf;
  mode "first":        member 0 only (sanity check: must reproduce the single-model evaluation).
No training, no dev-set selection. Oracle-of-two on dev is ≈ 95.4 vs ≈ 93.1–93.2 for the singles (per-scene CSVs of
2026-10-21), so there is headroom for a good selection rule; whether predicted confidence captures any of it is what the
dev evaluation measures. Zero changes to the vendored tree.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

MODES = ("argmax_conf", "mean_if_close", "first")


def select_trajectory(trajs: torch.Tensor, conf: torch.Tensor, mode: str, close_m: float = 0.5):
    """trajs [B, M, T, 3] (x, y, heading), conf [B, M] -> (trajectory [B, T, 3], choice [B] (-1 = averaged), close_mask [B])."""
    if mode not in MODES:
        raise ValueError(f"unknown ensemble mode {mode!r}; choose one of {MODES}")
    B, M = conf.shape
    ar = torch.arange(B, device=trajs.device)
    if mode == "first":
        return trajs[:, 0], torch.zeros(B, dtype=torch.long, device=trajs.device), torch.zeros(B, dtype=torch.bool, device=trajs.device)
    choice = conf.argmax(dim=1)
    picked = trajs[ar, choice]
    if mode == "argmax_conf":
        return picked, choice, torch.zeros(B, dtype=torch.bool, device=trajs.device)
    # mean_if_close: pairwise max displacement (xy, over poses) between members
    xy = trajs[..., :2]
    disp = (xy[:, :, None] - xy[:, None, :]).norm(dim=-1)          # [B, M, M, T]
    close = disp.amax(dim=(1, 2, 3)) <= close_m                        # [B]
    mean_xy = xy.mean(dim=1)                                           # [B, T, 2]
    hd = trajs[..., 2]
    mean_hd = torch.atan2(torch.sin(hd).mean(dim=1), torch.cos(hd).mean(dim=1))  # circular mean [B, T]
    averaged = torch.cat([mean_xy, mean_hd[..., None]], dim=-1)
    out = torch.where(close[:, None, None], averaged, picked)
    choice = torch.where(close, torch.full_like(choice, -1), choice)
    return out, choice, close


class DriveJEPAEnsembleAgent(AbstractAgent):
    def __init__(self, config: DriveJEPAConfig, checkpoint_paths: List[str], mode: str = "argmax_conf", close_m: float = 0.5,
                 lr: float = 1e-4):
        super().__init__()
        if len(checkpoint_paths) < 2:
            raise ValueError("an ensemble needs >= 2 checkpoint_paths")
        if mode not in MODES:
            raise ValueError(f"unknown ensemble mode {mode!r}")
        self._config, self._mode, self._close_m = config, mode, float(close_m)
        self.checkpoint_paths = list(checkpoint_paths)
        self.members = nn.ModuleList([DriveJEPAAgent(config, lr=lr, checkpoint_path=p) for p in checkpoint_paths])

    def name(self) -> str:
        return f"DriveJEPAEnsembleAgent[{self._mode},{len(self.members)}]"

    def initialize(self) -> None:
        for m in self.members:
            m.initialize()

    def get_sensor_config(self):
        return self.members[0].get_sensor_config()

    def get_feature_builders(self):
        return self.members[0].get_feature_builders()

    def get_target_builders(self):
        return self.members[0].get_target_builders()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # DriveJEPAModel.forward reassigns features['lidar2img'] -> every member gets its own dict with cloned tensors
        outs = [m.forward({k: (v.clone() if torch.is_tensor(v) else v) for k, v in features.items()}) for m in self.members]
        trajs = torch.stack([o["trajectory"] for o in outs], dim=1)                      # [B, M, T, 3]
        conf = torch.stack([o["pdm_score"].max(dim=1).values for o in outs], dim=1)       # [B, M]
        trajectory, choice, close = select_trajectory(trajs, conf, self._mode, self._close_m)
        return {"trajectory": trajectory, "pdm_score": conf, "ensemble_choice": choice, "ensemble_close": close,
                "member_trajectories": trajs}
