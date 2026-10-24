"""Validation loss (the agent's own l1_length_normalized_loss) of perception-free checkpoints on the compact val cache
(official devkit val logs), to calibrate the E1 learning curve against the official weights.
usage: python val_loss_cache.py <ckpt> [<ckpt> ...]   (env: drivejepa_env.sh; PYTHONPATH incl. frontdrive)"""
import os, sys, time, json
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, "/home/henry/Desktop/Qwen_Drive/frontdrive")
from navsim_ext.compact_cache import CompactCacheOnlyDataset
from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import DriveJEPAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

ROOT = os.environ["NAVSIM_DEVKIT_ROOT"]
E50 = os.environ["NAVSIM_EXP_ROOT"] + "/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
CACHE = os.environ["NAVSIM_LOCAL"] + "/cache/navtrain_compact"
split = yaml.safe_load(open(ROOT + "/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"))
val_logs = list(split["val_logs"])
res = {}
for ckpt in sys.argv[1:]:
    t0 = time.time()
    agent = DriveJEPAAgent(trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5), pretrain_pt_path=E50,
                           image_architecture="vit_large", tf_d_model=256, tf_d_ffn=1024, tf_num_layers=3, tf_num_head=8,
                           tf_dropout=0.0, num_keyval=129, lr=1e-4, checkpoint_path=ckpt, front_only=True,
                           freeze_encoder=False, double_image=True)
    agent.initialize()
    agent = agent.cuda().eval()
    ds = CompactCacheOnlyDataset(cache_path=CACHE, feature_builders=agent.get_feature_builders(),
                                 target_builders=agent.get_target_builders(), log_names=val_logs)
    dl = DataLoader(ds, batch_size=16, num_workers=6, shuffle=False)
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for feats, targets in dl:
            feats = {k: v.cuda(non_blocking=True) for k, v in feats.items()}
            targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}
            pred = agent.forward(feats)
            loss = agent.compute_loss(feats, targets, pred)
            b = next(iter(targets.values())).shape[0]
            tot += float(loss) * b; n += b
    res[ckpt] = {"val_loss": tot / n, "n": n, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "sec": time.time() - t0}
    print(json.dumps({os.path.basename(ckpt): res[ckpt]}), flush=True)
    del agent; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print("ALL", json.dumps(res, indent=1))
