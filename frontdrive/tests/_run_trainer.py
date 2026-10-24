"""Subprocess driver for tests/test_train_motion.py: run one trainer (vendored app.vjepa.train or the frontdrive fork
pretrain.train_motion) on a yaml config inside a fresh process with a world-size-1 process group.

    python tests/_run_trainer.py --trainer vendored|motion --config <yaml> [--port N]

A fresh process per run keeps the module-level RNG seeding (random / numpy / torch at import) and the CUDA / NCCL state
identical between the two trainers, which the step-for-step equivalence test relies on."""
import argparse
import os
import socket
import sys

import torch
import torch.distributed as dist
import yaml


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", choices=["vendored", "motion"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    with open(a.config) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs(params["folder"], exist_ok=True)
    use_cuda = torch.cuda.is_available() and not str(params.get("meta", {}).get("device", "cuda:0")).startswith("cpu")
    if use_cuda:
        torch.cuda.set_device(0)
    # like app/main.py: a single-process group so the DistributedDataParallel wrappers in the trainers work
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method=f"tcp://127.0.0.1:{a.port or _free_port()}",
        rank=0,
        world_size=1,
    )
    if a.trainer == "vendored":
        from app.vjepa.train import main as trainer_main  # hardcodes cuda:0 when CUDA is available
    else:
        from pretrain.train_motion import main as trainer_main
    trainer_main(args=params)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
