# source this before running the perception-based Drive-JEPA code (venv-pb: upstream pins, mmcv).
# Same paths as drivejepa_env.sh; only the interpreter differs. Valid on the server through the
# /home/henry/Desktop/Qwen_Drive symlink and on the local machine.
export FRONTDRIVE=/home/henry/Desktop/Qwen_Drive/frontdrive
export DRIVEJEPA_ROOT=$FRONTDRIVE/third_party/Drive-JEPA/navsim_v1
export NAVSIM_DEVKIT_ROOT=$DRIVEJEPA_ROOT
export NAVSIM_LOCAL=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim
export OPENSCENE_DATA_ROOT=$NAVSIM_LOCAL/dataset
export NUPLAN_MAPS_ROOT=$NAVSIM_LOCAL/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=$NAVSIM_LOCAL/exp
export PYTHONPATH=$DRIVEJEPA_ROOT:$DRIVEJEPA_ROOT/vjepa2:${PYTHONPATH:-}
if [ -f "$FRONTDRIVE/env/.use_conda_cvpr" ]; then
  source "$FRONTDRIVE/env/conda_cvpr_activate.sh"      # server: conda env "cvpr" runs all code
else
  export PATH=$FRONTDRIVE/env/drivejepa_pb/bin:$PATH
fi
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}   # torch 2.1 = first release with expandable_segments; export PYTORCH_CUDA_ALLOC_CONF= to disable
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OMEGACONF_MAX_YAML_EXPANDED_NODES=${OMEGACONF_MAX_YAML_EXPANDED_NODES:-10000000}   # omegaconf>=2.4 node limit vs the 14,561-log split yaml
