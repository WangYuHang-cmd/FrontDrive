# source this before running Drive-JEPA navsim_v1 code on the local machine
export FRONTDRIVE=/home/henry/Desktop/Qwen_Drive/frontdrive
export DRIVEJEPA_ROOT=$FRONTDRIVE/third_party/Drive-JEPA/navsim_v1
export NAVSIM_DEVKIT_ROOT=$DRIVEJEPA_ROOT
export NAVSIM_LOCAL=/home/henry/Desktop/Qwen_Drive/benchmarks/navsim          # local copies (logs, maps, sparse blobs, caches)
export OPENSCENE_DATA_ROOT=$NAVSIM_LOCAL/dataset
export NUPLAN_MAPS_ROOT=$NAVSIM_LOCAL/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=$NAVSIM_LOCAL/exp
export PYTHONPATH=$DRIVEJEPA_ROOT:$DRIVEJEPA_ROOT/vjepa2:${PYTHONPATH:-}
if [ -f "$FRONTDRIVE/env/.use_conda_cvpr" ] && [ "${FRONTDRIVE_USE_VENV:-0}" != 1 ]; then   # FRONTDRIVE_USE_VENV=1: keep the uv venv (torch 2.8 / PL 2.5.6) to continue runs started with it
  source "$FRONTDRIVE/env/conda_cvpr_activate.sh"      # server: conda env "cvpr" runs all code
else
  export PATH=$FRONTDRIVE/env/drivejepa/bin:$PATH
fi
export HYDRA_FULL_ERROR=1
export OMEGACONF_MAX_YAML_EXPANDED_NODES=${OMEGACONF_MAX_YAML_EXPANDED_NODES:-10000000}   # omegaconf>=2.4 node limit vs the 14,561-log split yaml
