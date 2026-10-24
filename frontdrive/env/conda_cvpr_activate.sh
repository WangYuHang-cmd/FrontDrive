# Activate the conda environment `cvpr` (Henry 2026-10-17: run all code on the server in conda env "cvpr").
# Sourced by env/drivejepa_env.sh and env/drivejepa_pb_env.sh when env/.use_conda_cvpr exists. Fails loudly.
_CONDA_BASE=${CONDA_CVPR_BASE:-/home/exouser/henry/tools/miniforge3}
if [ ! -f "$_CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "conda_cvpr_activate: $_CONDA_BASE not found (env/.use_conda_cvpr is set; run scripts/h100/bootstrap_conda_cvpr.sh or delete the marker)" >&2
  unset _CONDA_BASE; return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$_CONDA_BASE/etc/profile.d/conda.sh"
conda activate cvpr || { echo "conda_cvpr_activate: 'conda activate cvpr' failed; run scripts/h100/bootstrap_conda_cvpr.sh" >&2; unset _CONDA_BASE; return 1 2>/dev/null || exit 1; }
unset _CONDA_BASE
