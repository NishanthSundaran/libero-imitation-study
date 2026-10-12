# Source before running anything: `source env.sh`
# Keeps this project isolated from the system ROS 2 / Python 3.10 setup.
unset PYTHONPATH
export LIBERO_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.libero"
export MUJOCO_GL=egl
# Large, re-downloadable data lives on the second drive (root disk is nearly full).
# `outputs/` in the repo is a symlink to $ML_DATA/outputs.
export ML_DATA=/media/prapanch/Acer/ml_data
export HF_HOME="$ML_DATA/huggingface"
[ -d "$ML_DATA" ] || echo "WARNING: $ML_DATA not mounted; open the Acer drive in Files first" >&2
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # egl-probe (robomimic dep) predates CMake 4
# Pinned data revision: success rates are only comparable on the same data.
export LIBERO_DATASET_REVISION=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
