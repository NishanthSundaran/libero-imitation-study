# Source before running anything: `source env.sh`
# Keeps this project isolated from the system ROS 2 / Python 3.10 setup.
unset PYTHONPATH
export LIBERO_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.libero"
export MUJOCO_GL=egl
# Dataset cache and checkpoints. `outputs/` in the repo is a symlink to $ML_DATA/outputs.
# On this laptop ~/ml_data links to the Windows (NTFS) partition. That is only safe with Windows
# hibernation / Fast Startup off (`powercfg /h off`): otherwise Windows resumes with its old view
# of the disk and Linux-written files are lost (happened once, see results/log.md).
export ML_DATA="${ML_DATA:-$HOME/ml_data}"
export HF_HOME="$ML_DATA/huggingface"
if [ ! -d "$ML_DATA/" ]; then
  echo "WARNING: $ML_DATA not available; mount the data drive (open it in Files) first" >&2
elif [ "$(head -c4 "$(findmnt -no TARGET -T "$ML_DATA/" 2>/dev/null)/hiberfil.sys" 2>/dev/null)" = "HIBR" ]; then
  echo "WARNING: Windows is hibernated; do not write to the data drive" >&2
else
  mkdir -p "$ML_DATA/outputs" "$HF_HOME"
fi
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # egl-probe (robomimic dep) predates CMake 4
# Pinned data revision: success rates are only comparable on the same data.
export LIBERO_DATASET_REVISION=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
