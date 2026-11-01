#!/usr/bin/env bash
# Reproducible install. Tested on Ubuntu 22.04, RTX 3060 Laptop (6 GB), NVIDIA driver 535.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
unset PYTHONPATH
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # egl-probe (robomimic dep) fails to configure under CMake 4

uv python install 3.12                    # lerobot 0.6.x requires Python >= 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "lerobot[libero,smolvla,diffusion]==0.6.1"

# The default PyPI torch wheel targets CUDA 13 and needs driver >= 580.
# Driver 535 supports CUDA 12.x, so swap to the cu128 build of the same version.
uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchvision \
  "torch==2.11.0" "torchvision==0.26.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# MuJoCo >= 3.3.3 renders LIBERO's floor ~28% darker than the frames in the training data
# (bisected with tools/render_probe.py); 3.3.2 matches the dataset's mean floor colour to within ~3 RGB levels.
uv pip install --python .venv/bin/python "mujoco==3.3.2"

# LIBERO asks an interactive question on first import; answer "no custom path" once.
mkdir -p .libero
echo n | LIBERO_CONFIG_PATH="$PWD/.libero" .venv/bin/python -I -c "import libero.libero" >/dev/null

.venv/bin/python -I -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('ok', torch.__version__)"
