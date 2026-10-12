# LIBERO Imitation Study

How much do demonstrations matter, and how do a vision-language-action model (SmolVLA) and two
classic imitation-learning policies (ACT, Diffusion Policy) compare when trained on the same LIBERO data?
Everything runs on a single 6 GB laptop GPU (RTX 3060 Laptop) plus free cloud GPUs for the VLA fine-tune.

Status: work in progress. Numbers below are only what has been measured so far.

## Before trusting any number: reproducing a published result

The first step was not training anything but checking the evaluation itself, by evaluating the public
checkpoint `lerobot/smolvla_libero` (trained on the same dataset) and comparing with the published score.

| Check | Result |
| --- | --- |
| Published SmolVLA (0.45B), LIBERO-Object | 96.0 % |
| `lerobot-eval`, 10 tasks x 5 episodes, seed 1000 | **56 %** (28/50) |
| Same run, 1st episode of each task only | 90 % (9/10) |
| Same run, episodes 3 to 5 | 40 % (12/30) |

What explains the gap, tested one variable at a time with `tools/eval_libero.py` (which reproduces
`lerobot-eval` episode for episode):

1. **Not environment reuse.** A fresh simulator per episode gives identical outcomes.
2. **Not the sampling seed.** Re-seeding before every episode, 3 seeds x 20 init states: 59 of 60
   outcomes are the same for all three seeds. The first episode only looked better because the
   official init state 0 happens to be easy (12/12) while init state 3 is never solved (0/12).
3. **The MuJoCo version.** The training frames were rendered with an older MuJoCo. Bisecting
   versions with `tools/render_probe.py`: 3.3.2 and earlier reproduce the dataset's floor colour,
   3.3.3 and later (LeRobot installs 3.8.1) render it about 28 % darker. Pinning `mujoco==3.3.2`
   changes nothing else.

| Same checkpoint, same protocol (10 tasks x 5 episodes, seed 1000) | Success |
| --- | --- |
| MuJoCo 3.8.1 (default install) | 56 % (28/50) |
| MuJoCo 3.3.2 (matches training renders) | **72 % (36/50)** |

On the 20-state grid, every episode solved under 3.8.1 is also solved under 3.3.2 (9/20 to 16/20).
A 24 point gap to the published 96 % remains.

## Setup findings worth knowing

* **Camera orientation.** LIBERO renders images rotated 180 degrees relative to the dataset. LeRobot's
  `LiberoProcessorStep` flips them back at evaluation time; verified by comparing the policy's input
  frame with a dataset frame of the same task and init state (robot state agrees to about 5 mm).
* **Rendering differs from the data.** With MuJoCo 3.8.1 the floor renders about 28 % darker than in the
  dataset frames (RGB 136/123/111 vs 188/180/170) while objects match. The texture file is identical,
  so the difference comes from the renderer, not the assets. MuJoCo 3.3.2 gives 190/181/173, so
  `scripts/setup.sh` pins it (`tools/render_probe.py` reproduces the measurement).
* **Language matters for the comparison.** All ten LIBERO-Goal tasks start from the same scene, so
  policies without language input (ACT, Diffusion Policy) cannot tell them apart. The comparison is
  therefore done per task on LIBERO-Object, whose scenes differ per task.

## Reproduce

```bash
scripts/setup.sh           # Python 3.12 venv, lerobot[libero,smolvla]==0.6.1, torch cu128
source env.sh
python -I scripts/suite_episodes.py      # episode -> suite/task map (results/suite_episodes.json)
```

Dataset: `lerobot/libero` pinned at revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.

## Hardware notes

NVIDIA driver 535 supports CUDA 12.x only; the default PyPI torch wheel targets CUDA 13, so
`scripts/setup.sh` installs the cu128 build of the same torch version.
