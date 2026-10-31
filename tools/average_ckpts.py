"""Uniform weight averaging of several checkpoints of the same run (no retraining).

Success between neighbouring checkpoints swings by several points while the loss keeps falling;
averaging the late checkpoints is a cheap way to remove that noise.

Usage: python -I tools/average_ckpts.py --out <dir> <ckpt>/pretrained_model [<ckpt>/pretrained_model ...]
The config and processor files are copied from the last checkpoint given.
"""
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("ckpts", nargs="+")
    args = p.parse_args()

    states = [load_file(str(Path(c) / "model.safetensors")) for c in args.ckpts]
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            avg[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype)
        else:
            avg[k] = v  # integer buffers (e.g. counters) are taken from the first checkpoint
            assert all(torch.equal(s[k], v) for s in states), f"non-float tensor differs: {k}"

    out = Path(args.out)
    shutil.copytree(args.ckpts[-1], out, dirs_exist_ok=True)
    save_file(avg, str(out / "model.safetensors"))
    print(f"averaged {len(states)} checkpoints, {len(avg)} tensors -> {out}")


if __name__ == "__main__":
    main()
