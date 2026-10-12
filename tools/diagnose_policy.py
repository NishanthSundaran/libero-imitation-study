"""Open-loop sanity check for a trained policy.

1. Dataset check: feed demo frames to the policy, compare predicted action chunks with the
   recorded actions. Small error here means training and normalisation work.
2. Sim check: build the same observation the evaluator builds at an episode start and compare its
   statistics (state, image brightness) with dataset start frames, plus the first predicted actions.
A policy that is fine on (1) but wrong on (2) has an input mismatch between data and simulator.
"""
import argparse
import json
import os

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--rename-map", default="{}")
    args = p.parse_args()
    rename_map = json.loads(args.rename_map)
    torch.manual_seed(0)

    env_cfg = LiberoEnvConfig(task="libero_object", task_ids=[args.task_id])
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename_map).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename_map}},
    )

    # ---- 1. dataset frames -------------------------------------------------------------------
    eps = json.load(open("results/suite_episodes.json"))["libero_object"]["task_ids"][str(args.task_id)]
    ds = LeRobotDataset("lerobot/libero", revision=os.environ["LIBERO_DATASET_REVISION"], episodes=eps[:3])
    meta = ds.meta.episodes.to_pandas() if hasattr(ds.meta.episodes, "to_pandas") else ds.meta.episodes
    errs, starts = [], []
    for ep in eps[:3]:
        row = meta[meta.episode_index == ep].iloc[0]
        a, b = int(row.dataset_from_index), int(row.dataset_to_index)
        idx_all = ds.hf_dataset["index"]
        local = [i for i, g in enumerate(idx_all) if a <= int(g) < b]
        acts = torch.stack([ds[i]["action"] for i in local])
        for j in np.linspace(0, len(local) - 1, args.n_frames).astype(int)[:-1]:
            s = ds[local[j]]
            if j == 0:
                starts.append(s)
            obs = {k: v.unsqueeze(0) for k, v in s.items() if k.startswith("observation.")}
            obs["task"] = [s["task"]]
            with torch.no_grad():
                policy.reset()
                act = post(policy.select_action(pre(obs))).cpu()[0]
            errs.append((j, (act - acts[j]).abs().numpy()))
            if j == 0:
                print(f"ep {ep} t=0  pred {act.numpy().round(3)}\n           true {acts[0].numpy().round(3)}")
    e = np.stack([x[1] for x in errs])
    print("dataset open-loop |pred - true| per dim (mean):", e.mean(0).round(3))
    print("dataset action std per dim                  :", acts.std(0).numpy().round(3))

    # ---- 2. simulator start ----------------------------------------------------------------
    env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    vec = next(iter(make_env(env_cfg, n_envs=1)[env_cfg.task].values()))
    raw, _ = vec.reset(seed=0)
    obs = env_pre(preprocess_observation(raw))
    obs["task"] = list(vec.call("task_description"))
    d = starts[0]
    print("state  sim ", obs["observation.state"][0].numpy().round(3))
    print("state  data", d["observation.state"].numpy().round(3))
    for k in ["observation.images.image", "observation.images.image2"]:
        print(f"{k:28s} sim shape {tuple(obs[k].shape[1:])} mean {obs[k].mean().item():.3f}   "
              f"data shape {tuple(d[k].shape)} mean {d[k].mean().item():.3f}")
    with torch.no_grad():
        policy.reset()
        act = post(policy.select_action(pre(dict(obs)))).cpu()[0]
    print("first sim action", act.numpy().round(3))


if __name__ == "__main__":
    main()
