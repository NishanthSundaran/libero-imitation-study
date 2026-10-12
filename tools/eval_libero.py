"""Evaluate a LeRobot policy on LIBERO with explicit control over init states and env reuse.

Uses LeRobot's own `rollout()` and processors, so results are comparable with `lerobot-eval`.
Differences from `lerobot-eval`:
  * `--init-states` picks exactly which official init states are run (lerobot-eval always starts at 0).
  * `--env-mode fresh` builds a new simulator per episode; `reuse` keeps one env per task and
    resets it between episodes, which is what lerobot-eval does with batch_size=1.
  * Writes one JSON row per episode (task, init state, success, steps) for later analysis.

Example:
  python -I tools/eval_libero.py --policy lerobot/smolvla_libero --suite libero_object \
      --task-ids 0 1 2 --init-states 0 1 2 3 4 --env-mode fresh \
      --rename-map '{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
      --out outputs/eval/smolvla_fresh.json
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from libero.libero import benchmark

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.random_utils import set_seed


def build_vec_env(env_cfg, suite, suite_name, task_id, episode_index):
    def make():
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=env_cfg.camera_name,
            init_states=True,
            episode_index=episode_index,
            n_envs=1,
            control_mode=env_cfg.control_mode,
            **{k: v for k, v in env_cfg.gym_kwargs.items() if k != "task_ids"},
        )

    return gym.vector.SyncVectorEnv([make], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)


def save_mp4(frames, path, fps=20):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="Hub id or local pretrained_model dir")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--init-states", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--env-mode", choices=["fresh", "reuse"], default="fresh")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="if given: run every init state once per seed, re-seeding all RNGs before each episode "
                        "(fresh env only), so the outcome depends only on (init state, seed)")
    p.add_argument("--obs-size", type=int, default=360, help="render size; lerobot-eval default is 360")
    p.add_argument("--rename-map", default="{}")
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--temporal-ensemble", type=float, default=None,
                   help="ACT only: temporal ensembling coefficient (e.g. 0.01); forces n_action_steps=1")
    p.add_argument("--out", required=True)
    p.add_argument("--video-dir", default=None,
                   help="save agent-view mp4s of every failure and of the first success per task")
    args = p.parse_args()

    rename_map = json.loads(args.rename_map)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    env_cfg = LiberoEnvConfig(
        task=args.suite, task_ids=args.task_ids,
        observation_height=args.obs_size, observation_width=args.obs_size,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy_cfg.pretrained_path = args.policy
    if args.n_action_steps is not None:
        policy_cfg.n_action_steps = args.n_action_steps
    if args.temporal_ensemble is not None:
        policy_cfg.temporal_ensemble_coeff = args.temporal_ensemble
        policy_cfg.n_action_steps = 1
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for task_id in args.task_ids:
        saved_success = False
        shared = None
        if args.env_mode == "reuse":
            # Same as lerobot-eval: one env whose init-state counter starts at the first requested
            # index and advances by one on every reset.
            shared = build_vec_env(env_cfg, suite, args.suite, task_id, args.init_states[0])
        if args.seeds is not None:
            assert args.env_mode == "fresh", "--seeds requires --env-mode fresh"
            episodes = [(init_state, seed) for init_state in args.init_states for seed in args.seeds]
        else:
            episodes = [(init_state, args.seed + k) for k, init_state in enumerate(args.init_states)]
        for k, (init_state, ep_seed) in enumerate(episodes):
            env = shared or build_vec_env(env_cfg, suite, args.suite, task_id, init_state)
            if args.seeds is not None:
                set_seed(ep_seed)
            frames = []
            record = (lambda e: frames.append(e.call("render")[0])) if args.video_dir else None
            t0 = time.time()
            with torch.no_grad():
                info = rollout(
                    env=env, policy=policy,
                    env_preprocessor=env_pre, env_postprocessor=env_post,
                    preprocessor=preprocessor, postprocessor=postprocessor,
                    seeds=[ep_seed],
                    render_callback=record,
                )
            success = bool(info["success"][0].any())
            done = info["done"][0].numpy()
            steps = int(np.argmax(done)) + 1 if done.any() else int(done.shape[0])
            row = dict(policy=args.policy, suite=args.suite, task_id=task_id, init_state=init_state,
                       env_mode=args.env_mode, episode_in_env=k, seed=ep_seed, reseeded=args.seeds is not None,
                       success=success, steps=steps, wall_s=round(time.time() - t0, 1))
            if args.video_dir and frames and (not success or not saved_success):
                tag = "ok" if success else "fail"
                save_mp4(frames, Path(args.video_dir) / f"t{task_id}_init{init_state}_s{ep_seed}_{tag}.mp4")
                saved_success |= success
            rows.append(row)
            print(json.dumps(row), flush=True)
            if shared is None:
                env.close()
        if shared is not None:
            shared.close()

    json.dump(rows, open(out, "w"), indent=1)
    rate = np.mean([r["success"] for r in rows])
    print(f"SUMMARY {args.env_mode}: {sum(r['success'] for r in rows)}/{len(rows)} = {rate:.1%}")


if __name__ == "__main__":
    main()
