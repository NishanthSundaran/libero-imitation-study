"""Print the episode list (JSON) with N demos per task for the demo-count study.

Subsets are nested (the 10-demo set is contained in the 20-demo set) and random with a fixed seed,
so the result does not depend on how the dataset happens to order its demos.
Usage: python -I tools/demo_subset.py --per-task 10 [--suite libero_object] [--seed 0]
"""
import argparse
import json
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-task", type=int, required=True)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tasks = json.load(open("results/suite_episodes.json"))[args.suite]["task_ids"]
    eps = []
    for tid in sorted(tasks, key=int):
        order = sorted(tasks[tid])
        random.Random(args.seed * 1000 + int(tid)).shuffle(order)  # same order for every N: nested subsets
        eps += order[: args.per_task]
    print(json.dumps(sorted(eps)))


if __name__ == "__main__":
    main()
