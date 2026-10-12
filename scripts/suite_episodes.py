"""Map every episode of lerobot/libero to its LIBERO suite, write results/suite_episodes.json."""
import json, os
from collections import Counter
from libero.libero import benchmark
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REV = os.environ["LIBERO_DATASET_REVISION"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

def norm(s):
    return " ".join(s.lower().replace("_", " ").split())

lang2suite = {}
for suite in SUITES:
    b = benchmark.get_benchmark_dict()[suite]()
    for i in range(b.n_tasks):
        lang = norm(b.get_task(i).language)
        assert lang not in lang2suite, f"duplicate instruction across suites: {lang}"
        lang2suite[lang] = (suite, i)

ds = LeRobotDataset("lerobot/libero", revision=REV)
idx2task = {int(r.task_index): norm(t) for t, r in ds.meta.tasks.iterrows()}
cols = ds.hf_dataset.select_columns(["episode_index", "task_index"]).to_pandas().drop_duplicates("episode_index")

out = {s: {"episodes": [], "task_ids": {}} for s in SUITES}
missing = Counter()
for ep, ti in zip(cols.episode_index, cols.task_index):
    lang = idx2task[int(ti)]
    if lang not in lang2suite:
        missing[lang] += 1
        continue
    suite, tid = lang2suite[lang]
    out[suite]["episodes"].append(int(ep))
    out[suite]["task_ids"].setdefault(str(tid), []).append(int(ep))

for s in SUITES:
    per = {k: len(v) for k, v in sorted(out[s]["task_ids"].items(), key=lambda kv: int(kv[0]))}
    print(s, len(out[s]["episodes"]), "episodes; per task:", per)
print("unmatched instructions:", dict(missing))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/suite_episodes.json", "w"), indent=1)
