#!/usr/bin/env bash
# Every reported held-out number, re-run in deterministic mode (tools/eval_libero.py default since
# 2026-11-02) with failure videos, one evaluation at a time. Writes results/eval/final/*.json and
# outputs/eval_videos/final_*/. Ends with two repeat runs that must match exactly.
set -u
cd "$(dirname "$0")/.."
source env.sh >/dev/null
I="systemd-inhibit --what=sleep:idle:handle-lid-switch --who=libero-eval --why=final-heldout"
T=outputs/train
SMOL_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
H="--task-ids 0 1 2 3 4 5 6 7 8 9 --init-states 10 11 12 13 14 15 16 17 18 19"
OUT=results/eval/final
mkdir -p "$OUT" outputs/final_logs

run() {  # run <name> <seeds> <policy args...>
  local name=$1 seeds=$2; shift 2
  $I python -I tools/eval_libero.py "$@" $H --seeds $seeds --out "$OUT/$name.json" \
    --video-dir "outputs/eval_videos/final_$name" > "outputs/final_logs/$name.log" 2>&1
  python -I -c "
import json, collections
c = collections.Counter()
for r in json.load(open('$OUT/$name.json')): c[r['seed']] += r['success']
print('$name', dict(sorted(c.items())), sum(c.values()), flush=True)"
}

run act_aug_75k "3000 3001 3002" --policy $T/act_obj_all_aug_r1/checkpoints/075000/pretrained_model --obs-size 256 --n-action-steps 10
run dp_aug_50k "3000 3001 3002" --policy $T/dp_obj_all_aug_r1/checkpoints/050000/pretrained_model --obs-size 256
run smolvla "3000 3001 3002" --policy lerobot/smolvla_libero --rename-map "$SMOL_MAP" --n-action-steps 10
run act_demo10_25k 3000 --policy $T/act_obj_demo10_aug_r1/checkpoints/025000/pretrained_model --obs-size 256 --n-action-steps 10
run act_demo20_50k 3000 --policy $T/act_obj_demo20_aug_r1/checkpoints/050000/pretrained_model --obs-size 256 --n-action-steps 10
run act_noaug_100k 3000 --policy $T/act_obj_all_r2/checkpoints/100000/pretrained_model --obs-size 256 --n-action-steps 10

# Reproducibility: repeat a stochastic policy and a weak model; every episode must match.
rep() {  # rep <name> <seed> <policy args...>
  local name=$1 seed=$2; shift 2
  $I python -I tools/eval_libero.py "$@" $H --seeds $seed --out "outputs/final_logs/repeat_$name.json" \
    > "outputs/final_logs/repeat_$name.log" 2>&1
  python -I -c "
import json
k = lambda r: (r['task_id'], r['init_state'], r['seed'])
a = {k(r): (r['success'], r['steps']) for r in json.load(open('outputs/final_logs/repeat_$name.json'))}
b = {k(r): (r['success'], r['steps']) for r in json.load(open('$OUT/$name.json')) if k(r) in a}
print('repeat $name: mismatches', sum(a[x] != b[x] for x in a), 'of', len(a), flush=True)"
}
rep dp_aug_50k 3000 --policy $T/dp_obj_all_aug_r1/checkpoints/050000/pretrained_model --obs-size 256
rep act_demo10_25k 3000 --policy $T/act_obj_demo10_aug_r1/checkpoints/025000/pretrained_model --obs-size 256 --n-action-steps 10
