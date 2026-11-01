#!/usr/bin/env bash
# Demo-count study: the winning ACT + augmentation recipe, retrained on 10 and 20 demos per task
# (nested random subsets, tools/demo_subset.py). 50k steps each (more epochs per demo than the
# full-data run, whose dev score already peaked at 50k). Checkpoints 25k and 50k are scored on dev;
# the pick (best dev, ties to the earliest) is scored once on held-out.
# Usage: bash scripts/demo_count_study.sh 10 20
set -u
cd "$(dirname "$0")/.."
source env.sh
REF=outputs/train/act_obj_all_aug_r1/checkpoints/075000/pretrained_model/train_config.json
INHIBIT="systemd-inhibit --what=sleep:idle:handle-lid-switch --who=libero-study --why=training"

ev() {  # ev <ckpt dir> <out json> <init states...> -- seed
  local ck=$1 out=$2 seed=$3; shift 3
  $INHIBIT python -I tools/eval_libero.py --policy "$ck/pretrained_model" --init-states "$@" \
    --seed "$seed" --obs-size 256 --n-action-steps 10 --out "$out" > "outputs/$(basename "${out%.json}").log" 2>&1
}

for N in "$@"; do
  RUN=act_obj_demo${N}_aug_r1
  D=outputs/train/$RUN
  EPS=$(python -I tools/demo_subset.py --per-task "$N")
  mkdir -p outputs/train
  $INHIBIT lerobot-train --config_path="$REF" --dataset.episodes="$EPS" \
    --steps=50000 --save_freq=25000 --output_dir="$D" --job_name="$RUN" --resume=false \
    > "outputs/train/$RUN.log" 2>&1
  rm -rf "$D"/checkpoints/*/training_state  # optimizer state, not needed for evaluation
  [ -f "$D/checkpoints/050000/pretrained_model/model.safetensors" ] || { echo "$RUN: training failed"; exit 1; }

  best=""; bestn=-1
  for s in 025000 050000; do
    out=results/eval/dev_${RUN}_$((10#$s/1000))k_nas10.json
    ev "$D/checkpoints/$s" "$out" 2000 0 1 2 3 4 5 6 7 8 9
    n=$(python -I -c "import json;print(sum(r['success'] for r in json.load(open('$out'))))")
    echo "$RUN $s dev: $n/100"
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$s; fi
  done
  out=results/eval/heldout_${RUN}_$((10#$best/1000))k_s3000_nas10.json
  ev "$D/checkpoints/$best" "$out" 3000 10 11 12 13 14 15 16 17 18 19
  echo "$RUN pick $best held-out: $(python -I -c "import json;print(sum(r['success'] for r in json.load(open('$out'))))")/100"
done
