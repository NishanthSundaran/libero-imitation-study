#!/usr/bin/env bash
# Disk saver while lerobot-train runs: keep training_state (optimizer, needed to resume) only in
# the newest checkpoint; older checkpoints keep pretrained_model, which is all evaluation needs.
# Usage: tools/prune_ckpts.sh outputs/train/<run>
D=$1
while ps aux | grep -q "[b]in/lerobot-train"; do
  ls -d "$D"/checkpoints/[0-9]* 2>/dev/null | sort | head -n -1 | while read -r c; do
    rm -rf "$c/training_state"
  done
  sleep 120
done
