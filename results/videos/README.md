# Rollout videos

All from `tools/eval_libero.py --video-dir`, LIBERO-Object, MuJoCo 3.3.2, 256 px, replan every 10.
"dev" = init states 0-9 (seed 2000), "heldout" = init states 10-19 (seed 3000).

| File | What it shows |
| --- | --- |
| `compare_milk_noaug_vs_aug.mp4` | Same episode side by side: no augmentation hovers beside the milk carton until timeout; with augmentation it is placed in the basket |
| `compare_ketchup_noaug_vs_aug.mp4` | Same held-out episode: no augmentation releases the bottle next to the basket; with augmentation it goes in |
| `ketchup_heldout11_noaug100k_knocked_over.mp4` | Second failure mode without augmentation: bottle knocked over on approach |
| `creamcheese_dev3_noaug50k_fail.mp4` | Grasp slightly off the small cream cheese box |
| `soup_task0only_40k_init42_success.mp4` | Single-task ACT (44 demos, 40k steps), 50/50 run |
