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

GIFs embedded in the main README (`results/gifs/`, 12 fps):

| File | What it shows |
| --- | --- |
| `three_policies_tomato_sauce.gif` | Held-out task 5, init 10, seed 3000: SmolVLA (126 steps), ACT + aug 75k (120), Diffusion Policy + aug 50k (116), all succeed |
| `diffusion_policy_four_tasks.gif` | Diffusion Policy 50k, held-out init 10 of tasks 0, 3, 4, 7 |
| `augmentation_milk.gif` | GIF of `compare_milk_noaug_vs_aug.mp4` |
| `pudding_heldout12_demo10_hovers_over_ketchup.mp4` | ACT trained on 10 demos per task: the pudding box sits behind the ketchup bottle; the arm hovers over the bottle until timeout |
