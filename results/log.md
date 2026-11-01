# Experiment log

One entry per cycle: what was run, the measured result, the decision it led to.
All evals: LIBERO-Object, `tools/eval_libero.py` (matches `lerobot-eval` episode for episode),
MuJoCo 3.3.2 unless stated, dataset `lerobot/libero@a1aaacb`.

## Cycle 0: can the evaluation be trusted? (2026-10-12)

* `lerobot/smolvla_libero`, lerobot-eval 10x5, seed 1000: 56 % (MuJoCo 3.8.1) vs 96 % published.
* Fresh env per episode = identical outcomes. Reseeding (3 seeds x 20 init states): 59/60 outcomes
  seed-independent; init state decides (state 0: 12/12, state 3: 0/12).
* MuJoCo bisect: <= 3.3.2 renders the floor like the dataset, >= 3.3.3 about 28 % darker.
* Same protocol with MuJoCo 3.3.2: 72 % (36/50). 20-state grid: 9/20 -> 16/20, no regressions.
* Ablations on the 20-state grid (MuJoCo 3.3.2): 256 px render 15/20, replan every 10 steps 17/20.
  Neither is a lever at this sample size.
* Decision: pin MuJoCo 3.3.2; verify with a larger independent sample (10x10, seed 2000).

## Cycle 1: verification + first own policy

* Running: SmolVLA reference 10 tasks x 10 init states, seed 2000, MuJoCo 3.3.2 and 3.8.1.
* Running: ACT run 1, task 0 only (44 demos), batch 16, 40k steps, chunk 50, ckpt every 10k.

### Cycle 1 results

* Verification, independent sample (10 tasks x init states 0-9, seed 2000, MuJoCo 3.3.2):
  SmolVLA reference **78/100**, consistent with 36/50 from cycle 0.
* Determinism: rerunning task 3 with video recording gives the identical outcome string.
* Failure analysis (videos, task 3): the arm reaches the target but the grasp misses (gripper closes
  on air or clips the tall bottle), then the policy keeps executing the place motion. The checkpoint
  executes all 50 actions of a chunk (2.5 s) open loop, so it cannot notice and retry.
* Fix tested: replan every 10 actions (`n_action_steps=10`, inference-time only, same weights).
  Same 100 episodes: **94/100** (20 fixed, 4 broken, McNemar exact p = 0.0015).
* Caveat: chosen after looking at seed-2000 data, so confirming on held-out init states 10-19, seed 3000.
* Held-out confirmation (init states 10-19, seed 3000, never inspected): replan-10 **97/100**.

## Cycle 2: ACT, task 0 (alphabet soup), 44 demos

* Run 1 (batch 16, chunk 50, 10k of 40k steps), all 50 official init states, seed 4000:
  **0/50**. Videos: arm rises and drifts away, never descends. Systematic, so not undertraining.
* `tools/diagnose_policy.py`: open-loop on dataset frames the predictions are close to the
  recorded actions (mean abs error 0.04 vs action std 0.48 on y), so training is fine. In sim the
  observation is 360 px but the policy trained on 256 px. SmolVLA resizes internally, ACT does not
  (ResNet feature map 12x12 instead of 8x8). State and image brightness match.
* Fix: render at 256 px. Same checkpoint, same 50 states: **5/50**. Videos: reaches the right
  object, grasp misses, lifts and re-approaches until timeout (same open-loop chunk issue as SmolVLA).
* Testing: replan every 10 actions vs ACT temporal ensembling (coeff 0.01).
* Replan every 10 actions, same 10k checkpoint, 256 px, same 50 states: **38/50 (76 %)**.
* Interrupted (2026-10-13): all background jobs were stopped because system RAM ran
  critically low (3 GPU jobs with 6 dataloader workers each, plus Firefox using ~9 GB).
  Lost: ACT run 1 at step ~20k (only the 10k checkpoint was saved), SmolVLA held-out replan-50 and
  replan-5 runs, ACT temporal-ensembling test.
* Resumed (2026-10-13 06:38): ACT run 1 from the 10k checkpoint, `--resume=true --num_workers=2`,
  running alone. Loss at resume 0.143, about 5.6 steps/s, data loading not a bottleneck
  (data_s 0.016), RAM about 5 GB used. Plan: evaluate 20k, 30k, 40k checkpoints one at a time,
  256 px, replan every 10, same 50 init states, seed 4000.
* `tools/summarize.py` (paired comparison + McNemar): replan-10 vs open-loop chunks on the 10k ACT
  checkpoint, same 50 states: fixed 34, broken 1, p < 0.0001. Remaining failures: init states
  0, 2, 6, 9, 12, 15, 23, 30, 32, 40, 41, 42 (all time out at 280 steps).

### Cycle 2 results: ACT task 0, checkpoints (256 px, replan every 10, 50 init states, seed 4000)

| Checkpoint | Train loss | Success |
| --- | --- | --- |
| 10k | 0.143 | 38/50 |
| 20k | 0.094 | 39/50 |
| 30k | 0.076 | 38/50 |
| 40k (final, planned) | 0.067 | **50/50** |

* 40k vs 10k paired: fixed 12, broken 0, McNemar exact p = 0.0005. 20k and 30k are not different
  from 10k (fixed 6 / broken 5, fixed 8 / broken 8): failures move between init states.
* Verified by video (init states 0, 6, 42, rerun deterministically): grasps the alphabet soup,
  places it in the basket. 40k was the pre-planned final checkpoint, not picked from the four.
* Caveat: one task, 44 demos, every official init state used for evaluation, so there is no
  held-out set of init states for this task.

## Cycle 3: one ACT for all 10 LIBERO-Object tasks (454 demos)

* Check: every task has a different object set (bddl files), so the scene identifies the task
  without language. Fair comparison with SmolVLA, which is also one model for the suite.
* Run `act_obj_all_r1`: batch 32, 100k steps (about 47 epochs, task-0 run was 93), lr 1e-5,
  chunk 50, ckpt every 20k, seed 1000, 2 workers. 2.7 steps/s, about 10 h. GPU 4.6 GB.
* Planned eval: final checkpoint, 10 tasks x init states 10-19, seed 3000, 256 px, replan 10
  (same episodes as the SmolVLA held-out 97/100).
* Multi-task ACT finished 2026-10-14 06:35 (100k steps, loss 0.083, ckpts 20k to 100k).
  Held-out eval of the 100k checkpoint (10 tasks x init states 10-19, seed 3000) was started and
  then stopped on request before it finished; no result yet.

## Data loss and restart (2026-10-30)

* All checkpoints and the dataset cache on the Acer NTFS partition were gone. `hiberfil.sys` has
  the `HIBR` signature: Windows was hibernated (Fast Startup) while Linux wrote to the partition,
  so Windows' resume discarded the Linux-written files. Eval JSONs, code and log were in git on
  the Linux disk and are intact. Data now lives in `~/ml_data` (dataset is only 1.9 GB).
* Lost: ACT task-0 checkpoints, multi-task ACT run 1 (never evaluated).

## Cycle 3 (restart): one ACT for all 10 LIBERO-Object tasks

* Run `act_obj_all_r2`: same recipe as run 1 (batch 32, 100k steps, lr 1e-5, chunk 50, seed 1000)
  plus `use_amp=true`: 3.5 steps/s instead of 2.45. Ckpt every 25k; `tools/prune_ckpts.sh` drops
  optimizer state from older checkpoints to save disk.
* Protocol from here on: dev set = 10 tasks x init states 0-9, seed 2000 (for choosing settings),
  held-out = init states 10-19, seed 3000 (final numbers only). Same episodes as SmolVLA.
* Laptop suspended 16:50 to 20:00 at step 62k; after resume the training hung and no new process
  could create a CUDA context ("CUDA unknown error", Xid 31 in the kernel log). Fixed by
  reloading `nvidia_uvm` (`pkexec sh -c "rmmod nvidia_uvm && modprobe nvidia_uvm"`), resumed from
  the 50k checkpoint. Training now runs under `systemd-inhibit --what=sleep:idle:handle-lid-switch`.
* Dev set, 50k checkpoint, replan 10: **76/100** (SmolVLA reference on the same episodes: 94/100).
  Per task: 8 6 8 10 9 8 7 3 8 9. Task 7 (milk) 3/10, task 1 (cream cheese) 6/10.
* Videos (task 7 init 2/3/4, task 1 init 3/5): always goes to the correct object, so the scene
  does identify the task without language. Failures are imprecise grasps: gripper beside the tall
  milk carton, or slightly off the small cream cheese box, then hovering until timeout.
* Next: 100k checkpoint on the dev set; then inference-time precision levers (replan every 5,
  temporal ensembling) on the dev set before any retraining.
* Inference levers on the 50k checkpoint, dev set (per task in brackets):
  replan 10: 76/100 (8 6 8 10 9 8 7 3 8 9); replan 5: 71/100 (6 5 8 9 8 7 6 4 9 9), fixed 5 /
  broken 10, p = 0.30; temporal ensembling 0.01: 45/100 (6 2 7 3 9 4 3 0 4 7), fixed 3 / broken 34,
  p < 0.0001. Ensembling averages overlapping chunks and blurs the late grasp corrections.
  Decision: keep replan 10; the next gain has to come from training.

### Cycle 3 results: multi-task ACT, dev set (10 tasks x init 0-9, seed 2000, 256 px, replan 10)

| Checkpoint | Train loss | Dev success | Per task 0..9 |
| --- | --- | --- | --- |
| 50k | 0.115 | 76/100 | 8 6 8 10 9 8 7 3 8 9 |
| 75k | | 85/100 | 8 8 9 9 7 7 10 7 10 10 |
| 100k | 0.088 | **89/100** | 10 8 10 10 8 10 9 7 8 9 |

* 100k vs 50k: fixed 18, broken 5, McNemar p = 0.011. SmolVLA reference on the same 100: 94.
* Still improving with training; 100k steps = 48 epochs, while the single-task ACT needed 93
  epochs for 50/50. ACT uses a constant lr, so extending the run equals training longer.

## Cycle 4: extend multi-task ACT to 200k steps (about 95 epochs)

* Resume from 100k with `--steps=200000` (resumed at loss 0.088). Ckpts 125k/150k/175k/200k.
* Then dev eval of 150k and 200k, pick on dev, final number on held-out.
* 150k, dev: **82/100** (per task 10 9 10 9 9 7 7 7 7 7). vs 100k: fixed 4, broken 11, p = 0.12.
  Train loss keeps falling (0.078 at 136k) but success between neighbouring checkpoints swings by
  about 7 points (task-0 ACT showed the same: 38, 39, 38, 50). Persistent failures: task 7 init
  states 3, 4, 6 fail at both 100k and 150k.
* Plan for cycle 5: average the weights of the 100k to 200k checkpoints (no retraining) to remove
  checkpoint-to-checkpoint noise; compare on dev against the single 200k checkpoint.

### Cycle 4/5 results (dev set, replan 10)

| Model | Train loss | Dev | Per task 0..9 | vs 100k |
| --- | --- | --- | --- | --- |
| 100k | 0.088 | **89** | 10 8 10 10 8 10 9 7 8 9 | |
| 150k | 0.078 | 82 | 10 9 10 9 9 7 7 7 7 7 | fixed 4, broken 11, p = 0.12 |
| 200k | 0.068 | 83 | 9 10 9 7 6 8 8 9 10 7 | fixed 7, broken 13, p = 0.26 |
| mean of 100k..200k weights | | 83 | 10 8 9 9 6 7 9 8 8 9 | fixed 6, broken 12, p = 0.24 |

* Plateau after 100k: true rate about 85 %, 100 dev episodes give a standard error of about 3.5
  points, so 89 is probably a favourable draw. Lower loss without better success: fitting the
  demos more closely, not generalising better to new object positions. Weight averaging no gain.
* Decision: keep 100k as the pick (chosen on dev) and record its held-out number. Next lever
  targets generalisation: image augmentation.

## Cycle 6: multi-task ACT with image augmentation

* Run `act_obj_all_aug_r1`: identical to run 2 (AMP, batch 32, 100k steps, seed 1000) plus
  `--dataset.image_transforms.enable=true` (LeRobot defaults: up to 3 of brightness, contrast,
  saturation, hue, sharpness, affine +-5 deg / 5 % shift). data_s 0.02 vs updt_s 0.29: no loader
  bottleneck.
* **Held-out, 100k checkpoint (picked on dev): 78/100** (per task 9 9 7 10 4 8 9 6 8 8).
  SmolVLA on the same 100: 97/100; paired fixed 1, broken 20, p < 0.0001. Dev 89 was optimistic
  (selection among checkpoints plus a favourable draw). Init states (0,14) and (8,10) fail for both.
* Videos, task 4 (ketchup) held-out init 10/11: (10) grasps and carries the bottle but releases
  it short, bottle ends up standing next to the basket; (11) knocks the bottle over on approach,
  no recovery. Tall, tippable objects (ketchup, milk) are the weak spot.
* **Augmentation, 50k checkpoint, dev: 96/100** (per task 8 10 10 10 10 8 10 10 10 10).
  vs no-augmentation 50k (76/100): fixed 23, broken 3, McNemar p = 0.0001. SmolVLA on the same
  dev episodes: 94/100. Train loss is higher with augmentation (0.158 vs 0.107 at 61k).
* Videos: milk (dev init 3, failed at every no-aug checkpoint) and ketchup (held-out init 10,
  failed before by releasing next to the basket) are now grasped and placed in the basket.
  Note: the ketchup check used one held-out episode that was already inspected above.
* Interpretation: the earlier failures were overfitting to exact pixel positions; small random
  shifts/rotations and colour jitter make the grasp generalise to new object placements.
* Augmentation run finished (100k, loss 0.114). Dev: 50k 96, 75k 95, 100k 93 (per task 75k:
  10 10 10 9 10 7 10 10 10 9; 100k: 10 9 10 8 10 7 9 10 10 10). The automatic pick compared only
  75k and 100k (50k had been evaluated separately); all three are within dev noise.
* **Held-out, augmentation 75k: 97/100** (per task 10 10 10 9 10 9 9 10 10 10).
  vs SmolVLA reference (97/100): fixed 3, broken 3, p = 1.0, a tie with a model about 9x smaller.
  vs no-augmentation ACT 100k (78/100): fixed 21, broken 2, p = 0.0001.
  Remaining failures (3,17), (5,14), (6,15) are on three different tasks.

## Storage (2026-11-01)

* Windows hibernation turned off by the user (`powercfg /h off`; `hiberfil.sys` gone). Data moved
  back to the NTFS drive (`~/ml_data` is now a symlink to it, copy verified with `diff -r`).
  `env.sh` warns if the drive is missing or Windows is hibernated again.

## Cycle 7: Diffusion Policy, one model for all 10 LIBERO-Object tasks

* Installed the `diffusion` extra (adds only diffusers 0.39.0 + importlib-metadata; torch/mujoco unchanged).
* LeRobot defaults (horizon 64, no resize, 278M params) run out of memory on 6 GB. Settings follow
  the Diffusion Policy paper instead: n_obs_steps 2, horizon 16, n_action_steps 8, resize 128 px +
  random crop 0.9, U-Net down_dims (256, 512, 1024) = 89M params, DDIM with 10 inference steps,
  lr 1e-4 cosine. Same data, batch 32, 100k steps, AMP, seed 1000, and the same LeRobot image
  augmentation as the winning ACT run. 4 dataloader workers (2 frames x 2 cameras per sample).
* Smoke test: 1.9 GB GPU, evaluator works with n_obs_steps 2 (24.5 s per 280-step episode).
* Full run `dp_obj_all_aug_r1`: 3.4 steps/s (data_s 0.15 vs updt_s 0.14), about 8 h.
* Plan: dev eval of 50k, 75k, 100k; held-out on the dev pick.
* 50k dev: **100/100** (ACT + aug 50k: 96/100; paired: DP fixes 4, breaks 0, McNemar p = 0.125, so
  not yet significant). Mean steps when solved 128 (ACT 135), max 164 (ACT 279). The four ACT failures
  (task 0 init 6, 7; task 5 init 3, 5) were re-rendered with DP at obs 256: all four succeed, and the
  frames show a clean grasp, carry and release into the basket.
* Selection rule, fixed before 75k/100k are scored: best dev score, ties go to the earliest checkpoint.
  A later checkpoint can at best tie 100/100, so 50k is the pick and goes to held-out now; 75k and
  100k dev runs are a stability check only.
* Held-out 50k: **97/100**. vs ACT + aug (97): 3 fixed, 3 broken, p = 1.0. vs SmolVLA (97): 3 fixed,
  3 broken, p = 1.0. Three-way tie; dev 100/100 was partly luck of the dev draw.
* Failures, from the videos (all three are 280-step timeouts):
  - task 2 init 12: lifts the salad dressing bottle, it slips out, the arm carries nothing to the basket and hovers.
  - task 3 init 15: grasp closes beside the tall bottle (empty), arm goes to the basket and hovers.
  - task 8 init 11: knocks a tall bottle over, then wanders near it.
  Same failure class as ACT and SmolVLA: tall bottles, grasp robustness. No recovery behaviour after a
  missed grasp (demos never contain a retry), which explains the hovering.

## Cycle 8: demo-count study (ACT + augmentation, 10 and 20 demos per task)

* All three policies tie at 97/100 on held-out with ~45 demos per task, so the full-data benchmark no
  longer separates them. Next question: how much data does the winning recipe need?
* Same config as `act_obj_all_aug_r1` (via its train_config.json), only the episodes change: nested
  random subsets of 10 and 20 demos per task (`tools/demo_subset.py`, seed 0). 50k steps each.
  Dev at 25k and 50k, pick by best dev (ties to the earliest), held-out once on the pick.
  Runner: `scripts/demo_count_study.sh 10 20`.
* Protocol slip found (2026-11-01): the DP held-out run above used `--seed 3000` (one seed at start)
  instead of `--seeds 3000` (re-seed every episode), which ACT and SmolVLA used. Same init states,
  so 97/100 is a valid score, but its pairing with ACT/SmolVLA was not on identical episodes. Re-running
  with `--seeds 3000`; that result replaces this one. The same slip was fixed in the study runner
  before it started. Also: the watcher queued for the 75k/100k dev runs matched its own command line
  and never fired; re-queued as a script file.
