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
* Interrupted (2026-10-13): Claude Code stopped all background jobs because system RAM ran
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
