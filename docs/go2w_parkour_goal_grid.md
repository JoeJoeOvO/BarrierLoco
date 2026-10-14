# Go2W goal-grid expansion

Baseline preserved on `feature/go2w-omnijump-mlp`; work proceeds on `codex/go2w-parkour-goal-grid`.
The retained checkpoint and seven-motion bank are unchanged.

The branch includes the retained seven-motion `model_4999` policy bundle and
the expanded bank/code. It does not contain a finalized 30-motion policy while
that training is ongoing. Binary references and the retained policy use scoped
Git LFS; run `git lfs pull` after cloning to obtain actual files, not pointers.

## Candidate references

The 27 requested flat-ground goals are the Cartesian product of base rise
15/20/25 cm, first-landing X 40/60/80 cm, and Y -15/0/15 cm. Entry speed remains 1 m/s.
The existing Go2W dynamics, .25/10 action scales, 23.5/23.5/35 Nm leg limits and
30/30/20 rad/s leg speed limits are retained.

Each goal is optimized with actuator/time variables and independently replayed.
Height error <=2 cm and landing error <=5 cm remain mandatory. Rejected
trajectories do not enter the new training bank.

The first search uses 128 candidates/goal and 80+40 iterations. If its accepted
references do not cover independent height/XY combinations, one bounded local
refinement uses 64 candidates/goal and 60+20 iterations. Its coordinate and
coupled front/hind proposals change controls only; acceptance is identical.
Every promoted candidate still receives a fresh single-environment video replay.

### Explicit goal-first acceptance v4

Both v3 searches finished with 0/27 accepted. Hind-last push timing alone rejected
every candidate; the retained user-approved flat motions also fail that old style
gate. The new flat grid opts into `criteria_version=4`; default v3, the retained
seven-entry bank and physical-gap/tilted-ramp acceptance are unchanged.

Hard checks retain H/XY accuracy, an observed first landing after >=0.20 s of
simultaneous four-wheel flight, peak simultaneous clearance >=8 cm, ordinary
stable rolling entry, COM takeoff vx >=0.8 m/s, all existing physical and body
attitude limits, <=0.30 rad landing pitch, <=40 ms rebound and >=0.15 s terminal
loaded stable recovery. Preparatory hops longer than 30 ms still fail.

Hind-last delay, fore-first landing order, airborne XY fractions, the virtual flat
crossing interval, precise apex contraction/foot-path/landing-extension/reach and
compression depth are optimization preferences, not admission gates. Real gap
and ramp geometry remains mandatory in its separate original metric functions.

Search ordering is accepted-first, physical safety, hard flight/recovery/attitude
violation, H/XY tolerance violation, then normalized H/XY error plus a small bounded
style cost. Per-preference contributions are capped; style weighs 0.05 versus
normalized target error. Valid v4 goals keep optimizing while other goals search.
No state trajectory is scaled, mirrored or retimed to satisfy a new command.

The new 128-candidate/goal, 80+40 local-refinement run starts from the best actual
controls among both previous libraries and the retained flat seeds, ranked by v4.
Stored states only select initial guesses: all 27 goals get fresh dynamics/video
replays with frozen source hashes. `goal_first_status.json` records this TO-only
workflow; it builds a bank only with sufficient independent coverage and does
not automatically start RL before review.

## Matched sampling

The schema-2 bank embeds the seven retained entries unchanged and appends only
accepted flat-ground replays. Public goals use measured height/first-touchdown XY.
No requested command is attached to an unrelated reference.

Allocation retains 30% rolling lanes. Among jumping references, 25% is reserved
for the seven retained motions, and 25% each for the three new height levels,
uniformly across accepted XY cells at that height. Flat tiles resample a compatible
teacher at reset. Physical gap and ramp references remain attached to their terrain.
Reset state, phase, command, reward reference and landing checks use the same ID.

The bank builder requires at least three accepted goals at each height and at
least four XY cells with multiple heights. Missing coverage stops the pipeline
before RL; it never falls back to the old seven-motion bank under the new task name.

## Training and validation

Task: `OmniBot-Go2W-Parkour-GoalGrid-MLP-v0`.
The actor remains a 57-input MLP, critic 312 inputs, actions 16; the jump interface
and fixed 2.2-second sequence remain unchanged. Speed sampling is [0.2,1.5] m/s.

After bank validation, the pipeline checks real reset/command/reference alignment
and first-landing accounting in 100 native environments, then discards an 8-update
PPO/video smoke run. Formal training starts with random weights: one GPU,
4096 environments, 5000 updates, online W&B and video. Smoke weights are never resumed.

The original solver/terrain/collision tests, variable-bank layout and sampling
tests, inactive-control preservation, schedule bounds and scratch initialization
checks cover the new code. The completed v4 library independently accepts 23/27
goals (9/7/7 at 15/20/25 cm) and produces a 30-motion bank including the retained
seven entries. The four rejected short diagonal goals are not sampled.

The 100-environment, 320-step native audit observed every motion ID, verified
reset/actor-command/critic-command/reference alignment and recorded 70 first
landings without repeated event rewards. The discarded 8-update PPO smoke
checkpoint is finite and its video was decoded successfully.

On 2026-10-13 the formal run starts from random actor/critic weights and an empty
optimizer on GPU 3, with 4096 environments and 5000 updates. The 30-motion bank
SHA-256 is `096cf1a0d110a09d903abcd1247154a95b63171ac510f1c7ff5643ea045c9c7d`.
Online W&B and videos (1000 steps / 20 s every 500 updates) are enabled.
Run: `2026-10-13_19-34-32_parkour_goalgrid_v4_30to_scratch_4096_gpu3_5k_video`.
[W&B run](https://wandb.ai/ssh45441-xi-an-jiaotong-university-/omnibot-go2w/runs/nsbqgei5).
`run_scripts/training_v4_status.json` records the controller state; each run's
`params/` contains initialization evidence, both reference banks, their hashes
and frozen training sources. Smoke weights are not loaded for formal training.

From an existing configured Isaac Lab environment, after `git lfs pull`:

```bash
cd locolab
python scripts/locomotion/train.py --task OmniBot-Go2W-Parkour-GoalGrid-MLP-v0 \
  --from_scratch --num_envs 4096 --max_iterations 5000 --logger wandb \
  --video --video_interval 500 --video_length 1000 --headless
```

Stable publication evidence is under `artifacts/go2w_parkour/20261013_goal_grid_v4/`.

Current experiment and controller records are under
`artifacts/go2w/parkour_goal_grid_20261013/` in the main workspace and
`OmniBot-go2w-parkour-goal-grid-20261013-r1/run_scripts/` on the server.
