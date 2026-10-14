# Go2W gait: Walk These Ways command and contact schedule

Current retained result and commands: [Multiskill](multiskill_v1.md). Historical training logs and intermediate models have been removed; this document describes the technical mechanism.

Task: `OmniBot-Go2W-Gait-Flat-MLP-v0`.

Current implementation, completed evaluation and cleanup scope:
[Go2W 步态开发整理](go2w_gait_overview.md). Dated launch notes below
are historical records, not alternative current configurations.

The task is named Gait because it controls rolling, trot, bound and body height,
not a general multi-skill policy. Its configuration classes use `Go2WGait*` and
new logs use `omnibot_go2w_gait_flat_mlp_v0`. Historical run/snapshot/artifact paths
retain their original names. This rename does not change policy inputs, outputs,
reward weights or network parameters; existing gait checkpoints remain compatible.

This implements a shared rolling/trot/bound policy with the parameter-conditioned
contact schedule from **Walk These Ways: Tuning Robot Control for Generalization
with Multiplicity of Behavior** (Margolis and Agrawal, CoRL 2022). It ports the
locomotion interface/rewards to Go2W and Isaac Lab; it is not a claim to reproduce
the complete Go1 RMA policy, curriculum, dynamics, or experimental results.

## Evidence and conventions

Checked 2026-09-16:

- [Paper, arXiv v1, Appendix D](https://arxiv.org/html/2212.03238v1#A4): named-foot phase equation.
- [Conference PDF](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf): paper identity and method.
- [Official schedule implementation](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot.py): duration warp, sine clocks, normal-CDF contact smoothing.
- [Official rewards](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py): swing force, stance velocity, triangular clearance and Raibert targets.
- [Official training configuration](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py): duration 0.5, phase cost weights, disabled joint-default reward.

The paper and released source do not use identical phase parameterizations.
Upstream `play.py` also labels bound/pace differently from the default training
category code. We resolve this explicitly in favor of **Appendix D**, per the
requested paper convention. Do not substitute upstream play preset labels.

## Ten-dimensional command

The single command term is named `base_velocity` for compatibility with the
existing velocity tracking rewards, which read only columns 0:2 and 2.

| Column | Name | Units / convention |
|---|---|---|
| 0–2 | lin_vel_x, lin_vel_y, ang_vel_z | m/s, m/s, rad/s |
| 3 | body_height | absolute base height above plane, m |
| 4 | gait_frequency | cycles/s; continuously sampled in [1.5,4] for all three modes |
| 5–7 | gait_phase, gait_offset, gait_bound | paper theta1, theta2, theta3, in cycles; signed offsets accepted in [-1,1) |
| 8 | gait_duration | stance fraction in (0,1) for stepping; 1 for full-stance rolling |
| 9 | footswing_height | maximum wheel-bottom swing height, m |

There is no categorical gait observation. Height follows the paper's absolute
height convention. In the upstream code, column 3 is instead added to a nominal
height; upstream values must therefore be converted before using this interface.

Heading mode is enabled (`heading_command=True`) using the Go2W Flat convention:
80% of sampled environments receive a target heading in [-pi,pi], and their
column-2 yaw-rate command is updated as `clip(2 * wrap_to_pi(target-current),
-pi,pi)` every control step. The remaining 20% use directly sampled yaw rates.
The separate 10% standing mask has priority and forces all three velocities to
zero. Heading targets resample along with the other commands every 10 seconds.
The actor still sees a yaw-rate command, not an extra heading input. Explicit
`set_command(...)` calls retain direct yaw-rate semantics and disable automatic
heading/standing masks for the selected environments until reset/resampling.
This heading controller is a requested Go2W extension to the WTW interface.

Use the explicitly resolved foot order **FR, FL, RR, RL**:

```text
t <- (t + frequency * control_dt) mod 1
[t_FR, t_FL, t_RR, t_RL] =
  [t + theta2 + theta3, t + theta1 + theta3, t + theta1, t + theta2] mod 1
```

| Sampling center | theta1, theta2, theta3 | Synchronized feet at the center |
|---|---|---|
| trot | 0.5, 0, 0 | FR+RL; FL+RR |
| bound | 0, 0, 0.5 | FR+FL; RR+RL |
| rolling | 0, 0, 0 | all in stance; frequency in [1.5,4], duration=1, swing height=0 |

Every command resample adds uniform proposals in +/-5 degrees (5/360 cycles)
to each of the three phase parameters, for all three categories including
standing's rolling command. Proposals are rejected when any of the six wheel
pairs differs from its preset relative phase by more than 10 degrees, measured
with circular distance. These are `phase_jitter_deg=5.0` and
`max_pair_phase_error_deg=10.0`; setting jitter to zero restores exact presets.
The accepted distribution is uniform over the constrained neighborhood, not
three independent uniforms after rejection. Sampling has a 32-round retry bound
and raises an error rather than silently accepting an invalid sample.

Commands retain signed center-plus-offset values (e.g. -5/360 near zero), so
actor and critic do not see an artificial jump to 355/360 across zero. Only the
periodic schedule takes modulo one. Parameters stay fixed until the next command
resample/reset or explicit setter call; no per-step parameter noise is added.
Rewards and clock inputs use the actual sampled phases, not the preset center.
The neighborhood limits apply to training sampling; external `set_command` still
accepts general valid phase offsets and does not snap them to a named gait.

Both actor and critic receive the ten scaled commands and four sine clocks.
Stepping uses the released duration warp. Rolling retains the unwarped internal
phase because full stance has no swing interval, but its four clock observations
are zero for both actor and critic. No sin/cos duplication, one-hot or extra
contact-target observation is used. Existing proprioceptive and critic
observations are inherited from Go2W Flat; dimensions remain 64/318.

The duration warp maps stance to [0,0.5) and swing to [0.5,1). Contact targets use
the released normal CDF with kappa=0.07. Appendix D calls Phi a CDF but prints a
PDF formula; this port follows the **CDF implementation**, not that printed typo.
Periodic phases wrap modulo one, rather than saturating at one.

Rolling is selected by duration=1 and swing height=0, not by zero frequency.
Its internal phase advances at the sampled frequency while all four desired
contacts stay exactly one and all four observed clocks stay zero. Frequency is
an internal phase parameter in this mode, not a physical stepping frequency.
Small phase-command perturbations do not introduce rolling swing targets. The
10% standing samples use the same positive frequency distribution with zero
velocity, full support and masked clocks. Legacy
manual commands with frequency=0, duration=1 and swing height=0 remain valid,
but are no longer produced by training sampling. A zero swing height with
duration<1 still has a periodic swing-contact schedule; it is not automatically
converted into rolling.

This clock mask removes a periodic leg cue; it does not guarantee a previously
trained policy will stop shaking. The changed phase-command distribution and
rolling observations must be evaluated with a newly trained/fine-tuned policy.
Old checkpoints remain shape-compatible but have different observation semantics.

Reads do not advance time. CommandManager advances once per control step; the
command/schedule observed by the actor is also used to score that action's outcome.
Episode reset sets time to zero; ordinary 10-second resampling retains it. There
is no added transition state machine or mandatory parameter interpolation.

## Go2W adaptations

- Keep Flat action scales: twelve leg position targets at 0.25, four wheel
  velocity targets at 10.0. Leg action offsets remain the fixed asset defaults;
  height-conditioned IK is used only by rewards. No added action clipping or new motor model.
- Wheel cylinder radius 0.086 m, width 0.045 m, axial centers at +/-0.048 m,
  axis +Y, taken from the repository Go2W URDF. Use wheel-link origin velocities,
  not COM velocities. Tests check the dimensions and axes against the asset.
- The stance-velocity cost evaluates `v_link + omega_link cross r_contact` at
  the lowest wheel material point. Thus normal rolling is allowed and lateral
  slip is penalized. Upstream point-foot center velocity would prevent rolling.
- Clearance is measured at the wheel bottom, replacing the Go1 2 cm foot-radius
  correction. The phase trajectory and swing weighting follow the released code.
- Swing wheel-speed cost (-0.002) is a Go2W extension. It uses the smooth swing
  probability and vanishes in rolling mode.
- Raibert targets use Go2W nominal stance dimensions; disabled for full stance
  (duration=1) regardless of frequency, to allow continuous rolling.
- Height/velocity/angular-rate/clearance/phase costs use the released functional
  forms. Costs here are positive with negative weights (equivalent to upstream
  negative costs with positive weights). Height -50 (increased from -10 on
  2026-09-20), phase force/velocity -4/-4,
  clearance -30, Raibert -10, vertical velocity -0.02, roll/pitch rate -0.001,
  orientation -5. Hip-default L1 cost retains Go2W Flat's weight
  -0.5 and applies in every gait/phase against the reward-only height reference
  (the solved hip angles remain zero). Thigh/calf default and survival rewards
  are disabled.
  Standstill has weight -0.5 and penalizes leg deviation from the height-conditioned nominal pose
  only when the norm of the first three velocity commands is below 0.1 and
  duration is one. This is a requested Go2W addition; wheels are excluded.
- Other torque, limit, action and collision regularizers retain Go2W Flat values;
  this is not a literal copy of all Go1 reward weights or reward composition.
- Keep Flat randomization and 20 s episodes; each episode can contain the 10 s
  command switch. Standing is sampled with probability 0.10; otherwise rolling,
  trot and bound have equal probability. Labels are sampler metadata only.
- Speed ranges are x +/-1.0, y +/-0.5, yaw +/-pi. Absolute body height
  0.22–0.46 m (minimum matches Crawl), frequency 1.5–4 Hz, swing height
  0–0.10 m for stepping, duration fixed 0.5. Rolling retains swing height 0
  and duration 1 but samples the same 1.5–4 Hz frequency range.
  The paper's Go1 height/clearance extremes are not assumed feasible on Go2W.
- Existing global Flat velocity curriculum is disabled: it assumes a different
  command class and could promote all gaits on rolling-only success. The paper's
  gait-conditioned grid curriculum and RMA adaptation are not part of this port.

## Height-conditioned reward joint pose

Enabled by `commands.base_velocity.height_conditioned_reward_reference=True` (2026-09-20).
The original hip/thigh/calf reference `(0, 0.8, -1.5)` rad geometrically places
the base at about 0.40756 m when the four wheels touch a level plane. It remains
the fixed action/actor-observation reference at every target height; only the
reward reference is replaced with the height-matched pose.

The Go2W-specific analytical IK keeps the body level, hips at zero and each wheel
center's original base-frame x/y. It solves the negative-knee branch for wheel
center z = `wheel_radius - target_height`, using upper/lower lengths 0.213/0.2264 m
and radius 0.086 m. It does not prescribe a gait swing trajectory or provide
gravity/compliance compensation; the policy still learns joint targets relative
to the fixed asset defaults.

| Target height | Hip | Thigh | Calf |
|---|---|---|---|
| 0.22 m | 0 | 1.408996 | -2.523846 |
| 0.34 m | 0 | 1.025235 | -1.909676 |
| 0.46 m | 0 | 0.589986 | -1.105222 |

Angles are radians; all four legs have these values for the current symmetric
asset defaults. The reward-only per-environment reference is updated on command
sampling or manual command changes and is independent of gait phase:

- Leg action remains `q_target = q_asset_default + 0.25 * action`.
- Actor leg observation remains `q_actual - q_asset_default`, with unchanged noise,
  ordering and dimensions. Critic joint positions remain absolute, as before.
- Standstill/default-pose costs use `q_reward(height)`; reward weights and
  existing standstill gates are unchanged. No thigh/calf-default reward was added.
  Standstill is inactive during commanded motion/stepping, and the solved hip
  reference is still zero. Thus this change does not introduce a new thigh/calf
  pose cost during moving rolling, trot or bound; their height cost remains active.
- The immutable asset default pose and Flat reset/randomization events remain
  unchanged. In particular, this is not a height-matched physical reset: the
  initial body state still settles toward the sampled height command.
- Reachability and soft joint limits are checked for sampled/configured/manual
  heights. Invalid requests are rejected, not silently projected or action-clipped.
  A rejected manual command leaves both command and reference buffers unchanged.
- Command height changes still take effect immediately; no new interpolation,
  height curriculum or command ranges were introduced.

**Checkpoint/deployment compatibility:** actor/critic dimensions stay 64/318 and
the action dimension stays 16. Action and observation meanings are unchanged
from the phase-neighborhood/rolling-clock-mask training. Its existing
Compatible 64-input gait MLP checkpoints can be evaluated without a compatibility switch;
changing rewards does not improve an already-trained policy without further learning.
Height IK is not required in the deployment action or observation path.
Setting `height_conditioned_reward_reference=False` restores only the old fixed
reward targets, without changing the policy interface.

## Running and external control

Use the configured `robot_lab` environment after verifying its interpreter exists.
Do not resume an ordinary Flat checkpoint directly: the actor input dimension has
changed. Training from scratch is the default.

```sh
cd locolab
python scripts/locomotion/smoke_env.py --task OmniBot-Go2W-Gait-Flat-MLP-v0 --num_envs 3 --steps 64 --headless --device cuda:0
python scripts/locomotion/train.py --task OmniBot-Go2W-Gait-Flat-MLP-v0 --num_envs 4096 --max_iterations 5000 --headless --device cuda:0 --logger wandb --video
```

When resuming a historical gait run, keep the new task ID and supply
`--resume_experiment_name omnibot_go2w_multi_skill_flat_mlp_v0` together with
`--resume --load_run <existing-run> --checkpoint model_4999.pt`. This uses the old
log group only as the checkpoint source; new output uses the Gait log group.
For playback, pass the existing checkpoint's full path with `--checkpoint`.
Archived launch scripts below target their original snapshots and keep the old
task ID; they should not be used to launch this renamed working tree.

External callers supply all ten values. The setter validates finite values and
the rolling/periodic schedule contract, preserves the clock, recomputes targets
and suspends random resampling until episode reset:

```python
term = env.unwrapped.command_manager.get_term("base_velocity")
term.set_command([0.5, 0, 0, 0.42, 2.0, 0.5, 0, 0, 0.5, 0.08])  # trot
term.set_command([0.5, 0, 0, 0.42, 2.0, 0, 0, 0.5, 0.5, 0.08])  # bound
term.set_command([0.5, 0, 0, 0.42, 2.0, 0, 0, 0, 1, 0])        # rolling, same frequency
# Recompute observations before asking the policy for its next action.
obs = env.unwrapped.observation_manager.compute()
```

## Verification boundary

CPU tests exercise named-leg schedules, CDF smoothing, duration/clearance,
rolling edge cases, command changes and subset resets, force/velocity cost
direction, and wheel contact kinematics. Existing registration/runner regressions
are included. GPU training quality, actual contact timing and sim-to-real behavior
must be evaluated separately; a passing numerical test is not locomotion evidence.

At implementation time the specified `robot_lab` interpreter was not found in
the local Conda registrations or usual paths. CPU tests use the already installed
`env_isaaclab` Python without launching a simulator or allocating GPU training.
