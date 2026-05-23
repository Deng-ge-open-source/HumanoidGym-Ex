# Test Report

## 2026-05-22 Reward-Preserved IsaacGym/IsaacLab Alignment

This pass kept IsaacGym and IsaacLab reward names, default scales, and formulas aligned. No backend-specific default reward tuning was applied.

See also [MULTISEED_ALIGNMENT.md](MULTISEED_ALIGNMENT.md) for the `seed=1..5` long-training statistics.

Validation commands completed:

```bash
python -m py_compile humanoid_gym_ex/envs/robots/xbot/isaaclab_env.py humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py humanoid_gym_ex/scripts/check_reward_parity.py
python humanoid_gym_ex/scripts/check_reward_parity.py
python humanoid_gym_ex/scripts/compare_asset_summaries.py --isaacgym /tmp/hgex_asset_isaacgym.json --isaaclab /tmp/hgex_asset_isaaclab.json
bash humanoid_gym_ex/scripts/validate_smoke.sh
```

Results:

- `check_reward_parity.py`: `reward parity ok`.
- Full smoke validation passed, including IsaacGym train smoke, IsaacLab randomization smoke, IsaacLab rough terrain measured-heights smoke, IsaacLab PPO smoke, and IsaacLab play/export smoke.

The IsaacLab gait phase was aligned to the original Humanoid-Gym/IsaacGym reset timing by compensating DirectRLEnv's first-observation episode-length offset. The first random-rollout step now matches:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| first-step phase mean | 0.031250 | 0.031250 |
| first-step left stance rate | 1.000000 | 1.000000 |
| first-step right stance rate | 0.000000 | 0.000000 |

Random-action rollout, `64 envs x 240 steps`, seed `1`:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| reward/step | 0.030169 | 0.030331 |
| done/step | 0.262500 | 0.266667 |
| phase mean | 1.063871 | 1.073222 |
| feet contact rate | 0.699870 | 0.694531 |
| torque abs | 19.634413 | 19.848343 |

Seed `1`, plane, `64 envs x 60 steps x 200 iterations` training:

| Backend | Tail10 mean reward | Tail10 episode length | Final reward | Final episode length |
| --- | ---: | ---: | ---: | ---: |
| IsaacGym | 2.690 | 151.678 | 2.720 | 150.450 |
| IsaacLab | 2.997 | 155.626 | 2.880 | 149.290 |

Interpretation: the control-state and reward-input paths are now closely aligned, but 200-iteration PPO training still has about an 11% tail10 reward gap on seed `1`. The final iteration episode length is effectively matched, while tail10 episode length differs by about 2.6%. This remaining training-curve gap should be treated as a residual backend/physics and PPO stochasticity difference, not a reward-definition mismatch.

### Deterministic Action Trace Probe

Added and verified backend-neutral action trace diagnostics:

```bash
python humanoid_gym_ex/scripts/generate_action_trace.py --output /tmp/hgex_actions_seed1_120x64x12.npz --steps 120 --num_envs 64 --num_actions 12 --seed 1
python humanoid_gym_ex/scripts/compare_rollout_traces.py --left_summary /tmp/hgex_det_clean_isaacgym_120.json --right_summary /tmp/hgex_det_clean_isaaclab_120.json --left_trace /tmp/hgex_det_clean_isaacgym_120_trace.json --right_trace /tmp/hgex_det_clean_isaaclab_120_trace.json
```

In diagnostic mode, action delay and action noise are disabled. The processed action stream matches exactly across backends (`0.500890` mean abs in both). The first meaningful drift appears at step `7` in foot contact force/contact occupancy, while phase and action remain matched:

| Step | Field | IsaacGym | IsaacLab |
| --- | --- | ---: | ---: |
| 7 | processed_action_abs | 0.509216 | 0.509216 |
| 7 | phase_mean | 0.140625 | 0.140625 |
| 7 | feet_force_norm_mean | 13.872292 | 0.000000 |
| 7 | reward_mean | 0.044860 | 0.044545 |

This confirms the remaining early divergence is contact modeling/reporting, not reward or action preprocessing.

### Asset And Deterministic Reset Probe

Added asset/body diagnostics and `compare_asset_summaries.py`. The exported IsaacGym/IsaacLab XBot asset summaries show:

| Check | Result |
| --- | --- |
| body-name set | equal |
| joint-name set | equal |
| body order | different, mapped by name |
| joint order | different, canonicalized by name |
| total mass delta | `1.15e-7` |
| max inertia delta by body name | `2.38e-7` |
| shape count | `61` in both |
| friction/restitution/contact/rest offset | matched to numerical precision |

Added diagnostic-only deterministic reset support so fixed-action replay can remove reset joint perturbation as a confounder. With fixed actions and deterministic reset, the first drift still appears at step `7` in foot contact onset:

| Metric | IsaacGym | IsaacLab | Delta |
| --- | ---: | ---: | ---: |
| processed action abs | 0.500890 | 0.500890 | 0.000000 |
| reward/step | 0.030186 | 0.029907 | -0.000279 |
| done/step | 0.025000 | 0.033333 | 0.008333 |
| base_z | 0.841963 | 0.841706 | -0.000257 |
| base_vx | -0.197227 | -0.271509 | -0.074281 |
| feet contact rate | 0.676107 | 0.651367 | -0.024740 |
| feet force norm mean | 274.830286 | 258.765426 | -16.064861 |
| torque abs | 22.644442 | 22.704467 | 0.060025 |

Interpretation: the large static asset parameters are now aligned. The remaining nonzero gap is dominated by contact onset/reporting and later physics divergence, not reward definitions, action preprocessing, phase, mass, inertia, or nominal material settings.

## Environment Discovery

Date: 2026-05-21

Detected conda environments:

- `legged_gym`: Python 3.8.20, `isaacgym` available, `torch` available, `mujoco` available.
- `/home/cra02/anaconda3/envs/rlgpu`: Python 3.8.20, `isaacgym` available, `torch` available.
- `/home/cra02/anaconda3/envs/env_isaaclab`: Python 3.11.14, `isaaclab` available.
- Installing this package in `legged_gym` follows the upstream dependency pin and installs `mujoco==2.3.6`.

## Planned IsaacGym Baseline Tests

```bash
conda activate legged_gym
pip install -e .
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 512 --max_iterations 20 --run_name migration_check
python humanoid_gym_ex/scripts/play.py --task=humanoid_ppo --run_name migration_check
python humanoid_gym_ex/scripts/sim2sim.py --load_model logs/XBot_ppo/exported/policies/policy_example.pt
```

## Completed IsaacGym Tests

All commands below used:

```bash
conda run -n legged_gym
WANDB_MODE=offline
```

### Import And Config

Command:

```bash
python -c "import humanoid_gym_ex; from humanoid_gym_ex.envs import task_registry; env_cfg, train_cfg = task_registry.get_cfgs('humanoid_ppo'); print(env_cfg.env.num_actions, env_cfg.env.num_observations, env_cfg.env.num_privileged_obs)"
```

Result:

```text
task ['humanoid_ppo']
num_actions 12
num_obs 705
num_privileged_obs 219
experiment XBot_ppo
```

### One Iteration Smoke Test

Command:

```bash
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

### Upstream One Iteration Comparison

Command:

```bash
PYTHONPATH=/tmp/humanoid-gym-upstream python humanoid/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1
```

Result: matched HumanoidGym-Ex exactly for model dimensions, losses, mean reward, mean episode length, and reward terms. FPS differed only by normal runtime variance.

### 20 Iteration Comparison

HumanoidGym-Ex command:

```bash
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 20 --run_name migration_compare_64_20
```

Upstream command:

```bash
PYTHONPATH=/tmp/humanoid-gym-upstream python humanoid/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 20 --run_name migration_compare_64_20
```

Final iteration comparison:

| Metric | HumanoidGym-Ex | Upstream Humanoid-Gym |
| --- | ---: | ---: |
| iteration | 19/20 | 19/20 |
| total timesteps | 76800 | 76800 |
| mean reward | 1.90 | 1.90 |
| mean episode length | 127.57 | 127.57 |
| value function loss | 0.0185 | 0.0185 |
| surrogate loss | -0.0097 | -0.0097 |
| mean action noise std | 0.99 | 0.99 |

Reward terms at the final iteration also matched to the printed precision. FPS and total wall time differed slightly, which is expected.

### Play / Export Smoke Test

Command:

```bash
HUMANOID_GYM_EX_RENDER=0 HUMANOID_GYM_EX_PLOT_STATES=0 HUMANOID_GYM_EX_PLAY_STEPS=10 \
python humanoid_gym_ex/scripts/play.py --task=humanoid_ppo --headless
```

Result:

```text
loaded logs/XBot_ppo/May21_04-31-12_migration_compare_64_20/model_20.pt
exported logs/XBot_ppo/exported/policies/policy_1.pt
completed 10 policy steps
```

### Sim2Sim Artifact Check

Command:

```bash
python -c "import torch, mujoco; torch.jit.load('logs/XBot_ppo/exported/policies/policy_example.pt'); print(mujoco.__version__)"
```

Result:

```text
policy_example.pt loaded
mujoco 2.3.6
```

The full `sim2sim.py` viewer run was not launched in this pass because it runs a 60 second MuJoCo viewer session.

## Completed IsaacLab Discovery And Smoke

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python -c "import isaaclab; from humanoid_gym_ex.envs.backends.isaaclab_backend import IsaacLabBackend; print(IsaacLabBackend.__name__)"
```

Result:

```text
isaaclab import ok
IsaacLabBackend
```

IsaacLab training is not implemented in Phase 0. This check confirms that the IsaacLab environment can import the future backend placeholder without requiring IsaacGym.

### IsaacLab Direct XBot Smoke

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/isaaclab_smoke.py --headless --num_envs 2 --steps 2
```

Filtered result:

```text
[HumanoidGym-Ex] IsaacLab reset policy obs: (2, 705)
[HumanoidGym-Ex] IsaacLab reset critic obs: (2, 219)
[HumanoidGym-Ex] step=0 reward_mean=1.818731 done_count=0 policy_obs=(2, 705)
[HumanoidGym-Ex] step=1 reward_mean=1.818731 done_count=0 policy_obs=(2, 705)
```

Notes:

- XBot-L URDF import succeeds through `sim_utils.UrdfFileCfg`.
- Contact sensor initialization requires `activate_contact_sensors=True` on the spawn config.
- IsaacLab logs many URDF merge warnings for fixed links in the original XBot model; the smoke run still exits successfully.
- This is Direct workflow, not Manager-based workflow.
- This is not full IsaacLab PPO parity yet.

### IsaacLab PPO Smoke

Small no-log data-flow command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 2 --num_steps_per_env 4 --max_iterations 1 --no_log
```

Result: completed rollout and PPO update without writing logs/checkpoints.

Logged short command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 2 --num_steps_per_env 4 --max_iterations 1 --run_name isaaclab_ppo_smoke
```

Filtered result:

```text
Learning iteration 0/1
Value function loss: 0.4905
Surrogate loss: -0.0235
Mean action noise std: 1.00
Total timesteps: 8
log_dir: logs/XBot_isaaclab/May21_14-49-22_isaaclab_ppo_smoke
```

IsaacGym-scale rollout command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 --max_iterations 1 --run_name isaaclab_ppo_64x60
```

Filtered result before reward parity:

```text
Learning iteration 0/1
Value function loss: 42.0366
Surrogate loss: -0.0071
Mean action noise std: 1.00
Total timesteps: 3840
log_dir: logs/XBot_isaaclab/May21_14-50-35_isaaclab_ppo_64x60
```

Comparison note: this matches the IsaacGym one-iteration rollout size (`3840` timesteps), but the loss/reward numbers are not expected to match yet because the IsaacLab env currently implements only a minimal reward subset and lacks full reset, randomization, and reward parity.

### IsaacLab PPO Smoke After Reward Parity Pass

The IsaacLab env now dispatches reward functions by the original XBot `cfg.rewards.scales` names and logs the same reward term keys as IsaacGym.

Small logged command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 2 --num_steps_per_env 4 --max_iterations 1 --run_name isaaclab_reward_parity_smoke
```

Filtered result:

```text
Learning iteration 0/1
Value function loss: 0.0042
Surrogate loss: -0.0306
Mean action noise std: 1.00
Mean episode rew_tracking_lin_vel: 0.0006
Mean episode rew_orientation: 0.0009
Mean episode rew_torques: -0.0008
Total timesteps: 8
log_dir: logs/XBot_isaaclab/May21_15-02-19_isaaclab_reward_parity_smoke
```

IsaacGym-scale command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 --max_iterations 1 --run_name isaaclab_reward_parity_64x60
```

Filtered result:

```text
Learning iteration 0/1
Value function loss: 0.0017
Surrogate loss: -0.0083
Mean action noise std: 1.00
Mean episode rew_dof_acc: -0.1131
Mean episode rew_orientation: 0.0045
Mean episode rew_torques: -0.0157
Mean episode rew_tracking_lin_vel: 0.0076
Total timesteps: 3840
log_dir: logs/XBot_isaaclab/May21_15-02-47_isaaclab_reward_parity_64x60
```

Comparison note: IsaacLab now logs the same reward term names as IsaacGym, but the numerical values are not yet expected to match. Remaining differences come from reset/randomization gaps, IsaacLab URDF-to-USD import behavior, contact-sensor differences, and simulator dynamics.

### IsaacLab Randomization Smoke

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/isaaclab_smoke.py --headless --num_envs 2 --steps 2 --check_randomization
```

Filtered result:

```text
[HumanoidGym-Ex] IsaacLab reset policy obs: (2, 705)
[HumanoidGym-Ex] IsaacLab reset critic obs: (2, 219)
[HumanoidGym-Ex] randomization friction=True mass=True push_xy_mean=0.075735
[HumanoidGym-Ex] step=0 reward_mean=0.034671 done_count=0 policy_obs=(2, 705)
[HumanoidGym-Ex] step=1 reward_mean=0.036410 done_count=0 policy_obs=(2, 705)
```

This confirms that IsaacLab friction and base-mass randomization write into PhysX views successfully, and root-velocity push produces a non-zero perturbation.

### IsaacLab PPO Smoke After Randomization Pass

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 --max_iterations 1 --run_name isaaclab_rand_64x60
```

Filtered result:

```text
Learning iteration 0/1
Value function loss: 0.0012
Surrogate loss: -0.0080
Mean action noise std: 1.00
Mean episode rew_orientation: 0.0045
Mean episode rew_torques: -0.0139
Mean episode rew_tracking_lin_vel: 0.0074
Total timesteps: 3840
log_dir: logs/XBot_isaaclab/May21_21-07-25_isaaclab_rand_64x60
```

## Completed Phase 1 IsaacGym Backend Regression

Phase 1 introduced `IsaacGymBackend` as a thin adapter and routed low-level simulator calls through it while preserving upstream reward/obs/reset/command code.

### One Iteration Backend Regression

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name phase1_backend_smoke
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

This matches the Phase 0 and upstream one-iteration result.

### 20 Iteration Backend Regression

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 20 --run_name phase1_backend_compare_64_20
```

Final iteration:

| Metric | Phase 1 Backend | Phase 0 / Upstream Baseline |
| --- | ---: | ---: |
| iteration | 19/20 | 19/20 |
| total timesteps | 76800 | 76800 |
| mean reward | 1.90 | 1.90 |
| mean episode length | 127.57 | 127.57 |
| value function loss | 0.0185 | 0.0185 |
| surrogate loss | -0.0097 | -0.0097 |
| mean action noise std | 0.99 | 0.99 |

Reward terms matched the baseline to printed precision. FPS and wall time varied within normal runtime noise.

### Play / Export Backend Regression

Command:

```bash
HUMANOID_GYM_EX_RENDER=0 HUMANOID_GYM_EX_PLOT_STATES=0 HUMANOID_GYM_EX_PLAY_STEPS=10 \
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/play.py --task=humanoid_ppo --headless
```

Result:

```text
loaded logs/XBot_ppo/May21_14-00-32_phase1_backend_compare_64_20/model_20.pt
exported logs/XBot_ppo/exported/policies/policy_1.pt
completed 10 policy steps
```

## Completed Phase 2 IsaacGym Regression

After adding the IsaacLab Direct smoke path, the IsaacGym training path was re-tested.

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name phase2_regression_smoke
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

This matches the upstream, Phase 0, and Phase 1 one-iteration baseline.

### IsaacGym Regression After IsaacLab PPO Wrapper

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name phase2_isaaclab_ppo_regression
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

This confirms the IsaacLab PPO adapter did not change the default IsaacGym training behavior.

### IsaacGym Regression After IsaacLab Reward Parity Pass

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name phase2_reward_parity_regression
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

This still matches the upstream one-iteration baseline.

### IsaacGym Regression After IsaacLab Randomization Pass

Command:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name phase2_rand_regression
```

Result:

```text
completed
total timesteps: 3840
mean reward: 0.60
mean episode length: 36.00
value function loss: 0.0144
surrogate loss: -0.0052
```

Default IsaacGym behavior remains unchanged.

## IsaacLab Play / Export

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/play_isaaclab.py --headless --steps 2 --load_run May21_21-27-13_play_source --export_policy --fix_command
```

Filtered result:

```text
[HumanoidGym-Ex] exported IsaacLab policy: logs/XBot_isaaclab/exported/policies/policy_isaaclab.pt
[HumanoidGym-Ex] IsaacLab play steps=2 reward_mean=0.028599 checkpoint=logs/XBot_isaaclab/May21_21-27-13_play_source/model_1.pt
```

## Automatic Validation

Command:

```bash
bash humanoid_gym_ex/scripts/validate_smoke.sh
```

Result:

```text
[validate] compile
[validate] isaacgym train smoke
[validate] isaaclab randomization smoke
[validate] isaaclab rough terrain heights smoke
[validate] isaaclab ppo smoke
[validate] isaaclab play/export smoke
[validate] ok
```

The validation script checks:

- Python compilation under the IsaacGym environment.
- IsaacGym one-iteration smoke with `3840` total timesteps and the expected `0.60` mean reward.
- IsaacLab randomization smoke with `705` policy observations and successful friction/mass randomization.
- IsaacLab rough terrain and measured-heights smoke with original-width `705` policy observations and `780` critic observations.
- IsaacLab PPO smoke checkpoint creation.
- IsaacLab checkpoint load, policy rollout, and TorchScript export.

## IsaacLab Rough Terrain And Measured Heights

Command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/isaaclab_smoke.py --headless --num_envs 4 --steps 2 \
  --terrain rough --measure_heights --terrain_curriculum
```

Filtered result:

```text
[HumanoidGym-Ex] IsaacLab reset policy obs: (4, 705)
[HumanoidGym-Ex] IsaacLab reset critic obs: (4, 780)
[HumanoidGym-Ex] terrain=rough measure_heights=True terrain_curriculum=True
[HumanoidGym-Ex] step=0 reward_mean=0.031679 done_count=0 policy_obs=(4, 705) height_mean=0.045764
[HumanoidGym-Ex] step=1 reward_mean=0.028605 done_count=0 policy_obs=(4, 705) height_mean=0.045815
```

Implementation notes:

- `--terrain rough`, `--terrain heightfield`, and `--terrain trimesh` map to IsaacLab `TerrainImporterCfg(terrain_type="generator")` with `ROUGH_TERRAINS_CFG`.
- `--terrain_curriculum` uses `TerrainImporter.update_env_origins(...)` during reset.
- `--measure_heights` attaches an IsaacLab `RayCaster` to `base_link`.
- The original XBot plane observations are `47 x 15 = 705` policy and `73 x 3 = 219` critic. The original Humanoid-Gym XBot rough measured-height mode keeps the actor at `47 x 15 = 705` and appends `187` height samples per frame only to the critic, giving `(73 + 187) x 3 = 780`.

Short PPO data-flow command:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 16 --num_steps_per_env 24 \
  --max_iterations 2 --run_name rough_heights_smoke --terrain rough --measure_heights --terrain_curriculum --no_log
```

Result: completed rollout and PPO update without simulator or shape errors.

## IsaacGym vs IsaacLab 50-Iteration Curve Check

Commands:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 50 \
  > /tmp/hgex_isaacgym_50.log 2>&1

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 \
  --max_iterations 50 --run_name curve_align_plane_50_jointorder --terrain plane \
  > /tmp/hgex_isaaclab_plane_50_jointorder.log 2>&1

python humanoid_gym_ex/scripts/compare_training_curves.py \
  --isaacgym-log /tmp/hgex_isaacgym_50.log \
  --isaaclab-log /tmp/hgex_isaaclab_plane_50_jointorder.log \
  --csv /tmp/hgex_curve_compare_50_jointorder.csv
```

Summary:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 50 | 192000 | 1.8500 | 126.0900 | 0.014672 | 1.9700 | 128.5520 | 0.015325 |
| IsaacLab | 50 | 192000 | 2.4400 | 150.8300 | 0.016177 | 2.4320 | 149.4720 | 0.016271 |

Interpretation:

- This is no longer a smoke-only check; both backends completed the same `64 envs x 60 steps x 50 iterations` rollout budget.
- IsaacGym remains the numerical compatibility baseline and still matches upstream Humanoid-Gym on the earlier 20-iteration comparison.
- IsaacLab Direct is stable through the longer run. After joint-order canonicalization, tail10 reward/step is `0.016271` vs IsaacGym `0.015325`.
- The major fix was recomputing IsaacLab PD torques on every physics substep inside `_apply_action`, matching IsaacGym's decimation loop. Before this, IsaacLab held one torque for all 10 substeps and produced excessive `dof_acc` and torque penalties.
- IsaacLab termination now uses the max over `ContactSensor.data.net_forces_w_history` for termination bodies, instead of only the latest sensor sample. This reduced the final episode length from `158.6900` to `149.6200`.
- IsaacLab joint state/action/reward tensors are canonicalized to IsaacGym joint order. IsaacLab imports joints and bodies in a left/right interleaved order, while IsaacGym exposes all left-leg joints before right-leg joints.
- Remaining difference: IsaacLab episodes still run longer (`149.472` tail10 length vs `128.552`). Random-action contact diagnostics below show similar termination-contact rates, so the remaining gap is more likely policy-after-training dynamics and PhysX/contact behavior than a simple contact-force threshold mismatch.

Additional fix after this run: `compute_ref_state()` now keeps `ref_dof_pos` in the same IsaacGym canonical joint order as `dof_pos`, observations, and reward functions. Before that fix, IsaacLab canonicalized the live joint tensors but rebuilt the reference pose in IsaacLab sim-order.

## IsaacGym vs IsaacLab 200-Iteration Curve Check

Commands:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 200 \
  --run_name curve_align_plane_200_after_reforder \
  > /tmp/hgex_isaacgym_200_after_reforder.log 2>&1

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 \
  --max_iterations 200 --run_name curve_align_plane_200_after_reforder --terrain plane \
  > /tmp/hgex_isaaclab_200_after_reforder.log 2>&1
```

Summary:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 200 | 768000 | 2.8600 | 153.3100 | 0.018655 | 2.8580 | 151.7750 | 0.018831 |
| IsaacLab | 200 | 768000 | 3.7900 | 196.1700 | 0.019320 | 3.7310 | 188.2990 | 0.019814 |

Interpretation:

- The reward-rate gap is moderate: IsaacLab tail10 reward/step is about `5.2%` higher than IsaacGym.
- The episode-length gap is not negligible: IsaacLab tail10 episode length is about `24.1%` longer.
- This means the current default IsaacLab backend is suitable for API/authoring compatibility and smoke-to-medium PPO validation, but should not yet be advertised as numerically equivalent to IsaacGym for long training curves.
- The strongest remaining suspect is simulator/import/contact behavior, not reward naming or PPO plumbing: body count, mass, joint mapping, reward dispatch, and trained-policy replay are now aligned closely enough to expose the residual dynamics gap.

Optional strict termination remains an experiment, not the default. `--parity_termination_profile isaacgym_like` currently maps to `--termination_base_height 0.80`. A 50-iteration scan showed that this can reduce IsaacLab episode length, but it raises reward/step and is therefore not a physical parity fix:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 \
  --max_iterations 50 --run_name termscan_base080 --terrain plane --termination_base_height 0.80
```

| Backend | final episode length | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: |
| IsaacGym baseline | 126.0900 | 128.5520 | 0.015325 |
| IsaacLab default before ref-order fix | 150.8300 | 149.4720 | 0.016271 |
| IsaacLab `--termination_base_height 0.80` | 120.7700 | 117.6940 | 0.020791 |

This is useful as a comparability experiment, but the elevated reward/step means it should remain an explicit opt-in.

### IsaacLab Shape/Material Parity Pass

The IsaacLab env now explicitly writes actor rigid-shape defaults to match the IsaacGym XBot baseline before optional domain randomization:

- static friction: `1.0`
- dynamic friction: `1.0`
- restitution: `0.0`
- contact offset: `0.01`
- rest offset: `0.0`

The diagnostic path also disables domain randomization before env initialization, so shape/material summaries reflect nominal values rather than random friction samples.

Replay summary with the same IsaacGym `model_50.pt` checkpoint:

| Metric | IsaacGym | IsaacLab after shape/material parity |
| --- | ---: | ---: |
| reward/step | 0.029891 | 0.026446 |
| done_count / step | 0.323333 | 0.318333 |
| base_z | 0.826206 | 0.831702 |
| dof_vel_abs | 0.613922 | 0.517321 |
| torque_abs | 15.962492 | 14.182302 |
| termination_contact_gt_1.0 | 0.005052 | 0.004687 |

50-iteration curve after shape/material parity:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 50 | 192000 | 1.8500 | 126.0900 | 0.014672 | 1.9700 | 128.5520 | 0.015325 |
| IsaacLab shape/material parity | 50 | 192000 | 2.6400 | 159.6800 | 0.016533 | 2.3720 | 150.6290 | 0.015747 |

Interpretation:

- Shape/material parity improved the reward-rate alignment: tail10 reward/step is now `0.015747` vs IsaacGym `0.015325`, about `2.8%` high.
- Episode length is still materially longer: `150.629` vs `128.552`, about `17.2%` high.
- Therefore this pass fixed part of the contact/material mismatch, but the remaining long-episode behavior likely needs dedicated termination/contact timing diagnostics rather than more reward tuning.

### IsaacLab PhysX Solver Parity Pass

After the shape/material pass, trained-policy termination diagnostics showed that termination causes already matched: both backends ended episodes through `base_link` contact, with no timeout, base-height, or orientation termination. The remaining mismatch came from trajectory dynamics: the same IsaacLab-trained policy survived much longer in IsaacLab than in IsaacGym.

The IsaacLab XBot config now aligns the main IsaacGym PhysX settings:

- global solver type: TGS (`1`)
- position iterations: `4`
- velocity iterations: `1`
- bounce threshold velocity: `0.1`
- GPU rigid contact buffer: `2**23`
- actor max depenetration velocity: `1.0`
- articulation/body solver iteration counts: `4/1`

Replay of the same IsaacLab-trained `model_50.pt` for 600 steps:

| Metric | IsaacGym | IsaacLab before PhysX solver parity | IsaacLab after PhysX solver parity |
| --- | ---: | ---: | ---: |
| reward/step | 0.029556 | 0.029883 | 0.029268 |
| done_count / step | 0.425000 | 0.320000 | 0.413333 |
| termination events | 255 | 192 | 248 |
| contact terminations | 255 | 192 | 248 |
| episode length mean | 138.114 | 167.849 | 143.238 |
| episode length p50 | 137 | 168 | 142 |
| base_z at termination mean | 0.214107 | 0.297925 | 0.232337 |

50-iteration curve after PhysX solver parity:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 50 | 192000 | 1.8500 | 126.0900 | 0.014672 | 1.9700 | 128.5520 | 0.015325 |
| IsaacLab PhysX solver parity | 50 | 192000 | 2.2300 | 141.5900 | 0.015750 | 2.1150 | 137.4140 | 0.015391 |

Interpretation:

- Reward-rate alignment is now very close: IsaacLab tail10 reward/step is `0.015391` vs IsaacGym `0.015325`, about `0.43%` high.
- Episode-length alignment improved substantially: IsaacLab tail10 episode length is `137.414` vs IsaacGym `128.552`, about `6.9%` high, down from `17.2%`.
- This is no longer a large first-order mismatch for short 50-iteration checks, but it is still not exact equivalence. The next evidence needed is a repeated long run after PhysX parity, using the same 200-iteration protocol as the earlier comparison.

200-iteration curve after PhysX solver parity:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 \
  --max_iterations 200 --run_name curve_align_plane_200_physxalign --terrain plane \
  > /tmp/hgex_isaaclab_200_physxalign.log 2>&1
```

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 200 | 768000 | 2.8600 | 153.3100 | 0.018655 | 2.8580 | 151.7750 | 0.018831 |
| IsaacLab PhysX solver parity | 200 | 768000 | 2.6300 | 148.4100 | 0.017721 | 2.6710 | 153.0040 | 0.017457 |

Interpretation:

- The previous long-run episode-length mismatch is effectively fixed: IsaacLab tail10 episode length is now `153.004` vs IsaacGym `151.775`, about `0.8%` high. Before PhysX solver parity it was `188.299`, about `24.1%` high.
- The remaining long-run mismatch has moved to reward/learning dynamics: IsaacLab tail10 reward/step is now about `7.3%` lower than IsaacGym (`0.017457` vs `0.018831`).
- This means termination/contact timing is no longer the main blocker for the 200-iteration protocol. The next investigation should compare reward-term distributions and learned-policy replay across both backends after 200 iterations.

Cross-backend replay of 200-iteration checkpoints:

```bash
HGEX_DIAG_STEPS=600 HGEX_DIAG_CHECKPOINT=logs/XBot_ppo/May22_00-29-34_curve_align_plane_200_after_reforder/model_200.pt \
HGEX_DIAG_FIX_COMMAND=1 HGEX_DIAG_OUTPUT=/tmp/hgex_replay_isaacgym_gympolicy200.json \
HGEX_DIAG_TRACE_OUTPUT=/tmp/hgex_replay_isaacgym_gympolicy200_trace.json \
conda run -n legged_gym python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 600 \
  --checkpoint_path logs/XBot_ppo/May22_00-29-34_curve_align_plane_200_after_reforder/model_200.pt \
  --fix_command --output /tmp/hgex_replay_isaaclab_gympolicy200.json \
  --trace_output /tmp/hgex_replay_isaaclab_gympolicy200_trace.json
```

The same commands were repeated with `logs/XBot_isaaclab/May22_02-05-15_curve_align_plane_200_physxalign/model_200.pt`.

| Policy | Backend | reward/step | done/step | events | event len mean | event len p50 | base_z avg | dof_vel_abs | torque_abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym 200 | IsaacGym | 0.031636 | 0.326667 | 196 | 163.490 | 161 | 0.809198 | 0.484421 | 14.578374 |
| IsaacGym 200 | IsaacLab | 0.030737 | 0.338333 | 203 | 158.611 | 155 | 0.809344 | 0.497378 | 14.305114 |
| IsaacLab 200 | IsaacGym | 0.031747 | 0.426667 | 256 | 138.859 | 138 | 0.747030 | 0.584773 | 13.811639 |
| IsaacLab 200 | IsaacLab | 0.032161 | 0.420000 | 252 | 142.861 | 142 | 0.751509 | 0.557380 | 13.545282 |

Interpretation:

- Cross-backend execution is now close for the same policy. The IsaacGym-trained policy has `0.326667` done/step in IsaacGym and `0.338333` in IsaacLab; the IsaacLab-trained policy has `0.426667` in IsaacGym and `0.420000` in IsaacLab.
- The remaining difference is mainly policy quality/gait, not a backend termination bug. The IsaacLab-trained policy falls more often in both simulators and runs at lower average base height.
- Reward term comparison inside each backend points to the same pattern: the IsaacLab-trained policy trades lower `joint_pos` and `feet_contact_number` reward for better `tracking_ang_vel` and smaller contact/torque penalties. That explains why short replay reward can look similar while long training reward/step is lower.

Seed parity fix:

- IsaacGym training calls `set_seed(env_cfg.seed)` before environment creation.
- IsaacLab training now mirrors this without importing IsaacGym-only helpers: `train_isaaclab.py` sets Python, NumPy, Torch, CUDA, `PYTHONHASHSEED`, and `env_cfg.seed` before constructing `XBotIsaacLabEnv`.
- Smoke validation confirms IsaacLab reports `Environment seed: 5` and no longer logs `Seed not set`.

50-iteration seeded check after this fix:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 50 | 192000 | 1.8500 | 126.0900 | 0.014672 | 1.9700 | 128.5520 | 0.015325 |
| IsaacLab seeded PhysX parity | 50 | 192000 | 2.1300 | 137.1000 | 0.015536 | 2.0540 | 137.4340 | 0.014945 |

Interpretation: seed parity is a reproducibility fix, not a direct learning-quality fix. The 50-iteration seeded run keeps episode length near the previous PhysX parity result, while reward/step moves slightly below the IsaacGym baseline. The next learning-quality target remains PPO/randomization and reward-term sensitivity rather than termination behavior.

Friction randomization parity fix:

- IsaacGym samples `256` friction buckets at environment creation, then assigns one bucket id per environment. Friction is not resampled on every reset.
- IsaacLab now mirrors this: `_randomize_friction()` creates one `256`-bucket table and per-env bucket ids, writes the chosen value to both static and dynamic material friction, and keeps `env_frictions` aligned with the sampled bucket value.
- Base-mass randomization and root reset velocity timing already matched the original behavior: both are one-time initialization/reset writes rather than per-step randomization, and reset root velocity remains the default zero state for the non-fixed-base XBot task.
- Full smoke validation passed after the bucketed friction change.

50-iteration seeded bucket-friction check:

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 50 | 192000 | 1.8500 | 126.0900 | 0.014672 | 1.9700 | 128.5520 | 0.015325 |
| IsaacLab seeded bucket friction | 50 | 192000 | 2.0900 | 136.3200 | 0.015332 | 2.1110 | 136.9760 | 0.015411 |

Interpretation: bucketed friction improves the 50-iteration reward-rate match compared with the seeded-only run. Tail10 reward/step is now about `0.56%` higher than IsaacGym (`0.015411` vs `0.015325`). Episode length remains about `6.6%` higher at 50 iterations.

200-iteration seeded bucket-friction check:

```bash
PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex WANDB_MODE=offline \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 64 --num_steps_per_env 60 \
  --max_iterations 200 --run_name curve_align_plane_200_seeded_bucketfriction --terrain plane \
  > /tmp/hgex_isaaclab_200_seeded_bucketfriction.log 2>&1
```

| Backend | iterations | final timesteps | final reward | final episode length | final reward/step | tail10 reward | tail10 episode length | tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 200 | 768000 | 2.8600 | 153.3100 | 0.018655 | 2.8580 | 151.7750 | 0.018831 |
| IsaacLab seeded bucket friction | 200 | 768000 | 2.9900 | 154.6500 | 0.019334 | 2.9990 | 159.8010 | 0.018767 |

Interpretation:

- The previous 200-iteration reward-rate gap is effectively fixed: IsaacLab tail10 reward/step is now only about `0.34%` lower than IsaacGym (`0.018767` vs `0.018831`). Before the seed and friction-bucket parity fixes, IsaacLab was about `7.3%` lower.
- Episode length is still higher at 200 iterations: `159.801` vs `151.775`, about `5.3%` high. This is much smaller than the pre-PhysX-parity `24.1%` gap, but still worth tracking.
- The remaining mismatch is now mostly gait/episode-length behavior, not reward-rate scale.

Cross-backend replay with the latest seeded bucket-friction IsaacLab `model_200.pt`:

```bash
HGEX_DIAG_STEPS=600 \
HGEX_DIAG_CHECKPOINT=logs/XBot_isaaclab/May22_10-55-43_curve_align_plane_200_seeded_bucketfriction/model_200.pt \
HGEX_DIAG_FIX_COMMAND=1 \
HGEX_DIAG_OUTPUT=/tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction.json \
HGEX_DIAG_TRACE_OUTPUT=/tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction_trace.json \
conda run -n legged_gym python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 600 \
  --checkpoint_path logs/XBot_isaaclab/May22_10-55-43_curve_align_plane_200_seeded_bucketfriction/model_200.pt \
  --fix_command \
  --output /tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction.json \
  --trace_output /tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction_trace.json
```

| Policy | Backend | reward/step | done/step | events | event len mean | event len p50 | base_z avg | dof_vel_abs | torque_abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym 200 | IsaacGym | 0.031636 | 0.326667 | 196 | 163.490 | 161 | 0.809198 | 0.484421 | 14.578374 |
| IsaacGym 200 | IsaacLab | 0.030737 | 0.338333 | 203 | 158.611 | 155 | 0.809344 | 0.497378 | 14.305114 |
| IsaacLab seeded bucket friction 200 | IsaacGym | 0.033911 | 0.423333 | 254 | 137.327 | 135 | 0.780359 | 0.557796 | 13.479199 |
| IsaacLab seeded bucket friction 200 | IsaacLab | 0.035351 | 0.421667 | 253 | 138.968 | 138 | 0.784142 | 0.529198 | 12.691100 |

Interpretation:

- The latest IsaacLab-trained policy transfers consistently between backends: `0.423333` done/step on IsaacGym and `0.421667` on IsaacLab, with all recorded termination events caused by contact termination rather than timeout, base-height, or orientation checks.
- This confirms that the remaining fixed-command replay difference is not primarily a backend execution mismatch. The IsaacLab policy still falls more often than the IsaacGym policy under the fixed replay command, even though the 200-iteration training reward/step is now aligned.
- Treat the 200-iteration reward curve as close enough for the current migration milestone, but do not treat fixed-command policy behavior as fully equivalent yet. The next useful work is gait/contact robustness alignment, not another broad backend abstraction pass.

Replay robustness analysis pass:

```bash
python humanoid_gym_ex/scripts/analyze_replay_alignment.py \
  --gym_policy_on_gym /tmp/hgex_replay_isaacgym_gympolicy200_v2.json \
  --gym_policy_on_lab /tmp/hgex_replay_isaaclab_gympolicy200_v2.json \
  --lab_policy_on_gym /tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction_v2.json \
  --lab_policy_on_lab /tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction_v2.json \
  --gym_policy_on_gym_trace /tmp/hgex_replay_isaacgym_gympolicy200_v2_trace.json \
  --gym_policy_on_lab_trace /tmp/hgex_replay_isaaclab_gympolicy200_v2_trace.json \
  --lab_policy_on_gym_trace /tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction_v2_trace.json \
  --lab_policy_on_lab_trace /tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction_v2_trace.json \
  --output docs/REPLAY_ALIGNMENT.md
```

Key v2 metrics after adding raw-action, base velocity, and foot-contact diagnostics:

| Case | reward/step | done/step | event len p50 | base_z | base_vx | action delta | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym policy on IsaacGym | 0.031636 | 0.326667 | 161 | 0.809198 | 0.621608 | 0.057490 | 0.702500 |
| IsaacGym policy on IsaacLab | 0.030735 | 0.330000 | 157 | 0.811960 | 0.626922 | 0.039190 | 0.318125 |
| IsaacLab policy on IsaacGym | 0.033911 | 0.423333 | 135 | 0.780359 | 0.854383 | 0.053243 | 0.729336 |
| IsaacLab policy on IsaacLab | 0.035242 | 0.426667 | 138 | 0.783465 | 0.856413 | 0.039527 | 0.311419 |

Interpretation:

- Same-policy backend transfer remains close after seeding IsaacLab diagnostics: done/step differs by only `0.003333` for each policy.
- The IsaacLab-trained policy moves substantially faster under the fixed `0.5 m/s` command: about `+0.23 m/s` base velocity on both backends. It also falls earlier and at lower base height.
- The largest reward-term deltas are stable across both backends: higher `tracking_ang_vel`, higher `default_joint_pos`, lower `feet_contact_number`, and slightly lower `tracking_lin_vel`/`joint_pos`. This points to learned gait/contact tradeoff, not a backend state mapping error.
- Absolute foot-contact occupancy differs between IsaacGym net contact forces and IsaacLab ContactSensor history, so foot-contact rate should be compared primarily within the same backend.

Reward tuning follow-up:

- Added optional `--reward_scale NAME=VALUE` and `--reward_param NAME=VALUE` overrides to `train_isaaclab.py`; defaults remain unchanged.
- Added `rewards.high_speed_penalty = 0.0`, preserving the original Humanoid-Gym `low_speed` behavior by default.
- Ran 50- and 200-iteration IsaacLab tuning checks. Full details are in `docs/REWARD_TUNING.md`.

Best short-run result:

| Policy | done/step | event len p50 | base_vx | base_z | reward/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| IsaacLab seeded bucket 200 | 0.426667 | 138 | 0.856413 | 0.783465 | 0.035242 |
| Mild tuning + `high_speed_penalty=1.0`, 50 | 0.323333 | 161 | 0.659919 | 0.808862 | 0.037654 |

Long-run result:

| Policy | done/step | event len p50 | base_vx | base_z | reward/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| IsaacLab seeded bucket 200 | 0.426667 | 138 | 0.856413 | 0.783465 | 0.035242 |
| Mild tuning + `high_speed_penalty=1.0`, 200 | 0.396667 | 145 | 0.824049 | 0.791510 | 0.032145 |

Interpretation: explicit overspeed penalty is the right direction, but the constant penalty is not sufficient over 200 iterations. `high_speed_penalty=2.0` regressed the 50-iteration learning curve (`0.013958` tail10 reward/step), so it is not selected. The next experiment should use a shaped penalty proportional to speed error, not a larger constant penalty.

### Zero/Random Action Diagnostics

The diagnostic scripts disable observation noise, command motion, push, friction randomization, and base-mass randomization:

```bash
HGEX_DIAG_STEPS=120 HGEX_DIAG_ACTION=zero HGEX_DIAG_OUTPUT=/tmp/hgex_diag_isaacgym_zero.json \
conda run -n legged_gym python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 120 --action_mode zero \
  --output /tmp/hgex_diag_isaaclab_zero_after_pd.json
```

After the PD substep fix, zero-action averages are close:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| reward/step | 0.050264 | 0.046927 |
| base_z | 0.859634 | 0.862924 |
| dof_vel_abs | 0.240759 | 0.165055 |
| torque_abs | 11.439625 | 10.659807 |
| contact_norm | 39.951544 | 39.975942 |

Random-action averages are also close:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| reward/step | 0.031600 | 0.029409 |
| base_z | 0.843286 | 0.836787 |
| dof_vel_abs | 0.911663 | 0.792239 |
| torque_abs | 19.981427 | 18.256818 |
| scaled dof_acc reward | -0.000320 | -0.000332 |
| scaled torques reward | -0.001180 | -0.001087 |

After switching IsaacLab termination to contact history, a longer 240-step random-action diagnostic shows termination-contact rates are close:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| done_count / step | 0.262500 | 0.275000 |
| termination_contact_mean | 30.810335 | 53.670832 |
| termination_contact_gt_0.1 | 0.004102 | 0.004036 |
| termination_contact_gt_1.0 | 0.004102 | 0.004036 |

### Trained-Policy Replay And Asset Order

The same IsaacGym `model_50.pt` checkpoint was replayed for 600 steps in both backends with fixed command. The diagnostic scripts now also export per-step trace JSON and rigid shape/material summaries:

```bash
HGEX_DIAG_STEPS=600 HGEX_DIAG_CHECKPOINT=logs/XBot_ppo/May21_21-40-58_/model_50.pt HGEX_DIAG_FIX_COMMAND=1 \
HGEX_DIAG_OUTPUT=/tmp/hgex_replay_isaacgym_policy50_orderfix.json \
HGEX_DIAG_TRACE_OUTPUT=/tmp/hgex_replay_isaacgym_policy50_orderfix_trace.json \
conda run -n legged_gym python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py --task=humanoid_ppo --headless --num_envs 64

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py --headless --num_envs 64 --steps 600 \
  --checkpoint_path logs/XBot_ppo/May21_21-40-58_/model_50.pt --fix_command \
  --output /tmp/hgex_replay_isaaclab_policy50_orderfix.json \
  --trace_output /tmp/hgex_replay_isaaclab_policy50_orderfix_trace.json
```

Replay summary after joint/ref-order canonicalization:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| reward/step | 0.029891 | 0.026415 |
| done_count / step | 0.323333 | 0.325000 |
| base_z | 0.826206 | 0.830437 |
| dof_vel_abs | 0.613922 | 0.519882 |
| torque_abs | 15.962492 | 14.029982 |
| termination_contact_gt_1.0 | 0.005052 | 0.004505 |

First done frame in the 600-step trace:

| Backend | first done step | base_z_mean | base_z_min | max_abs_roll_pitch | termination_contact_gt_1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 131 | 0.728367 | 0.321476 | 1.176777 | 0.015625 |
| IsaacLab | 136 | 0.689542 | 0.287927 | 1.257980 | 0.000000 |

Asset discovery:

| Item | IsaacGym | IsaacLab |
| --- | --- | --- |
| body count | 13 | 13 |
| total mass env0 | 53.03625858 | 53.03625870 |
| termination bodies | `base_link` | `base_link` |
| feet bodies | `left_ankle_roll_link`, `right_ankle_roll_link` | `left_ankle_roll_link`, `right_ankle_roll_link` |
| joint order | left leg joints, then right leg joints | left/right interleaved |
| rigid shapes | 61 | 61 |
| friction/material | friction mean `1.0` | static/dynamic friction mean `1.11023` |
| contact offset | mean `0.0100` | mean `0.0010`, max `0.0080` |

The mass/body identity is close, but joint/body order and collision/material parameters differ. The IsaacLab env now maps joint state/action/torque/reference tensors to the IsaacGym canonical order before upper-level reward, observation, and action logic sees them. The remaining contact-offset and material differences are likely contributors to long-run curve mismatch.

## Upstream Comparison Protocol

Run upstream `roboterax/humanoid-gym` and HumanoidGym-Ex in the same conda environment, same GPU, same seed, same `num_envs`, and same `max_iterations`.

Compare:

- task registration
- action dimension
- observation dimension
- privileged observation dimension
- one-iteration training completion
- 20-iteration mean reward and mean episode length
- exported policy loadability
- checkpoint path format

Exact bitwise equality is not required for GPU PhysX training, but shapes, reward names, checkpoint structure, and training entry behavior should match.

## IsaacLab ContactSensor Body-Order Fix

Date: 2026-05-22

Issue found during Route A parity probing: IsaacLab `ContactSensor.body_names` did not match `Articulation.body_names` for XBot-L. The robot articulation body order is left/right interleaved, while the contact sensor reports the original imported body order. Using articulation body indices directly on `ContactSensor.data.net_forces_w` made foot-contact rewards and observations read the wrong bodies.

Fix:

- `IsaacLabBackend.get_contact_forces()` now reorders contact sensor forces into robot body order.
- `XBotIsaacLabEnv` uses the same reorder for contact-history termination and foot-contact diagnostics.
- Reward formulas, reward names, and default reward scales were not changed.

Post-fix zero-action contact probe:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| feet_contact_rate | 0.919401 | 0.911198 |
| left_foot_contact_rate | 0.919271 | 0.911589 |
| right_foot_contact_rate | 0.919531 | 0.910807 |
| feet_force_norm_mean | 259.090257 | 257.100527 |

Post-fix random-action probe:

| Metric | IsaacGym | IsaacLab |
| --- | ---: | ---: |
| reward/step | 0.031600 | 0.031820 |
| done_count / step | 0.016667 | 0.016667 |
| feet_contact_rate | 0.744987 | 0.722266 |
| feet_force_norm_mean | 267.776801 | 250.819562 |
| processed_action_abs | 0.389777 | 0.389122 |

Post-fix 50-iteration default IsaacLab training:

| Backend | Tail10 mean reward | Tail10 episode length |
| --- | ---: | ---: |
| IsaacGym baseline | 1.970 | 128.552 |
| IsaacLab ContactSensor mapping fix | 2.193 | 139.039 |

Post-fix 200-iteration default IsaacLab training:

| Backend | Tail10 mean reward | Tail10 episode length |
| --- | ---: | ---: |
| IsaacGym baseline | 2.858 | 151.775 |
| IsaacLab ContactSensor mapping fix | 2.855 | 151.700 |

Post-fix fixed-command replay, `64 envs x 600 steps`, seed `5`, command `lin_vel_x=0.5`:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym policy on IsaacGym | 0.031636 | 0.326667 | 196 | 0.809198 | 0.621608 | 0.638970 | 14.578374 | 0.702500 |
| IsaacGym policy on IsaacLab | 0.032153 | 0.330000 | 198 | 0.811960 | 0.626922 | 0.608523 | 14.363855 | 0.701016 |
| IsaacLab ContactSensor policy on IsaacGym | 0.033405 | 0.400000 | 240 | 0.795448 | 0.822486 | 0.711925 | 13.985397 | 0.737018 |
| IsaacLab ContactSensor policy on IsaacLab | 0.033748 | 0.355000 | 213 | 0.809665 | 0.708755 | 0.637380 | 14.277981 | 0.723060 |

Conclusion:

- The broad 200-iteration training-curve mismatch is resolved under the default reward route.
- The remaining difference is not a default reward inconsistency. It is concentrated in fixed-command learned gait robustness: the IsaacLab-trained policy still falls more often and tends toward a faster gait than the IsaacGym-trained policy.
- Reward-tuning hooks remain default-off. Route A should continue with multi-seed and rough-terrain parity using shared default rewards.

## Route A Multi-Seed And Rough-Terrain Follow-Up

Date: 2026-05-22

Small compatibility fixes before running the matrix:

- `train_isaaclab.py` now accepts `--seed`.
- IsaacGym `--seed` now updates env cfg as well as train cfg before simulator creation.
- IsaacGym rollout diagnostics now apply `HGEX_DIAG_SEED` to env cfg before environment creation.
- IsaacGym train now accepts `--terrain`, `--measure_heights`, and `--terrain_curriculum`.
- IsaacGym measured-height privileged observations now append height samples to the current privileged observation, not the stale `self.obs_buf`.
- IsaacGym critic history is rebuilt when measured-height mode changes the privileged-observation width.

Validation:

```bash
bash humanoid_gym_ex/scripts/validate_smoke.sh
```

Result: `[validate] ok`.

The validation now includes static reward parity:

```bash
python humanoid_gym_ex/scripts/check_reward_parity.py
```

Result: `reward parity ok`. This checks that every default nonzero XBot reward scale has an IsaacGym and IsaacLab implementation and that default-off reward extensions remain disabled.

IsaacGym rough measured-heights smoke:

```bash
WANDB_MODE=offline conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless \
  --num_envs 4 --max_iterations 1 --seed 1 \
  --terrain rough --measure_heights --run_name rough_cli_smoke_fixed6
```

Result: completed one PPO iteration. Actor input stayed `705`; critic input expanded to `780`.

Plane 200-iteration multi-seed check:

| Seed | Backend | Tail10 mean reward | Tail10 episode length | Notes |
| ---: | --- | ---: | ---: | --- |
| 1 | IsaacGym | 2.690 | 151.678 | CLI seed confirmed as `1`. |
| 1 | IsaacLab | 3.082 | 160.071 | Same default reward, but higher reward and longer episodes. |
| 5 | IsaacGym | 2.858 | 151.775 | Existing ContactSensor-fix baseline. |
| 5 | IsaacLab | 2.855 | 151.700 | Closely aligned with IsaacGym for this seed. |

Rough measured-heights 50-iteration check, seed `1`, `64 envs x 60 steps`:

| Backend | Tail10 mean reward | Tail10 episode length | Terrain level |
| --- | ---: | ---: | ---: |
| IsaacGym | 1.818 | 118.770 | 0.0 |
| IsaacLab | 1.718 | 130.110 | not logged |

Seed `1` fixed-command replay, `64 envs x 600 steps`, command `lin_vel_x=0.5`:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym seed1 policy on IsaacGym | 0.031706 | 0.365000 | 219 | 0.809257 | 0.687142 | 0.626114 | 15.147251 | 0.640247 |
| IsaacGym seed1 policy on IsaacLab | 0.031665 | 0.365000 | 219 | 0.814243 | 0.618381 | 0.607897 | 15.460887 | 0.658451 |
| IsaacLab seed1 policy on IsaacGym | 0.036764 | 0.388333 | 233 | 0.777798 | 0.810515 | 0.701024 | 12.865049 | 0.764505 |
| IsaacLab seed1 policy on IsaacLab | 0.037410 | 0.363333 | 218 | 0.785402 | 0.764769 | 0.687618 | 12.607409 | 0.765052 |

Conclusion:

- Route A remains correct: the default reward path is shared and should stay shared.
- Plane seed `5` is very well aligned, but seed `1` shows non-trivial seed sensitivity. The difference is not ignorable for a benchmark claim.
- Seed `1` replay still supports the same high-level diagnosis: same-policy backend transfer is close, while the IsaacLab-trained policy chooses a faster, lower gait than the IsaacGym-trained policy.
- Rough measured-height training now runs in both backends. At 50 iterations it is a functional parity check, not a convergence parity claim.
- Next useful validation is a 3-seed or 5-seed plane 200/500-iteration matrix, plus rough 200-iteration runs once terrain-level logging is mirrored in IsaacLab.

## IsaacLab Timeout Bootstrap Parity

Date: 2026-05-22

Issue: the IsaacGym PPO path forwards `infos["time_outs"]` so `PPO.process_env_step()` can bootstrap timeout transitions. The IsaacLab VecEnv previously merged `truncated` into `done` but did not forward `time_outs`, so timeout transitions were treated as terminal for return computation.

Fix:

- `IsaacLabRslRlVecEnv.step()` now sets `extras["time_outs"] = truncated`.
- IsaacLab timeout comparison now follows the IsaacGym condition: `episode_length_buf > max_episode_length`.
- Reward parity remains unchanged and is checked by `check_reward_parity.py`.

Seed `1` plane 200-iteration training after the timeout fix:

| Run | Tail10 mean reward | Tail10 episode length |
| --- | ---: | ---: |
| IsaacGym seed1 baseline | 2.690 | 151.678 |
| IsaacLab seed1 before timeout fix | 3.082 | 160.071 |
| IsaacLab seed1 timeout fix | 2.970 | 155.618 |

Fixed-command replay of the IsaacLab seed `1` policy before and after the timeout fix:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before fix on IsaacGym | 0.036764 | 0.388333 | 233 | 0.777798 | 0.810515 | 0.701024 | 12.865049 | 0.764505 |
| Before fix on IsaacLab | 0.037410 | 0.363333 | 218 | 0.785402 | 0.764769 | 0.687618 | 12.607409 | 0.765052 |
| Timeout fix on IsaacGym | 0.036410 | 0.376667 | 226 | 0.794192 | 0.845014 | 0.698219 | 13.949369 | 0.744622 |
| Timeout fix on IsaacLab | 0.036887 | 0.350000 | 210 | 0.801025 | 0.779176 | 0.674677 | 13.891590 | 0.735143 |

Interpretation:

- Timeout bootstrap was a real non-reward mismatch. Fixing it moves seed `1` IsaacLab training and replay toward the IsaacGym baseline.
- It does not fully close the policy gap: the IsaacLab policy still runs faster than the IsaacGym policy under fixed command.
- Next non-reward targets are reset observation timing, command resampling timing, and episode-info logging parity.
