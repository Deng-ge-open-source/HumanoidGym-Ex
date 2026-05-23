# HumanoidGym-Ex Architecture Review

This document records the initial architecture review for HumanoidGym-Ex. The current repository is empty except for Git metadata, so the review uses the upstream public Humanoid-Gym repository as the baseline:

- Upstream repository: `https://github.com/roboterax/humanoid-gym`
- Reviewed upstream commit: `ae46e20 Update README.md`
- Review date: 2026-05-21

HumanoidGym-Ex should start as a conservative extension of Humanoid-Gym, not a rewrite. The first milestone should preserve the original script-centered workflow, config inheritance style, reward function naming, task registry usage, and PPO environment API before introducing backend boundaries.

## Upstream Project Shape

Upstream Humanoid-Gym has a compact layout:

```text
humanoid-gym/
├── humanoid/
│   ├── algo/
│   │   ├── vec_env.py
│   │   └── ppo/
│   │       ├── actor_critic.py
│   │       ├── on_policy_runner.py
│   │       ├── ppo.py
│   │       └── rollout_storage.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── base_config.py
│   │   │   ├── base_task.py
│   │   │   ├── legged_robot.py
│   │   │   └── legged_robot_config.py
│   │   └── custom/
│   │       ├── humanoid_config.py
│   │       └── humanoid_env.py
│   ├── scripts/
│   │   ├── train.py
│   │   ├── play.py
│   │   └── sim2sim.py
│   └── utils/
│       ├── helpers.py
│       ├── logger.py
│       ├── math.py
│       ├── task_registry.py
│       └── terrain.py
├── resources/
│   └── robots/XBot/
│       ├── meshes/
│       ├── mjcf/
│       ├── terrain/
│       └── urdf/
├── logs/
└── setup.py
```

The project is intentionally script-centered. Users run:

```bash
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096
python scripts/play.py --task=humanoid_ppo --run_name v1
python scripts/sim2sim.py --load_model /path/to/policy_1.pt
```

## Main Modules

### Environment Class

The main IsaacGym environment is split into:

- `humanoid/envs/base/base_task.py`
- `humanoid/envs/base/legged_robot.py`
- `humanoid/envs/custom/humanoid_env.py`

`BaseTask` owns generic IsaacGym setup:

- `gymapi.acquire_gym()`
- sim device parsing
- common buffers: `obs_buf`, `privileged_obs_buf`, `rew_buf`, `reset_buf`, `episode_length_buf`, `extras`
- viewer and camera creation
- `render()`

`LeggedRobot` owns most locomotion mechanics:

- `step()`
- `post_physics_step()`
- `reset_idx()`
- `check_termination()`
- reward dispatch
- command sampling and command curriculum
- terrain curriculum
- IsaacGym tensor acquisition and refresh
- actor creation and rigid body / DOF property processing
- PD torque computation
- root and DOF reset

`XBotLFreeEnv` extends `LeggedRobot` for the humanoid task:

- gait phase and reference action computation
- humanoid-specific observation construction
- action delay and action noise
- XBot-specific rewards
- custom terrain creation through `HumanoidTerrain`

The key style to preserve is that env logic is readable in one place. Rewards, observations, reset behavior, command logic, and robot-specific locomotion assumptions are not scattered through a manager system.

### Robot Config

Configuration uses nested Python classes, not YAML:

- `LeggedRobotCfg`
- `LeggedRobotCfgPPO`
- `XBotLCfg`
- `XBotLCfgPPO`

Important config groups:

- `env`: number of envs, obs sizes, action count, frame stacking, episode length
- `terrain`: plane / heightfield / trimesh, terrain curriculum, height sampling
- `commands`: command ranges, heading command, resampling time, command curriculum
- `init_state`: base pose and default joint angles
- `control`: PD stiffness, damping, action scale, decimation
- `asset`: URDF path, body name filters, contact body names, IsaacGym asset options
- `domain_rand`: friction, base mass, pushes, action delay, action noise
- `rewards.scales`: names that map directly to `_reward_<name>` methods
- `normalization`: observation scales and clipping
- `noise`: observation noise
- `sim`: IsaacGym PhysX settings

This config format is central to the Humanoid-Gym user experience and should remain the first-class authoring format in HumanoidGym-Ex.

### Train Script

`humanoid/scripts/train.py` is deliberately minimal:

```python
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
ppo_runner.learn(...)
```

The script relies on:

- `humanoid.envs.__init__` importing and registering tasks
- `get_args()` parsing IsaacGym and custom CLI arguments
- `task_registry` constructing both env and PPO runner

HumanoidGym-Ex should keep this path intact for Phase 1.

### Play Script

`humanoid/scripts/play.py`:

- gets env/train cfg from `task_registry`
- overrides runtime settings for evaluation
- creates one environment by default
- resumes the PPO runner
- gets an inference policy
- optionally exports a TorchScript actor to `logs/<experiment>/exported/policies/policy_1.pt`
- optionally records camera frames through IsaacGym camera sensors
- logs joint, command, velocity, torque, and contact force signals

This script is not backend-neutral. It directly accesses `env.gym`, `env.sim`, `env.envs`, `env.contact_forces`, `env.dof_pos`, and `env.torques`. Phase 1 can preserve that; Phase 2 needs a compatibility layer for rendering and state access.

### Reward Functions

Reward dispatch is config-name based:

1. Convert `cfg.rewards.scales` to a dict.
2. Remove zero-scale rewards.
3. Multiply scales by `dt`.
4. For each remaining key, call `self._reward_<key>()`.

`XBotLFreeEnv` defines humanoid rewards:

- `joint_pos`
- `feet_distance`
- `knee_distance`
- `foot_slip`
- `feet_air_time`
- `feet_contact_number`
- `orientation`
- `feet_contact_forces`
- `default_joint_pos`
- `base_height`
- `base_acc`
- `vel_mismatch_exp`
- `track_vel_hard`
- `tracking_lin_vel`
- `tracking_ang_vel`
- `feet_clearance`
- `low_speed`
- `torques`
- `dof_vel`
- `dof_acc`
- `collision`
- `action_smoothness`

The reward functions read tensors directly from the env object:

- `self.dof_pos`
- `self.dof_vel`
- `self.torques`
- `self.root_states`
- `self.rigid_state`
- `self.contact_forces`
- `self.commands`
- `self.base_lin_vel`
- `self.base_ang_vel`
- `self.base_euler_xyz`
- `self.projected_gravity`
- `self.feet_indices`
- `self.knee_indices`

For multi-backend support, the most important compatibility target is preserving these tensor attributes with the same shapes and semantics.

### Observation Construction

`XBotLFreeEnv.compute_observations()` constructs:

- command input: gait phase sin/cos plus scaled velocity commands
- actor obs: joint position offsets, joint velocity, previous actions, base angular velocity, base euler angles
- privileged obs: actor obs plus reference diff, base linear velocity, pushes, friction, body mass, stance mask, contact mask
- frame-stacked observation through `obs_history`
- frame-stacked privileged observation through `critic_history`

The default XBot config uses:

- `num_single_obs = 47`
- `frame_stack = 15`
- `num_observations = 705`
- `single_num_privileged_obs = 73`
- `c_frame_stack = 3`
- `num_privileged_obs = 219`
- `num_actions = 12`

The observation logic should stay in `humanoid_env.py` in Phase 1 and remain visually close in Phase 2.

### Reset Logic

Reset is split between generic and robot-specific layers:

- `LeggedRobot.reset()` resets all envs and performs one zero-action step.
- `LeggedRobot.reset_idx(env_ids)` handles curriculum, DOF reset, root reset, command resampling, buffer clearing, episode logging, and timeout reporting.
- `XBotLFreeEnv.reset_idx(env_ids)` calls `super()` and clears observation histories.

IsaacGym reset writes directly into:

- `self.dof_state`
- `self.root_states`

then calls:

- `gym.set_dof_state_tensor_indexed(...)`
- `gym.set_actor_root_state_tensor_indexed(...)`

Phase 2 IsaacLab support must keep the same higher-level `reset_idx(env_ids)` flow while replacing the final state write operations.

### Command Curriculum

Commands are sampled in `_resample_commands(env_ids)`:

- `lin_vel_x`
- `lin_vel_y`
- `ang_vel_yaw`, or `heading` if heading mode is enabled

Small planar commands are zeroed. During `_post_physics_step_callback()`, commands are periodically resampled and heading commands are converted to yaw-rate commands.

`update_command_curriculum(env_ids)` expands the `lin_vel_x` range when tracking reward is high enough. This is simple and should remain in the upper env class, not in a backend.

### Domain Randomization

Domain randomization is currently mixed into IsaacGym env creation and stepping:

- `_process_rigid_shape_props()`: friction bucket randomization per env
- `_process_rigid_body_props()`: base mass randomization
- `_post_physics_step_callback()`: periodic pushes
- `XBotLFreeEnv.step()`: action delay and action noise

This is a major backend-boundary risk. In Phase 1 it can remain IsaacGym-native. In Phase 2, friction, mass, and push APIs need backend-specific implementations while action delay/noise can stay in the upper env.

### Terrain

`humanoid/utils/terrain.py` builds heightfield/trimesh data with IsaacGym `terrain_utils`.

`LeggedRobot` adds terrain through:

- `_create_ground_plane()`
- `_create_heightfield()`
- `_create_trimesh()`

Terrain origins feed reset placement and terrain curriculum. IsaacLab migration should first support `plane`, then add heightfield/trimesh mapping after the basic policy loop is verified.

### Sim-to-Sim / MuJoCo

`humanoid/scripts/sim2sim.py` is a separate deployment script, not part of the training env:

- loads exported TorchScript policy
- loads XBot MJCF in MuJoCo
- manually reconstructs the same 47-D single-frame observation
- maintains frame stack
- applies PD control with configured gains and torque limits
- runs viewer rendering

The script is useful for deployment compatibility but should not drive the backend abstraction. Keep it as a separate tool.

### rsl_rl PPO Interface

Humanoid-Gym vendors a compact rsl_rl-style PPO implementation under `humanoid/algo/ppo`.

The runner expects the env to provide:

- `num_envs`
- `num_obs`
- `num_privileged_obs`
- `num_actions`
- `max_episode_length`
- `episode_length_buf`
- `reset_buf`
- `device`
- `reset() -> (obs, privileged_obs)`
- `step(actions) -> (obs, privileged_obs, rewards, dones, infos)`
- `get_observations()`
- `get_privileged_observations()`

This is the practical minimum compatibility contract for Phase 1 and Phase 2.

## Current Coupling Points

These points make a direct backend swap risky:

- `BaseTask` directly creates IsaacGym sim, viewer, and camera.
- `LeggedRobot` directly calls IsaacGym tensor acquire/refresh/set APIs.
- Env creation directly uses IsaacGym asset options and actor handles.
- Play/render code directly uses `env.gym`, `env.sim`, `env.envs`.
- Terrain generation imports IsaacGym `terrain_utils`.
- Domain randomization modifies IsaacGym rigid shape/body props during env creation.
- Reward code assumes tensor attributes exist with IsaacGym-like shapes.

The recommended strategy is to preserve upper-level tensor attributes and move only simulator state acquisition/writes behind a narrow backend object.

## Proposed HumanoidGym-Ex Directory Design

The target layout should be familiar to Humanoid-Gym users while separating backend-specific code:

```text
humanoid_gym_ex/
├── humanoid_gym_ex/
│   ├── __init__.py
│   ├── algo/
│   │   ├── vec_env.py
│   │   └── ppo/
│   │       ├── actor_critic.py
│   │       ├── on_policy_runner.py
│   │       ├── ppo.py
│   │       └── rollout_storage.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── backend_interface.py
│   │   │   ├── humanoid_config.py
│   │   │   └── humanoid_env.py
│   │   ├── robots/
│   │   │   ├── h1/
│   │   │   │   ├── h1_config.py
│   │   │   │   └── h1_env.py
│   │   │   ├── xbot/
│   │   │   │   ├── xbot_config.py
│   │   │   │   └── xbot_env.py
│   │   │   ├── g1/
│   │   │   │   ├── g1_config.py
│   │   │   │   └── g1_env.py
│   │   │   └── custom/
│   │   │       └── README.md
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── isaacgym_backend.py
│   │       ├── isaaclab_backend.py
│   │       └── genesis_backend.py
│   ├── scripts/
│   │   ├── train.py
│   │   ├── play.py
│   │   ├── export_policy.py
│   │   └── sim2sim.py
│   ├── utils/
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   ├── math.py
│   │   ├── task_registry.py
│   │   └── terrain.py
│   └── assets/
│       └── robots/
├── docs/
│   ├── ARCHITECTURE_REVIEW.md
│   ├── BACKEND_INTERFACE.md
│   ├── DESIGN_GOALS.md
│   ├── MIGRATION_ISAACLAB.md
│   └── ROADMAP.md
├── tests/
├── README.md
├── CHANGELOG.md
└── setup.py
```

Two naming choices are worth keeping conservative:

- Keep `humanoid_env.py` and `humanoid_config.py` names in `envs/base/` so existing users recognize the pattern.
- Keep robot configs as Python nested classes. Do not move to YAML or IsaacLab Manager-based config objects in the first migration.

## Minimal Backend Interface

The backend boundary should stay small:

```python
class BackendInterface:
    def create_sim(self): ...
    def create_envs(self): ...
    def step(self, actions): ...
    def reset(self, env_ids): ...
    def get_root_states(self): ...
    def get_dof_pos(self): ...
    def get_dof_vel(self): ...
    def get_contact_forces(self): ...
    def set_dof_targets(self, targets_or_torques): ...
    def apply_domain_randomization(self, env_ids=None): ...
    def render_or_viewer_step(self): ...
```

Important constraint: this interface should not become a plugin system. It should only hide simulator mechanics needed to keep `humanoid_env.py` readable.

Recommended ownership:

- Upper env owns reward, obs, command sampling, curriculum, episode buffers, PPO-facing API.
- Backend owns sim creation, env creation, simulator stepping, tensor refresh/access, reset writes, actuator target writes, viewer stepping, simulator-specific domain randomization.

## Phase Plan

### Phase 0: Import Baseline Safely

Goal: get a runnable Humanoid-Gym baseline into this repository before abstraction.

Actions:

- Import upstream package with minimal renaming.
- Keep `train.py`, `play.py`, task registry, PPO, XBot config, assets, and terrain code running.
- Add project README and compatibility notes.
- Add `CHANGELOG.md`.

Exit criteria:

- `python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1` reaches PPO rollout/update on a machine with IsaacGym installed.
- `python humanoid_gym_ex/scripts/play.py --task=humanoid_ppo` can load a checkpoint and export policy.

### Phase 1: IsaacGym Backend Boundary

Goal: introduce `BackendInterface` and `IsaacGymBackend` without changing user-facing task behavior.

Actions:

- Move only low-level IsaacGym operations from env/base task into `isaacgym_backend.py`.
- Keep tensor attributes on env with the same names: `root_states`, `dof_pos`, `dof_vel`, `contact_forces`, `rigid_state`.
- Keep reward names and config fields unchanged.
- Keep train/play command style unchanged.

Exit criteria:

- Same Phase 0 train/play commands still run.
- Diff in reward/obs logic is minimal.

### Phase 2: IsaacLab Direct Workflow

Goal: add an IsaacLab backend using Direct workflow, not Manager-based workflow.

Actions:

- Implement `IsaacLabBackend` with DirectRLEnv-style stepping and `Articulation` data.
- Map IsaacLab tensors into the same upper env attributes.
- Preserve reward/obs/reset functions as much as possible.
- Write `MIGRATION_ISAACLAB.md` with IsaacGym-to-IsaacLab API mapping.

Exit criteria:

- A small env count can step with IsaacLab Direct workflow.
- At least the plane terrain task can run through PPO rollout.
- Shared reward functions work for IsaacGym and IsaacLab for the supported tensor set.

### Phase 3: Genesis Placeholder

Goal: reserve space without contaminating IsaacGym or IsaacLab code.

Actions:

- Add `GenesisBackend` stub that raises clear `NotImplementedError`.
- Document expected tensor contract.
- Avoid adding Genesis-specific assumptions to upper env.

Exit criteria:

- Importing the package does not require Genesis.
- Selecting Genesis fails with a clear message.

## Files To Create First

For the next implementation step, keep the first code import small:

```text
README.md
CHANGELOG.md
setup.py
humanoid_gym_ex/__init__.py
humanoid_gym_ex/envs/__init__.py
humanoid_gym_ex/envs/base/humanoid_config.py
humanoid_gym_ex/envs/base/humanoid_env.py
humanoid_gym_ex/envs/base/backend_interface.py
humanoid_gym_ex/envs/backends/isaacgym_backend.py
humanoid_gym_ex/envs/robots/xbot/xbot_config.py
humanoid_gym_ex/envs/robots/xbot/xbot_env.py
humanoid_gym_ex/scripts/train.py
humanoid_gym_ex/scripts/play.py
humanoid_gym_ex/scripts/export_policy.py
humanoid_gym_ex/scripts/sim2sim.py
humanoid_gym_ex/utils/task_registry.py
humanoid_gym_ex/utils/helpers.py
humanoid_gym_ex/utils/terrain.py
humanoid_gym_ex/utils/math.py
humanoid_gym_ex/algo/vec_env.py
humanoid_gym_ex/algo/ppo/*
docs/DESIGN_GOALS.md
docs/BACKEND_INTERFACE.md
docs/MIGRATION_ISAACLAB.md
docs/ROADMAP.md
```

For Phase 0, it is acceptable to import `algo`, `utils`, `scripts`, and XBot assets close to upstream. The backend file can exist as a documented placeholder until the Phase 1 boundary is introduced.

## Risk Points

- The local repository is currently empty, so there is no existing project code to preserve. The first implementation step will need to import or reconstruct the baseline.
- IsaacGym Preview 4 is old and environment-specific. CI may not be able to run real simulator tests unless self-hosted GPU runners are available.
- `play.py` is more IsaacGym-coupled than `train.py` because it directly accesses viewer and camera APIs.
- Terrain generation depends on IsaacGym `terrain_utils`; IsaacLab support should initially target plane terrain.
- Reward functions assume exact tensor shapes and body index semantics. Backend tensor mapping must be validated carefully.
- Domain randomization is split across env creation and runtime stepping. Friction and mass randomization will be backend-specific.
- Upstream default `get_args()` uses default task `"XBotL_free"` while registered task is `"humanoid_ppo"`; HumanoidGym-Ex should normalize this before release.
- Upstream PPO runner imports `wandb` and initializes it when logging. For open-source usability, this should be optional or disabled by config/env var.
- IsaacLab Direct workflow has different lifecycle assumptions from IsaacGym. Avoid forcing IsaacLab concepts into reward/obs code.

## Minimum Runnable Route

The minimum route should avoid a large rewrite:

1. Import upstream Humanoid-Gym code into `humanoid_gym_ex` with package path updates.
2. Keep `humanoid_ppo` XBot task registered and runnable on IsaacGym.
3. Fix obvious packaging and task-name mismatches only.
4. Add documentation that compatibility is currently IsaacGym-first.
5. Add a narrow `BackendInterface` document and stub, but do not route all code through it until the baseline train/play path works.
6. Introduce `IsaacGymBackend` by moving simulator calls in small batches:
   - sim/viewer creation
   - tensor acquisition and refresh
   - DOF/root reset writes
   - action/torque application
   - terrain creation
7. Only after Phase 1 passes train/play, add IsaacLab Direct workflow.

This sequence keeps the project useful after each step and avoids breaking the original Humanoid-Gym user experience while the backend abstraction is still forming.
