# Backend Interface

The backend interface is intentionally small. It exists to keep simulator mechanics out of the upper humanoid environment while preserving Humanoid-Gym-style reward and observation code.

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

Phase 1 routes the IsaacGym training path through `IsaacGymBackend` for the low-level simulator operations that are most important for future backend replacement:

- sim creation
- env creation call boundary
- torque/action target writes
- physics stepping
- state tensor acquisition
- state tensor refresh
- DOF indexed reset writes
- root indexed reset writes
- full root state writes used by push randomization
- viewer step and viewer/camera creation

The backend is intentionally still a thin adapter. Asset loading, terrain creation, rigid property callbacks, reward code, observation construction, and command curriculum remain in the Humanoid-Gym-style env classes until a later refactor has a concrete need to move them.

The upper env should keep these common tensor attributes:

- `root_states`
- `dof_pos`
- `dof_vel`
- `contact_forces`
- `rigid_state`
- `torques`

Reward and observation functions should read those attributes directly, matching upstream Humanoid-Gym style.

## IsaacLab Direct Status

Phase 2 adds `IsaacLabBackend` as a tensor adapter around a DirectRLEnv and Articulation:

- `get_root_states()` -> `Articulation.data.root_state_w`
- `get_dof_pos()` -> `Articulation.data.joint_pos`
- `get_dof_vel()` -> `Articulation.data.joint_vel`
- `get_contact_forces()` -> `ContactSensor.data.net_forces_w`
- `set_dof_targets()` -> effort targets by default, position targets when configured

The first IsaacLab env is intentionally local and explicit:

- [isaaclab_env.py](/home/cra02/Documents/GitHub/HumanoidGym-Ex/humanoid_gym_ex/envs/robots/xbot/isaaclab_env.py)
- [isaaclab_vec_env.py](/home/cra02/Documents/GitHub/HumanoidGym-Ex/humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py)
- [isaaclab_smoke.py](/home/cra02/Documents/GitHub/HumanoidGym-Ex/humanoid_gym_ex/scripts/isaaclab_smoke.py)
- [train_isaaclab.py](/home/cra02/Documents/GitHub/HumanoidGym-Ex/humanoid_gym_ex/scripts/train_isaaclab.py)

It does not introduce a plugin system or Manager-based task decomposition. Reward, observation, command, and reset code remain in one DirectRLEnv subclass so Humanoid-Gym users can follow it.

The IsaacLab env uses the same reward term names from `cfg.rewards.scales` and keeps reward functions as `_reward_<name>` methods on the env class, matching the original Humanoid-Gym style.

Domain randomization is still kept local to the DirectRLEnv class. The current IsaacLab path implements:

- action delay/noise before torque computation
- root velocity push through `write_root_velocity_to_sim`
- rigid material friction randomization through the articulation PhysX view
- base mass randomization through `set_masses` with inertia scaling

Terrain support is intentionally narrow and script-driven:

- `--terrain plane` keeps the original `705/219` observation contract.
- `--terrain rough`, `--terrain heightfield`, and `--terrain trimesh` use IsaacLab `TerrainImporterCfg(terrain_type="generator")` with `ROUGH_TERRAINS_CFG`.
- `--terrain_curriculum` uses `TerrainImporter.update_env_origins(...)` from reset logic.
- `--measure_heights` attaches a `RayCaster` to `base_link` and appends `187` sampled heights per observation frame.

The backend smoke validation is scripted in [validate_smoke.sh](/home/cra02/Documents/GitHub/HumanoidGym-Ex/humanoid_gym_ex/scripts/validate_smoke.sh).
