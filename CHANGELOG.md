# Changelog

## Unreleased - Phase 2 IsaacLab Direct Backend

- Added the IsaacLab Direct workflow path while preserving the Humanoid-Gym user model: centralized robot config, centralized rewards, `train.py`/`play.py` style scripts, and local PPO runner integration.
- Implemented `IsaacLabBackend`, `XBotIsaacLabEnv`, and `IsaacLabRslRlVecEnv` for DirectRLEnv/Articulation based training and play.
- Added IsaacLab train/play/export entry points without introducing a Manager-based task split.
- Preserved XBot policy, critic, and action dimensions in the default IsaacLab plane path: `705/219/12`.
- Migrated XBot reward dispatch and default reward names so IsaacGym and IsaacLab use the same default reward config.
- Added IsaacLab rough terrain, `heightfield`/`trimesh` aliases, terrain curriculum, and measured-height support. In measured-height mode, actor observations stay at `705` and critic observations expand to `780`, matching the original Humanoid-Gym pattern.
- Added IsaacLab action delay/noise, root-velocity push randomization, friction randomization, base-mass randomization, command resampling, heading command, and command curriculum.
- Fixed IsaacLab PD torque application so torque is recomputed on every physics substep, matching the IsaacGym decimation loop.
- Aligned IsaacLab joint state, action, torque, reference DOF, and contact-force body ordering with the IsaacGym canonical XBot order.
- Aligned IsaacLab timeout handling, gait phase timing, seeded training setup, rigid-shape material/contact defaults, PhysX solver settings, and friction bucket sampling with the IsaacGym baseline.
- Fixed IsaacLab XBot-L self-collision mapping by applying the Humanoid-Gym collision-filter flag to both `UrdfFileCfg.self_collision` during URDF conversion and `ArticulationRootPropertiesCfg.enabled_self_collisions` on the articulation root.
- Added `asset.isaaclab_self_collisions = 1` as the XBot-L IsaacLab default while keeping the original IsaacGym `asset.self_collisions` field unchanged.
- Regressed IsaacLab self-collision on/off with a 600-step fixed-command trained-policy rollout. On the current XBot-L asset/checkpoint, both settings produced the same measured rollout (`base_z_last_mean=0.8942`, `foot_fz_last_mean=262.2729`, zero resets), so the reported floating issue was not reproduced locally under this scenario.
- Re-ran seed `42` plane training after the self-collision mapping fix at `64 envs x 60 steps x 200 iterations` on IsaacGym and IsaacLab. Tail10 reward/step alignment is `98.90%` (`0.019166` IsaacGym vs `0.018955` IsaacLab), tail10 reward alignment is `98.22%`, and tail10 episode-length alignment is `99.31%`.
- Completed long plane and rough-terrain validation across IsaacGym and IsaacLab. Plane training is closely aligned; rough measured-height terrain runs end-to-end with matching reward definitions and observation dimensions, but long rough-terrain convergence remains a known residual gap.
- Kept optional parity/tuning switches default-off so the published default route uses the same reward names and default reward values across IsaacGym and IsaacLab.

## Unreleased - Phase 1 IsaacGym Backend Adapter

- Implemented `IsaacGymBackend` as a thin adapter around IsaacGym calls.
- Routed simulator creation, env creation call, torque writes, physics stepping, tensor refresh, tensor acquisition, DOF reset writes, root reset writes, push root-state writes, viewer stepping, viewer creation, and camera creation through the backend adapter.
- Kept reward, observation, reset, command curriculum, domain-randomization callbacks, terrain creation, and robot-specific config style close to upstream Humanoid-Gym.
- Verified IsaacGym one-iteration and 20-iteration regression against the Phase 0/upstream baseline.
- Verified headless play/export smoke test after backend routing.

## 0.1.0 - Phase 0 Baseline Migration

- Imported upstream Humanoid-Gym code into `humanoid_gym_ex`.
- Migrated XBot-L assets, images, MuJoCo sim2sim files, PPO implementation, and example exported policy.
- Renamed Python imports from `humanoid` to `humanoid_gym_ex`.
- Moved the upstream custom humanoid task under `humanoid_gym_ex/envs/robots`.
- Normalized the default task name to `humanoid_ppo`.
- Added initial architecture, design, backend, roadmap, migration, and test documentation.
