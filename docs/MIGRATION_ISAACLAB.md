# IsaacLab Migration Notes

IsaacLab support will use Direct workflow. Manager-based tasks are out of scope for HumanoidGym-Ex because they would break the Humanoid-Gym-style authoring model.

## Current Status

Phase 2 has an IsaacLab Direct environment for XBot-L:

- `humanoid_gym_ex/envs/robots/xbot/isaaclab_env.py`
- `humanoid_gym_ex/envs/robots/xbot/isaaclab_vec_env.py`
- `humanoid_gym_ex/scripts/isaaclab_smoke.py`
- `humanoid_gym_ex/scripts/train_isaaclab.py`
- `humanoid_gym_ex/scripts/play_isaaclab.py`
- `humanoid_gym_ex/scripts/validate_smoke.sh`

The path uses `DirectRLEnv`, `Articulation`, `InteractiveScene`, `TerrainImporterCfg`, `ContactSensor`, and optional `RayCaster` height scanning. It loads the migrated XBot-L URDF, keeps the original observation dimensions (`705` policy observations and `219` critic observations) on plane terrain, keeps the rough-terrain actor policy observation at `705`, expands the rough-terrain critic to `780` with measured-height samples, applies the original PD-style torque calculation, and steps IsaacLab headless.

The IsaacLab PPO path uses the local Humanoid-Gym PPO runner through `IsaacLabRslRlVecEnv`. It verifies rollout/update plumbing and uses the same reward term names/scales as the IsaacGym XBot task. The production-compatible train/play entry points remain IsaacGym-first; IsaacLab is exposed through explicit `*_isaaclab.py` scripts while numerical curve parity is still being tightened.

## API Mapping

| Humanoid-Gym / IsaacGym | IsaacLab Direct Workflow Target |
| --- | --- |
| `gym.acquire_actor_root_state_tensor(sim)` | `Articulation.data.root_state_w` |
| `gym.acquire_dof_state_tensor(sim)` | `Articulation.data.joint_pos`, `Articulation.data.joint_vel` |
| `gym.acquire_net_contact_force_tensor(sim)` | `ContactSensor.data.net_forces_w` with `activate_contact_sensors=True` on spawn config |
| `gym.acquire_rigid_body_state_tensor(sim)` | `Articulation.data.body_state_w` |
| `gym.set_dof_actuation_force_tensor(sim, ...)` | `Articulation.set_joint_effort_target(...)` |
| `gym.set_dof_position_target_tensor(sim, ...)` | `Articulation.set_joint_position_target(...)` |
| `gym.set_dof_state_tensor_indexed(sim, ...)` | `Articulation.write_joint_state_to_sim(..., env_ids=...)` |
| `gym.set_actor_root_state_tensor_indexed(sim, ...)` | `Articulation.write_root_pose_to_sim(...)` and `Articulation.write_root_velocity_to_sim(...)` |
| `gym.refresh_*_tensor(sim)` | IsaacLab updates `Articulation.data` during DirectRLEnv stepping |
| `gym.add_ground(...)` | `TerrainImporterCfg(terrain_type="plane")` |
| `gym.add_heightfield(...)` | `TerrainImporterCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG...)` with height-field sub-terrains |
| `gym.add_triangle_mesh(...)` | `TerrainImporterCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG...)` with generated trimesh import |
| Humanoid-Gym measured heights from height samples | `RayCasterCfg` attached to `base_link`, `ray_hits_w[..., 2]` |
| rigid shape/body property callbacks | IsaacLab spawn config, event/randomization utilities, or direct property writes where supported |
| `sim.physx.solver_type`, solver iteration counts, `bounce_threshold_velocity`, contact buffers | `SimulationCfg(physx=PhysxCfg(...))` plus `RigidBodyPropertiesCfg` and `ArticulationRootPropertiesCfg` on robot spawn |
| domain randomization push via root tensor write | DirectRLEnv root velocity write through `Articulation.write_root_velocity_to_sim(...)` |
| terrain curriculum origin update | `TerrainImporter.update_env_origins(env_ids, move_up, move_down)` |
| IsaacGym viewer/headless flags | `AppLauncher.add_app_launcher_args(...)`, `--headless`, and `env.render()` |

## Phase 2 Scope

- Completed: plane terrain smoke.
- Completed: XBot-L URDF spawn in IsaacLab Direct workflow.
- Completed: `705/219/12` policy observation, critic observation, and action dimensions.
- Completed: minimal `IsaacLabBackend` tensor adapter.
- Completed: IsaacLab Direct VecEnv wrapper for the local PPO runner.
- Completed: IsaacLab PPO smoke at `64 envs x 60 steps x 1 iteration`, matching the IsaacGym smoke rollout size of `3840` total timesteps.
- Completed: centralized IsaacLab reward dispatch using original `cfg.rewards.scales` names.
- Completed: migrated XBot reward terms for joint tracking, gait/contact terms, velocity tracking, base pose, energy penalties, collision, and action smoothness.
- Completed: action delay/noise randomization.
- Completed: push randomization using `Articulation.write_root_velocity_to_sim(...)`.
- Completed: friction randomization using `Articulation.root_physx_view.set_material_properties(...)`.
- Completed: base mass randomization using `Articulation.root_physx_view.set_masses(...)` and inertia scaling.
- Completed: command handling. Random command resampling, heading command, and command curriculum are present behind the original config switches.
- Completed: rough terrain generator support through IsaacLab `ROUGH_TERRAINS_CFG`.
- Completed: `heightfield` and `trimesh` script aliases mapped to IsaacLab generated rough terrain.
- Completed: terrain curriculum origin updates through `TerrainImporter.update_env_origins(...)`.
- Completed: RayCaster measured heights. With `--measure_heights`, policy observation stays at the original XBot actor width `705`, while critic observation expands from `219` to `780`.
- Completed: reset handling for terrain origins, DOF reset noise, and command resampling.
- Completed: IsaacLab checkpoint play/export smoke.
- Completed: local automatic validation for IsaacGym train, IsaacLab randomization, IsaacLab PPO, and IsaacLab play/export smoke.
- Completed: 50-iteration IsaacGym vs IsaacLab curve comparison at `64 envs x 60 steps x 50 iterations`.
- Completed: IsaacLab PD torque is recomputed on every physics substep, matching IsaacGym's decimation loop.
- Completed: IsaacLab VecEnv forwards `episode_length_buf` writes to DirectRLEnv, so random initial episode lengths work like IsaacGym.
- Completed: zero/random action diagnostics now show close IsaacGym/IsaacLab reward-rate, torque, dof velocity, and dof acceleration scales.
- Completed: IsaacLab termination uses contact-force history for termination bodies, not only the latest `ContactSensor` sample.
- Completed: IsaacLab maps joint state/action/torque tensors to the IsaacGym canonical joint order. IsaacLab imports the URDF in a left/right interleaved joint order, while IsaacGym exposes left leg then right leg.
- Completed: IsaacLab keeps reference DOF targets in the same canonical joint order as observations, rewards, and actions.
- Completed: trained-policy replay and asset discovery diagnostics. Body count, total mass, termination body, and foot bodies match; joint order differs and is now handled.
- Completed: optional strict fall termination switches (`--termination_base_height`, `--termination_orientation`) and `--parity_termination_profile isaacgym_like` for comparability experiments.
- Completed: 200-iteration IsaacGym vs IsaacLab curve comparison at `64 envs x 60 steps x 200 iterations`.
- Completed: IsaacLab rigid-shape material/contact defaults are explicitly written before optional randomization: friction `1.0`, restitution `0.0`, contact offset `0.01`, rest offset `0.0`.
- Completed: IsaacLab PhysX solver parity for the XBot baseline: TGS solver, `4/1` position/velocity iterations, `0.1` bounce threshold, rigid contact buffer, actor max depenetration velocity, and articulation/body solver iteration counts.
- Remaining difference: IsaacLab is close but not numerically identical. After seed and bucketed friction parity, 50-iteration tail10 reward/step is close (`0.015411` vs IsaacGym `0.015325`) and tail10 episode length is about `6.6%` longer (`136.976` vs `128.552`). At 200 iterations, tail10 reward/step is effectively aligned (`0.018767` vs `0.018831`) while tail10 episode length remains about `5.3%` longer (`159.801` vs `151.775`).
- Completed: 200-iteration cross-backend replay, including the latest seeded bucket-friction IsaacLab checkpoint. The same policy behaves similarly across IsaacGym and IsaacLab; the latest IsaacLab policy records `0.423333` done/step on IsaacGym and `0.421667` on IsaacLab under fixed commands. The remaining gap is mainly that the IsaacLab-trained policy learns a less stable gait than the IsaacGym-trained policy under the current PPO/randomization settings.
- Completed: IsaacLab train seed parity. `train_isaaclab.py` now sets Python, NumPy, Torch, CUDA, `PYTHONHASHSEED`, and `env_cfg.seed` before creating the DirectRLEnv, matching the IsaacGym train entry behavior without importing IsaacGym-only utilities.
- Completed: 50-iteration seeded PhysX parity recheck. Seed parity improves reproducibility but does not by itself remove the remaining reward/learning gap.
- Completed: IsaacLab friction randomization now mirrors IsaacGym's `256`-bucket sampling scheme instead of sampling continuous per-env friction directly.
- Completed: 50-iteration seeded bucket-friction recheck. Tail10 reward/step is within about `0.56%` of IsaacGym; 50-iteration episode length remains about `6.6%` high.
- Completed: 200-iteration seeded bucket-friction recheck. Tail10 reward/step is within about `0.34%` of IsaacGym; 200-iteration episode length remains about `5.3%` high.
- Completed: replay robustness analysis through `humanoid_gym_ex/scripts/analyze_replay_alignment.py` and `docs/REPLAY_ALIGNMENT.md`.
- Completed: IsaacLab ContactSensor body-order mapping. ContactSensor forces are reordered into `Articulation.body_names` order before upper reward, observation, and reset logic reads them. This fixed a foot-contact data mismatch while preserving the same default reward names and values as IsaacGym.
- Completed: post-ContactSensor mapping default-reward curve recheck. The 200-iteration default protocol is now aligned: IsaacLab tail10 mean reward is `2.855` vs IsaacGym `2.858`, and IsaacLab tail10 episode length is `151.700` vs IsaacGym `151.775`.
- Completed: post-ContactSensor mapping fixed-command replay. The IsaacGym reference policy transfers closely across backends (`0.326667` vs `0.330000` done/step). The IsaacLab policy improved versus the older seeded-bucket checkpoint but still falls more often than the IsaacGym policy (`0.355000` done/step in IsaacLab, `0.400000` in IsaacGym).
- Completed: CLI seed parity follow-up. `train_isaaclab.py` accepts `--seed`, and IsaacGym `--seed` now updates env cfg before simulator creation. A seed `1` 200-iteration plane check shows remaining seed sensitivity: IsaacGym `2.690 / 151.678`, IsaacLab `3.082 / 160.071`.
- Completed: IsaacGym rough measured-height train entry fix. The IsaacGym rough path now appends measured heights to the current privileged observation and builds a `780`-dim critic input for the XBot rough-height setting.
- Completed: rough measured-height 1000-iteration functional check with identical reward definitions and `705/780` actor/critic dimensions. IsaacGym reaches tail10 `138.464 / 2357.588`; IsaacLab reaches tail10 `97.214 / 2030.989`. This verifies long-run execution, but not convergence equivalence.
- Completed: initial reward/command shaping experiments in `docs/REWARD_TUNING.md`. These are now treated as experiment history after the ContactSensor fix. Optional tuning hooks remain default-off.
- Note: exact episode-return equality is not expected because IsaacLab imports the URDF through a USD conversion path, uses IsaacLab contact sensors, and runs on different PhysX/Direct workflow dynamics. The next target under Route A is policy/gait robustness analysis with shared default rewards, multiple seeds, and rough-terrain parity, not backend-specific default reward tuning.
