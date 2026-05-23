# Roadmap

## Phase 0: Baseline Migration

- Import upstream Humanoid-Gym examples and assets.
- Rename package to `humanoid_gym_ex`.
- Verify IsaacGym train/play/sim2sim entry points.
- Compare short training runs with upstream.

## Phase 1: IsaacGym Backend

- Add `BackendInterface`.
- Move IsaacGym sim creation, tensor refresh, action writes, reset writes, and viewer stepping behind `IsaacGymBackend`. Initial adapter completed; deeper terrain/asset cleanup remains optional.
- Keep reward, observation, reset, and command code centralized.
- Re-run baseline comparison.

## Phase 2: IsaacLab Direct Backend

- Add `IsaacLabBackend`. Initial tensor adapter completed.
- Use Direct workflow, `Articulation`, and scene APIs. Initial XBot-L smoke completed.
- Keep upper reward and observation APIs stable. Initial smoke preserves `705/219/12` dimensions.
- Produce IsaacGym-to-IsaacLab mapping and test report. Initial docs completed.
- IsaacLab Direct has been routed into a PPO-compatible VecEnv wrapper.
- IsaacLab reward dispatch now uses original XBot reward names and scales.
- IsaacLab action delay/noise, push, friction, and base-mass randomization are implemented.
- IsaacLab command curriculum and checkpoint play/export are implemented.
- IsaacLab rough terrain, generated heightfield/trimesh aliases, terrain curriculum, and RayCaster measured heights are implemented.
- 50-iteration and 200-iteration IsaacGym vs IsaacLab curve comparisons are available through `compare_training_curves.py`.
- IsaacLab PD torque now matches IsaacGym's per-physics-substep recomputation model.
- IsaacLab termination now uses contact-force history for termination bodies.
- IsaacLab joint/action/torque tensors are canonicalized to IsaacGym joint order.
- IsaacLab reference DOF targets are canonicalized to the same joint order.
- IsaacLab nominal rigid-shape material/contact defaults are aligned with the IsaacGym XBot baseline before optional randomization.
- IsaacLab PhysX solver settings are aligned with the IsaacGym XBot baseline for solver type, iteration counts, bounce threshold, contact buffer, and max depenetration velocity.
- Optional strict fall termination switches and `--parity_termination_profile isaacgym_like` are available for comparability experiments.
- Rollout diagnostics for zero/random and trained-policy actions are available for backend alignment.
- Trained-policy diagnostics export per-step traces and shape/material summaries.
- IsaacLab train now sets the same default seed before environment creation as the IsaacGym train entry.
- Local automatic smoke validation is available through `humanoid_gym_ex/scripts/validate_smoke.sh`.
- IsaacLab ContactSensor forces are reordered into `Articulation.body_names` order before rewards, observations, and termination checks use them. This fixed a left/right foot-contact data mismatch without changing default reward values.
- Current status: broad IsaacGym/IsaacLab curve parity is effectively resolved for the 200-iteration default protocol. After the ContactSensor mapping fix, IsaacLab tail10 mean reward is `2.855` vs IsaacGym `2.858`, and tail10 episode length is `151.700` vs `151.775`. Same-policy replay of the IsaacGym checkpoint is also close across backends (`0.326667` vs `0.330000` done/step).
- Follow-up seed check: seed `1` is less aligned than seed `5` at 200 iterations (`2.690 / 151.678` IsaacGym vs `3.082 / 160.071` IsaacLab). Treat the seed `5` result as a strong positive signal, not a full benchmark claim.
- Seed `1` fixed-command replay confirms same-policy backend transfer remains close, but the IsaacLab-trained policy still selects a faster and lower gait than the IsaacGym-trained policy.
- IsaacLab timeout bootstrap parity is fixed. `truncated` is now forwarded as `infos["time_outs"]`, matching IsaacGym PPO return handling. Seed `1` IsaacLab 200-iteration tail10 improved from `3.082 / 160.071` to `2.970 / 155.618`, closer to IsaacGym `2.690 / 151.678`.
- IsaacGym rough measured-height training now works from CLI and builds a `780`-dim critic, matching the intended rough-height critic shape.
- Rough measured-height 1000-iteration check is complete (`138.464 / 2357.588` IsaacGym tail10 vs `97.214 / 2030.989` IsaacLab tail10). This verifies long-run training functionality, but rough-terrain convergence equivalence remains open.
- Next: keep Route A as the default. Do not add backend-specific default reward tuning. Remaining work should focus on a 3-seed or 5-seed 200/500-iteration plane matrix, mirrored IsaacLab terrain-level logging, rough 200-iteration runs, and fixed-command replay for each trained checkpoint. Optional reward-shaping hooks stay experimental and default-off.

## Phase 3: Genesis Placeholder

- Add a clear `GenesisBackend` stub.
- Document expected tensor contract.
- Do not let Genesis requirements complicate IsaacGym or IsaacLab code.
