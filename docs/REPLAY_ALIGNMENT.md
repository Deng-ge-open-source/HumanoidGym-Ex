# Replay Alignment Report

## Reward-Preserved Phase/Timeout Alignment Follow-Up

Date: 2026-05-22

This pass kept the default reward implementation shared between IsaacGym and IsaacLab. The changes were limited to non-reward parity:

- IsaacLab forwards `truncated` as PPO `infos["time_outs"]`, matching IsaacGym timeout bootstrap semantics.
- IsaacLab timeout termination uses the same `episode_length_buf > max_episode_length` condition as IsaacGym.
- IsaacLab gait phase compensates DirectRLEnv's first-observation episode-length offset so phase-dependent observations, reference actions, gait masks, and reward inputs see the same control-step timing as Humanoid-Gym.
- Replay diagnostics now report reset snapshots only on steps that actually reset an environment.

Static parity check:

```text
reward parity ok
```

Random-action timing probe, `64 envs x 240 steps`, seed `1`:

| Metric | IsaacGym | IsaacLab | Delta |
| --- | ---: | ---: | ---: |
| first-step phase mean | 0.031250 | 0.031250 | 0.000000 |
| first-step left stance rate | 1.000000 | 1.000000 | 0.000000 |
| first-step right stance rate | 0.000000 | 0.000000 | 0.000000 |
| rollout reward/step | 0.030169 | 0.030331 | 0.000162 |
| rollout done/step | 0.262500 | 0.266667 | 0.004167 |
| feet contact rate | 0.699870 | 0.694531 | -0.005339 |
| torque abs | 19.634413 | 19.848343 | 0.213930 |

Seed `1`, plane, `64 envs x 60 steps x 200 iterations` training after the phase fix:

| Backend | Tail10 mean reward | Tail10 episode length | Final reward | Final episode length |
| --- | ---: | ---: | ---: | ---: |
| IsaacGym | 2.690 | 151.678 | 2.720 | 150.450 |
| IsaacLab | 2.997 | 155.626 | 2.880 | 149.290 |

Fixed-command replay, `64 envs x 300 steps`, seed `1`, command `lin_vel_x=0.5`:

| Case | reward/step | done/step | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym policy on IsaacGym | 0.032048 | 0.303333 | 0.814237 | 0.656003 | 0.613973 | 15.193214 | 0.644036 |
| IsaacGym policy on IsaacLab | 0.031281 | 0.313333 | 0.818472 | 0.564146 | 0.571093 | 15.394991 | 0.647630 |
| IsaacLab policy on IsaacGym | 0.032503 | 0.356667 | 0.794385 | 0.777684 | 0.673830 | 13.801786 | 0.658776 |
| IsaacLab policy on IsaacLab | 0.032677 | 0.273333 | 0.808790 | 0.590913 | 0.608759 | 13.972515 | 0.666875 |

Interpretation:

- The same IsaacGym policy transfers closely: reward differs by about `2.4%`, done/step by `0.010`, base height by `0.004`, and foot-contact occupancy by `0.004`.
- The same IsaacLab policy has close reward but larger done-rate spread in this short replay window. It also runs faster and lower on IsaacGym, which points to residual contact/termination boundary sensitivity rather than reward-formula mismatch.
- The remaining gap is not safe to call fully solved for all seeds, but it is now narrow enough for IsaacLab to be usable as a default-reward backend while continuing physics/contact parity work.

## Deterministic Action Trace Probe

Date: 2026-05-22

Added backend-neutral deterministic replay tooling:

- `humanoid_gym_ex/scripts/generate_action_trace.py` writes an `npz` action tensor with shape `[steps, num_envs, num_actions]`.
- `diagnose_isaacgym_rollout.py` can consume it through `HGEX_DIAG_ACTION_FILE`.
- `diagnose_isaaclab_rollout.py` can consume it through `--action_file`.
- `humanoid_gym_ex/scripts/compare_rollout_traces.py` compares per-step trace JSON files and reports the first field that crosses a drift threshold.

The diagnostic mode now disables observation noise, domain randomization, pushes, action delay, and action noise, so the replayed raw and processed actions are identical across backends.

Command pattern:

```bash
python humanoid_gym_ex/scripts/generate_action_trace.py \
  --output /tmp/hgex_actions_seed1_120x64x12.npz --steps 120 --num_envs 64 --num_actions 12 --seed 1

HGEX_DIAG_STEPS=120 HGEX_DIAG_ACTION_FILE=/tmp/hgex_actions_seed1_120x64x12.npz \
HGEX_DIAG_SEED=1 HGEX_DIAG_OUTPUT=/tmp/hgex_det_clean_isaacgym_120.json \
HGEX_DIAG_TRACE_OUTPUT=/tmp/hgex_det_clean_isaacgym_120_trace.json \
conda run -n legged_gym python humanoid_gym_ex/scripts/diagnose_isaacgym_rollout.py \
  --task=humanoid_ppo --headless --num_envs 64

PYTHONPATH=/home/cra02/Documents/GitHub/HumanoidGym-Ex \
conda run -p /home/cra02/anaconda3/envs/env_isaaclab \
python humanoid_gym_ex/scripts/diagnose_isaaclab_rollout.py \
  --headless --num_envs 64 --steps 120 --seed 1 \
  --action_file /tmp/hgex_actions_seed1_120x64x12.npz \
  --output /tmp/hgex_det_clean_isaaclab_120.json \
  --trace_output /tmp/hgex_det_clean_isaaclab_120_trace.json
```

Summary, fixed action trace, `64 envs x 120 steps`, seed `1`:

| Metric | IsaacGym | IsaacLab | Delta |
| --- | ---: | ---: | ---: |
| processed action abs | 0.500890 | 0.500890 | 0.000000 |
| reward/step | 0.028153 | 0.028286 | 0.000133 |
| done/step | 0.050000 | 0.033333 | -0.016667 |
| base_z | 0.835077 | 0.832547 | -0.002530 |
| base_vx | -0.217734 | -0.283732 | -0.065998 |
| feet contact rate | 0.660742 | 0.638802 | -0.021940 |
| feet force norm mean | 273.295111 | 250.388928 | -22.906182 |
| torque abs | 22.679612 | 22.669585 | -0.010028 |

First drift reported by `compare_rollout_traces.py` with threshold `0.05`:

| Step | Field | IsaacGym | IsaacLab | Delta |
| --- | --- | ---: | ---: | ---: |
| 7 | feet_force_norm_mean | 13.872292 | 0.000000 | -13.872292 |
| 7 | feet_contact_rate | 0.031250 | 0.000000 | -0.031250 |
| 7 | processed_action_abs | 0.509216 | 0.509216 | 0.000000 |
| 7 | phase_mean | 0.140625 | 0.140625 | 0.000000 |
| 7 | reward_mean | 0.044860 | 0.044545 | -0.000315 |

Interpretation:

- With actions, action preprocessing, and phase matched, the first clear divergence is foot contact onset/measurement.
- Reward remains nearly identical at the drift step, so this is not a reward-definition issue.
- The next useful parity target is asset/contact modeling: foot collision shape import, contact sensor reporting, contact offset/rest offset behavior, and any IsaacLab/Isaac Sim URDF merge side effects around foot links.

## Asset And Reset-Isolated Contact Probe

Date: 2026-05-22

Added two diagnostic-only alignment tools:

- `compare_asset_summaries.py` compares the `asset` section exported by the Gym/Lab rollout diagnostics.
- `HGEX_DIAG_DETERMINISTIC_RESET=1` and IsaacLab `--deterministic_reset` remove reset-time joint perturbation from replay diagnostics.

Asset comparison results:

| Field | Result |
| --- | --- |
| body-name set | equal |
| joint-name set | equal |
| body order | different, expected from IsaacLab import and handled by mapping |
| joint order | different, handled by canonical joint mapping |
| mass sum delta | `1.15e-7` |
| max inertia delta by body | `2.38e-7` |
| rigid shapes | `61` in both |
| material/contact offsets | matched to numerical precision |

Deterministic-reset replay, `64 envs x 120 steps`, fixed action trace:

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

First drift is still step `7`, and still in foot contact force/contact occupancy. This narrows the remaining mismatch to contact onset/reporting and downstream PhysX divergence. It is not caused by reward scale/formula differences, action preprocessing, gait phase, reset joint noise, mass, inertia, or nominal material/contact-offset settings.

## ContactSensor Mapping Update

Date: 2026-05-22

IsaacLab `ContactSensor.body_names` can be ordered differently from `Articulation.body_names` for the same URDF. XBot imports in IsaacLab with an interleaved left/right articulation body order, while the contact sensor reports the original imported body order. Before the fix, indexing `ContactSensor.data.net_forces_w` with robot body indices caused left-foot contact metrics to read the wrong body.

The IsaacLab backend now reorders contact forces from `ContactSensor.body_names` into `robot.body_names` order before upper-level reward, observation, and termination logic uses them. This is a Route A fix: it keeps the default reward formulas and scales identical, and corrects the data fed into those formulas.

Post-fix zero/random action probes:

| Probe | IsaacGym | IsaacLab | Delta |
| --- | ---: | ---: | ---: |
| zero feet contact rate | 0.919401 | 0.911198 | -0.008203 |
| zero feet force norm mean | 259.090257 | 257.100527 | -1.989730 |
| random reward/step | 0.031600 | 0.031820 | 0.000221 |
| random done/step | 0.016667 | 0.016667 | 0.000000 |
| random feet contact rate | 0.744987 | 0.722266 | -0.022721 |
| random feet force norm mean | 267.776801 | 250.819562 | -16.957239 |
| random processed action abs | 0.389777 | 0.389122 | -0.000654 |

Post-fix 200-iteration default-reward training:

| Backend | Tail10 mean reward | Tail10 episode length |
| --- | ---: | ---: |
| IsaacGym baseline | 2.858 | 151.775 |
| IsaacLab ContactSensor mapping fix | 2.855 | 151.700 |

Post-fix fixed-command replay, `64 envs x 600 steps`, seed `5`, command `lin_vel_x=0.5`:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact | feet force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym policy on IsaacGym | 0.031636 | 0.326667 | 196 | 0.809198 | 0.621608 | 0.638970 | 14.578374 | 0.702500 | 240.154474 |
| IsaacGym policy on IsaacLab | 0.032153 | 0.330000 | 198 | 0.811960 | 0.626922 | 0.608523 | 14.363855 | 0.701016 | 222.494156 |
| IsaacLab ContactSensor policy on IsaacGym | 0.033405 | 0.400000 | 240 | 0.795448 | 0.822486 | 0.711925 | 13.985397 | 0.737018 | 211.414168 |
| IsaacLab ContactSensor policy on IsaacLab | 0.033748 | 0.355000 | 213 | 0.809665 | 0.708755 | 0.637380 | 14.277981 | 0.723060 | 217.086422 |

Interpretation after the fix:

- Same-policy backend transfer is now close for the IsaacGym reference policy: done rate, base height, forward velocity, torque, and foot-contact occupancy are all in the same band.
- The default 200-iteration training curves are aligned closely enough that broad backend/reward plumbing is no longer the main issue.
- The remaining non-ignorable difference is learned-policy gait robustness under fixed command. The IsaacLab-trained policy still falls more often than the IsaacGym policy and tends to choose a faster gait, especially when replayed in IsaacGym.
- Default reward values remain shared across IsaacGym and IsaacLab. The optional reward tuning hooks are experimental and should not be promoted to defaults unless the project intentionally chooses backend-specific policy shaping.

## Seed 1 Replay Follow-Up

The seed `1` 200-iteration plane curves were less aligned than seed `5`, so the checkpoints were replayed under the same fixed command protocol:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym seed1 policy on IsaacGym | 0.031706 | 0.365000 | 219 | 0.809257 | 0.687142 | 0.626114 | 15.147251 | 0.640247 |
| IsaacGym seed1 policy on IsaacLab | 0.031665 | 0.365000 | 219 | 0.814243 | 0.618381 | 0.607897 | 15.460887 | 0.658451 |
| IsaacLab seed1 policy on IsaacGym | 0.036764 | 0.388333 | 233 | 0.777798 | 0.810515 | 0.701024 | 12.865049 | 0.764505 |
| IsaacLab seed1 policy on IsaacLab | 0.037410 | 0.363333 | 218 | 0.785402 | 0.764769 | 0.687618 | 12.607409 | 0.765052 |

Interpretation:

- Same-policy backend transfer remains close for seed `1`. The IsaacGym-trained policy has identical done rate on both backends, and the IsaacLab-trained policy is close across backends.
- The policy-quality gap remains: the IsaacLab-trained policy runs faster and lower than the IsaacGym-trained policy.
- This reinforces Route A: backend execution is close enough to expose learned gait differences, so default reward values should remain shared while multi-seed PPO/terrain behavior is measured.

After fixing IsaacLab timeout bootstrap (`truncated` forwarded as `infos["time_outs"]`), the seed `1` IsaacLab checkpoint improved:

| Case | reward/step | done/step | events | base_z | base_vx | vx abs err | torque_abs | feet contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacLab seed1 before timeout fix on IsaacGym | 0.036764 | 0.388333 | 233 | 0.777798 | 0.810515 | 0.701024 | 12.865049 | 0.764505 |
| IsaacLab seed1 before timeout fix on IsaacLab | 0.037410 | 0.363333 | 218 | 0.785402 | 0.764769 | 0.687618 | 12.607409 | 0.765052 |
| IsaacLab seed1 timeout fix on IsaacGym | 0.036410 | 0.376667 | 226 | 0.794192 | 0.845014 | 0.698219 | 13.949369 | 0.744622 |
| IsaacLab seed1 timeout fix on IsaacLab | 0.036887 | 0.350000 | 210 | 0.801025 | 0.779176 | 0.674677 | 13.891590 | 0.735143 |

| Case | reward/step | done/step | events | event len mean | event len p50 | base_z | base_vx | dof_vel_abs | torque_abs | action_abs | action_delta_abs | feet contact | feet force | first done step | peak done/25 steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IsaacGym policy on IsaacGym | 0.031636 | 0.326667 | 196 | 163.490 | 161 | 0.809198 | 0.621608 | 0.484421 | 14.578374 | 0.448625 | 0.057490 | 0.702500 | 240.154474 | 129 | 36 |
| IsaacGym policy on IsaacLab | 0.030735 | 0.330000 | 198 | 160.641 | 157 | 0.811960 | 0.626922 | 0.496105 | 14.363855 | 0.441567 | 0.039190 | 0.318125 | 82.198981 | 129 | 41 |
| IsaacLab policy on IsaacGym | 0.033911 | 0.423333 | 254 | 137.327 | 135 | 0.780359 | 0.854383 | 0.557796 | 13.479199 | 0.445416 | 0.053243 | 0.729336 | 209.988899 | 119 | 55 |
| IsaacLab policy on IsaacLab | 0.035242 | 0.426667 | 256 | 138.379 | 138 | 0.783465 | 0.856413 | 0.531568 | 12.684235 | 0.429255 | 0.039527 | 0.311419 | 90.533151 | 123 | 56 |

## Same-Policy Backend Transfer

| Policy | Delta reward | Delta done | Delta base_z | Delta base_vx | Delta dof_vel_abs | Delta torque_abs | Delta action_delta | Delta feet contact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IsaacGym policy | -0.000901 | 0.003333 | 0.002763 | 0.005314 | 0.011684 | -0.214519 | -0.018300 | -0.384375 |
| IsaacLab policy | 0.001331 | 0.003333 | 0.003105 | 0.002030 | -0.026229 | -0.794964 | -0.013716 | -0.417917 |

## Same-Backend Policy Difference

| Backend | LabPolicy-GymPolicy reward | done | base_z | base_vx | dof_vel_abs | torque_abs | action_delta | feet contact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IsaacGym | 0.002276 | 0.096667 | -0.028839 | 0.232775 | 0.073375 | -1.099176 | -0.004247 | 0.026836 |
| IsaacLab | 0.004507 | 0.096667 | -0.028496 | 0.229491 | 0.035463 | -1.679620 | 0.000337 | -0.006706 |

## Top Reward-Term Deltas


### IsaacGym backend

| Reward term | IsaacGym policy | IsaacLab policy | Delta |
| --- | --- | --- | --- |
| tracking_ang_vel | 0.006545 | 0.009095 | 0.002550 |
| feet_contact_number | 0.006822 | 0.004922 | -0.001901 |
| default_joint_pos | 0.001333 | 0.002391 | 0.001058 |
| feet_contact_forces | -0.001303 | -0.000254 | 0.001048 |
| track_vel_hard | -0.000543 | -0.000007 | 0.000536 |
| tracking_lin_vel | 0.004600 | 0.004091 | -0.000510 |
| joint_pos | 0.006125 | 0.005694 | -0.000431 |
| feet_distance | 0.001375 | 0.001786 | 0.000411 |

### IsaacLab backend

| Reward term | IsaacGym policy | IsaacLab policy | Delta |
| --- | --- | --- | --- |
| tracking_ang_vel | 0.006478 | 0.009201 | 0.002723 |
| default_joint_pos | 0.001322 | 0.003044 | 0.001722 |
| feet_contact_number | 0.005194 | 0.004588 | -0.000606 |
| track_vel_hard | -0.000505 | 0.000079 | 0.000585 |
| feet_distance | 0.001352 | 0.001831 | 0.000479 |
| tracking_lin_vel | 0.004598 | 0.004164 | -0.000434 |
| vel_mismatch_exp | 0.002002 | 0.001628 | -0.000374 |
| joint_pos | 0.006153 | 0.005879 | -0.000274 |

## Termination Causes

| Case | contact | time_out | base_height | orientation |
| --- | --- | --- | --- | --- |
| IsaacGym policy on IsaacGym | 196 | 0 | 0 | 0 |
| IsaacGym policy on IsaacLab | 198 | 0 | 0 | 0 |
| IsaacLab policy on IsaacGym | 254 | 0 | 0 | 0 |
| IsaacLab policy on IsaacLab | 256 | 0 | 0 | 0 |

## Interpretation

- Same-policy backend transfer is close: the IsaacLab policy has nearly identical done rate on IsaacGym and IsaacLab, so the current replay gap is not mainly a backend execution mismatch.
- Same-backend policy comparison shows the IsaacLab-trained policy is less robust under the fixed replay command: done rate is higher and average base height is lower on both backends.
- Foot-contact rate should be compared mainly within the same backend. IsaacGym net contact forces and IsaacLab ContactSensor history report different absolute contact occupancy, but the same-backend policy delta is small.
- The largest reward-term shifts should be inspected before changing environment physics. They point to gait/contact and command-tracking tradeoffs in the learned policy.
