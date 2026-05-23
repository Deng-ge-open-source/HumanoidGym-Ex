# Multi-Seed IsaacGym / IsaacLab Alignment

Date: 2026-05-22

This report keeps the default IsaacGym and IsaacLab reward definitions identical. No backend-specific reward scale or formula changes were used.

## Setup

- Robot/task: XBot plane task.
- Backends: IsaacGym and IsaacLab Direct workflow.
- Seeds: `1, 2, 3, 4, 5`.
- Training horizon: `64 envs x 60 steps x 200 iterations`.
- Comparison window: tail 10 printed PPO iterations.
- Equivalence metric: `min(gym, lab) / max(gym, lab) * 100`.

Seed `1` reused the phase-aligned 200-iteration logs because the later changes only affected diagnostics and rough/generated-terrain reset placement, not plane reward or training logic. Seeds `2..5` were freshly run in this pass.

## Results

| Seed | Gym Tail Reward | Lab Tail Reward | Reward Eq. | Gym Tail Len | Lab Tail Len | Len Eq. | Gym Reward/Step | Lab Reward/Step | Reward/Step Eq. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.690 | 2.997 | 89.76% | 151.678 | 155.626 | 97.46% | 0.017735 | 0.019258 | 92.09% |
| 2 | 2.937 | 2.933 | 99.86% | 157.008 | 156.332 | 99.57% | 0.018706 | 0.018761 | 99.71% |
| 3 | 2.909 | 3.015 | 96.48% | 155.372 | 156.144 | 99.51% | 0.018723 | 0.019309 | 96.96% |
| 4 | 2.725 | 3.074 | 88.65% | 148.315 | 157.598 | 94.11% | 0.018373 | 0.019505 | 94.20% |
| 5 | 2.858 | 3.055 | 93.55% | 151.775 | 158.534 | 95.74% | 0.018831 | 0.019270 | 97.72% |

Aggregate:

| Metric | Mean Eq. | Min Eq. | Max Eq. |
| --- | ---: | ---: | ---: |
| Tail reward | 93.66% | 88.65% | 99.86% |
| Tail episode length | 97.28% | 94.11% | 99.57% |
| Tail reward/step | 96.13% | 92.09% | 99.71% |

## Interpretation

The multi-seed result is mixed:

- If the acceptance metric is tail reward/step, the mean equivalence is above 95%.
- If the acceptance metric is raw tail episode reward, the mean equivalence is below 95%, and seeds `1`, `4`, and `5` do not meet the requested threshold.
- The Lab runs are usually higher reward and longer episode length, so the mismatch is not random sign noise.

The largest final-iteration reward-term deltas on failing seeds are contact and contact-adjacent terms:

| Seed | Largest Deltas |
| --- | --- |
| 1 | `joint_pos`, `feet_contact_number`, `tracking_ang_vel`, `feet_contact_forces` |
| 4 | `feet_contact_forces`, `feet_contact_number`, `joint_pos`, `tracking_lin_vel`, `tracking_ang_vel` |
| 5 | `low_speed`, `feet_contact_number`, `tracking_ang_vel`, `joint_pos`, `feet_contact_forces` |

This matches the deterministic fixed-action finding: actions, phase, static asset parameters, mass, inertia, friction, and contact offsets are aligned, but the first rollout drift appears at foot contact onset/reporting.

## Required Fix Direction

Do not fix this by changing reward scales. The next fixes should keep reward formulas identical and make IsaacLab provide more IsaacGym-like inputs:

1. Replace IsaacLab's upper-level `contact_forces` source with a Gym-like aggregated contact tensor derived from `ContactSensor.net_forces_w_history`, then rerun deterministic step-7 and multi-seed training.
2. Add a foot-only contact microbenchmark to compare first-contact frame, peak normal force, and contact duration for IsaacGym vs IsaacLab.
3. Export and compare foot collision shape pose/geometry ownership, not just global shape count/materials. If foot collision poses differ after URDF fixed-joint merge, fix the IsaacLab asset import or add an asset preprocessing path.
4. If contact tensor aggregation improves deterministic contact onset but training still misses the threshold, run a small PhysX contact-parameter sweep limited to non-reward settings: contact offset, rest offset, solver velocity iterations, and contact sensor update/history semantics.

Current status: usable, but not yet passing a strict `>95%` raw tail reward multi-seed equivalence criterion.
