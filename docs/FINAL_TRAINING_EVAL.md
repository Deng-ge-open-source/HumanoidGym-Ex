# Final Training Evaluation

This note records the automatic IsaacGym vs IsaacLab evaluation for the 1000-iteration XBot viewer-training runs completed on 2026-05-23.

## Runs

- IsaacGym: `logs/XBot_ppo/May22_21-02-51_viewer_gym_4096_seed42_1000/model_1000.pt`
- IsaacLab: `logs/XBot_isaaclab/May22_21-03-02_viewer_lab_4096_seed42_1000/model_1000.pt`
- Training scale: `4096 envs x 60 steps x 1000 iterations`
- Fixed-command replay scale: `64 envs x 600 steps`, command `lin_vel_x=0.5`, no randomization/noise/delay

## Training Curve Equivalence

The final training curves are aligned at the PPO-log level.

| Metric | IsaacGym | IsaacLab | Equivalence |
| --- | ---: | ---: | ---: |
| final reward | 166.8300 | 164.7000 | 98.72% |
| final episode length | 2401.0000 | 2374.0800 | 98.88% |
| final reward / step | 0.069484 | 0.069374 | 99.84% |
| tail10 reward | 165.7760 | 165.1530 | 99.62% |
| tail10 episode length | 2401.0000 | 2384.6570 | 99.32% |
| tail10 reward / step | 0.069045 | 0.069257 | 99.69% |

Source files:

- `/tmp/hgex_viewer_gym_4096_seed42_1000.log`
- `/tmp/hgex_viewer_lab_4096_seed42_1000.log`
- `/tmp/hgex_viewer_train_equivalence_1000.json`
- `/tmp/hgex_final_training_curve_compare_1000.csv`

## Native Final-Policy Replay

Each final checkpoint was replayed in its native backend.

| Metric | Gym policy on Gym | Lab policy on Lab | Equivalence |
| --- | ---: | ---: | ---: |
| reward / step | 0.073606 | 0.073871 | 99.64% |
| base lin vel x | 0.380029 | 0.369778 | 97.30% |
| lin vel x abs error | 0.121771 | 0.130415 | 93.37% |
| base z | 0.896806 | 0.895060 | 99.81% |
| feet contact rate | 0.538503 | 0.541003 | 99.54% |
| feet force norm mean | 261.240102 | 260.628291 | 99.77% |
| torque abs | 15.335913 | 15.193900 | 99.07% |
| processed action abs | 0.751597 | 0.699734 | 93.10% |
| done count | 0 | 0 | 100.00% |

Mean equivalence excluding done count: `97.70%`.

## Cross-Backend Policy Replay

The checkpoints were also loaded into the opposite backend to separate policy differences from backend differences.

Same IsaacGym backend:

| Metric | Gym policy | Lab policy | Equivalence |
| --- | ---: | ---: | ---: |
| reward / step | 0.073606 | 0.073704 | 99.87% |
| base lin vel x | 0.380029 | 0.400141 | 94.97% |
| lin vel x abs error | 0.121771 | 0.105513 | 86.65% |
| feet contact rate | 0.538503 | 0.549336 | 98.03% |
| feet force norm mean | 261.240102 | 259.976989 | 99.52% |

Same IsaacLab backend:

| Metric | Gym policy | Lab policy | Equivalence |
| --- | ---: | ---: | ---: |
| reward / step | 0.066882 | 0.073871 | 90.54% |
| base lin vel x | 0.344821 | 0.369778 | 93.25% |
| lin vel x abs error | 0.160729 | 0.130415 | 81.14% |
| feet contact rate | 0.520234 | 0.541003 | 96.16% |
| feet force norm mean | 262.091714 | 260.628291 | 99.44% |

## Interpretation

The final training outcome is equivalent for practical use by training-curve criteria: tail reward, episode length, and reward per step are all above `99%` equivalence. Native final-policy replay is also strong, with a mean equivalence of `97.70%` across core behavior metrics and no resets in either backend over the fixed-command rollout.

The remaining non-negligible gap is concentrated in locomotion style rather than task success:

- forward velocity tracking error,
- processed action amplitude,
- cross-policy behavior when a Gym-trained checkpoint is replayed in IsaacLab.

Contact aggregate metrics are already close in the final native replay. This is better than earlier diagnostic runs where contact reporting was the dominant mismatch. At this point, the final trained policies are usable for viewer comparison and development, but exact per-policy transfer is not fully solved.

## Raw Outputs

- `/tmp/hgex_eval_gym_policy_gym_backend_1000.json`
- `/tmp/hgex_eval_lab_policy_lab_backend_1000.json`
- `/tmp/hgex_eval_lab_policy_gym_backend_1000.json`
- `/tmp/hgex_eval_gym_policy_lab_backend_1000.json`
