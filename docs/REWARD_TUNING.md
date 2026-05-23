# IsaacLab Reward Tuning Notes

Date: 2026-05-22

This note tracks narrow IsaacLab reward/command-shaping experiments for the remaining fixed-command gait/contact gap. These experiments do not change the default Humanoid-Gym-compatible reward values.

## Status After Route A Contact Fix

The reward-tuning results below are kept as experiment history. They were run before the IsaacLab ContactSensor body-order mapping fix.

After that fix, default IsaacLab reward values are still shared with IsaacGym, and the 200-iteration default-reward curve is aligned with the IsaacGym baseline (`2.855` vs `2.858` tail10 mean reward, `151.700` vs `151.775` tail10 episode length in the current log scale). Fixed-command replay also improved: the default IsaacLab `model_200.pt` now records `0.355000` done/step in IsaacLab, compared with `0.421667` for the older seeded-bucket policy.

Do not promote the optional tuning hooks to defaults. Route A should continue to fix data, physics, ordering, and randomization mismatches first, while keeping reward names and default values identical across IsaacGym and IsaacLab.

## Optional Tuning Hooks

`train_isaaclab.py` now supports local reward tuning without editing config files:

```bash
--reward_scale tracking_lin_vel=1.5
--reward_scale track_vel_hard=0.65
--reward_scale low_speed=0.3
--reward_scale feet_contact_number=1.4
--reward_scale tracking_ang_vel=1.0
--reward_param high_speed_penalty=1.0
```

`high_speed_penalty` is an extension parameter in `XBotLCfg.rewards`. Its default is `0.0`, which preserves the original Humanoid-Gym `low_speed` behavior where overspeed gets zero reward rather than a negative reward.

## Training Curves

Baseline references:

| Run | Iterations | Tail10 reward/step | Tail10 episode length |
| --- | ---: | ---: | ---: |
| IsaacGym baseline | 50 | 0.015325 | 128.552 |
| IsaacLab seeded bucket friction | 50 | 0.015411 | 136.976 |
| IsaacGym baseline | 200 | 0.018831 | 151.775 |
| IsaacLab seeded bucket friction | 200 | 0.018767 | 159.801 |

Tuning results:

| Run | Iterations | Tail10 reward/step | Tail10 episode length | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Strong lin/contact scale tuning | 50 | 0.015411 | 135.552 | Reduced fixed-command speed too much in replay; not selected. |
| Mild lin/contact scale tuning | 50 | 0.015792 | 142.798 | Improved short replay, but 200-iteration replay regressed to high speed. |
| Mild lin/contact scale tuning | 200 | 0.019352 | 154.710 | Good curve, but fixed-command replay still overspeeds. |
| Mild tuning + `high_speed_penalty=1.0` | 50 | 0.016181 | 136.520 | Best short replay result. |
| Mild tuning + `high_speed_penalty=1.0` | 200 | 0.019088 | 155.752 | Partial long-run replay improvement, but still overspeeds. |
| Mild tuning + `high_speed_penalty=2.0` | 50 | 0.013958 | 133.116 | Too strong; training reward/step regressed. |

## Fixed-Command Replay

All rows use IsaacLab backend, `64 envs x 600 steps`, fixed command `lin_vel_x=0.5`, seed `5`.

| Policy | done/step | event len p50 | base_vx | base_z | reward/step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IsaacGym 200 policy | 0.330000 | 157 | 0.626922 | 0.811960 | 0.030735 | Current reference behavior. |
| IsaacLab seeded bucket 200 policy | 0.426667 | 138 | 0.856413 | 0.783465 | 0.035242 | Overspeeds and falls more often. |
| Mild tuning 50 policy | 0.345000 | 153 | 0.689128 | 0.801994 | 0.031656 | Good short-run signal. |
| Mild tuning + `high_speed_penalty=1.0`, 50 policy | 0.323333 | 161 | 0.659919 | 0.808862 | 0.037654 | Best short-run replay result. |
| Mild tuning 200 policy | 0.426667 | 135 | 0.857958 | 0.787704 | 0.033563 | Long-run policy reverts to overspeed. |
| Mild tuning + `high_speed_penalty=1.0`, 200 policy | 0.396667 | 145 | 0.824049 | 0.791510 | 0.032145 | Partial long-run improvement. |

## Current Conclusion

The remaining gap is not fixed by reward-scale changes alone. The new `high_speed_penalty` hook confirms that explicit overspeed penalty is the right direction, but `1.0` is not strong enough over 200 iterations and `2.0` hurts the 50-iteration learning curve.

Do not promote the tuning values to defaults yet. The next useful experiment is a shaped overspeed penalty that grows with speed error, instead of a constant penalty inside `low_speed`.
