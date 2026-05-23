# Original Scene Matrix

This report records the original Humanoid-Gym XBot rough-terrain scene validation after porting the IsaacLab Direct path.

## Scope

- Robot: XBot-L migrated from the original Humanoid-Gym tree.
- Scene: rough terrain with measured heights and terrain curriculum.
- Rewards: default XBot reward names and scales are shared by IsaacGym and IsaacLab.
- Actor/critic observations: `705 / 780` in both backends for rough measured-height mode.
- Training: `4096` envs, `60` steps per env, `1000` iterations, seed `42`.
- Replay: final checkpoints, `64` envs, `600` fixed-command steps, command `x=0.5`.
- Extra scene checks: `heightfield` and `trimesh` aliases, one PPO iteration each.

## Artifacts

| Artifact | Path |
| --- | --- |
| IsaacGym rough log | `/tmp/hgex_rough_heights_curric_gym_4096_seed42_1000.log` |
| IsaacLab rough log | `/tmp/hgex_rough_heights_curric_lab_critic_only_4096_seed42_1000.log` |
| Matrix output dir | `/tmp/hgex_original_scene_matrix` |
| IsaacGym checkpoint | `logs/XBot_ppo/May23_01-27-07_rough_heights_curric_gym_4096_seed42_1000/model_1000.pt` |
| IsaacLab checkpoint | `logs/XBot_isaaclab/May23_01-32-55_rough_heights_curric_lab_critic_only_4096_seed42_1000/model_1000.pt` |

## Training Curves

| Backend | Final reward | Final ep len | Final reward/step | Tail10 reward | Tail10 ep len | Tail10 reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IsaacGym | 138.180 | 2346.220 | 0.058895 | 138.464 | 2357.588 | 0.058731 |
| IsaacLab | 93.170 | 1947.270 | 0.047846 | 97.214 | 2030.989 | 0.047865 |
| Equivalence | 67.43% | 83.00% | 81.24% | 70.21% | 86.15% | 81.50% |

The rough-terrain long run is functional but not equivalent. The gap is too large to ignore if the goal is a strict IsaacGym reproduction on rough terrain.

## Final Checkpoint Replay

| Policy/backend | Reward | Done/step | Base z | Vel x | Vel x abs err | Torque abs | Foot force mean | Action abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gym policy on Gym | 0.063207 | 0.000000 | 0.882592 | 0.382241 | 0.142760 | 14.911716 | 259.087410 | 0.760616 |
| Gym policy on Lab | 0.048904 | 0.111667 | 0.866756 | 0.331553 | 0.326908 | 16.709065 | 249.424714 | 0.829394 |
| Lab policy on Gym | 0.053092 | 0.000000 | 0.882542 | 0.289847 | 0.217177 | 16.499144 | 260.312120 | 0.748381 |
| Lab policy on Lab | 0.048751 | 0.020000 | 0.887020 | 0.255516 | 0.285034 | 16.280939 | 258.361645 | 0.742581 |

Same-policy backend transfer is mixed:

| Pair | Core metric mean equivalence |
| --- | ---: |
| Gym policy, Gym vs Lab backend | 83.32% |
| Lab policy, Gym vs Lab backend | 93.26% |

The IsaacLab-trained policy transfers more consistently across Gym/Lab than the IsaacGym-trained rough policy, but its long-run training curve is still below the IsaacGym curve.

## Heightfield And Trimesh Smoke

| Terrain alias | IsaacGym | IsaacLab |
| --- | --- | --- |
| `heightfield` | Passed one PPO iteration, reward `0.53`, episode length `29.50` | Passed one PPO iteration, reward `0.53`, episode length `28.50` |
| `trimesh` | Passed one PPO iteration, reward `0.39`, episode length `22.33` | Passed one PPO iteration, reward `0.53`, episode length `28.50` |

Full smoke validation also passed:

```text
[validate] compile
[validate] reward parity
[validate] isaacgym train smoke
[validate] isaaclab randomization smoke
[validate] isaaclab rough terrain heights smoke
[validate] isaaclab ppo smoke
[validate] isaaclab play/export smoke
[validate] ok
```

## Conclusion

The original rough-terrain training scene is migrated far enough to run long training, checkpoint replay, heightfield/trimesh aliases, measured heights, and terrain curriculum in both IsaacGym and IsaacLab. The reward implementation remains shared and default reward values are unchanged.

It is not yet a strict rough-terrain reproduction. Plane training was already above the 95% equivalence target, but rough terrain is currently below that target, especially in tail reward and velocity tracking.

## Next Fixes Without Changing Rewards

1. Compare terrain sampling distributions directly: terrain level, type, env origin, roughness scale, height sample grid, and command distribution over time.
2. Run deterministic action-trace comparison on rough terrain, as was done for plane terrain, to locate the first divergence between terrain height sampling, contact onset, and reset placement.
3. Compare IsaacGym heightfield/trimesh geometry against the generated IsaacLab USD mesh numerically: height grid min/max/mean, border size, horizontal/vertical scale, and curriculum terrain type layout.
4. Add rough-terrain replay with recorded command and terrain-origin traces so both backends replay the same terrain cells, not only the same command.
5. Inspect contact reporting on rough terrain. The remaining plane gap was contact-onset related; rough terrain magnifies the same class of difference.
