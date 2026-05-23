# Design Goals

HumanoidGym-Ex should preserve the Humanoid-Gym workflow first and add backend support second.

## Keep

- Script-centered training and play commands.
- Nested Python config classes.
- `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`.
- Reward functions named `_reward_<name>`.
- Centralized observation, reward, reset, and command logic in humanoid env classes.
- rsl_rl-style PPO environment API.

## Avoid

- Large rewrites before the IsaacGym baseline is verified.
- Manager-based IsaacLab task decomposition.
- Plugin systems or deep simulator abstractions.
- Scattering reward and observation terms across many small files.

## Backend Direction

The backend interface should expose simulator state tensors and a few state write operations. The upper environment should continue to look like Humanoid-Gym.
