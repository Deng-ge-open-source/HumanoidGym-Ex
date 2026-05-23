# HumanoidGym-Ex

**中文 | [English](#english)**

HumanoidGym-Ex is not a new humanoid learning framework from scratch. It is a Humanoid-Gym-style extension framework that preserves the original Humanoid-Gym user experience while enabling future IsaacLab and Genesis backends.

The main work in this repository is the migration of the original Humanoid-Gym codebase plus a Humanoid-Gym-style IsaacLab Direct backend. IsaacGym remains the compatibility baseline; IsaacLab / Isaac Sim is the primary extension path.

HumanoidGym-Ex 是 **Humanoid-Gym 库的移植版，它使得HumanoidGym项目可以在目前最新的IsaacLab/Sim环境上进行训练，原始的老版本Humanoid-Gym库只支持Isaac Gym训练，在工具链上已经便老旧，因此我进行了移植，希望能够对大家开发人形机器人有帮助。（原仓库（https://github.com/roboterax/humanoid-gym））**：

移植版本被命名为HumanoidGym-Ex，它保留原版 Humanoid-Gym 的脚本入口、配置方式、reward 写法、PPO 接口和使用习惯，可以同时支持IsaacLab训练和原版Isaac Gym的训练和训练脚本编写方式。你可以直接继承原版Humanoid-Gym /Legged Gym的脚本风格，无缝把原属于Humanoid-Gym 的Isaac Gym训练工程该用Isaac Lab进行训练。

为了确保移植的可靠性，我进行了我移植版本的HumanoidGym-Ex(在Isaac Lab中)和原版HumanoidGym（在Isaac Gym中）的训练结果对比，对比表如下：

【AI请补充对比表】

对比结果表明，移植后的库非常好用，可以用于接近原版的HumanoidGym 项目在Isaac Lab/sim上的部署和训练。


本项目由 **灯哥开源** 移植与维护。

- B 站主页：https://space.bilibili.com/493192058
- 迁移目标：让熟悉 Humanoid-Gym 的用户可以用接近原版的方式进入 IsaacLab / Isaac Sim 生态。

## 效果动图

以下HumanoidGym-Ex 在 IsaacLab训练后的结果，与 IsaacGym 原版HumanoidGym的训练结果并列播放，大家可以直接看到训练效果：

![IsaacLab and IsaacGym walking policies](docs/assets/isaaclab_isaacgym_walk.gif)

静态截图：

HumanoidGym 原版训练后策略播放：

![IsaacGym play](docs/assets/isaacgym_play_viewer.png)

HumanoidGym-Ex在IsaacLab / Isaac Sim 训练后的策略播放：

![IsaacLab play](docs/assets/isaaclab_play_viewer.png)

## 为什么做 HumanoidGym-Ex

原版 Humanoid-Gym 的优点是非常直接：

- `train.py` / `play.py` 脚本中心化，学习成本低。
- robot config、reward scale、observation、reset、command curriculum 都在熟悉的位置。
- reward 函数集中在环境类里，便于调试和快速实验。
- PPO 接口基于 `rsl_rl` 风格，训练链路简单。

IsaacLab / Isaac Sim 的生态更现代，但 它的新的工作流（Manager-based workflow ）对很多从 Humanoid-Gym / Legged-Gym 迁移过来的用户来说会显得有学习成本，因此我做了这个HumanoidGym-Ex项目，实现了保持原本HumanoidGym风格的工程模式下，把训练器和仿真器从Isaac Gym换成了Isaac Lab，使其适用于最新的仿真科技。同步的，HumanoidGym-Ex在能够实现Isaac Lab/sim训练方式的前提下，也同步支持原生Isaac Gym的train和play，实现了一套框架，同步支持Isaac Lab/sim和Isaac Gym强化学习后端。



## 项目目录结构

```text
humanoid_gym_ex/
├── humanoid_gym_ex/
│   ├── algo/                 # PPO / actor-critic / runner
│   ├── envs/
│   │   ├── base/             # Humanoid-Gym style base env/config/backend interface
│   │   ├── backends/         # IsaacGym / IsaacLab / Genesis adapters
│   │   └── robots/           # XBot-L and future humanoids
│   ├── resources/            # URDF / meshes / MuJoCo resources
│   ├── scripts/              # train/play/sim2sim/diagnostics
│   └── utils/
├── docs/
├── images/
├── logs/
└── README.md
```

## 环境安装

### IsaacGym 环境

HumanoidGym-Ex支持Isaacgym环境和IsaacLab/Sim双环境，如果你需要使用IsaacGym来启动训练，安装方式可以参考isaacGym官方仓库或者我的视频：https://www.bilibili.com/video/BV1kYo8BhEkN/?vd_source=5d20af79ff500db9da1b6e7e1213da51

### IsaacLab / Isaac Sim 环境

Isaac Lab可以参考官方的安装方式：IsaacLab安装链接，【Ai你帮我找好后补充】

## 启动训练--使用IsaacLab 
这里是基于IsaacLab启动HumanoidGym -Ex自带的Xbot机器人例子进行训练的方法

启动平地训练
```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
WANDB_MODE=offline \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name xbot_isaaclab_1000 \
  --device cuda:0
```

### 启动粗糙地形训练

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
WANDB_MODE=offline \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name rough_heights_curric_lab \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --device cuda:0
```

### Isaac Lab上进行训练结果播放

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/play_isaaclab.py \
  --load_run <run_dir_name> \
  --checkpoint 1000 \
  --fix_command \
  --follow_camera \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --device cuda:0
```


### 启动训练--使用IsaacGym 
这里是启动HumanoidGym-Ex 自带的Xbot机器人例子进行训练的方法,基于IsaacGym，这体现了HumanoidGym-Ex 同时兼容IsaacLab和IsaacGym的特性
```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --run_name xbot_isaacgym_1000 \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

### 启动粗糙地形训练

```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --run_name rough_heights_curric_gym \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

### Isaac Gym上进行训练结果播放

```bash
python humanoid_gym_ex/scripts/play.py \
  --task=humanoid_ppo \
  --load_run <run_dir_name> \
  --checkpoint 1000
```

### 当用粗糙地形进行训练时，用这样的方式启动

```bash
HUMANOID_GYM_EX_EXPORT_POLICY=0 \
HUMANOID_GYM_EX_RENDER=0 \
HUMANOID_GYM_EX_PLOT_STATES=0 \
HUMANOID_GYM_EX_FOLLOW_CAMERA=1 \
python humanoid_gym_ex/scripts/play.py \
  --task=humanoid_ppo \
  --load_run <run_dir_name> \
  --checkpoint 1000 \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum
```

## 致谢与声明

本项目基于原版 `roboterax/humanoid-gym` 进行迁移和扩展，并保留迁移源文件中的 BSD-3-Clause license headers。

特别感谢：

- Humanoid-Gym 原作者与 RobotEra XBot-L 示例。
- Legged-Gym / rsl_rl 生态。
- NVIDIA IsaacGym、IsaacLab、Isaac Sim。
- 灯哥

灯哥开源 B 站视频号：

https://space.bilibili.com/493192058

---

<a id="english"></a>

# HumanoidGym-Ex English README

HumanoidGym-Ex is not a new humanoid learning framework from scratch. It is a Humanoid-Gym-style extension framework that preserves the original Humanoid-Gym user experience while enabling future IsaacLab and Genesis backends.

The main work in this repository is the migration of the original Humanoid-Gym codebase plus a Humanoid-Gym-style IsaacLab Direct backend. IsaacGym remains the compatibility baseline; IsaacLab / Isaac Sim is the primary extension path.

This project is an open-source migration by **Deng Ge Open Source**.

- Bilibili: https://space.bilibili.com/493192058
- Goal: keep the original Humanoid-Gym workflow familiar while making IsaacLab / Isaac Sim migration practical.

## Screenshots

Side-by-side trained policy playback in IsaacLab Direct and the IsaacGym-compatible path:

![IsaacLab and IsaacGym walking policies](docs/assets/isaaclab_isaacgym_walk.gif)

Static screenshots:

Trained policy play in IsaacGym:

![IsaacGym play](docs/assets/isaacgym_play_viewer.png)

Trained policy play in IsaacLab / Isaac Sim:

![IsaacLab play](docs/assets/isaaclab_play_viewer.png)

## Motivation

The original Humanoid-Gym is practical because it is script-centered and easy to modify:

- `train.py` and `play.py` are the main user entry points.
- Robot config, reward scales, observations, resets, and commands are easy to find.
- Reward functions are written in one environment class instead of being split across many manager terms.
- The PPO runner follows the familiar `rsl_rl` style.

HumanoidGym-Ex keeps that style. It does not try to replace Humanoid-Gym with a new framework. Instead, it adds a conservative backend path toward IsaacLab Direct Workflow and, later, Genesis.

## Current Status

| Component | Status |
| --- | --- |
| IsaacGym backend | Train, play, and policy export are active |
| IsaacLab Direct backend | Smoke, train, play, export, rough terrain are active |
| Genesis backend | Interface reserved, not fully implemented |
| XBot-L example | Migrated from upstream Humanoid-Gym |
| MuJoCo sim2sim | Migrated from upstream |
| PPO | Local Humanoid-Gym / rsl_rl-style runner |
| Reward parity | Default reward names and scales are shared |
| Rough terrain | Runs in both backends; long-run convergence is not fully equivalent yet |
| Measured heights | Actor `705`, critic `780` on rough terrain |

Default task:

```bash
humanoid_ppo
```

Default example robot:

```text
RobotEra XBot-L
```

## Design Principles

1. **Preserve the Humanoid-Gym user experience**  
   Users should still recognize train, play, config, reward, observation, reset, and command code.

2. **Keep backend abstraction small**  
   The project intentionally avoids a complex plugin system.

3. **Keep reward and observation code centralized**  
   The IsaacLab path does not split rewards into Manager-based terms.

4. **Use IsaacLab Direct Workflow**  
   The IsaacLab implementation uses `DirectRLEnv`, `Articulation`, `Scene`, `ContactSensor`, and `RayCaster`, not Manager-based tasks.

5. **Keep default rewards identical**  
   IsaacGym and IsaacLab use the same default reward names and reward scales. Current alignment work focuses on state ordering, contact reporting, terrain sampling, reset placement, randomization, and physics settings before any backend-specific reward shaping.

## Installation

For IsaacGym:

```bash
conda activate legged_gym
pip install -e .
```

For IsaacLab / Isaac Sim, use your IsaacLab conda environment and set:

```bash
export PYTHONPATH=/path/to/HumanoidGym-Ex
```

## IsaacLab Direct Quick Start

Original IsaacGym train / play commands are still preserved, but they are folded here: [Original Humanoid-Gym / IsaacGym compatible commands](#isaacgym-original-compatible-commands-en).

PPO smoke:

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
WANDB_MODE=offline \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py \
  --headless \
  --num_envs 64 \
  --num_steps_per_env 60 \
  --max_iterations 1 \
  --run_name isaaclab_ppo_smoke
```

Train rough terrain:

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
WANDB_MODE=offline \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name rough_heights_curric_lab \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --device cuda:0
```

Play:

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/play_isaaclab.py \
  --load_run <run_dir_name> \
  --checkpoint 1000 \
  --fix_command \
  --follow_camera \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --device cuda:0
```

The IsaacLab viewer uses a close, zoomable follow camera by default:

```text
--camera_zoom 0.45
```

Mouse-wheel zoom is preserved after the viewport stabilizes.

<details>
<summary id="isaacgym-original-compatible-commands-en">Original Humanoid-Gym / IsaacGym compatible commands (click to expand)</summary>

This section documents compatibility with the original Humanoid-Gym / IsaacGym workflow. The main README path focuses on the IsaacLab Direct migration above.

### Smoke test

```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 1
```

### Train

```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --run_name xbot_isaacgym_1000 \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

### Play

```bash
python humanoid_gym_ex/scripts/play.py \
  --task=humanoid_ppo \
  --load_run <run_dir_name> \
  --checkpoint 1000
```

</details>

## Compatibility Target

HumanoidGym-Ex aims to preserve:

- upstream task registry style
- nested Python config style
- `cfg.rewards.scales` reward dispatch
- observation and privileged-observation dimensions
- PPO runner interface
- XBot-L assets and MuJoCo sim2sim example
- script-centered user workflow

Default XBot-L dimensions:

```text
num_actions = 12
num_single_obs = 47
frame_stack = 15
num_observations = 705
single_num_privileged_obs = 73
c_frame_stack = 3
num_privileged_obs = 219
rough measured-height critic obs = 780
```

## Documentation

- [Architecture review](docs/ARCHITECTURE_REVIEW.md)
- [Backend interface](docs/BACKEND_INTERFACE.md)
- [Design goals](docs/DESIGN_GOALS.md)
- [IsaacLab migration notes](docs/MIGRATION_ISAACLAB.md)
- [Roadmap](docs/ROADMAP.md)

## Credits

This project is derived from `roboterax/humanoid-gym` and preserves BSD-3-Clause license headers in migrated source files.

Thanks to:

- the original Humanoid-Gym authors and RobotEra XBot-L example
- Legged-Gym / rsl_rl
- NVIDIA IsaacGym, IsaacLab, and Isaac Sim
- Deng Ge Open Source community

Deng Ge Open Source Bilibili:

https://space.bilibili.com/493192058
