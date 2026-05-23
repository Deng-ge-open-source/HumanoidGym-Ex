# HumanoidGym-Ex

**中文 | [English](#english)**

HumanoidGym-Ex is not a new humanoid learning framework from scratch. It is a Humanoid-Gym-style extension framework that preserves the original Humanoid-Gym user experience while enabling future IsaacLab and Genesis backends.

HumanoidGym-Ex 不是一个从零重写的人形机器人强化学习框架。它是一个 **Humanoid-Gym 风格的扩展框架**：尽量保留原版 Humanoid-Gym 的脚本入口、配置方式、reward 写法、PPO 接口和使用习惯，同时逐步支持 IsaacGym、IsaacLab Direct Workflow，并为 Genesis 后端预留接口。

本项目由 **灯哥开源** 移植与维护。

- B 站主页：https://space.bilibili.com/493192058
- 迁移目标：让熟悉 Humanoid-Gym 的用户可以用接近原版的方式进入 IsaacLab / Isaac Sim 生态。

## 项目截图

IsaacGym 训练后策略播放：

![IsaacGym play](docs/assets/isaacgym_play.png)

IsaacLab / Isaac Sim 训练后策略播放：

![IsaacLab play](docs/assets/isaaclab_play.png)

## 为什么做 HumanoidGym-Ex

原版 Humanoid-Gym 的优点是非常直接：

- `train.py` / `play.py` 脚本中心化，学习成本低。
- robot config、reward scale、observation、reset、command curriculum 都在熟悉的位置。
- reward 函数集中在环境类里，便于调试和快速实验。
- PPO 接口基于 `rsl_rl` 风格，训练链路简单。

IsaacLab / Isaac Sim 的生态更现代，但 Manager-based workflow 对很多从 Humanoid-Gym / Legged-Gym 迁移过来的用户来说会显得拆分较多。HumanoidGym-Ex 的目标不是把 Humanoid-Gym 改造成一个全新的大框架，而是在保留原有使用体验的前提下，逐步接入新后端。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| IsaacGym 后端 | 已可训练、play、导出策略 |
| IsaacLab Direct 后端 | 已可 smoke、训练、play、导出策略 |
| Genesis 后端 | 已预留接口，暂未完整实现 |
| XBot-L 原版示例 | 已迁移 |
| MuJoCo sim2sim | 已迁移原版脚本 |
| PPO | 保留本地 Humanoid-Gym / rsl_rl 风格接口 |
| reward parity | 默认 reward 名称和默认 reward scale 保持一致 |
| rough terrain | Gym / Lab 均可运行，长训练曲线仍有差距 |
| measured heights | Gym / Lab 均支持，actor `705`，critic `780` |

默认任务名：

```bash
humanoid_ppo
```

默认示例机器人：

```text
RobotEra XBot-L
```

## 设计原则

HumanoidGym-Ex 的核心原则是：

1. **不重写用户习惯**  
   熟悉 Humanoid-Gym 的用户应该能快速找到 train、play、cfg、reward、obs、reset、commands。

2. **不引入复杂插件系统**  
   backend interface 只抽象必要接口，不把项目做成过度工程化的大平台。

3. **不把 reward / obs 拆成 Manager-based 风格**  
   reward 仍然集中、直观，便于从原版代码迁移。

4. **IsaacLab 使用 Direct Workflow**  
   IsaacLab 路径基于 `DirectRLEnv`、`Articulation`、`Scene`、`ContactSensor` 和 `RayCaster`，不使用 Manager-based task。

5. **默认 reward 保持一致**  
   IsaacGym 和 IsaacLab 默认使用同一套 reward 名称和 reward scale。当前调试优先做数据顺序、接触、地形、reset、随机化和物理参数对齐，而不是为某个后端单独改 reward。

## 目录结构

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

## 安装

### IsaacGym 环境

假设你已经安装 IsaacGym Preview 4，并有一个类似 `legged_gym` 的 conda 环境：

```bash
conda activate legged_gym
pip install -e .
```

### IsaacLab / Isaac Sim 环境

IsaacLab 建议使用单独 conda 环境。本仓库测试时使用：

```bash
/home/cra02/anaconda3/envs/env_isaaclab
```

运行 IsaacLab 脚本时建议显式设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/HumanoidGym-Ex
```

## IsaacGym 使用方式

### Smoke test

```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 1
```

### 训练

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

### rough terrain + measured heights + terrain curriculum

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

### Play

```bash
python humanoid_gym_ex/scripts/play.py \
  --task=humanoid_ppo \
  --load_run <run_dir_name> \
  --checkpoint 1000
```

### Play with rough terrain

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

## IsaacLab Direct 使用方式

IsaacLab 路径不使用 Manager-based task，而是保留 Humanoid-Gym 风格的上层 env / reward / obs / reset 写法。

### Smoke test

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/isaaclab_smoke.py \
  --headless \
  --num_envs 2 \
  --steps 2
```

期望输出包含：

```text
IsaacLab reset policy obs: (2, 705)
IsaacLab reset critic obs: (2, 219)
```

### IsaacLab PPO smoke

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

### IsaacLab 训练

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

### IsaacLab rough terrain + measured heights

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

rough measured-height 模式下：

```text
actor policy obs: 705
critic obs:      780
```

这与原版 Humanoid-Gym XBot 路径一致：actor 不拼接 height samples，critic privileged obs 拼接 height samples。

### IsaacLab Play

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

IsaacLab viewer 默认开启近距离可缩放跟随相机：

- 默认 `--camera_zoom 0.45`
- 鼠标滚轮可以继续自由缩放
- 相机会跟随机器人 root，但不会强制覆盖你手动调整后的 zoom

## 多 GPU 示例

两张 A6000 上可以让 IsaacGym 和 IsaacLab 同时训练：

```bash
# GPU0: IsaacGym
CUDA_VISIBLE_DEVICES=0 \
conda run -n legged_gym \
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name rough_gym_seed42 \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --sim_device cuda:0 \
  --rl_device cuda:0

# GPU1: IsaacLab
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/path/to/HumanoidGym-Ex \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/train_isaaclab.py \
  --headless \
  --num_envs 4096 \
  --num_steps_per_env 60 \
  --max_iterations 1000 \
  --seed 42 \
  --run_name rough_lab_seed42 \
  --terrain rough \
  --measure_heights \
  --terrain_curriculum \
  --device cuda:0
```

## 验证与对齐结果

本仓库包含多种验证脚本：

```bash
bash humanoid_gym_ex/scripts/validate_smoke.sh
```

该脚本覆盖：

- Python compile
- reward parity
- IsaacGym train smoke
- IsaacLab randomization smoke
- IsaacLab rough terrain measured-heights smoke
- IsaacLab PPO smoke
- IsaacLab play/export smoke

reward parity 可单独检查：

```bash
python humanoid_gym_ex/scripts/check_reward_parity.py
```

### 已完成的长训练验证

已完成 `4096 envs x 1000 iterations` 的 rough terrain + measured heights + terrain curriculum 对比：

| Backend | Final reward | Final episode length | Tail10 reward | Tail10 episode length |
| --- | ---: | ---: | ---: | ---: |
| IsaacGym | 138.180 | 2346.220 | 138.464 | 2357.588 |
| IsaacLab | 93.170 | 1947.270 | 97.214 | 2030.989 |

结论：

- IsaacLab 路径已经可以完整训练、play 和导出。
- 默认 reward 名称和默认 reward scale 与 IsaacGym 保持一致。
- plane 训练对齐已经较好。
- rough terrain 长训练仍未达到严格等价，当前 tail reward-per-step 等价约 `81.50%`。
- 剩余差异主要集中在 rough terrain 几何、接触报告、height sampling、reset placement 和 PhysX 细节。

详细报告：

- [Original scene matrix](docs/ORIGINAL_SCENE_MATRIX.md)
- [Final training evaluation](docs/FINAL_TRAINING_EVAL.md)
- [Multi-seed alignment](docs/MULTISEED_ALIGNMENT.md)
- [Replay alignment](docs/REPLAY_ALIGNMENT.md)

## 与原版 Humanoid-Gym 的兼容目标

HumanoidGym-Ex 尽量保持：

- 原版 task registry 风格
- 原版 nested Python cfg 风格
- 原版 `cfg.rewards.scales` reward dispatch
- 原版 observation / privileged observation 维度
- 原版 PPO runner 使用方式
- 原版 XBot-L 资源路径和 MuJoCo sim2sim 示例
- 原版脚本中心化体验

默认 XBot-L 维度：

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

## 后端抽象

当前 backend interface 保持极简：

```text
create_sim()
create_envs()
step(actions)
reset(env_ids)
get_root_states()
get_dof_pos()
get_dof_vel()
get_contact_forces()
set_dof_targets()
apply_domain_randomization()
render_or_viewer_step()
```

设计目的不是构建复杂插件系统，而是让上层 Humanoid-Gym 风格环境尽量不感知 IsaacGym / IsaacLab 的底层差异。

## 文档

- [Architecture review](docs/ARCHITECTURE_REVIEW.md)
- [Backend interface](docs/BACKEND_INTERFACE.md)
- [Design goals](docs/DESIGN_GOALS.md)
- [IsaacLab migration notes](docs/MIGRATION_ISAACLAB.md)
- [Roadmap](docs/ROADMAP.md)
- [Test report](docs/TEST_REPORT.md)
- [Original scene matrix](docs/ORIGINAL_SCENE_MATRIX.md)

## Roadmap

- Phase 0: 原版 Humanoid-Gym 代码迁移。
- Phase 1: IsaacGym backend adapter，保持原版训练和 play 可运行。
- Phase 2: IsaacLab Direct Workflow backend，支持 XBot smoke / train / play / rough terrain。
- Phase 3: Genesis backend 接口预留。
- Phase 4: 更多机器人、更多原版场景和跨后端长训练统计。

## 致谢与声明

本项目基于原版 `roboterax/humanoid-gym` 进行迁移和扩展，并保留迁移源文件中的 BSD-3-Clause license headers。

特别感谢：

- Humanoid-Gym 原作者与 RobotEra XBot-L 示例。
- Legged-Gym / rsl_rl 生态。
- NVIDIA IsaacGym、IsaacLab、Isaac Sim。
- 灯哥开源社区。

灯哥开源 B 站视频号：

https://space.bilibili.com/493192058

---

<a id="english"></a>

# HumanoidGym-Ex English README

HumanoidGym-Ex is not a new humanoid learning framework from scratch. It is a Humanoid-Gym-style extension framework that preserves the original Humanoid-Gym user experience while enabling future IsaacLab and Genesis backends.

This project is an open-source migration by **Deng Ge Open Source**.

- Bilibili: https://space.bilibili.com/493192058
- Goal: keep the original Humanoid-Gym workflow familiar while making IsaacLab / Isaac Sim migration practical.

## Screenshots

Trained policy play in IsaacGym:

![IsaacGym play](docs/assets/isaacgym_play.png)

Trained policy play in IsaacLab / Isaac Sim:

![IsaacLab play](docs/assets/isaaclab_play.png)

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

## IsaacGym Quick Start

Smoke test:

```bash
python humanoid_gym_ex/scripts/train.py \
  --task=humanoid_ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 1
```

Train:

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

Play:

```bash
python humanoid_gym_ex/scripts/play.py \
  --task=humanoid_ppo \
  --load_run <run_dir_name> \
  --checkpoint 1000
```

## IsaacLab Direct Quick Start

Smoke test:

```bash
PYTHONPATH=/path/to/HumanoidGym-Ex \
conda run -p /path/to/env_isaaclab \
python humanoid_gym_ex/scripts/isaaclab_smoke.py \
  --headless \
  --num_envs 2 \
  --steps 2
```

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

## Validation

Run all local smoke checks:

```bash
bash humanoid_gym_ex/scripts/validate_smoke.sh
```

Check reward parity:

```bash
python humanoid_gym_ex/scripts/check_reward_parity.py
```

Long rough-terrain training has been tested for `4096 envs x 1000 iterations` on IsaacGym and IsaacLab. The IsaacLab backend runs end-to-end, but rough-terrain convergence is not yet strictly equivalent. Tail reward-per-step equivalence is about `81.50%`. Remaining differences are most likely related to rough-terrain geometry, contact reporting, height sampling, reset placement, and PhysX details.

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
- [Test report](docs/TEST_REPORT.md)
- [Original scene matrix](docs/ORIGINAL_SCENE_MATRIX.md)

## Credits

This project is derived from `roboterax/humanoid-gym` and preserves BSD-3-Clause license headers in migrated source files.

Thanks to:

- the original Humanoid-Gym authors and RobotEra XBot-L example
- Legged-Gym / rsl_rl
- NVIDIA IsaacGym, IsaacLab, and Isaac Sim
- Deng Ge Open Source community

Deng Ge Open Source Bilibili:

https://space.bilibili.com/493192058
