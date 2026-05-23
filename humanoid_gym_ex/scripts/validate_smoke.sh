#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ISAACLAB_ENV="${ISAACLAB_ENV:-/home/cra02/anaconda3/envs/env_isaaclab}"
ISAACGYM_ENV="${ISAACGYM_ENV:-legged_gym}"

echo "[validate] compile"
conda run -n "$ISAACGYM_ENV" python -m compileall humanoid_gym_ex >/tmp/humanoidgym_ex_compile.log

echo "[validate] reward parity"
conda run -n "$ISAACGYM_ENV" python humanoid_gym_ex/scripts/check_reward_parity.py >/tmp/humanoidgym_ex_reward_parity.log
grep -q "reward parity ok" /tmp/humanoidgym_ex_reward_parity.log

echo "[validate] isaacgym train smoke"
WANDB_MODE=offline conda run -n "$ISAACGYM_ENV" \
  python humanoid_gym_ex/scripts/train.py --task=humanoid_ppo --headless --num_envs 64 --max_iterations 1 --run_name validate_isaacgym \
  >/tmp/humanoidgym_ex_isaacgym.log
grep -q "Mean reward: 0.60" /tmp/humanoidgym_ex_isaacgym.log
grep -q "Total timesteps: 3840" /tmp/humanoidgym_ex_isaacgym.log

echo "[validate] isaaclab randomization smoke"
PYTHONPATH="$ROOT_DIR" conda run -p "$ISAACLAB_ENV" \
  python humanoid_gym_ex/scripts/isaaclab_smoke.py --headless --num_envs 2 --steps 2 --check_randomization \
  >/tmp/humanoidgym_ex_isaaclab_smoke.log 2>&1
grep -q "IsaacLab reset policy obs: (2, 705)" /tmp/humanoidgym_ex_isaaclab_smoke.log
grep -q "randomization friction=True mass=True" /tmp/humanoidgym_ex_isaaclab_smoke.log

echo "[validate] isaaclab rough terrain heights smoke"
PYTHONPATH="$ROOT_DIR" conda run -p "$ISAACLAB_ENV" \
  python humanoid_gym_ex/scripts/isaaclab_smoke.py --headless --num_envs 2 --steps 1 --terrain rough --measure_heights --terrain_curriculum \
  >/tmp/humanoidgym_ex_isaaclab_rough.log 2>&1
grep -q "IsaacLab reset policy obs: (2, 705)" /tmp/humanoidgym_ex_isaaclab_rough.log
grep -q "IsaacLab reset critic obs: (2, 780)" /tmp/humanoidgym_ex_isaaclab_rough.log
grep -q "terrain=rough measure_heights=True terrain_curriculum=True" /tmp/humanoidgym_ex_isaaclab_rough.log

echo "[validate] isaaclab ppo smoke"
PYTHONPATH="$ROOT_DIR" WANDB_MODE=offline conda run -p "$ISAACLAB_ENV" \
  python humanoid_gym_ex/scripts/train_isaaclab.py --headless --num_envs 2 --num_steps_per_env 4 --max_iterations 1 --run_name validate_isaaclab \
  >/tmp/humanoidgym_ex_isaaclab_train.log 2>&1
grep -q "Total timesteps: 8" /tmp/humanoidgym_ex_isaaclab_train.log
ISAACLAB_RUN="$(grep "IsaacLab PPO log_dir:" /tmp/humanoidgym_ex_isaaclab_train.log | awk '{print $NF}')"
test -f "$ISAACLAB_RUN/model_1.pt"

echo "[validate] isaaclab play/export smoke"
PYTHONPATH="$ROOT_DIR" conda run -p "$ISAACLAB_ENV" \
  python humanoid_gym_ex/scripts/play_isaaclab.py --headless --steps 2 --load_run "$(basename "$ISAACLAB_RUN")" --export_policy --fix_command \
  >/tmp/humanoidgym_ex_isaaclab_play.log 2>&1
grep -q "IsaacLab play steps=2" /tmp/humanoidgym_ex_isaaclab_play.log
grep -q "exported IsaacLab policy" /tmp/humanoidgym_ex_isaaclab_play.log

echo "[validate] ok"
