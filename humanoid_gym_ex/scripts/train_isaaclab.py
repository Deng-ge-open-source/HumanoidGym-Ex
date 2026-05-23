"""Train the IsaacLab Direct XBot smoke env with the local Humanoid-Gym PPO runner."""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train XBot-L in IsaacLab Direct workflow with local PPO.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--max_iterations", type=int, default=1)
parser.add_argument("--num_steps_per_env", type=int, default=60)
parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to XBotLCfgPPO.seed.")
parser.add_argument("--run_name", type=str, default="isaaclab_direct")
parser.add_argument("--no_log", action="store_true", help="Disable wandb/tensorboard logging and checkpoint save.")
parser.add_argument("--terrain", choices=["plane", "rough", "heightfield", "trimesh"], default="plane")
parser.add_argument("--measure_heights", action="store_true")
parser.add_argument("--terrain_curriculum", action="store_true")
parser.add_argument("--termination_base_height", type=float, default=None)
parser.add_argument("--termination_orientation", type=float, default=None)
parser.add_argument("--parity_termination_profile", choices=["none", "isaacgym_like"], default="none")
parser.add_argument(
    "--reward_scale",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="Override one XBot reward scale for local tuning, e.g. --reward_scale tracking_lin_vel=1.8.",
)
parser.add_argument(
    "--tracking_sigma",
    type=float,
    default=None,
    help="Override XBot reward tracking_sigma for local tuning.",
)
parser.add_argument(
    "--reward_param",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="Override one XBot reward parameter for local tuning, e.g. --reward_param high_speed_penalty=1.0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_gym_ex.algo.ppo.on_policy_runner import OnPolicyRunner  # noqa: E402
from humanoid_gym_ex.envs.robots.humanoid_config import XBotLCfg, XBotLCfgPPO  # noqa: E402
from humanoid_gym_ex.envs.robots.xbot.isaaclab_env import (  # noqa: E402
    XBotIsaacLabEnv,
    XBotIsaacLabEnvCfg,
    configure_xbot_isaaclab_parity_termination,
    configure_xbot_isaaclab_terrain,
)
from humanoid_gym_ex.envs.robots.xbot.isaaclab_vec_env import IsaacLabRslRlVecEnv  # noqa: E402


def set_isaaclab_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed), flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, list):
            result[key] = [class_to_dict(item) for item in value]
        else:
            result[key] = class_to_dict(value)
    return result


def _apply_reward_overrides(xbot_cfg, reward_scale_overrides, reward_param_overrides, tracking_sigma):
    if tracking_sigma is not None:
        xbot_cfg.rewards.tracking_sigma = tracking_sigma
        print("[HumanoidGym-Ex] override rewards.tracking_sigma={}".format(tracking_sigma), flush=True)
    for item in reward_param_overrides:
        if "=" not in item:
            raise ValueError("--reward_param must use NAME=VALUE, got {!r}".format(item))
        name, value_text = item.split("=", 1)
        name = name.strip()
        if not hasattr(xbot_cfg.rewards, name):
            raise ValueError("Unknown reward parameter {!r}".format(name))
        value = float(value_text)
        setattr(xbot_cfg.rewards, name, value)
        print("[HumanoidGym-Ex] override rewards.{}={}".format(name, value), flush=True)
    for item in reward_scale_overrides:
        if "=" not in item:
            raise ValueError("--reward_scale must use NAME=VALUE, got {!r}".format(item))
        name, value_text = item.split("=", 1)
        name = name.strip()
        if not hasattr(xbot_cfg.rewards.scales, name):
            raise ValueError("Unknown reward scale {!r}".format(name))
        value = float(value_text)
        setattr(xbot_cfg.rewards.scales, name, value)
        print("[HumanoidGym-Ex] override rewards.scales.{}={}".format(name, value), flush=True)


def main():
    os.environ.setdefault("WANDB_MODE", "offline")
    train_cfg = XBotLCfgPPO()
    if args_cli.seed is not None:
        train_cfg.seed = args_cli.seed
    seed = set_isaaclab_seed(train_cfg.seed)
    env_cfg = XBotIsaacLabEnvCfg()
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    configure_xbot_isaaclab_terrain(
        env_cfg,
        terrain=args_cli.terrain,
        measure_heights=args_cli.measure_heights,
        terrain_curriculum=args_cli.terrain_curriculum,
    )
    env_cfg.termination_base_height = args_cli.termination_base_height
    env_cfg.termination_orientation = args_cli.termination_orientation
    env_cfg.parity_termination_profile = args_cli.parity_termination_profile
    configure_xbot_isaaclab_parity_termination(env_cfg, args_cli.parity_termination_profile)
    _apply_reward_overrides(XBotLCfg, args_cli.reward_scale, args_cli.reward_param, args_cli.tracking_sigma)
    direct_env = XBotIsaacLabEnv(env_cfg)
    vec_env = IsaacLabRslRlVecEnv(direct_env)

    train_cfg.runner.max_iterations = args_cli.max_iterations
    train_cfg.runner.num_steps_per_env = args_cli.num_steps_per_env
    train_cfg.runner.run_name = args_cli.run_name
    train_cfg.runner.experiment_name = "XBot_isaaclab"
    cfg_dict = class_to_dict(train_cfg)

    log_dir = None
    if not args_cli.no_log:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)
        os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(vec_env, cfg_dict, log_dir=log_dir, device=args_cli.device)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    vec_env.close()
    if log_dir is not None:
        print("[HumanoidGym-Ex] IsaacLab PPO log_dir:", log_dir, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
