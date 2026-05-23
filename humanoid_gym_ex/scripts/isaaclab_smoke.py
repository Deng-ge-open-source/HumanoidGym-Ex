"""Minimal IsaacLab Direct smoke test for the XBot task.

This script intentionally stays separate from the IsaacGym-centered train.py
until the Direct backend reaches PPO parity.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a minimal IsaacLab Direct XBot smoke test.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--check_randomization", action="store_true")
parser.add_argument("--terrain", choices=["plane", "rough", "heightfield", "trimesh"], default="plane")
parser.add_argument("--measure_heights", action="store_true")
parser.add_argument("--terrain_curriculum", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from humanoid_gym_ex.envs.robots.xbot.isaaclab_env import (  # noqa: E402
    XBotIsaacLabEnv,
    XBotIsaacLabEnvCfg,
    configure_xbot_isaaclab_terrain,
)


def main():
    cfg = XBotIsaacLabEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    configure_xbot_isaaclab_terrain(
        cfg,
        terrain=args_cli.terrain,
        measure_heights=args_cli.measure_heights,
        terrain_curriculum=args_cli.terrain_curriculum,
    )
    env = XBotIsaacLabEnv(cfg)
    obs, _ = env.reset()
    print("[HumanoidGym-Ex] IsaacLab reset policy obs:", tuple(obs["policy"].shape), flush=True)
    print("[HumanoidGym-Ex] IsaacLab reset critic obs:", tuple(obs["critic"].shape), flush=True)
    print(
        "[HumanoidGym-Ex] terrain={} measure_heights={} terrain_curriculum={}".format(
            args_cli.terrain, args_cli.measure_heights, args_cli.terrain_curriculum
        ),
        flush=True,
    )
    if args_cli.check_randomization:
        env.backend.apply_domain_randomization()
        env._push_robots()
        print(
            "[HumanoidGym-Ex] randomization friction={} mass={} push_xy_mean={:.6f}".format(
                env.material_randomization_enabled,
                env.mass_randomization_enabled,
                env.rand_push_force[:, :2].abs().mean().item(),
            ),
            flush=True,
        )
    actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    for step in range(args_cli.steps):
        obs, rewards, terminated, truncated, _ = env.step(actions)
        done_count = int((terminated | truncated).sum().item())
        print(
            "[HumanoidGym-Ex] step={} reward_mean={:.6f} done_count={} policy_obs={} height_mean={:.6f}".format(
                step,
                rewards.mean().item(),
                done_count,
                tuple(obs["policy"].shape),
                env.measured_heights.mean().item() if args_cli.measure_heights else 0.0,
            ),
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
