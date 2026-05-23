"""Play an IsaacLab Direct checkpoint with the local Humanoid-Gym policy."""

import argparse
import copy
import os
import types
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play XBot-L IsaacLab Direct policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--load_run", type=str, default="-1")
parser.add_argument("--checkpoint", type=int, default=-1)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--export_policy", action="store_true")
parser.add_argument("--fix_command", action="store_true")
parser.add_argument("--terrain", choices=["plane", "rough", "heightfield", "trimesh"], default="plane")
parser.add_argument("--measure_heights", action="store_true")
parser.add_argument("--terrain_curriculum", action="store_true")
parser.add_argument("--follow_camera", action="store_true")
parser.add_argument("--camera_zoom", type=float, default=0.45)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_gym_ex.algo.ppo.on_policy_runner import OnPolicyRunner  # noqa: E402
from humanoid_gym_ex.envs.robots.humanoid_config import XBotLCfgPPO  # noqa: E402
from humanoid_gym_ex.envs.robots.xbot.isaaclab_env import (  # noqa: E402
    XBotIsaacLabEnv,
    XBotIsaacLabEnvCfg,
    configure_xbot_isaaclab_terrain,
)
from humanoid_gym_ex.envs.robots.xbot.isaaclab_vec_env import IsaacLabRslRlVecEnv  # noqa: E402


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


def get_load_path(root, load_run="-1", checkpoint=-1):
    root = Path(root)
    if load_run == "-1":
        runs = sorted([path for path in root.iterdir() if path.is_dir() and path.name != "exported"])
        if not runs:
            raise ValueError(f"No runs in {root}")
        run_dir = runs[-1]
    else:
        run_dir = root / load_run
    if checkpoint == -1:
        models = sorted(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
        if not models:
            raise ValueError(f"No checkpoints in {run_dir}")
        return str(models[-1])
    return str(run_dir / f"model_{checkpoint}.pt")


def export_policy(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    policy = copy.deepcopy(actor_critic.actor).to("cpu")
    traced = torch.jit.script(policy)
    out_path = os.path.join(path, "policy_isaaclab.pt")
    traced.save(out_path)
    return out_path


def follow_camera_pose(root, zoom):
    zoom = max(float(zoom), 0.05)
    eye = root + torch.tensor([1.8 * zoom, -1.8 * zoom, 1.15], device=root.device)
    target = root + torch.tensor([0.0, 0.0, 0.7], device=root.device)
    return eye.cpu().tolist(), target.cpu().tolist()


def translate_follow_viewport(root, state):
    try:
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
    except Exception:
        return
    root_cpu = root.detach().cpu()
    if state["last_root"] is None:
        state["last_root"] = root_cpu.clone()
        return
    delta = root_cpu - state["last_root"]
    state["last_root"] = root_cpu.clone()
    if float(torch.linalg.norm(delta).item()) < 1e-6:
        return
    camera_state = ViewportCameraState()
    position = torch.as_tensor(camera_state.position_world, dtype=torch.float32) + delta
    target = torch.as_tensor(camera_state.target_world, dtype=torch.float32) + delta
    camera_state.set_position_world(position.tolist(), False)
    camera_state.set_target_world(target.tolist(), True)


def enable_zoomable_asset_follow(env, asset_name, zoom):
    """Track an IsaacLab asset while preserving mouse-wheel zoom/orbit edits."""
    controller = getattr(env, "viewport_camera_controller", None)
    if controller is None:
        return False

    zoom = max(float(zoom), 0.05)
    initial_eye = np.array([1.8 * zoom, -1.8 * zoom, 1.15], dtype=float)
    initial_lookat = np.array([0.0, 0.0, 0.7], dtype=float)
    controller.default_cam_eye = initial_eye.copy()
    controller.default_cam_lookat = initial_lookat.copy()
    controller._hgex_follow_updates = 0
    controller._hgex_preserve_after_updates = 90
    original_update = controller.update_view_to_asset_root

    def update_view_to_asset_root_preserving_zoom(self, next_asset_name):
        self._hgex_follow_updates += 1
        if self._hgex_follow_updates <= self._hgex_preserve_after_updates:
            self.default_cam_eye = initial_eye.copy()
            self.default_cam_lookat = initial_lookat.copy()
        else:
            try:
                from omni.kit.viewport.utility.camera_state import ViewportCameraState

                camera_state = ViewportCameraState()
                origin = self.viewer_origin.detach().cpu().numpy()
                eye_offset = np.asarray(camera_state.position_world, dtype=float) - origin
                lookat_offset = np.asarray(camera_state.target_world, dtype=float) - origin
                if np.all(np.isfinite(eye_offset)) and np.all(np.isfinite(lookat_offset)):
                    distance = np.linalg.norm(eye_offset - lookat_offset)
                    if 0.15 <= distance <= 20.0:
                        self.default_cam_eye = eye_offset
                        self.default_cam_lookat = lookat_offset
            except Exception as exc:
                if not getattr(self, "_hgex_camera_warning_printed", False):
                    print("[HumanoidGym-Ex] IsaacLab zoom-preserving camera fallback:", exc, flush=True)
                    self._hgex_camera_warning_printed = True
        return original_update(next_asset_name)

    original_update(asset_name)
    controller.update_view_to_asset_root = types.MethodType(update_view_to_asset_root_preserving_zoom, controller)
    return True


def main():
    env_cfg = XBotIsaacLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    configure_xbot_isaaclab_terrain(
        env_cfg,
        terrain=args_cli.terrain,
        measure_heights=args_cli.measure_heights,
        terrain_curriculum=args_cli.terrain_curriculum,
    )
    direct_env = XBotIsaacLabEnv(env_cfg)
    env = IsaacLabRslRlVecEnv(direct_env)
    if args_cli.follow_camera:
        if not enable_zoomable_asset_follow(direct_env, "robot", args_cli.camera_zoom):
            eye, target = follow_camera_pose(torch.zeros(3, device=direct_env.device), args_cli.camera_zoom)
            direct_env.sim.set_camera_view(eye=eye, target=target)
    follow_camera_state = {"last_root": None}

    train_cfg = XBotLCfgPPO()
    train_cfg.runner.experiment_name = "XBot_isaaclab"
    cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=args_cli.device)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    load_path = get_load_path(log_root, args_cli.load_run, args_cli.checkpoint)
    runner.load(load_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    if args_cli.export_policy:
        export_dir = os.path.join(log_root, "exported", "policies")
        out_path = export_policy(runner.alg.actor_critic, export_dir)
        print("[HumanoidGym-Ex] exported IsaacLab policy:", out_path, flush=True)

    obs = env.get_observations()
    total_reward = torch.zeros(env.num_envs, device=env.device)
    for step in range(args_cli.steps):
        actions = policy(obs)
        if args_cli.fix_command:
            direct_env.commands[:, 0] = 0.5
            direct_env.commands[:, 1] = 0.0
            direct_env.commands[:, 2] = 0.0
            direct_env.commands[:, 3] = 0.0
        obs, _, rewards, _, _ = env.step(actions)
        if args_cli.follow_camera and getattr(direct_env, "viewport_camera_controller", None) is None and step % 5 == 0:
            root = direct_env.root_states[0, :3].detach()
            translate_follow_viewport(root, follow_camera_state)
        total_reward += rewards
    print(
        "[HumanoidGym-Ex] IsaacLab play steps={} reward_mean={:.6f} checkpoint={}".format(
            args_cli.steps, total_reward.mean().item() / max(args_cli.steps, 1), load_path
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
