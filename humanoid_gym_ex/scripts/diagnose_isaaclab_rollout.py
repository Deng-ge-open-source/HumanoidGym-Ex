"""Collect IsaacLab XBot rollout statistics for backend alignment."""

import argparse
import json
import os
import random

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--action_mode", choices=["zero", "random"], default="zero")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--trace_output", type=str, default="")
parser.add_argument("--terrain", choices=["plane", "rough", "heightfield", "trimesh"], default="plane")
parser.add_argument("--measure_heights", action="store_true")
parser.add_argument("--terrain_curriculum", action="store_true")
parser.add_argument("--termination_base_height", type=float, default=None)
parser.add_argument("--termination_orientation", type=float, default=None)
parser.add_argument("--parity_termination_profile", choices=["none", "isaacgym_like"], default="none")
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--fix_command", action="store_true")
parser.add_argument("--action_file", type=str, default="", help="NPZ file with actions[steps, num_envs, num_actions].")
parser.add_argument("--deterministic_reset", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402

from humanoid_gym_ex.envs.robots.xbot.isaaclab_env import (  # noqa: E402
    XBotIsaacLabEnv,
    XBotIsaacLabEnvCfg,
    configure_xbot_isaaclab_parity_termination,
    configure_xbot_isaaclab_terrain,
)
from humanoid_gym_ex.algo.ppo.on_policy_runner import OnPolicyRunner  # noqa: E402
from humanoid_gym_ex.envs.robots.humanoid_config import XBotLCfgPPO  # noqa: E402
from humanoid_gym_ex.envs.robots.xbot.isaaclab_vec_env import IsaacLabRslRlVecEnv  # noqa: E402


def _mean_abs(tensor):
    return float(torch.mean(torch.abs(tensor)).item())


def _mean_norm(tensor):
    return float(torch.mean(torch.norm(tensor, dim=-1)).item())


def _std(tensor):
    return float(torch.std(tensor.float(), unbiased=False).item())


def _mean(tensor):
    return float(torch.mean(tensor.float()).item())


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def _configure_for_diagnostics(env):
    env.xbot_cfg.noise.add_noise = False
    env.xbot_cfg.domain_rand.randomize_friction = False
    env.xbot_cfg.domain_rand.randomize_base_mass = False
    env.xbot_cfg.domain_rand.push_robots = False
    env.xbot_cfg.domain_rand.action_delay = 0.0
    env.xbot_cfg.domain_rand.action_noise = 0.0
    env.xbot_cfg.commands.heading_command = False
    env.xbot_cfg.commands.curriculum = False
    env.command_ranges["lin_vel_x"] = [0.0, 0.0]
    env.command_ranges["lin_vel_y"] = [0.0, 0.0]
    env.command_ranges["ang_vel_yaw"] = [0.0, 0.0]
    env.command_ranges["heading"] = [0.0, 0.0]
    env.commands.zero_()


def _tensor_stats(tensor):
    data = tensor.detach().cpu().float()
    return {
        "shape": list(data.shape),
        "mean": float(data.mean().item()) if data.numel() else None,
        "min": float(data.min().item()) if data.numel() else None,
        "max": float(data.max().item()) if data.numel() else None,
    }


def _collect_shape_material_summary(env):
    summary = {}
    view = getattr(env.robot, "root_physx_view", None)
    if view is None:
        return summary
    try:
        materials = view.get_material_properties()
        env0 = materials[0]
        summary["material_properties"] = _tensor_stats(env0)
        if env0.ndim >= 2 and env0.shape[-1] >= 3:
            summary["static_friction_mean"] = float(env0[..., 0].float().mean().item())
            summary["dynamic_friction_mean"] = float(env0[..., 1].float().mean().item())
            summary["restitution_mean"] = float(env0[..., 2].float().mean().item())
    except Exception as exc:
        summary["material_properties_error"] = str(exc)
    for method_name in ("get_contact_offsets", "get_rest_offsets"):
        if not hasattr(view, method_name):
            continue
        try:
            values = getattr(view, method_name)()
            summary[method_name.replace("get_", "")] = _tensor_stats(values[0])
        except Exception as exc:
            summary[method_name.replace("get_", "") + "_error"] = str(exc)
    return summary


def _as_list(tensor):
    return tensor.detach().cpu().float().tolist()


def _collect_body_properties(env):
    bodies = []
    body_names = list(env.robot.body_names)
    data = env.robot.data
    masses = data.default_mass[0].detach().cpu().float()
    inertias = getattr(data, "default_inertia", None)
    coms = getattr(data, "default_com_pos_b", None)
    for index, name in enumerate(body_names):
        body = {
            "index": index,
            "name": name,
            "mass": float(masses[index].item()),
        }
        if coms is not None:
            body["com"] = _as_list(coms[0, index])
        if inertias is not None:
            inertia = inertias[0, index]
            if inertia.numel() == 9:
                body["inertia"] = _as_list(inertia.reshape(3, 3))
            else:
                body["inertia"] = _as_list(inertia)
        bodies.append(body)
    return {"bodies": bodies}


def _snapshot_events(env, step, dones):
    snapshot = getattr(env, "last_termination_snapshot", {})
    done_ids = dones.nonzero(as_tuple=False).flatten()
    events = []
    if not snapshot:
        return events
    for env_id_tensor in done_ids.detach().cpu():
        env_id = int(env_id_tensor)
        termination_contact = snapshot["termination_contact"][env_id]
        events.append(
            {
                "step": int(step),
                "env_id": env_id,
                "episode_length": int(snapshot["episode_length"][env_id].item()),
                "contact": bool(snapshot["contact"][env_id].item()),
                "time_out": bool(snapshot["time_out"][env_id].item()),
                "base_height": bool(snapshot["base_height"][env_id].item()),
                "orientation": bool(snapshot["orientation"][env_id].item()),
                "base_z": float(snapshot["base_z"][env_id].item()),
                "roll": float(snapshot["base_euler_xyz"][env_id, 0].item()),
                "pitch": float(snapshot["base_euler_xyz"][env_id, 1].item()),
                "termination_contact_max": float(termination_contact.max().item()) if termination_contact.numel() else 0.0,
            }
        )
    return events


def main():
    _set_seed(args_cli.seed)
    cfg = XBotIsaacLabEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    configure_xbot_isaaclab_terrain(
        cfg,
        terrain=args_cli.terrain,
        measure_heights=args_cli.measure_heights,
        terrain_curriculum=args_cli.terrain_curriculum,
    )
    cfg.termination_base_height = args_cli.termination_base_height
    cfg.termination_orientation = args_cli.termination_orientation
    cfg.parity_termination_profile = args_cli.parity_termination_profile
    cfg.disable_domain_randomization = True
    cfg.deterministic_reset = args_cli.deterministic_reset
    configure_xbot_isaaclab_parity_termination(cfg, args_cli.parity_termination_profile)
    env = XBotIsaacLabEnv(cfg)
    _configure_for_diagnostics(env)
    vec_env = None
    policy = None
    if args_cli.checkpoint_path:
        train_cfg = XBotLCfgPPO()
        cfg_dict = class_to_dict(train_cfg)
        vec_env = IsaacLabRslRlVecEnv(env)
        runner = OnPolicyRunner(vec_env, cfg_dict, log_dir=None, device=args_cli.device)
        runner.load(args_cli.checkpoint_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=vec_env.device)
        obs = {"policy": vec_env.get_observations(), "critic": vec_env.get_privileged_observations()}
    else:
        obs, _ = env.reset()
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args_cli.seed)
    action_trace = None
    if args_cli.action_file:
        with np.load(args_cli.action_file) as data:
            action_trace = torch.as_tensor(data["actions"], dtype=torch.float32, device=env.device)
        if action_trace.ndim != 3 or action_trace.shape[1] < env.num_envs or action_trace.shape[2] != cfg.action_space:
            raise ValueError(
                "--action_file must contain actions with shape [steps, num_envs, num_actions], got {}".format(
                    tuple(action_trace.shape)
                )
            )
        if action_trace.shape[0] < args_cli.steps:
            raise ValueError("--action_file has {} steps but {} were requested".format(action_trace.shape[0], args_cli.steps))

    samples = []
    trace = []
    termination_events = []
    reward_terms = {}
    raw_reward_terms = {}
    previous_raw_actions = None
    for step in range(args_cli.steps):
        if action_trace is not None:
            actions = action_trace[step, : env.num_envs].clone()
        elif policy is not None:
            actions = policy(obs["policy"])
        elif args_cli.action_mode == "zero":
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        else:
            actions = torch.rand(env.num_envs, cfg.action_space, generator=generator, device=env.device) * 2.0 - 1.0
        if previous_raw_actions is None:
            action_delta_abs = torch.zeros((), device=env.device)
        else:
            action_delta_abs = torch.mean(torch.abs(actions - previous_raw_actions))
        raw_action_abs = torch.mean(torch.abs(actions))
        previous_raw_actions = actions.detach().clone()
        if args_cli.fix_command:
            env.commands[:, 0] = 0.5
            env.commands[:, 1:] = 0.0
        if vec_env is not None:
            obs_policy, _, rewards, dones, extras = vec_env.step(actions)
            obs = {"policy": obs_policy, "critic": vec_env.get_privileged_observations()}
        else:
            obs, rewards, terminated, truncated, extras = env.step(actions)
            dones = terminated | truncated
        step_events = _snapshot_events(env, step, dones)
        termination_events.extend(step_events)
        reset_snapshot = getattr(env, "last_reset_snapshot", {}) if int(dones.sum().item()) > 0 else {}
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
        for name, value in getattr(env, "last_reward_terms", {}).items():
            reward_terms.setdefault(name, []).append(float(value.mean().item()))
        for name, value in getattr(env, "last_raw_reward_terms", {}).items():
            raw_reward_terms.setdefault(name, []).append(float(value.mean().item()))
        contact_history = env._contact_sensor_history_robot_order()
        if contact_history is not None:
            termination_contact = torch.max(
                torch.norm(contact_history[:, :, env.termination_contact_indices, :], dim=-1), dim=1
            )[0]
        else:
            termination_contact = torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
        feet_forces = env.contact_forces[:, env.feet_indices[:2], :]
        feet_contact = feet_forces[:, :, 2] > 5.0
        feet_force_norm = torch.norm(feet_forces, dim=-1)
        phase = env._get_phase()
        stance_mask = env._get_gait_phase()
        action_preprocess = getattr(env, "last_action_preprocess", {})
        samples.append(
            {
                "reward": float(rewards.mean().item()),
                "done_count": int(dones.sum().item()),
                "command_x_mean": _mean(env.commands[:, 0]),
                "command_x_std": _std(env.commands[:, 0]),
                "command_y_mean": _mean(env.commands[:, 1]),
                "command_yaw_mean": _mean(env.commands[:, 2]),
                "phase_mean": _mean(phase),
                "phase_std": _std(phase),
                "stance_left_rate": _mean(stance_mask[:, 0]),
                "stance_right_rate": _mean(stance_mask[:, 1]),
                "base_z": float(env.root_states[:, 2].mean().item()),
                "base_lin_vel_x": float(env.base_lin_vel[:, 0].mean().item()),
                "lin_vel_x_error": float((env.base_lin_vel[:, 0] - env.commands[:, 0]).mean().item()),
                "lin_vel_x_abs_error": float(torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0]).mean().item()),
                "base_lin_vel_abs": _mean_abs(env.base_lin_vel),
                "base_ang_vel_abs": _mean_abs(env.base_ang_vel),
                "dof_pos_abs": _mean_abs(env.dof_pos - env.default_dof_pos),
                "dof_vel_abs": _mean_abs(env.dof_vel),
                "torque_abs": _mean_abs(env.torques),
                "raw_action_abs": float(raw_action_abs.item()),
                "raw_action_delta_abs": float(action_delta_abs.item()),
                "preprocess_delay_mean": float(action_preprocess.get("delay_mean", 0.0)),
                "preprocess_delay_max": float(action_preprocess.get("delay_max", 0.0)),
                "preprocess_noise_abs": float(action_preprocess.get("noise_abs", 0.0)),
                "processed_action_abs": float(action_preprocess.get("processed_action_abs", 0.0)),
                "contact_norm": _mean_norm(env.contact_forces),
                "feet_contact_rate": float(feet_contact.float().mean().item()),
                "left_foot_contact_rate": float(feet_contact[:, 0].float().mean().item()) if feet_contact.shape[1] > 0 else 0.0,
                "right_foot_contact_rate": float(feet_contact[:, 1].float().mean().item()) if feet_contact.shape[1] > 1 else 0.0,
                "feet_force_norm_mean": float(feet_force_norm.mean().item()),
                "feet_force_norm_max": float(feet_force_norm.max().item()),
                "termination_contact_mean": float(termination_contact.mean().item()),
                "termination_contact_max": float(termination_contact.max().item()),
                "termination_contact_gt_0p1": float((termination_contact > 0.1).float().mean().item()),
                "termination_contact_gt_1p0": float((termination_contact > 1.0).float().mean().item()),
            }
        )
        trace.append(
            {
                "step": step,
                "reward_mean": float(rewards.mean().item()),
                "done_count": int(dones.sum().item()),
                "extras_time_out_count": int(time_outs.sum().item()) if time_outs is not None else 0,
                "contact_done_count": sum(1 for item in step_events if item["contact"]),
                "timeout_done_count": sum(1 for item in step_events if item["time_out"]),
                "base_height_done_count": sum(1 for item in step_events if item["base_height"]),
                "orientation_done_count": sum(1 for item in step_events if item["orientation"]),
                "base_z_mean": float(env.root_states[:, 2].mean().item()),
                "base_z_min": float(env.root_states[:, 2].min().item()),
                "base_lin_vel_x_mean": float(env.base_lin_vel[:, 0].mean().item()),
                "command_x_mean": _mean(env.commands[:, 0]),
                "lin_vel_x_abs_error": float(torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0]).mean().item()),
                "phase_mean": _mean(phase),
                "stance_left_rate": _mean(stance_mask[:, 0]),
                "stance_right_rate": _mean(stance_mask[:, 1]),
                "roll_abs_mean": float(torch.abs(env.base_euler_xyz[:, 0]).mean().item()),
                "pitch_abs_mean": float(torch.abs(env.base_euler_xyz[:, 1]).mean().item()),
                "max_abs_roll_pitch": float(torch.abs(env.base_euler_xyz[:, :2]).max().item()),
                "raw_action_abs": float(raw_action_abs.item()),
                "raw_action_delta_abs": float(action_delta_abs.item()),
                "preprocess_delay_mean": float(action_preprocess.get("delay_mean", 0.0)),
                "preprocess_noise_abs": float(action_preprocess.get("noise_abs", 0.0)),
                "processed_action_abs": float(action_preprocess.get("processed_action_abs", 0.0)),
                "feet_contact_rate": float(feet_contact.float().mean().item()),
                "left_foot_contact_rate": float(feet_contact[:, 0].float().mean().item()) if feet_contact.shape[1] > 0 else 0.0,
                "right_foot_contact_rate": float(feet_contact[:, 1].float().mean().item()) if feet_contact.shape[1] > 1 else 0.0,
                "feet_force_norm_mean": float(feet_force_norm.mean().item()),
                "feet_force_norm_max": float(feet_force_norm.max().item()),
                "termination_contact_mean": float(termination_contact.mean().item()),
                "termination_contact_max": float(termination_contact.max().item()),
                "termination_contact_gt_1p0": float((termination_contact > 1.0).float().mean().item()),
                "reset_env_count": int(reset_snapshot.get("env_count", 0)),
                "reset_post_root_z_mean": float(reset_snapshot.get("post_root_z_mean", 0.0)),
                "reset_post_dof_vel_abs": float(reset_snapshot.get("post_dof_vel_abs", 0.0)),
                "reset_post_command_x_mean": float(reset_snapshot.get("post_command_x_mean", 0.0)),
            }
        )

    critic = obs.get("critic")
    summary = {
        "backend": "isaaclab",
        "steps": args_cli.steps,
        "action_mode": "trace" if action_trace is not None else ("policy" if policy is not None else args_cli.action_mode),
        "num_envs": env.num_envs,
        "num_obs": int(obs["policy"].shape[-1]),
        "num_privileged_obs": int(critic.shape[-1]) if critic is not None else None,
        "metrics": {},
        "reward_terms": {},
        "raw_reward_terms": {},
        "termination_events": termination_events,
        "termination_profile": args_cli.parity_termination_profile,
        "last_reset_snapshot": getattr(env, "last_reset_snapshot", {}),
            "asset": {
                "body_names": list(env.robot.body_names),
                "contact_sensor_body_names": list(getattr(env.contact_sensor, "body_names", [])),
                "joint_names": list(env.robot.joint_names),
            "termination_body_names": [env.robot.body_names[int(i)] for i in env.termination_contact_indices],
            "feet_body_names": [env.robot.body_names[int(i)] for i in env.feet_indices],
            "base_indices": [int(i) for i in env.termination_contact_indices],
            "default_mass_mean": float(env.robot.data.default_mass.mean().item()),
            "default_mass_sum_env0": float(env.robot.data.default_mass[0].sum().item()),
            "material_parity_enabled": bool(getattr(env, "material_parity_enabled", False)),
            "shape_offset_parity_enabled": bool(getattr(env, "shape_offset_parity_enabled", False)),
            "shape_material": _collect_shape_material_summary(env),
            "body_properties": _collect_body_properties(env),
        },
    }
    for key in samples[0]:
        summary["metrics"][key] = sum(item[key] for item in samples) / len(samples)
    for name, values in reward_terms.items():
        summary["reward_terms"][name] = sum(values) / len(values)
    for name, values in raw_reward_terms.items():
        summary["raw_reward_terms"][name] = sum(values) / len(values)

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args_cli.output:
        with open(args_cli.output, "w") as stream:
            stream.write(payload + "\n")
    if args_cli.trace_output:
        with open(args_cli.trace_output, "w") as stream:
            stream.write(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(payload, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
