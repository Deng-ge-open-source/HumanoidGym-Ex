"""Collect IsaacGym XBot rollout statistics for backend alignment."""

import json
import os

import isaacgym  # noqa: F401
import numpy as np
import torch

from humanoid_gym_ex.envs import task_registry
from humanoid_gym_ex.algo.ppo.on_policy_runner import OnPolicyRunner
from humanoid_gym_ex.utils import class_to_dict, get_args


def _mean_abs(tensor):
    return float(torch.mean(torch.abs(tensor)).item())


def _mean_norm(tensor):
    return float(torch.mean(torch.norm(tensor, dim=-1)).item())


def _std(tensor):
    return float(torch.std(tensor.float(), unbiased=False).item())


def _mean(tensor):
    return float(torch.mean(tensor.float()).item())


def _configure_for_diagnostics(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.action_delay = 0.0
    env_cfg.domain_rand.action_noise = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]


def _collect_shape_summary(env):
    summary = {}
    try:
        shape_props = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])
    except Exception as exc:
        summary["rigid_shape_properties_error"] = str(exc)
        return summary
    summary["num_shapes"] = len(shape_props)
    fields = (
        "friction",
        "restitution",
        "rolling_friction",
        "torsion_friction",
        "contact_offset",
        "rest_offset",
    )
    for field in fields:
        values = [float(getattr(prop, field)) for prop in shape_props if hasattr(prop, field)]
        if values:
            summary[field + "_mean"] = sum(values) / len(values)
            summary[field + "_min"] = min(values)
            summary[field + "_max"] = max(values)
    summary["shapes"] = [
        {
            field: float(getattr(prop, field))
            for field in fields
            if hasattr(prop, field)
        }
        for prop in shape_props
    ]
    return summary


def _vec3_to_list(value):
    if value is None:
        return None
    if all(hasattr(value, field) for field in ("x", "y", "z")):
        return [float(value.x), float(value.y), float(value.z)]
    return None


def _mat33_to_list(value):
    if value is None:
        return None
    columns = []
    for field in ("x", "y", "z"):
        column = _vec3_to_list(getattr(value, field, None))
        if column is None:
            return None
        columns.append(column)
    return columns


def _collect_body_properties(env):
    try:
        body_props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    except Exception as exc:
        return {"error": str(exc), "bodies": []}
    bodies = []
    body_names = list(getattr(env, "body_names", []))
    for index, prop in enumerate(body_props):
        body = {
            "index": index,
            "name": body_names[index] if index < len(body_names) else str(index),
            "mass": float(prop.mass),
        }
        com = _vec3_to_list(getattr(prop, "com", None))
        inertia = _mat33_to_list(getattr(prop, "inertia", None))
        if com is not None:
            body["com"] = com
        if inertia is not None:
            body["inertia"] = inertia
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
    args = get_args()
    steps = int(os.environ.get("HGEX_DIAG_STEPS", "120"))
    action_mode = os.environ.get("HGEX_DIAG_ACTION", "zero")
    seed = int(os.environ.get("HGEX_DIAG_SEED", "1"))
    output = os.environ.get("HGEX_DIAG_OUTPUT", "")
    trace_output = os.environ.get("HGEX_DIAG_TRACE_OUTPUT", "")
    checkpoint_path = os.environ.get("HGEX_DIAG_CHECKPOINT", "")
    fix_command = os.environ.get("HGEX_DIAG_FIX_COMMAND", "0") == "1"
    action_file = os.environ.get("HGEX_DIAG_ACTION_FILE", "")
    deterministic_reset = os.environ.get("HGEX_DIAG_DETERMINISTIC_RESET", "0") == "1"

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.seed = seed
    env_cfg.env.deterministic_reset = deterministic_reset
    train_cfg.seed = seed
    _configure_for_diagnostics(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    policy = None
    if checkpoint_path:
        runner = OnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=args.rl_device)
        runner.load(checkpoint_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
    else:
        obs, privileged_obs = env.reset()
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    action_trace = None
    if action_file:
        with np.load(action_file) as data:
            action_trace = torch.as_tensor(data["actions"], dtype=torch.float32, device=env.device)
        if action_trace.ndim != 3 or action_trace.shape[1] < env.num_envs or action_trace.shape[2] != env.num_actions:
            raise ValueError(
                "HGEX_DIAG_ACTION_FILE must contain actions with shape [steps, num_envs, num_actions], got {}".format(
                    tuple(action_trace.shape)
                )
            )
        if action_trace.shape[0] < steps:
            raise ValueError("HGEX_DIAG_ACTION_FILE has {} steps but {} were requested".format(action_trace.shape[0], steps))

    samples = []
    trace = []
    termination_events = []
    reward_terms = {}
    raw_reward_terms = {}
    previous_raw_actions = None
    for step in range(steps):
        if action_trace is not None:
            actions = action_trace[step, : env.num_envs].clone()
        elif policy is not None:
            actions = policy(obs.detach())
        elif action_mode == "zero":
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        elif action_mode == "random":
            actions = torch.rand(env.num_envs, env.num_actions, generator=generator, device=env.device) * 2.0 - 1.0
        else:
            raise ValueError("HGEX_DIAG_ACTION must be zero or random")
        if previous_raw_actions is None:
            action_delta_abs = torch.zeros((), device=env.device)
        else:
            action_delta_abs = torch.mean(torch.abs(actions - previous_raw_actions))
        raw_action_abs = torch.mean(torch.abs(actions))
        previous_raw_actions = actions.detach().clone()
        if fix_command:
            env.commands[:, 0] = 0.5
            env.commands[:, 1:] = 0.0
        obs, privileged_obs, rewards, dones, extras = env.step(actions)
        step_events = _snapshot_events(env, step, dones)
        termination_events.extend(step_events)
        reset_snapshot = getattr(env, "last_reset_snapshot", {}) if int(dones.sum().item()) > 0 else {}
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
        for name, value in getattr(env, "last_reward_terms", {}).items():
            reward_terms.setdefault(name, []).append(float(value.mean().item()))
        for name, value in getattr(env, "last_raw_reward_terms", {}).items():
            raw_reward_terms.setdefault(name, []).append(float(value.mean().item()))
        termination_contact = torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
        feet_forces = env.contact_forces[:, env.feet_indices, :]
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

    summary = {
        "backend": "isaacgym",
        "steps": steps,
        "action_mode": "trace" if action_trace is not None else ("policy" if policy is not None else action_mode),
        "num_envs": env.num_envs,
        "num_obs": int(obs.shape[-1]),
        "num_privileged_obs": int(privileged_obs.shape[-1]) if privileged_obs is not None else None,
        "metrics": {},
        "reward_terms": {},
        "raw_reward_terms": {},
        "termination_events": termination_events,
        "last_reset_snapshot": getattr(env, "last_reset_snapshot", {}),
        "asset": {
            "body_names": list(getattr(env, "body_names", [])),
            "joint_names": list(getattr(env, "dof_names", [])),
            "termination_body_names": [env.body_names[int(i)] for i in env.termination_contact_indices],
            "feet_body_names": [env.body_names[int(i)] for i in env.feet_indices],
            "base_indices": [int(i) for i in env.termination_contact_indices],
            "shape_material": _collect_shape_summary(env),
            "body_properties": _collect_body_properties(env),
        },
    }
    body_props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    masses = [float(prop.mass) for prop in body_props]
    summary["asset"]["default_mass_mean"] = sum(masses) / len(masses)
    summary["asset"]["default_mass_sum_env0"] = sum(masses)
    for key in samples[0]:
        summary["metrics"][key] = sum(item[key] for item in samples) / len(samples)
    for name, values in reward_terms.items():
        summary["reward_terms"][name] = sum(values) / len(values)
    for name, values in raw_reward_terms.items():
        summary["raw_reward_terms"][name] = sum(values) / len(values)

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if output:
        with open(output, "w") as stream:
            stream.write(payload + "\n")
    if trace_output:
        with open(trace_output, "w") as stream:
            stream.write(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
