from __future__ import annotations

import math
from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from humanoid_gym_ex import LEGGED_GYM_ROOT_DIR
from humanoid_gym_ex.envs.backends.isaaclab_backend import IsaacLabBackend
from humanoid_gym_ex.envs.robots.humanoid_config import XBotLCfg


def _quat_wxyz_to_euler_xyz(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _quat_rotate_inverse_wxyz(quat, vec):
    w = quat[:, 0]
    q_vec = quat[:, 1:4]
    a = vec * (2.0 * w * w - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, vec, dim=-1) * (-2.0 * w).unsqueeze(-1)
    c = q_vec * (2.0 * torch.sum(q_vec * vec, dim=-1, keepdim=True))
    return a + b + c


def _quat_apply_wxyz(quat, vec):
    w = quat[:, 0]
    q_vec = quat[:, 1:4]
    a = vec * (2.0 * w * w - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, vec, dim=-1) * (2.0 * w).unsqueeze(-1)
    c = q_vec * (2.0 * torch.sum(q_vec * vec, dim=-1, keepdim=True))
    return a + b + c


def _torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def configure_xbot_isaaclab_terrain(env_cfg, terrain="plane", measure_heights=False, terrain_curriculum=False):
    """Apply the small set of terrain switches exposed by the IsaacLab scripts."""
    env_cfg.measure_heights = measure_heights
    env_cfg.terrain_curriculum = terrain_curriculum
    if terrain == "plane":
        env_cfg.terrain.terrain_type = "plane"
        env_cfg.terrain.terrain_generator = None
        env_cfg.terrain.max_init_terrain_level = None
        env_cfg.terrain.use_terrain_origins = True
    elif terrain in ("heightfield", "trimesh", "rough"):
        rough_cfg = ROUGH_TERRAINS_CFG.replace(
            size=(XBotLCfg.terrain.terrain_length, XBotLCfg.terrain.terrain_width),
            num_rows=XBotLCfg.terrain.num_rows,
            num_cols=XBotLCfg.terrain.num_cols,
            curriculum=terrain_curriculum,
        )
        env_cfg.terrain.terrain_type = "generator"
        env_cfg.terrain.terrain_generator = rough_cfg
        env_cfg.terrain.max_init_terrain_level = XBotLCfg.terrain.max_init_terrain_level
        env_cfg.terrain.use_terrain_origins = True
    else:
        raise ValueError("Unsupported IsaacLab terrain '{}'. Use plane, rough, heightfield, or trimesh.".format(terrain))

    if measure_heights:
        num_height_points = len(XBotLCfg.terrain.measured_points_x) * len(XBotLCfg.terrain.measured_points_y)
        env_cfg.height_scanner = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # Match the original Humanoid-Gym XBot rough-terrain path: measured
        # heights are privileged critic inputs, while actor observations stay
        # at the plane-terrain policy width.
        env_cfg.observation_space = XBotLCfg.env.num_observations
        env_cfg.state_space = (XBotLCfg.env.single_num_privileged_obs + num_height_points) * XBotLCfg.env.c_frame_stack
    else:
        env_cfg.height_scanner = None
        env_cfg.observation_space = XBotLCfg.env.num_observations
        env_cfg.state_space = XBotLCfg.env.num_privileged_obs
    return env_cfg


def configure_xbot_isaaclab_parity_termination(env_cfg, profile="none"):
    """Apply opt-in termination profiles used only for backend parity experiments."""
    if profile in (None, "", "none"):
        return env_cfg
    if profile == "isaacgym_like":
        env_cfg.termination_base_height = 0.80
        env_cfg.termination_orientation = None
        return env_cfg
    raise ValueError("Unsupported IsaacLab parity termination profile '{}'. Use none or isaacgym_like.".format(profile))


@configclass
class XBotIsaacLabEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv config mirroring the original XBot-L Humanoid-Gym task."""

    episode_length_s = XBotLCfg.env.episode_length_s
    decimation = XBotLCfg.control.decimation
    action_scale = XBotLCfg.control.action_scale
    action_space = XBotLCfg.env.num_actions
    observation_space = XBotLCfg.env.num_observations
    state_space = XBotLCfg.env.num_privileged_obs
    measure_heights = False
    terrain_curriculum = False
    height_scanner = None
    termination_base_height = None
    termination_orientation = None
    parity_termination_profile = "none"
    disable_domain_randomization = False
    deterministic_reset = False
    rigid_shape_static_friction = 1.0
    rigid_shape_dynamic_friction = 1.0
    rigid_shape_restitution = XBotLCfg.terrain.restitution
    rigid_shape_contact_offset = XBotLCfg.sim.physx.contact_offset
    rigid_shape_rest_offset = XBotLCfg.sim.physx.rest_offset

    sim: SimulationCfg = SimulationCfg(
        dt=XBotLCfg.sim.dt,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=XBotLCfg.sim.physx.solver_type,
            min_position_iteration_count=XBotLCfg.sim.physx.num_position_iterations,
            max_position_iteration_count=XBotLCfg.sim.physx.num_position_iterations,
            min_velocity_iteration_count=XBotLCfg.sim.physx.num_velocity_iterations,
            max_velocity_iteration_count=XBotLCfg.sim.physx.num_velocity_iterations,
            bounce_threshold_velocity=XBotLCfg.sim.physx.bounce_threshold_velocity,
            gpu_max_rigid_contact_count=XBotLCfg.sim.physx.max_gpu_contact_pairs,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=XBotLCfg.terrain.static_friction,
            dynamic_friction=XBotLCfg.terrain.dynamic_friction,
            restitution=XBotLCfg.terrain.restitution,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=XBotLCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            fix_base=XBotLCfg.asset.fix_base_link,
            merge_fixed_joints=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=XBotLCfg.sim.physx.max_depenetration_velocity,
                solver_position_iteration_count=XBotLCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=XBotLCfg.sim.physx.num_velocity_iterations,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(XBotLCfg.asset.self_collisions),
                solver_position_iteration_count=XBotLCfg.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=XBotLCfg.sim.physx.num_velocity_iterations,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(XBotLCfg.init_state.pos),
            joint_pos=XBotLCfg.init_state.default_joint_angles,
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=XBotLCfg.sim.dt, track_air_time=True
    )


class XBotIsaacLabEnv(DirectRLEnv):
    cfg: XBotIsaacLabEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.xbot_cfg = XBotLCfg()
        if getattr(cfg, "disable_domain_randomization", False):
            self.xbot_cfg.domain_rand.randomize_friction = False
            self.xbot_cfg.domain_rand.randomize_base_mass = False
            self.xbot_cfg.domain_rand.push_robots = False
        super().__init__(cfg, render_mode, **kwargs)
        self.backend = IsaacLabBackend(self, self.robot, self.contact_sensor)
        self._init_humanoid_buffers()
        self._prepare_reward_function()

    def _contact_sensor_history_robot_order(self):
        contact_history = getattr(self.contact_sensor.data, "net_forces_w_history", None)
        if contact_history is None:
            return None
        sensor_body_names = getattr(self.contact_sensor, "body_names", None)
        if sensor_body_names and len(sensor_body_names) == len(self.robot.body_names):
            if not hasattr(self, "contact_sensor_to_robot_body_ids"):
                self.contact_sensor_to_robot_body_ids = [
                    sensor_body_names.index(name) for name in self.robot.body_names
                ]
            return contact_history[:, :, self.contact_sensor_to_robot_body_ids, :]
        return contact_history

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        if self.cfg.height_scanner is not None:
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner
        else:
            self.height_scanner = None
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene._terrain = self.terrain
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_humanoid_buffers(self):
        cfg = self.xbot_cfg
        self.canonical_joint_names = list(cfg.init_state.default_joint_angles.keys())
        joint_sim_ids = [self.robot.joint_names.index(name) for name in self.canonical_joint_names]
        self.joint_sim_ids = torch.tensor(joint_sim_ids, dtype=torch.long, device=self.device)
        self.joint_sim_ids_cpu = self.joint_sim_ids.detach().cpu()
        self.joint_canonical_to_sim_ids = torch.empty_like(self.joint_sim_ids)
        self.joint_canonical_to_sim_ids[self.joint_sim_ids] = torch.arange(len(self.joint_sim_ids), device=self.device)
        self.actions = torch.zeros(self.num_envs, cfg.env.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, cfg.commands.num_commands, device=self.device)
        self.command_ranges = {
            "lin_vel_x": list(cfg.commands.ranges.lin_vel_x),
            "lin_vel_y": list(cfg.commands.ranges.lin_vel_y),
            "ang_vel_yaw": list(cfg.commands.ranges.ang_vel_yaw),
            "heading": list(cfg.commands.ranges.heading),
        }
        self.commands_scale = torch.tensor(
            [cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.lin_vel, cfg.normalization.obs_scales.ang_vel],
            device=self.device,
        )
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.joint_sim_ids].clone()
        self.p_gains = torch.zeros(self.robot.num_joints, device=self.device)
        self.d_gains = torch.zeros(self.robot.num_joints, device=self.device)
        for i, name in enumerate(self.canonical_joint_names):
            for key, value in cfg.control.stiffness.items():
                if key in name:
                    self.p_gains[i] = value
            for key, value in cfg.control.damping.items():
                if key in name:
                    self.d_gains[i] = value
        self.torque_limits = torch.ones(self.robot.num_joints, device=self.device) * 200.0 * cfg.safety.torque_limit
        self.obs_scales = cfg.normalization.obs_scales
        self.noise_scale_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        self.measure_heights = bool(getattr(self.cfg, "measure_heights", False))
        self.num_height_points = len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
        self.obs_history = deque(maxlen=cfg.env.frame_stack)
        self.critic_history = deque(maxlen=cfg.env.c_frame_stack)
        for _ in range(cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, cfg.env.num_single_obs, device=self.device))
        for _ in range(cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, cfg.env.single_num_privileged_obs + self.measure_heights * self.num_height_points, device=self.device)
            )
        self.feet_indices, _ = self.robot.find_bodies(".*ankle_roll.*")
        self.knee_indices, _ = self.robot.find_bodies(".*knee.*")
        self.base_indices, _ = self.robot.find_bodies("base_link")
        self.penalised_contact_indices = self.base_indices
        self.termination_contact_indices = self.base_indices
        self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_frictions = torch.ones(self.num_envs, 1, device=self.device) * cfg.terrain.static_friction
        self.body_mass = torch.ones(self.num_envs, 1, device=self.device) * 30.0
        self.push_interval = max(1, math.ceil(cfg.domain_rand.push_interval_s / self.step_dt))
        self.material_randomization_enabled = False
        self.mass_randomization_enabled = False
        self.contact_forces = self.backend.get_contact_forces()
        self.rigid_state = self.robot.data.body_state_w
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        self.base_quat = self.robot.data.root_quat_w
        self.base_euler_xyz = _quat_wxyz_to_euler_xyz(self.base_quat)
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity = _quat_rotate_inverse_wxyz(self.base_quat, self.gravity_vec)
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        feet_count = max(len(self.feet_indices), 1)
        self.feet_air_time = torch.zeros(self.num_envs, feet_count, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, feet_count, dtype=torch.bool, device=self.device)
        self.feet_height = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_feet_z = torch.zeros(self.num_envs, 2, device=self.device) + 0.05
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        self._apply_rigid_shape_parity_properties()
        self.apply_domain_randomization()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _apply_rigid_shape_parity_properties(self):
        """Set IsaacLab rigid-shape defaults to the IsaacGym XBot baseline before optional randomization."""
        view = getattr(self.robot, "root_physx_view", None)
        if view is None:
            return
        try:
            materials = view.get_material_properties()
            materials[..., 0] = float(self.cfg.rigid_shape_static_friction)
            materials[..., 1] = float(self.cfg.rigid_shape_dynamic_friction)
            materials[..., 2] = float(self.cfg.rigid_shape_restitution)
            env_ids_cpu = torch.arange(self.num_envs, dtype=torch.long, device="cpu")
            view.set_material_properties(materials, env_ids_cpu)
            self.material_parity_enabled = True
        except Exception as exc:
            self.material_parity_enabled = False
            if not getattr(self, "_reported_material_parity_error", False):
                print("[HumanoidGym-Ex] IsaacLab material parity write failed:", exc, flush=True)
                self._reported_material_parity_error = True
        try:
            env_ids_cpu = torch.arange(self.num_envs, dtype=torch.long, device="cpu")
            contact_offsets = view.get_contact_offsets()
            contact_offsets[:] = float(self.cfg.rigid_shape_contact_offset)
            view.set_contact_offsets(contact_offsets, env_ids_cpu)
            rest_offsets = view.get_rest_offsets()
            rest_offsets[:] = float(self.cfg.rigid_shape_rest_offset)
            view.set_rest_offsets(rest_offsets, env_ids_cpu)
            self.shape_offset_parity_enabled = True
        except Exception as exc:
            self.shape_offset_parity_enabled = False
            if not getattr(self, "_reported_shape_offset_parity_error", False):
                print("[HumanoidGym-Ex] IsaacLab shape-offset parity write failed:", exc, flush=True)
                self._reported_shape_offset_parity_error = True

    def apply_domain_randomization(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.xbot_cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.xbot_cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)

    def _randomize_friction(self, env_ids):
        friction_range = self.xbot_cfg.domain_rand.friction_range
        if not hasattr(self, "friction_coeffs"):
            num_buckets = 256
            bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1), device=self.device)
            friction_buckets = _torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), self.device)
            self.friction_coeffs = friction_buckets[bucket_ids.squeeze(-1)]
        samples = self.friction_coeffs[env_ids]
        self.env_frictions[env_ids] = samples
        try:
            env_ids_cpu = env_ids.detach().cpu()
            materials = self.robot.root_physx_view.get_material_properties()
            material_samples = materials[env_ids_cpu].clone()
            material_samples[..., 0] = samples.detach().cpu()
            material_samples[..., 1] = samples.detach().cpu()
            materials[env_ids_cpu] = material_samples
            self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
            self.material_randomization_enabled = True
        except Exception as exc:
            if not getattr(self, "_reported_material_randomization_error", False):
                print("[HumanoidGym-Ex] IsaacLab friction randomization kept in buffers only:", exc, flush=True)
                self._reported_material_randomization_error = True

    def _randomize_base_mass(self, env_ids):
        mass_range = self.xbot_cfg.domain_rand.added_mass_range
        mass_delta = _torch_rand_float(mass_range[0], mass_range[1], (len(env_ids),), self.device)
        try:
            env_ids_cpu = env_ids.detach().cpu()
            body_id = int(self.base_indices[0]) if len(self.base_indices) else 0
            masses = self.robot.root_physx_view.get_masses()
            masses[env_ids_cpu, body_id] = self.robot.data.default_mass[env_ids_cpu, body_id] + mass_delta.detach().cpu()
            masses[env_ids_cpu, body_id] = torch.clamp(masses[env_ids_cpu, body_id], min=1e-6)
            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)
            inertias = self.robot.root_physx_view.get_inertias()
            ratios = masses[env_ids_cpu, body_id] / self.robot.data.default_mass[env_ids_cpu, body_id]
            inertias[env_ids_cpu, body_id] = self.robot.data.default_inertia[env_ids_cpu, body_id] * ratios.unsqueeze(-1)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
            self.body_mass[env_ids, 0] = masses[env_ids_cpu, body_id].to(self.device)
            self.mass_randomization_enabled = True
        except Exception as exc:
            default_mass = torch.ones(len(env_ids), device=self.device) * 30.0
            self.body_mass[env_ids, 0] = torch.clamp(default_mass + mass_delta, min=1e-6)
            if not getattr(self, "_reported_mass_randomization_error", False):
                print("[HumanoidGym-Ex] IsaacLab base-mass randomization kept in buffers only:", exc, flush=True)
                self._reported_mass_randomization_error = True

    def _push_robots(self):
        max_vel = self.xbot_cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.xbot_cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = _torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), self.device)
        self.rand_push_force[:, 2] = 0.0
        self.rand_push_torque[:] = _torch_rand_float(-max_push_angular, max_push_angular, (self.num_envs, 3), self.device)
        root_velocity = self.robot.data.root_vel_w.clone()
        root_velocity[:, 0:2] = self.rand_push_force[:, :2]
        root_velocity[:, 3:6] = self.rand_push_torque
        self.robot.write_root_velocity_to_sim(root_velocity)

    def _prepare_reward_function(self):
        cfg = self.xbot_cfg
        self.reward_scales = {}
        for name in dir(cfg.rewards.scales):
            if name.startswith("_"):
                continue
            value = getattr(cfg.rewards.scales, name)
            if callable(value) or value == 0:
                continue
            self.reward_scales[name] = value * self.step_dt
        self.reward_names = []
        self.reward_functions = []
        for name in self.reward_scales:
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales
        }

    def _update_state_cache(self):
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        self.base_quat = self.robot.data.root_quat_w
        self.base_euler_xyz = _quat_wxyz_to_euler_xyz(self.base_quat)
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.projected_gravity = _quat_rotate_inverse_wxyz(self.base_quat, self.gravity_vec)
        self.rigid_state = self.robot.data.body_state_w
        self.contact_forces = self.backend.get_contact_forces()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = _torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),), self.device
        )
        self.commands[env_ids, 1] = _torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),), self.device
        )
        if self.xbot_cfg.commands.heading_command:
            self.commands[env_ids, 3] = _torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids),), self.device
            )
        else:
            self.commands[env_ids, 2] = _torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids),), self.device
            )
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def update_command_curriculum(self, env_ids):
        if len(env_ids) == 0 or "tracking_lin_vel" not in self.episode_sums:
            return
        tracking_mean = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
        threshold = 0.8 * self.reward_scales["tracking_lin_vel"]
        if tracking_mean > threshold:
            max_curriculum = self.xbot_cfg.commands.max_curriculum
            self.command_ranges["lin_vel_x"][0] = max(self.command_ranges["lin_vel_x"][0] - 0.5, -max_curriculum)
            self.command_ranges["lin_vel_x"][1] = min(self.command_ranges["lin_vel_x"][1] + 0.5, max_curriculum)

    def _post_physics_step_callback(self):
        resample_interval = int(self.xbot_cfg.commands.resampling_time / self.step_dt)
        env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.xbot_cfg.commands.heading_command:
            forward = _quat_apply_wxyz(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            yaw_cmd = 0.5 * torch.atan2(torch.sin(self.commands[:, 3] - heading), torch.cos(self.commands[:, 3] - heading))
            self.commands[:, 2] = torch.clip(yaw_cmd, -1.0, 1.0)
        if self.measure_heights:
            self.measured_heights = self._get_heights()
        if self.xbot_cfg.domain_rand.push_robots and self.common_step_counter % self.push_interval == 0:
            self._push_robots()

    def _get_heights(self):
        if self.height_scanner is None:
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        return self.height_scanner.data.ray_hits_w[..., 2]

    def _update_terrain_curriculum(self, env_ids):
        if not getattr(self.cfg, "terrain_curriculum", False):
            return
        if not hasattr(self.terrain, "update_env_origins") or getattr(self.terrain, "terrain_origins", None) is None:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.terrain.env_origins[env_ids, :2], dim=1)
        move_up = distance > (XBotLCfg.terrain.terrain_length / 2.0)
        required_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down = (distance < required_distance) & ~move_up
        self.terrain.update_env_origins(env_ids, move_up, move_down)

    def _get_phase(self):
        # IsaacGym's reset path performs one zero-action step before the first
        # observation used by PPO, so Humanoid-Gym phase-dependent obs/rewards
        # see episode_length_buf one control step ahead of IsaacLab DirectRLEnv.
        return (self.episode_length_buf + 1) * self.step_dt / self.xbot_cfg.rewards.cycle_time

    def _get_gait_phase(self):
        sin_pos = torch.sin(2.0 * math.pi * self._get_phase())
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2.0 * math.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.xbot_cfg.rewards.target_joint_pos_scale
        scale_2 = 2.0 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_r[sin_pos_r < 0] = 0
        if self.ref_dof_pos.shape[1] >= 11:
            self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
            self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
            self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
            self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
            self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
            self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    def _pre_physics_step(self, actions):
        cfg = self.xbot_cfg
        actions = torch.clip(actions, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        raw_actions = actions.detach().clone()
        delay = torch.rand((self.num_envs, 1), device=self.device) * cfg.domain_rand.action_delay
        delayed_actions = (1.0 - delay) * actions + delay * self.actions
        noise = cfg.domain_rand.action_noise * torch.randn_like(delayed_actions) * delayed_actions
        actions = delayed_actions + noise
        self.last_action_preprocess = {
            "delay_mean": float(delay.mean().item()),
            "delay_max": float(delay.max().item()),
            "raw_action_abs": float(torch.mean(torch.abs(raw_actions)).item()),
            "delayed_action_abs": float(torch.mean(torch.abs(delayed_actions)).item()),
            "noise_abs": float(torch.mean(torch.abs(noise)).item()),
            "processed_action_abs": float(torch.mean(torch.abs(actions)).item()),
        }
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.actions[:] = actions

    def _apply_action(self):
        cfg = self.xbot_cfg
        targets = self.actions * cfg.control.action_scale + self.default_dof_pos
        joint_pos = self.robot.data.joint_pos[:, self.joint_sim_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_sim_ids]
        torques = self.p_gains * (targets - joint_pos) - self.d_gains * joint_vel
        self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self.backend.set_dof_targets(self.torques[:, self.joint_canonical_to_sim_ids])

    def _get_observations(self):
        cfg = self.xbot_cfg
        self._update_state_cache()
        self.compute_ref_state()
        phase = self._get_phase()
        sin_pos = torch.sin(2.0 * math.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2.0 * math.pi * phase).unsqueeze(1)
        command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        obs_now = torch.cat((command_input, q, dq, self.actions, self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz), dim=-1)
        diff = self.dof_pos - self.ref_dof_pos
        contact_mask = torch.zeros(self.num_envs, 2, device=self.device)
        if len(self.feet_indices) >= 2:
            contact_mask = self.contact_forces[:, self.feet_indices[:2], 2] > 5.0
        privileged_obs = torch.cat(
            (
                command_input,
                q,
                dq,
                self.actions,
                diff,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_euler_xyz,
                self.rand_push_force[:, :2],
                self.rand_push_torque,
                self.env_frictions,
                self.body_mass / 30.0,
                self._get_gait_phase(),
                contact_mask.float(),
            ),
            dim=-1,
        )
        if self.measure_heights:
            height_obs = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1.0, 1.0
            ) * self.obs_scales.height_measurements
            privileged_obs = torch.cat((privileged_obs, height_obs), dim=-1)
        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs)
        obs = torch.stack(list(self.obs_history), dim=1).reshape(self.num_envs, -1)
        critic = torch.cat(list(self.critic_history), dim=1)
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state
        return {"policy": obs, "critic": critic}

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)
        self.extras["episode"] = {}
        self.last_reward_terms = {}
        self.last_raw_reward_terms = {}
        for name, func in zip(self.reward_names, self.reward_functions):
            raw_rew = func()
            rew = raw_rew * self.reward_scales[name]
            rewards += rew
            self.episode_sums[name] += rew
            self.last_raw_reward_terms[name] = raw_rew.detach()
            self.last_reward_terms[name] = rew.detach()
            self.extras["episode"]["rew_" + name] = self.episode_sums[name].mean() / self.max_episode_length_s
        if self.xbot_cfg.rewards.only_positive_rewards:
            rewards = torch.clip(rewards, min=0.0)
        return rewards

    def _get_dones(self):
        self._update_state_cache()
        self._post_physics_step_callback()
        time_out = self.episode_length_buf > self.max_episode_length
        died = torch.zeros_like(time_out)
        contact_died = torch.zeros_like(time_out)
        termination_contact = torch.zeros(self.num_envs, len(self.termination_contact_indices), device=self.device)
        if len(self.termination_contact_indices) > 0:
            contact_history = self._contact_sensor_history_robot_order()
            if contact_history is not None:
                termination_forces = torch.norm(contact_history[:, :, self.termination_contact_indices, :], dim=-1)
                termination_contact = torch.max(termination_forces, dim=1)[0]
                contact_died = torch.any(termination_contact > 1.0, dim=1)
            else:
                termination_contact = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
                contact_died = torch.any(termination_contact > 1.0, dim=1)
            died = contact_died
        base_height_died = torch.zeros_like(time_out)
        if self.cfg.termination_base_height is not None:
            base_height_died = self.root_states[:, 2] < float(self.cfg.termination_base_height)
            died = died | base_height_died
        orientation_died = torch.zeros_like(time_out)
        if self.cfg.termination_orientation is not None:
            orientation_died = torch.any(torch.abs(self.base_euler_xyz[:, :2]) > float(self.cfg.termination_orientation), dim=1)
            died = died | orientation_died
        self.last_termination_snapshot = {
            "contact": contact_died.detach().clone(),
            "time_out": time_out.detach().clone(),
            "base_height": base_height_died.detach().clone(),
            "orientation": orientation_died.detach().clone(),
            "termination_contact": termination_contact.detach().clone(),
            "base_z": self.root_states[:, 2].detach().clone(),
            "base_euler_xyz": self.base_euler_xyz.detach().clone(),
            "episode_length": self.episode_length_buf.detach().clone(),
            "reset": (died | time_out).detach().clone(),
        }
        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        reset_snapshot = {}
        if hasattr(self, "root_states"):
            reset_snapshot = {
                "env_count": int(len(env_ids)),
                "pre_episode_length_mean": float(self.episode_length_buf[env_ids].float().mean().item()),
                "pre_root_z_mean": float(self.root_states[env_ids, 2].mean().item()),
                "pre_dof_pos_abs": float(torch.mean(torch.abs(self.dof_pos[env_ids] - self.default_dof_pos[env_ids])).item()),
                "pre_dof_vel_abs": float(torch.mean(torch.abs(self.dof_vel[env_ids])).item()),
                "pre_command_x_mean": float(self.commands[env_ids, 0].mean().item()),
                "pre_action_abs": float(torch.mean(torch.abs(self.actions[env_ids])).item()),
            }
        if (
            hasattr(self, "episode_sums")
            and self.xbot_cfg.commands.curriculum
            and self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)
        if hasattr(self, "root_states") and self.common_step_counter > 0:
            self._update_terrain_curriculum(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        if self.cfg.terrain.terrain_type == "generator":
            default_root_state[:, :2] += _torch_rand_float(-1.0, 1.0, (len(env_ids), 2), self.device)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        canonical_joint_pos = joint_pos[:, self.joint_sim_ids]
        if not self.cfg.deterministic_reset:
            canonical_joint_pos += _torch_rand_float(-0.1, 0.1, canonical_joint_pos.shape, self.device)
        joint_pos[:, self.joint_sim_ids] = canonical_joint_pos
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.last_rigid_state[env_ids] = 0.0
        for name in getattr(self, "episode_sums", {}):
            self.episode_sums[name][env_ids] = 0.0
        self._resample_commands(env_ids)
        for item in self.obs_history:
            item[env_ids] = 0.0
        for item in self.critic_history:
            item[env_ids] = 0.0
        self._update_state_cache()
        if reset_snapshot:
            reset_snapshot.update(
                {
                    "post_episode_length_mean": float(self.episode_length_buf[env_ids].float().mean().item()),
                    "post_root_z_mean": float(self.root_states[env_ids, 2].mean().item()),
                    "post_dof_pos_abs": float(torch.mean(torch.abs(self.dof_pos[env_ids] - self.default_dof_pos[env_ids])).item()),
                    "post_dof_vel_abs": float(torch.mean(torch.abs(self.dof_vel[env_ids])).item()),
                    "post_command_x_mean": float(self.commands[env_ids, 0].mean().item()),
                    "post_action_abs": float(torch.mean(torch.abs(self.actions[env_ids])).item()),
                    "post_last_action_abs": float(torch.mean(torch.abs(self.last_actions[env_ids])).item()),
                }
            )
            self.last_reset_snapshot = reset_snapshot

    def _reward_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        return torch.exp(-2.0 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

    def _reward_feet_distance(self):
        if len(self.feet_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)
        foot_pos = self.rigid_state[:, self.feet_indices[:2], :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.xbot_cfg.rewards.min_dist
        max_df = self.xbot_cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0.0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0

    def _reward_knee_distance(self):
        if len(self.knee_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)
        knee_pos = self.rigid_state[:, self.knee_indices[:2], :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = self.xbot_cfg.rewards.min_dist
        max_df = self.xbot_cfg.rewards.max_dist / 2.0
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - max_df, 0.0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0

    def _reward_foot_slip(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        foot_ids = self.feet_indices[:2]
        contact = self.contact_forces[:, foot_ids, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, foot_ids, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        foot_ids = self.feet_indices[:2]
        contact = self.contact_forces[:, foot_ids, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask.bool()), self.last_contacts[:, :2])
        self.last_contacts[:, :2] = contact
        first_contact = (self.feet_air_time[:, :2] > 0.0) * self.contact_filt
        self.feet_air_time[:, :2] += self.step_dt
        air_time = self.feet_air_time[:, :2].clamp(0, 0.5) * first_contact
        self.feet_air_time[:, :2] *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices[:2], 2] > 5.0
        stance_mask = self._get_gait_phase().bool()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10.0)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20.0)
        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_contact_forces(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices[:2], :], dim=-1) - self.xbot_cfg.rewards.max_contact_force).clip(0, 400),
            dim=1,
        )

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100.0) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        if len(self.feet_indices) < 2:
            return torch.exp(-torch.abs(self.root_states[:, 2] - self.xbot_cfg.rewards.base_height_target) * 100.0)
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(self.rigid_state[:, self.feet_indices[:2], 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.xbot_cfg.rewards.base_height_target) * 100.0)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        return torch.exp(-torch.norm(root_acc, dim=1) * 3.0)

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10.0)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)
        return (lin_mismatch + ang_mismatch) / 2.0

    def _reward_track_vel_hard(self):
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10.0)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10.0)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.xbot_cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.xbot_cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        if len(self.feet_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices[:2], 2] > 5.0
        feet_z = self.rigid_state[:, self.feet_indices[:2], 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1.0 - self._get_gait_phase()
        rew_pos = torch.abs(self.feet_height - self.xbot_cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = -self.xbot_cfg.rewards.high_speed_penalty
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.step_dt), dim=1)

    def _reward_collision(self):
        if len(self.penalised_contact_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2.0 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
