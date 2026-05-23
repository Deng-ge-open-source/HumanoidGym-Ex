from humanoid_gym_ex.envs.base.backend_interface import BackendInterface


class IsaacLabBackend(BackendInterface):
    """Thin tensor adapter for IsaacLab Direct workflow environments.

    The DirectRLEnv subclass still owns scene construction, reward, observation,
    command, and reset policy.  This adapter only gives the upper humanoid code
    the same state/action access points used by the IsaacGym backend.
    """

    def __init__(self, env=None, robot=None, contact_sensor=None, effort_control=True):
        self.env = env
        self.robot = robot
        self.contact_sensor = contact_sensor
        self.effort_control = effort_control

    def bind(self, env=None, robot=None, contact_sensor=None):
        if env is not None:
            self.env = env
        if robot is not None:
            self.robot = robot
        if contact_sensor is not None:
            self.contact_sensor = contact_sensor
        return self

    def _require_env(self):
        if self.env is None:
            raise RuntimeError("IsaacLabBackend requires a DirectRLEnv instance.")
        return self.env

    def _require_robot(self):
        if self.robot is None:
            env = self._require_env()
            self.robot = getattr(env, "robot", getattr(env, "_robot", None))
        if self.robot is None:
            raise RuntimeError("IsaacLabBackend requires an Articulation robot.")
        return self.robot

    def create_sim(self):
        return self._require_env().sim

    def create_envs(self):
        return getattr(self._require_env(), "scene", None)

    def step(self, actions):
        return self._require_env().step(actions)

    def reset(self, env_ids):
        env = self._require_env()
        if env_ids is None:
            return env.reset()
        return env._reset_idx(env_ids)

    def get_root_states(self):
        return self._require_robot().data.root_state_w

    def get_dof_pos(self):
        return self._require_robot().data.joint_pos

    def get_dof_vel(self):
        return self._require_robot().data.joint_vel

    def get_contact_forces(self):
        if self.contact_sensor is None and self.env is not None:
            self.contact_sensor = getattr(self.env, "contact_sensor", None)
        if self.contact_sensor is not None:
            forces = self.contact_sensor.data.net_forces_w
            robot = self._require_robot()
            sensor_body_names = getattr(self.contact_sensor, "body_names", None)
            robot_body_names = getattr(robot, "body_names", None)
            if sensor_body_names and robot_body_names and len(sensor_body_names) == len(robot_body_names):
                if not hasattr(self, "_contact_sensor_to_robot_ids"):
                    self._contact_sensor_to_robot_ids = [
                        sensor_body_names.index(name) for name in robot_body_names
                    ]
                return forces[:, self._contact_sensor_to_robot_ids]
            return forces
        robot = self._require_robot()
        return robot.data.body_state_w[..., 0:3] * 0.0

    def set_dof_targets(self, targets_or_torques):
        robot = self._require_robot()
        if self.effort_control:
            robot.set_joint_effort_target(targets_or_torques)
        else:
            robot.set_joint_position_target(targets_or_torques)

    def apply_domain_randomization(self, env_ids=None):
        env = self._require_env()
        if hasattr(env, "apply_domain_randomization"):
            return env.apply_domain_randomization(env_ids)
        return None

    def render_or_viewer_step(self):
        return self._require_env().render()
