from humanoid_gym_ex.envs.base.backend_interface import BackendInterface

from isaacgym import gymapi, gymtorch


class IsaacGymBackend(BackendInterface):
    """Thin IsaacGym adapter used by the Humanoid-Gym-style env classes."""

    def __init__(self, task):
        self.task = task
        self.gym = gymapi.acquire_gym()

    def create_sim(self):
        task = self.task
        return self.gym.create_sim(
            task.sim_device_id,
            task.graphics_device_id,
            task.physics_engine,
            task.sim_params,
        )

    def create_envs(self):
        return self.task._create_envs()

    def step(self, actions):
        del actions
        self.gym.simulate(self.task.sim)
        if self.task.device == "cpu":
            self.gym.fetch_results(self.task.sim, True)

    def reset(self, env_ids):
        self.set_dof_state_indexed(env_ids)
        self.set_actor_root_state_indexed(env_ids)

    def get_root_states(self):
        return self.task.root_states

    def get_dof_pos(self):
        return self.task.dof_pos

    def get_dof_vel(self):
        return self.task.dof_vel

    def get_contact_forces(self):
        return self.task.contact_forces

    def set_dof_targets(self, targets_or_torques):
        self.gym.set_dof_actuation_force_tensor(
            self.task.sim, gymtorch.unwrap_tensor(targets_or_torques)
        )

    def apply_domain_randomization(self, env_ids=None):
        del env_ids
        return None

    def render_or_viewer_step(self, sync_frame_time=True):
        task = self.task
        if task.viewer:
            if self.gym.query_viewer_has_closed(task.viewer):
                raise SystemExit

            for evt in self.gym.query_viewer_action_events(task.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    raise SystemExit
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    task.enable_viewer_sync = not task.enable_viewer_sync

            if task.device != "cpu":
                self.gym.fetch_results(task.sim, True)

            if task.enable_viewer_sync:
                self.gym.step_graphics(task.sim)
                self.gym.draw_viewer(task.viewer, task.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(task.sim)
            else:
                self.gym.poll_viewer_events(task.viewer)

    def prepare_sim(self):
        self.gym.prepare_sim(self.task.sim)

    def create_viewer(self, camera_properties):
        return self.gym.create_viewer(self.task.sim, camera_properties)

    def create_camera_sensor(self, env_handle, camera_properties):
        return self.gym.create_camera_sensor(env_handle, camera_properties)

    def subscribe_viewer_keyboard_event(self, key, action):
        self.gym.subscribe_viewer_keyboard_event(self.task.viewer, key, action)

    def refresh_dof_state(self):
        self.gym.refresh_dof_state_tensor(self.task.sim)

    def refresh_actor_root_state(self):
        self.gym.refresh_actor_root_state_tensor(self.task.sim)

    def refresh_contact_forces(self):
        self.gym.refresh_net_contact_force_tensor(self.task.sim)

    def refresh_rigid_body_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.task.sim)

    def refresh_state_tensors(self, include_dof=False):
        if include_dof:
            self.refresh_dof_state()
        self.refresh_actor_root_state()
        self.refresh_contact_forces()
        self.refresh_rigid_body_state()

    def acquire_state_tensors(self):
        sim = self.task.sim
        return {
            "actor_root_state": self.gym.acquire_actor_root_state_tensor(sim),
            "dof_state": self.gym.acquire_dof_state_tensor(sim),
            "net_contact_forces": self.gym.acquire_net_contact_force_tensor(sim),
            "rigid_body_state": self.gym.acquire_rigid_body_state_tensor(sim),
        }

    def set_dof_state_indexed(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=self.task.env_id_int_dtype)
        self.gym.set_dof_state_tensor_indexed(
            self.task.sim,
            gymtorch.unwrap_tensor(self.task.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def set_actor_root_state_indexed(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=self.task.env_id_int_dtype)
        self.gym.set_actor_root_state_tensor_indexed(
            self.task.sim,
            gymtorch.unwrap_tensor(self.task.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def set_actor_root_state(self, root_states):
        self.gym.set_actor_root_state_tensor(
            self.task.sim, gymtorch.unwrap_tensor(root_states)
        )
