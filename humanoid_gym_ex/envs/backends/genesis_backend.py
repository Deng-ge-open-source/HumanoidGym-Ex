from humanoid_gym_ex.envs.base.backend_interface import BackendInterface


class GenesisBackend(BackendInterface):
    """Reserved Genesis backend placeholder."""

    def create_sim(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def create_envs(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def step(self, actions):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def reset(self, env_ids):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def get_root_states(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def get_dof_pos(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def get_dof_vel(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def get_contact_forces(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def set_dof_targets(self, targets_or_torques):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def apply_domain_randomization(self, env_ids=None):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")

    def render_or_viewer_step(self):
        raise NotImplementedError("GenesisBackend is reserved for Phase 3.")
