from abc import ABC, abstractmethod


class BackendInterface(ABC):
    """Minimal simulator backend contract for future HumanoidGym-Ex phases."""

    @abstractmethod
    def create_sim(self):
        raise NotImplementedError

    @abstractmethod
    def create_envs(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids):
        raise NotImplementedError

    @abstractmethod
    def get_root_states(self):
        raise NotImplementedError

    @abstractmethod
    def get_dof_pos(self):
        raise NotImplementedError

    @abstractmethod
    def get_dof_vel(self):
        raise NotImplementedError

    @abstractmethod
    def get_contact_forces(self):
        raise NotImplementedError

    @abstractmethod
    def set_dof_targets(self, targets_or_torques):
        raise NotImplementedError

    @abstractmethod
    def apply_domain_randomization(self, env_ids=None):
        raise NotImplementedError

    @abstractmethod
    def render_or_viewer_step(self):
        raise NotImplementedError
