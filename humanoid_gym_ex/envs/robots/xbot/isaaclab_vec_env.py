from __future__ import annotations

import torch

from humanoid_gym_ex.algo.vec_env import VecEnv


class IsaacLabRslRlVecEnv(VecEnv):
    """Adapt an IsaacLab DirectRLEnv to the local Humanoid-Gym PPO interface."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.cfg.observation_space
        self.num_privileged_obs = env.cfg.state_space
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.extras = {}
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf[:] = value.to(self.env.device)

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = torch.logical_or(terminated, truncated)
        self.obs_buf = obs_dict["policy"]
        self.privileged_obs_buf = obs_dict.get("critic")
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras if extras is not None else {}
        self.extras["time_outs"] = truncated.to(self.device)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        if env_ids is None:
            obs_dict, extras = self.env.reset()
        else:
            self.env._reset_idx(env_ids)
            obs_dict = self.env._get_observations()
            extras = {}
        self.obs_buf = obs_dict["policy"]
        self.privileged_obs_buf = obs_dict.get("critic")
        self.extras = extras if extras is not None else {}
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def close(self):
        return self.env.close()
