from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
import numpy as np
import torch
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from helper.utilities import evenly_distribute_sensors
from gymnasium.spaces import Box, Discrete
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


class TargetTrackingEnv(ParallelEnv):
    metadata = {
        "name": "target_tracking_v0",
        "render_modes": ["human"]
    }

    def __init__(self, n_sensors=5, m_targets=6, area_size=3.0, rho_max=18.0/20.0, alpha_max=np.pi/3,
                 max_theta_delta=np.pi/36, max_steps=40, continuous_actions=False, render_mode=None):
        """
        Implements a multi-agent environment where fixed directional sensors track randomly moving targets.

        - Sensors: Fixed positions, controllable directions (rotate by delta_theta).
        - Targets: Move randomly each step.
        - Objective: Maximize unique tracked targets (within range rho_max and FoV alpha_max).
        - Reward: Shared, number of unique tracked targets per step.
        - Observations: Per sensor, (m_targets, 2) of (dist, rel_angle).
        - Actions:
          - If continuous_actions=True: Box(-1,1,(1,)) – normalized delta_theta, scaled to [-max_theta_delta, max_theta_delta].
          - If False: Discrete(3): 0=left (-max_theta_delta), 1=stay, 2=right (+max_theta_delta).
        - Episode: Fixed length max_steps (truncation).
        - Area: Square [0, area_size] x [0, area_size], clip boundaries.

        Logic Modifications for Continuous Actions:
        - Added 'continuous_actions' flag to match MPE-style (e.g., simple_spread_v3).
        - For continuous: Action is a scalar [-1,1], mapped to rotation delta [-max_theta_delta, +max_theta_delta].
          - Allows finer control (fractional turns) vs. discrete's fixed steps.
          - max_theta_delta defaults to pi/18 (~10 degrees) for max turn per step.
        - In step: For continuous, delta = action[0] * max_theta_delta; for discrete, fixed steps.
        - Directions always wrapped to [0, 2pi).
        - No other changes: Observations, rewards, rendering remain the same.
        - Rationale: Continuous enables gradient-based policies (e.g., DDPG); discrete for simpler baselines.
        """
        self.n_sensors = n_sensors
        self.m_targets = m_targets
        self.area_size = area_size
        self.rho_max = rho_max
        self.alpha_max = alpha_max
        self.max_theta_delta = max_theta_delta
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode

        self.possible_agents = [f"sensor_{i}" for i in range(n_sensors)]
        self.agents = self.possible_agents.copy()

        # Observation: (m_targets, 2) for (dist, rel_angle); unbounded but practical [-pi,pi] for angle
        obs_shape = m_targets * 2
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Action space
        if self.continuous_actions:
            self.action_spaces = {
                agent: Box(low=0, high=1.0, shape=(3,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            self.action_spaces = {
                agent: Discrete(3)  # 0=left, 1=stay, 2=right
                for agent in self.possible_agents
            }

        # Internal state
        self.sensor_pos = None  # (n, 2)
        self.sensor_dir = None  # (n,)
        self.target_pos = None  # (m, 2)
        self.current_step = 0

        # For render
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Random sensor positions (fixed per episode)
        self.sensor_pos = evenly_distribute_sensors(self.n_sensors, [self.area_size]*2)

        # Random initial directions [0, 2pi)
        self.sensor_dir = np.random.uniform(0, 2 * np.pi, self.n_sensors)

        # Random target positions
        self.target_pos = np.random.uniform(0, self.area_size, (self.m_targets, 2))

        self.current_step = 0

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        # Actions: dict {agent: array or int}
        for idx, agent in enumerate(self.agents):
            if self.continuous_actions:
                tensor_act_i = torch.tensor(actions[agent], dtype=torch.float32)
                act_i_cat = Categorical(F.softmax(tensor_act_i))
                act_i = act_i_cat.sample()
                # delta = actions[agent][0] * self.max_theta_delta # for act=(1,)
            else:
                act_i = actions[agent]

            if act_i == 0:  # left
                delta = -self.max_theta_delta
            elif act_i == 2:  # right
                delta = self.max_theta_delta
            else:  # stay (1)
                delta = 0.0
            self.sensor_dir[idx] += delta
            self.sensor_dir[idx] %= (2 * np.pi)

        # Move targets: random walk, sigma = 0.1 * area_size
        move_sigma = 0.1 * self.area_size
        self.target_pos += np.random.normal(0, move_sigma, self.target_pos.shape)
        self.target_pos = np.clip(self.target_pos, 0, self.area_size)

        # Compute tracking
        tracked = self._get_tracked()

        # Reward: number of unique tracked targets
        unique_tracked = len(np.unique(np.where(tracked)[1]))  # column indices (targets)
        reward = unique_tracked
        rewards = {agent: reward for agent in self.agents}

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        truncations = {agent: truncated for agent in self.agents}
        terminations = {agent: False for agent in self.agents}  # No terminal states

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        observations = {}
        for idx, agent in enumerate(self.agents):
            obs = np.zeros((self.m_targets, 2), dtype=np.float32)
            s_pos = self.sensor_pos[idx]
            for j in range(self.m_targets):
                t_pos = self.target_pos[j]
                dx, dy = t_pos - s_pos
                rho = np.sqrt(dx**2 + dy**2)
                alpha = np.arctan2(dy, dx) - self.sensor_dir[idx]
                alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
                obs[j] = [rho, alpha]
            observations[agent] = obs.flatten()
        return observations

    def _get_tracked(self):
        tracked = np.zeros((self.n_sensors, self.m_targets), dtype=bool)
        for i in range(self.n_sensors):
            s_pos = self.sensor_pos[i]
            s_dir = self.sensor_dir[i]
            for j in range(self.m_targets):
                t_pos = self.target_pos[j]
                dx, dy = t_pos - s_pos
                rho = np.sqrt(dx**2 + dy**2)
                if rho > self.rho_max:
                    continue
                alpha = np.arctan2(dy, dx) - s_dir
                alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
                if abs(alpha) <= self.alpha_max / 2:
                    tracked[i, j] = True
        return tracked

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_aspect('equal')

        # Plot targets
        self.ax.scatter(self.target_pos[:, 0], self.target_pos[:, 1], c='red', label='Targets')

        # Plot sensors and FoV
        for i in range(self.n_sensors):
            s_x, s_y = self.sensor_pos[i]
            s_dir_deg = np.degrees(self.sensor_dir[i])
            alpha_deg = np.degrees(self.alpha_max)

            # Sensor position
            self.ax.scatter(s_x, s_y, c='blue', label='Sensors' if i == 0 else None)

            # FoV wedge
            wedge = Wedge((s_x, s_y), self.rho_max, s_dir_deg - alpha_deg/2, s_dir_deg + alpha_deg/2,
                          width=self.rho_max, fill=True, alpha=0.3, color='blue')
            self.ax.add_patch(wedge)

            # Direction line
            dx = self.rho_max * np.cos(self.sensor_dir[i])
            dy = self.rho_max * np.sin(self.sensor_dir[i])
            self.ax.plot([s_x, s_x + dx], [s_y, s_y + dy], 'b--')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None