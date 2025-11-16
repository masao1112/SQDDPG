"""
Environment description:
Context: Initiate n_sensors at evenly distributed position in a 2D grid, given
a sensing range, the sensors have to decide which action to take to capture
the target in order to maximize the cummulative rewards over time.
- Actions: Box(0, 1, (3,)) for continuous actions, namely {Turn Left, Turn Right, Stay}
- States: Box(-inf, inf, (n_targets*2,)) --flattened bc networks require it to be
- Reward: Number of unique tracked target per step
- Terminate condition: When max_steps reached
"""

from pettingzoo import ParallelEnv
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from helper.utilities import evenly_distribute_sensors
from gymnasium.spaces import Box, Discrete
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle


class TargetTrackingEnv(ParallelEnv):
    metadata = {
        "name": "target_tracking_v0",
        "render_modes": ["human"]
    }

    def __init__(self, n_sensors=5, m_targets=6, area_size=3.0, sr_max=18.0/20.0, alpha_max=np.pi/3,
                 max_theta_delta=np.pi/36, max_steps=40, continuous_actions=False, render_mode=None):
        """
        Implements a multi-agent environment where fixed directional sensors track randomly moving targets.
        """
        self.n_sensors = n_sensors                      # number of sensors
        self.m_targets = m_targets                      # number of targets to keep track of
        self.area_size = area_size                      # grid size
        self.sr_max = sr_max                            # max sensing range
        self.alpha_max = alpha_max                      # max degree of sensing
        self.max_theta_delta = max_theta_delta          # sensor stepping angle
        self.max_steps = max_steps                      # max steps to terminate episode
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.best_case = n_sensors * m_targets * max_steps # best score per episode i.e covers all targets per time step
        self.possible_agents = [f"sensor_{i}" for i in range(n_sensors)]
        self.agents = self.possible_agents.copy()

        # Observation: (m_targets, 2) for coordinate x and y -> (m_targets * 2)
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

        # Fixed sensor positions (fixed per episode)
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
                # act_cat_i = Categorical(F.softmax(tensor_act_i))
                act_i = torch.argmax(F.softmax(tensor_act_i))
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

        # Reward: number of unique tracked targets per step
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
                if rho > self.sr_max:
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
            wedge = Wedge((s_x, s_y), self.sr_max, s_dir_deg - alpha_deg/2, s_dir_deg + alpha_deg/2,
                          width=self.sr_max, fill=True, alpha=0.3, color='blue')
            circle = Circle((s_x, s_y), self.sr_max, edgecolor='lightblue', facecolor='none',
                            linestyle='--', linewidth=1)
            self.ax.add_patch(wedge)
            self.ax.add_patch(circle)

            # Direction line
            dx = self.sr_max * np.cos(self.sensor_dir[i])
            dy = self.sr_max * np.sin(self.sensor_dir[i])
            self.ax.plot([s_x, s_x + dx], [s_y, s_y + dy], 'b--')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None