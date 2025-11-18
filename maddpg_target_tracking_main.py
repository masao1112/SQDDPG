import time
import numpy as np
from models.maddpg import MADDPG
from helper.memory_buffer import MultiAgentReplayBuffer
from helper.utilities import *
from custom_environment.env.WSN import TargetTrackingEnv

if __name__ == '__main__':

    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 50
    total_steps = 0
    score_history = []
    avg_score_history = []
    cov_rate_history = []
    avg_cov_rate_history = []
    evaluate = False
    best_score = -100
    batch_size = 128

    env = TargetTrackingEnv(
        n_sensors=5, m_targets=6, area_size=3.0, sr_max=18.0 / 20.0,
        continuous_actions=True,
        # render_mode="human",
        max_steps=MAX_STEPS
    )
    env.reset()
    best_case = env.best_case
    n_agents = len(env.agents)
    actor_dims = []
    action_dims = []
    for agent in env.possible_agents:
        obs_dim = env.observation_space(agent).shape[0]
        actor_dims.append(obs_dim)
        action_dims.append(env.action_space(agent).shape[0])

    critic_dims = sum(actor_dims)
    n_actions = action_dims[0]  # assume all continuous actions same dim

    print(f"\nEnvironment: simple_target_tracking")
    print(f"Number of agents: {n_agents}")
    print(f"Actor dims: {actor_dims}")
    print(f"Critic dims: {critic_dims}, n_actions: {n_actions}\n")

    maddpg_agents = MADDPG(critic_dims, actor_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=1e-3, beta=2e-3, gamma=0.99, tau=0.001,
                           chkpt_dir='tmp/maddpg/target_tracking/scene1',
                           evaluate=evaluate)

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, n_actions, n_agents, batch_size)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for episode in range(N_GAMES):
        obs_dict, _ = env.reset()
        obs = get_dict_value(obs_dict)
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                time.sleep(0.1)  # to slow down the action for the video
            # noise_std = 0.2 * (1 - episode / N_GAMES)
            actions = maddpg_agents.choose_action(obs)
            # perform rescaling as package required
            action_dict = {
                agent: np.array(actions[idx], dtype=np.float32)
                for idx, agent in enumerate(env.agents)
            }
            obs_, reward, done, info, _ = env.step(action_dict)
            obs_ = get_dict_value(obs_)
            reward = get_dict_value(reward)
            done = get_dict_value(done)
            info = get_dict_value(info)

            if episode_step >= MAX_STEPS:
                done = [True] * n_agents

            memory.store_transition(obs, actions, reward, obs_, done)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        # stats tracking
        cov_rate = score / best_case
        # cov_rate_percentage = round(cov_rate * 10000) / 100
        avg_score = np.mean(score_history[-100:])
        avg_cov_rate = np.mean(cov_rate_history[:-100])

        score_history.append(score)
        avg_score_history.append(avg_score)
        cov_rate_history.append(cov_rate)
        avg_cov_rate_history.append(avg_cov_rate)

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if episode % PRINT_INTERVAL == 0 and episode > 0:
            print('episode', episode, 'average score {:.1f}'.format(avg_score), "avg_cover_rate {:.2f}".format(avg_cov_rate))

    plot_rewards(avg_score_history, "mean_maddpg_target_sensor_rewards.png")
    plot_rewards(score_history, "original_maddpg_target_sensor_rewards.png")
    plot_rewards(avg_cov_rate_history, "mean_maddpg_target_sensor_cov_rate.png")
    plot_rewards(cov_rate_history, "original_maddpg_target_sensor_cov_rate.png")