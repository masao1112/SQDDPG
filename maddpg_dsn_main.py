import time
import numpy as np
import pandas as pd
from models.maddpg import MADDPG
from helper.memory_buffer import MultiAgentReplayBuffer
from helper.utilities import *
from custom_environment.env.pose_env_base import Pose_Env_Base  # Assuming PoseEnv is in the same location or adjust path


if __name__ == '__main__':

    PRINT_INTERVAL = 100
    N_GAMES = 25000
    MAX_STEPS = 100  # Matches PoseEnv default max_steps
    total_steps = 0
    score_history = []
    avg_score_history = []
    evaluate = False
    best_score = -100
    batch_size = 128

    env = Pose_Env_Base(
        render=False,
        render_save=False,
        continuous_action=True
    )
    state = env.reset()
    n_agents = env.n
    n_actions = env.action_space[0].shape[0]
    critic_dims = state.shape[0] * state.shape[1] # total critic dims
    actor_dims = [state.shape[1]] * n_agents # each actor's dim

    print(f"\nEnvironment: MADDPG(pose_env)")
    print(f"Number of agents: {n_agents}")
    print(f"Actor dims: {actor_dims}")
    print(f"Critic dims: {critic_dims}, n_actions: {n_actions}\n")

    maddpg_agents = MADDPG(critic_dims, actor_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=5e-5, beta=5e-5, gamma=0.9, tau=0.1,
                           chkpt_dir='tmp/maddpg/dsn',
                           evaluate=evaluate)

    memory = MultiAgentReplayBuffer(10000, critic_dims, actor_dims, n_actions, n_agents, batch_size)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for episode in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                time.sleep(0.1)  # to slow down the action for the video
            noise_std = 0.2 * (1 - episode / N_GAMES)
            actions = maddpg_agents.choose_action(obs, noise_std)
            # perform rescaling as package required
            obs_, rewards, coverage_rate, dones, _ = env.step(actions)

            if episode_step >= MAX_STEPS:
                done = [True] * n_agents

            memory.store_transition(obs, actions, rewards, obs_, done)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            score += coverage_rate  # Use single reward since shared
            total_steps += 1
            episode_step += 1

        # stats tracking
        avg_score = np.mean(score_history[-100:])
        score_history.append(score)
        avg_score_history.append(avg_score)
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        print(score)
        if episode % PRINT_INTERVAL == 0 and episode > 0:
            print('episode', episode, 'average score {:.1f}'.format(avg_score))

    # save output
    np.savetxt("maddpg_dsn_rewards.txt2", score_history)
    plot_rewards(avg_score_history, "mean_maddpg_pose_rewards.png")
    plot_rewards(score_history, "original_maddpg_pose_rewards.png")