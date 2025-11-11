import time
import numpy as np
from sqddpg import SQDDPG
from memory_buffer import MultiAgentReplayBuffer
from utilities import *
# from mpe2 import simple_spread_v3  # or simple_adversary_v3, simple_spread_v3
from custom_environment.env.WSN import TargetTrackingEnv


if __name__ == '__main__':
    
    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    avg_score_history = []
    evaluate = False
    best_score = -100
    batch_size = 128
    sample_size = 6
    
    # env = simple_adversary_v3.parallel_env(
    #     N=3,                 # total number of agents (1 adversary + 2 good)
    #     max_cycles=MAX_STEPS,
    #     continuous_actions=True ,  # use continuous control for MADDPG
    #     # render_mode="human"
    # )   
    env = TargetTrackingEnv(
        continuous_actions=True,
        render_mode="human"
    )
    env.reset()
    n_agents = len(env.agents) 
    actor_dims = []
    action_dims = []
    for agent in env.possible_agents:
        obs_dim = env.observation_space(agent).shape[0]
        actor_dims.append(obs_dim)
        action_dims.append(env.action_space(agent).shape[0])

    critic_dims = sum(actor_dims)
    n_actions = action_dims[0]  # assume all continuous actions same dim

    print(f"\nEnvironment: simple_adversary_v3")
    print(f"Number of agents: {n_agents}")
    print(f"Actor dims: {actor_dims}")
    print(f"Critic dims: {critic_dims}, n_actions: {n_actions}\n")

    sqddpg_agents = SQDDPG(critic_dims, actor_dims, n_agents, n_actions, 
                           batch_size=batch_size, sample_size=sample_size,
                           fc1=128, fc2=128,  
                           alpha=1e-4, beta=1e-3, gamma=0.99, tau=0.001,
                           chkpt_dir='tmp/sqddpg/',
                           evaluate=evaluate)

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size)

    if evaluate:
        sqddpg_agents.load_checkpoint()

    for episode in range(N_GAMES):
        obs_dict, _ = env.reset()
        obs = get_dict_value(obs_dict)
        score = 0       
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                time.sleep(0.1) # to slow down the action for the video
            noise_std = 0.2 * (1 - episode / N_GAMES)
            actions = sqddpg_agents.choose_action(obs, noise_std)
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
                done = [True]*n_agents

            memory.store_transition(obs, actions, reward, obs_, done)
            
            if total_steps % 10 == 0 and not evaluate:
                sqddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1
            
        print(score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        
        if not evaluate:
            if avg_score > best_score:
                sqddpg_agents.save_checkpoint()
                best_score = avg_score
        if episode % PRINT_INTERVAL == 0 and episode > 0:
            print('episode', episode, 'average score {:.1f}'.format(avg_score))

    plot_rewards(avg_score_history, "mean_sqddpg_rewards.png")
    plot_rewards(score_history, "original_sqddpg_rewards.png")