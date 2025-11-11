import time
import numpy as np
from maddpg import MADDPG
from memory_buffer import MultiAgentReplayBuffer
from utilities import *
from mpe2 import simple_spread_v3  # or simple_adversary_v3, simple_spread_v3


if __name__ == '__main__':
    
    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 40
    total_steps = 0
    score_history = []
    avg_score_history = []
    evaluate = False
    best_score = -100
    batch_size = 128
    # initiate environment
    env = simple_spread_v3.parallel_env(
        N=3, 
        max_cycles=MAX_STEPS, continuous_actions=True,
        dynamic_rescaling=True, #render_mode="human"
    )
    env.reset()

    n_agents = len(env.agents) # -1 for redundancy
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

    # action space is a list of arrays, assume each agent has same action space
    maddpg_agents = MADDPG(critic_dims, actor_dims, n_agents, n_actions, 
                           fc1=128, fc2=128,  
                           alpha=1e-4, beta=1e-3, gamma=0.99, tau=0.001,
                           chkpt_dir='tmp/maddpg',
                           evaluate=evaluate)

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size)


    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs_dict, _ = env.reset()
        obs = get_dict_value(obs_dict)
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            # landmarks_pos = [lm.state.p_pos for lm in env.unwrapped.world.landmarks]
            # print(f"Episode {i}, Step {episode_step}: Landmark positions: {landmarks_pos}")
            if evaluate:
                env.render()
                time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            # convert to dict bc env requires it
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
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        # keep track of stats
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
        
    plot_rewards(avg_score_history, "mean_maddpg_rewards_eval.png")
    plot_rewards(score_history, "original_maddpg_rewards_eval.png")
