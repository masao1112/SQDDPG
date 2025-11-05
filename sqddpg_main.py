import numpy as np
from sqddpg import SQDDPG
from memory_buffer import MultiAgentReplayBuffer
from utilities import *
from pettingzoo.mpe import simple_spread_v3  # or simple_adversary_v3, simple_spread_v3



if __name__ == '__main__':
    
    PRINT_INTERVAL = 10
    N_GAMES = 1500
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    avg_score_history = []
    evaluate = False
    best_score = 0
    batch_size = 32
    sample_size = 5
    
    env = simple_spread_v3.parallel_env(
            N=3,                 # total number of agents (1 adversary + 2 good)
        max_cycles=25,
        continuous_actions=True   # use continuous control for MADDPG
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

    maddpg_agents = SQDDPG(critic_dims, actor_dims, n_agents, n_actions, 
                           batch_size=batch_size, sample_size=sample_size,
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000, critic_dims, actor_dims, n_actions, n_agents, batch_size)


    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs_dict, _ = env.reset()
        obs = get_dict_value(obs_dict)
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            # perform rescaling as package required
            action_dict = {
                agent: np.clip(rescale_action(actions[idx]), 0.0, 1.0).astype(np.float32)
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

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        
        if not evaluate:
            if avg_score > best_score:
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

    plot_rewards(avg_score_history, "sqddpg_rewards.png")
