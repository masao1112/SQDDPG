# notebook-python
from mpe2 import simple_tag_v3
import numpy as np

# Hyperparameters
PRINT_INTERVAL = 100
N_GAMES = 10000
MAX_STEPS = 25  # default in simple_adversary_v3 is 25
total_steps = 0
score_history = []
avg_score_history = []

# Create environment
env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=MAX_STEPS,
        continuous_actions=True,
        dynamic_rescaling=True,
        render_mode="human"
)
env.reset(seed=42)

print(f"Agents: {env.agents}")
print(f"Action spaces: { {agent: env.action_space(agent) for agent in env.agents} }")
print(f"Observation spaces: { {agent: env.observation_space(agent).shape for agent in env.agents} }")

for episode in range(N_GAMES):
    observations, infos = env.reset()
    episode_reward = 0
    done = False

    for step in range(MAX_STEPS):
        # Sample random actions from each agent's action space
        actions = {
            agent: env.action_space(agent).sample() 
            for agent in env.agents
        }

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Sum rewards across all agents (common in coop/competitive settings)
        episode_reward += sum(rewards.values())

        # Check if episode is done (any termination or truncation)
        done = any(terminations.values()) or any(truncations.values())
        if done:
            break

    score_history.append(episode_reward)
    avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
    avg_score_history.append(avg_score)

    if episode % PRINT_INTERVAL == 0 and episode > 0:
        print(f"Episode {episode} | Avg Score (last 100): {avg_score:.2f} | Episode Reward: {episode_reward:.2f}")

env.close()