from pettingzoo.mpe import simple_adversary_v3
import numpy as np

score_history = []
avg_score_history = []
eps = 2000
PRINT_INTERVAL = 100
MAX_STEPS = 75

env = simple_adversary_v3.parallel_env(
    N=3,  # 1 adversary + 4 good agents? Wait — see note below
    max_cycles=MAX_STEPS,
    continuous_actions=True,
    # render_mode="human"
)
env.reset()

n_agents = len(env.agents)
print(f"Agents: {env.agents}")  # Debug: check agent names

for i in range(eps):
    observations, infos = env.reset()
    episode_step = 0
    score = 0
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not all(terminated.values()) and not all(truncated.values()):
        # Sample random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminated, truncated, infos = env.step(actions)

        score += sum(rewards.values())
        episode_step += 1

        # Optional: enforce max steps (though max_cycles should handle this)
        if episode_step >= MAX_STEPS:
            truncated = {agent: True for agent in env.agents}

    score_history.append(score)
    avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
    avg_score_history.append(avg_score)

    if i % PRINT_INTERVAL == 0 and i > 0:
        print('episode', i, 'average score {:.1f}'.format(avg_score))

env.close()