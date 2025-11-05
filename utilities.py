import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, out_path, window=100):
    n_eps = len(rewards)
    sum_rewards = np.zeros(n_eps)
    for t in range(n_eps):
        sum_rewards[t] = np.sum(rewards[max(0, t-window):(t+1)])
    
    plt.plot(sum_rewards, "b-")
    plt.xlabel('episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(out_path)
    
def _update_target_networks(target, source, tau):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(tau * sparam.data + (1.0 - tau) * tparam.data)
        
def get_dict_value(map):
    return list(map.values())

def rescale_action(action):
    return (action + 1) / 2.0