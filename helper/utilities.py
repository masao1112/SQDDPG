import matplotlib.pyplot as plt
import numpy as np
import math
import os

def plot_rewards(rewards, out_path, window=100):
    # n_eps = len(rewards)
    # sum_rewards = np.zeros(n_eps)
    # for t in range(n_eps):
    #     sum_rewards[t] = np.sum(rewards[max(0, t-window):(t+1)])
    
    plt.plot(rewards, "b-")
    plt.xlabel('episode')
    plt.ylabel('Mean Reward Episode')
    plt.legend()
    plt.savefig(out_path)
    
def _update_target_networks(target, source, tau):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(tau * sparam.data + (1.0 - tau) * tparam.data)
        
def get_dict_value(map):
    return list(map.values())

# Custom Env utilities
def evenly_distribute_sensors(n_sensors, area):
    """
    Evenly distributes n_sensors positions in a 2D area [width, height].

    Parameters:
    - n_sensors (int): Number of sensors to place.
    - area (list or np.array): [width, height] of the area.

    Returns:
    - np.array of shape (n_sensors, 2): Positions [x, y].

    Notes:
    - Places sensors in a grid-like pattern for even distribution.
    - rho_max could be used to check/enforce max spacing, e.g.:
      if x_step > 2 * rho_max or y_step > 2 * rho_max:
          print("Warning: Spacing may cause coverage gaps.")
    - For hexagonal or other patterns, extend as needed.
    """
    width, height = area
    # Compute grid dimensions to approximate square (balanced distribution)
    cols = math.ceil(math.sqrt(n_sensors))
    rows = math.ceil(n_sensors / cols)

    # Spacing with margins (positions centered in cells)
    x_step = width / (cols + 1)
    y_step = height / (rows + 1)

    # Optional: Check if spacing exceeds coverage (assuming circular for simplicity)
    # if x_step > 2 * rho_max or y_step > 2 * rho_max:
    #     # Could adjust rows/cols to densify, but would change effective n_sensors usage
    #     pass

    positions = []
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= n_sensors:
                break
            x = (col + 1) * x_step
            y = (row + 1) * y_step
            positions.append([x, y])

    return np.array(positions)