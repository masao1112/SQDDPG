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

    # Find minimal k such that number of checkerboard positions >= n_sensors
    k = 1
    while True:
        if k % 2 == 0:
            num_checker = k ** 2 // 2
        else:
            num_checker = (k ** 2 + 1) // 2
        if num_checker >= n_sensors:
            break
        k += 1

    # Spacing with margins (positions centered in cells)
    x_step = width / (k + 1)
    y_step = height / (k + 1)

    # Optional: Check if spacing exceeds coverage (assuming circular for simplicity)
    # if x_step > 2 * rho_max or y_step > 2 * rho_max:
    #     # Could adjust k to densify, but would change the pattern
    #     pass

    # Collect checkerboard positions (row + col even)
    checker_positions = []
    for row in range(k):
        for col in range(k):
            if (row + col) % 2 == 0:
                x = (col + 1) * x_step
                y = (row + 1) * y_step
                checker_positions.append([x, y])

    # Sort by row-major order (though already in order, but to confirm)
    checker_positions.sort(key=lambda p: (p[1] / y_step - 1) * k + (p[0] / x_step - 1))

    # Take first n_sensors
    positions = checker_positions[:n_sensors]

    return np.array(positions)