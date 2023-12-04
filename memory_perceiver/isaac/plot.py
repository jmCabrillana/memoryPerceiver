import matplotlib.pyplot as plt
import numpy as np

def two_levels_plot(data, ax, fig, horizon, path):
    ax.clear()
    recent_data=data[-horizon:]
    ax.plot(recent_data, alpha=0.5)
    ax.plot((np.convolve(data, np.ones(horizon), 'valid') / horizon)[::max(1,int(len(data)/horizon))])
    # ax.set_ylim(bottom=np.min(recent_data), top=np.max(recent_data))
    fig.savefig(path)

"""
self.fig_loss, ax_loss
fig_loss_critic, ax_loss_critic = plt.subplots(figsize=(8, 6))
fig_rewards, ax_rewards = plt.subplots(figsize=(8, 6))
"""
class Plotter:
    def __init__(self):
        self.fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
        self.path_plots = '/isaac-sim/memoryPerceiver/memory_perceiver/main/'

