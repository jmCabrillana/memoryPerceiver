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
    def __init__(self, path_plots='/isaac-sim/memoryPerceiver/memory_perceiver/main/', titles=[]):
        self.path_plots = path_plots
        self.title=titles
        self.figs=[]
        self.axes=[]
        for i in range(len(titles)):
            self.figs[i], self.axes[i] = plt.subplots(figsize=(8, 6))
        # fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
        # fig_loss_critic, ax_loss_critic = plt.subplots(figsize=(8, 6))
        # fig_rewards, ax_rewards = plt.subplots(figsize=(8, 6))

        
    def plot_all(self, series):
        for i, element in enumerate(series):
            if isinstance(element, list):
                self.axes[i].plot(element)
                self.figs[i].savefig(self.path_plots + self.title[i]+ '.png')

            if isinstance(element, tuple) and len(element)==2:
                serie, horizon = element
                two_levels_plot(serie, self.axes[i], self.figs[i], horizon=horizon, 
                                path=self.path_plots + self.title[i]+ '.png')

        # two_levels_plot(all_losses, ax_loss, fig_loss, horizon=160, path=self.path_plots+'loss.png')
        # two_levels_plot(all_losses_critic, ax_loss_critic, fig_loss_critic, horizon=160, path=self.path_plots+'loss_critic.png')
        # ax_rewards.plot(all_returns)
        # fig_rewards.savefig(self.path_plots+'returns.png')


