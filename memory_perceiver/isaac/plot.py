import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def two_levels_plot(data, ax, fig, horizon, path):
    ax.clear()
    recent_data=data[-horizon:]
    ax.plot(recent_data, alpha=0.5)
    ax.plot((np.convolve(data, np.ones(horizon), 'valid') / horizon)[::max(1,int(len(data)/horizon))])
    # ax.set_ylim(bottom=np.min(recent_data), top=np.max(recent_data))
    fig.savefig(path)

class Plotter:
    def __init__(self, plot_type, path_plots, titles=[]):
        self.plot_type = plot_type
        self.path_plots = path_plots
        if plot_type == 'matplotlib':
            self.titles=titles
            self.figs=[]
            self.axes=[]
            self.data={}
            for title in titles:
                fig, axe = plt.subplots(figsize=(8, 6))
                self.figs.append(fig)
                self.axes.append(axe)
                self.data[title] = []
        elif plot_type == 'tensorboard':
            self.writer = SummaryWriter(path_plots)

    def add(self, key, value, step=0):
        if self.plot_type == 'matplotlib':
            self.data[key].append(value)
        elif self.plot_type == 'tensorboard':
            self.writer.add_scalar(key, value, step)

        
    def plot_all(self, series):
        if self.plot_type == 'matplotlib':
            for i, element in enumerate(series):
                if isinstance(element, str):
                    title = element
                    if len(self.data[title]) > 0:
                        self.axes[i].plot(self.data[title])
                        self.figs[i].savefig(self.path_plots / 'plt_' / (self.titles[i]+ '.png') )

                if isinstance(element, tuple) and len(element)==2:
                    title, horizon = element
                    if len(self.data[title]) > 0:
                        two_levels_plot(self.data[title], self.axes[i], self.figs[i], horizon=horizon, 
                                        path=self.path_plots / 'plt_' / (self.titles[i]+ '.png') )


