import matplotlib.pyplot as plt
import numpy as np

def two_levels_plot(data, ax, fig, horizon, path):
    ax.clear()
    recent_data=data[-horizon:]
    ax.plot(recent_data, alpha=0.5)
    ax.plot((np.convolve(data, np.ones(horizon), 'valid') / horizon)[::max(1,int(len(data)/horizon))])
    # ax.set_ylim(bottom=np.min(recent_data), top=np.max(recent_data))
    fig.savefig(path)

class Plotter:
    def __init__(self, path_plots='/isaac-sim/memoryPerceiver/memory_perceiver/main/plots/', titles=[]):
        self.path_plots = path_plots
        self.titles=titles
        self.figs=[]
        self.axes=[]
        self.data={}
        for title in titles:
            fig, axe = plt.subplots(figsize=(8, 6))
            self.figs.append(fig)
            self.axes.append(axe)
            self.data[title] = []

        
    def plot_all(self, series):
        for i, element in enumerate(series):
            print(self.titles, self.axes, self.figs)
            if isinstance(element, list):
                self.axes[i].plot(element)
                self.figs[i].savefig(self.path_plots + self.titles[i]+ '.png')

            if isinstance(element, tuple) and len(element)==2:
                serie, horizon = element
                two_levels_plot(serie, self.axes[i], self.figs[i], horizon=horizon, 
                                path=self.path_plots + self.titles[i]+ '.png')


