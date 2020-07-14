from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


class LoopStatsTracker():
    def __init__(self, plot_freq=5):
        self.counter = 0
        self.plot_freq = plot_freq
        self._metrics = OrderedDict({})

    def _make_plot(self):
        fig, ax = plt.subplots(1, len(self._metrics.keys()), squeeze=False, figsize=(18, 4))
        for i, metric in enumerate(self._metrics.keys()):
            train_metric = self._metrics[metric]['train']
            valid_metric = self._metrics[metric]['valid']
            ax[0, i].plot(list(range(len(train_metric))), train_metric, c='b', label='Train', marker='.')
            ax[0, i].plot(list(range(len(valid_metric))), valid_metric, c='r', label='Valid', marker='.')
            ax[0, i].set_ylabel(metric)
            ax[0, i].set_xlabel('epochs')
            ax[0, i].legend(loc='upper right')

    def plot(self):
        # Will plot the current loss_graph every plot_freq^th call to plot
        self.counter += 1
        print(self.counter)
        print(self.counter % self.plot_freq)
        if self.counter % self.plot_freq == 0:
            self._make_plot()
            clear_output()
            plt.show()

    def save(self, path):
        # Saves the current loss_graph to the given path
        self._make_plot()
        plt.savefig(path)
        plt.close()

        def log(self, value, metric, setting='train'):
            assert (setting == 'train' or setting == 'valid')
            if metric not in self._metrics.keys():
                self._metrics[metric] = {'train': [], 'valid': []}
            self._metrics[metric][setting].append(value)


class Timer:

    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = 0

    def elapsed_seconds(self):
        return time.time() - self.start_time

    def elapsed_minutes_seconds(self):
        elapsed_time = self.elapsed_seconds()
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
