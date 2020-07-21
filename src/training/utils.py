from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


class TrainingTracker:

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


# currently only works for classification criteria
def run_training_epoch(
    data_loader,
    model,
    criterion,
    optimizer,
    train=True,
    using_cuda=True,
    predict=lambda output: output.data.argmax(1, keepdim=True),
):
    # qois (quantities of interest)
    loss_sum = 0.0
    correct = 0.0
    example_count = 0

    if train:
        model.train()
    else:
        model.eval()

    for i, (input, target) in enumerate(data_loader):
        if using_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        if train and optimizer is not None:
            # optimise loss for input
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update qois
        loss_sum += loss.data.item() * input.size(0)
        preds = predict(output)
        correct += preds.eq(target.data.view_as(preds)).sum().item()
        example_count += input.size(0)

    return {
        "loss": loss_sum / example_count,
        "accuracy": (correct / example_count) * 100.0,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
