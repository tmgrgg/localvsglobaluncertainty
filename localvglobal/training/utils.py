import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from PIL import Image
import numpy as np
import torch
import random

from tqdm import tqdm


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
        return fig, ax

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

    def get_image(self):
        fig, _ = self._make_plot()
        fig.canvas.draw()
        return Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


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


def seed(val=1234):
    torch.backends.cudnn.deterministic = True
    np.random.seed(val)
    random.seed(val)
    torch.manual_seed(val)


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def accuracy(outputs, targets):
    num_datapoints = targets.size(0)
    preds = outputs.data.argmax(1, keepdim=True)
    correct = preds.eq(targets.data.view_as(preds)).sum().item()
    return  100.0 * correct / num_datapoints

