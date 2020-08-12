import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np


class Ensembler:

    def __init__(self, data_loader):
        # do not want data_loader to shuffle!
        self.data_loader = DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, shuffle=False)
        self.data_loader = data_loader
        self.num_datapoints = len(data_loader.dataset)
        self.num_classes = len(np.unique(data_loader.dataset.targets))
        self.targets = torch.zeros(size=(self.num_datapoints,)).long()
        self.predictions = torch.zeros(size=(self.num_datapoints, self.num_classes))
        self.weights = []
        self.model_count = 0

    # TODO: handle non-equal weights?
    @torch.no_grad()
    def add_model(self, model, eval=True, using_cuda=True):
        model.eval()

        start_idx = 0
        for i, (input, target) in enumerate(self.data_loader):
            batch_size = input.size()[0]
            end_idx = start_idx + batch_size

            if using_cuda:
                input = input.cuda(non_blocking=True)

            output = model(input).cpu()
            self.predictions[start_idx:end_idx] = (self.model_count * self.predictions[start_idx:end_idx] + output) / (self.model_count + 1)
            if self.model_count == 0:
                self.targets[start_idx:end_idx] = target

            start_idx = end_idx

        self.model_count += 1

    def evaluate(self, f):
        return f(self.predictions, self.targets)
