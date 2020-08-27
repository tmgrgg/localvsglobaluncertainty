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
        self.targets = torch.tensor(self.data_loader.dataset.targets).long()
        self.predictions = torch.zeros(size=(self.num_datapoints, self.num_classes))
        self.weights = []
        self.model_count = 0

    # TODO: handle non-equal weights?
    @torch.no_grad()
    def add_model(self, model, activation=lambda output: F.softmax(output, dim=1), using_cuda=True):
        model.eval()
        predictions = torch.zeros(size=(self.num_datapoints, self.num_classes))
        start_idx = 0
        for i, (input, target) in enumerate(self.data_loader):
            batch_size = input.size()[0]
            end_idx = start_idx + batch_size

            if using_cuda:
                input = input.cuda(non_blocking=True)

            output = model(input).cpu()
            predictions[start_idx:end_idx] = output
            start_idx = end_idx
        self.add_predictions(predictions, activation=activation)

    @torch.no_grad()
    def add_predictions(self, predictions, activation=lambda x: x):
      numer = (self.model_count * self.predictions) + activation(predictions)
      self.predictions =  numer / (self.model_count + 1)
      self.model_count += 1

    def evaluate(self, f, pre_f=torch.log):
        return f(pre_f(self.predictions), self.targets)
