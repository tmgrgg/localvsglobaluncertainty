import torch
from collections import defaultdict
from abc import ABC, abstractmethod


class GradientDescentIterateSampler(ABC):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def step(self, closure=None):
        pass


class SWAGSampler(GradientDescentIterateSampler):

    def __init__(self, optimizer, rank, sampling_condtn=lambda: True, sample_freq=300):
        super(SWAGSampler, self).__init__(optimizer)
        self.rank = rank
        self.sampling_condtn = sampling_condtn
        self.sample_freq = sample_freq
        self.state = defaultdict(dict)
        self._counter = 0
        self.__setup__()

    def __setup__(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.state[p]
                assert (len(state) == 0)

                # SWAG statistics, (page 5 of
                # "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
                # https://arxiv.org/abs/1902.02476)
                state['num_swag_steps'] = 0
                state['mean'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state['sq_mean'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state['deviations'] = p.data.new_empty((0,) + p.data.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step, and an update to swag_parameters if sampling_condition is met
            with the given sample_frequency
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # perform optimization step
        loss = self.optimizer.step(closure=closure)

        if self.sampling_condtn():
            self._counter += 1
            if (self._counter % self.sample_freq) == 0:
                self._counter = 0

                # use parameter values to update running SWAG statistics
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.state[p]
                        mean = state['mean']
                        sq_mean = state['sq_mean']
                        deviations = state['deviations']
                        num_swag_steps = state['num_swag_steps']

                        # calculate
                        mean = (mean * num_swag_steps + p.data) / (num_swag_steps + 1)
                        sq_mean = (sq_mean * num_swag_steps + p.data ** 2) / (num_swag_steps + 1)

                        dev = (p.data - mean).unsqueeze(0)
                        deviations = torch.cat([deviations, dev], dim=0)
                        if deviations.shape[0] > self.rank:
                            deviations = deviations[1:, ...]

                        # update
                        state['mean'] = mean
                        state['sq_mean'] = sq_mean
                        state['num_swag_steps'] = num_swag_steps + 1
                        state['deviations'] = deviations
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()
