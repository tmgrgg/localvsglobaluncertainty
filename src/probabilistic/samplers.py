import torch
from collections import defaultdict


class SWAGSampler:

    def __init__(
        self,
        posterior,
        optimization,
        sample_freq=300,
        sampling_condtn = lambda: True,
        *args,
        **kwargs
    ):
        self._posterior = posterior
        self.rank = posterior.rank
        self.named_params = dict(posterior.model.named_parameters())
        self.sample_freq = sample_freq
        self.optimizer = optimization(self.named_params.values(), *args, **kwargs)
        self.stats = defaultdict(dict)
        self.sampling_condtn = sampling_condtn
        self._counter = 0
        self.__setup__()

    def __setup__(self):
        for name, param in self.named_params.items():
            stats = self.stats[name]
            assert (len(stats) == 0)

            # SWAG statistics, (page 5 of
            # "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
            # https://arxiv.org/abs/1902.02476)
            stats['num_swag_steps'] = 0
            stats['mean'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)
            stats['sq_mean'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)
            stats['deviations'] = param.data.new_empty((0,) + param.data.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step, and an update to swag_parameters (with given sample_freq)
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
                for name, param in self.named_params.items():
                    stats = self.stats[name]
                    mean = stats['mean']
                    sq_mean = stats['sq_mean']
                    deviations = stats['deviations']
                    num_swag_steps = stats['num_swag_steps']

                    # calculate
                    mean = (mean * num_swag_steps + param.data) / (num_swag_steps + 1)
                    sq_mean = (sq_mean * num_swag_steps + param.data ** 2) / (num_swag_steps + 1)

                    dev = (param.data - mean).unsqueeze(0)
                    deviations = torch.cat([deviations, dev], dim=0)
                    if deviations.shape[0] > self.rank:
                        deviations = deviations[1:, ...]

                    # update
                    stats['mean'] = mean
                    stats['sq_mean'] = sq_mean
                    stats['num_swag_steps'] = num_swag_steps + 1
                    stats['deviations'] = deviations
        return loss

    def collect(self):
        self._posterior.infer(self.stats)

    def zero_grad(self):
        self.optimizer.zero_grad()
