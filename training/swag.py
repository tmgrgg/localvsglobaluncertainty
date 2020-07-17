import torch
from collections import defaultdict


# make this a torch.nn.Optimizer?
class SWAGSampler:

    def __init__(self, optimizer, rank, sampling_condtn=lambda: True, sample_freq=300):
        self.rank = rank
        self.optimizer = optimizer
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


class SWAGPosterior(torch.nn.Module):

    def __init__(self, model, swag, var_clamp=1e-30):
        super(SWAGPosterior, self).__init__()
        self.model = model
        # each parameter in model.parameters() must map to a sq_mean, mean, and deviations
        # otherwise SWAG will break!
        self.state = swag.state
        self.var_clamp = var_clamp
        self._infer_sigmas()

    def _infer_sigmas(self):
        list_mean = []
        list_sq_mean = []
        list_deviations = []
        for p in self.model.parameters():
            state = self.state[p]
            if not all(x in state.keys() for x in ['mean', 'sq_mean', 'deviations']):
                raise RuntimeError('SWAGPosterior is missing required state.')

            # here's the part where we assume that model.parameters() doesn't decide to
            # change order under the hood...
            list_mean.append(state['mean'].view(-1))
            list_sq_mean.append(state['sq_mean'].view(-1))
            list_deviations.append(state['deviations'].view(state['deviations'].shape[0], -1))

        mean = torch.cat(list_mean).cpu()
        sq_mean = torch.cat(list_sq_mean).cpu()
        deviations = torch.cat(list_deviations, dim=1).cpu()

        self.mean = mean
        self.sigma_diag = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        self.sigma_low_rank = deviations.t()

    def sample(self, scale=0.5, diagonal_only=False):
        z1 = torch.randn_like(self.sigma_diag, requires_grad=False)
        diag_term = self.sigma_diag.sqrt() * z1

        if not diagonal_only:
            rank = self.sigma_low_rank.shape[1]
            z2 = torch.randn(rank, requires_grad=False)
            low_rank_term = self.sigma_low_rank.mv(z2)
            low_rank_term /= (rank - 1) ** 0.5
        else:
            low_rank_term = 0.0

        sample = self.mean + (diag_term + low_rank_term) / (scale ** 0.5)
        self._set_params(sample)

    def _set_params(self, sample):
        i = 0
        for p in self.model.parameters():
            shape = p.data.shape
            n = p.data.numel()
            p.data = sample[i: i + n].view(shape).to(p.device)
            i += n

    def mean(self):
        self._set_params(self.mean)
        
    def forward(self, *input):
        return self.model(*input)
