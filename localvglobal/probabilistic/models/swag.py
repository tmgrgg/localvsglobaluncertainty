import torch
from collections import defaultdict
from localvglobal.probabilistic.models import ProbabilisticModule
from localvglobal.training.utils import bn_update


class SWAGPosterior(ProbabilisticModule):

    def __init__(self, model, rank=30, var_clamp=1e-30):
        super().__init__(model)
        num_params = sum(p.numel() for p in model.parameters())
        self.var_clamp = var_clamp
        self.rank = rank
        self.register_buffer('mean', torch.zeros(size=(num_params,)))
        self.register_buffer('sigma_diag', torch.zeros(size=(num_params,)))
        self.register_buffer('sigma_low_rank', torch.zeros(size=(num_params, self.rank)))

    def infer(self, swag_stats, strict=False):
        list_mean = []
        list_sq_mean = []
        list_deviations = []
        for name, param in self.model.named_parameters():
            stats = swag_stats[name]
            if not all(x in stats.keys() for x in ['mean', 'sq_mean', 'deviations']):
                raise RuntimeError('SWAGPosterior is missing required statistics.')

            # here's the part where we assume that model.parameters() doesn't decide to
            # change order under the hood...
            list_mean.append(stats['mean'].view(-1))
            list_sq_mean.append(stats['sq_mean'].view(-1))
            list_deviations.append(stats['deviations'].view(stats['deviations'].shape[0], -1))

        mean = torch.cat(list_mean)
        sq_mean = torch.cat(list_sq_mean)
        deviations = torch.cat(list_deviations, dim=1)

        self.mean = mean
        self.sigma_diag = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        if self.sigma_low_rank.size() == deviations.t().size():
            self.sigma_low_rank = deviations.t()
        elif strict:
            raise RuntimeError('Only {} deviation samples in call to .infer for SWAG posterior of rank {}'.format(
                deviations.t().size()[1], self.rank))
        else:
            print('Only {} deviation samples in call to .infer for SWAG posterior of rank {}'.format(
                deviations.t().size()[1], self.rank))

    def sample(self, scale=0.5, diagonal_only=False):
        z1 = torch.randn_like(self.sigma_diag, requires_grad=False).to(self.sigma_diag.device)
        diag_term = self.sigma_diag.sqrt() * z1

        if not diagonal_only:
            rank = self.sigma_low_rank.shape[1]
            z2 = torch.randn(rank, requires_grad=False).to(self.sigma_low_rank.device)
            low_rank_term = self.sigma_low_rank.mv(z2)
            low_rank_term /= (rank - 1) ** 0.5
        else:
            low_rank_term = 0.0

        sample = self.mean + (diag_term + low_rank_term) / (scale ** 0.5)
        self._set_params(sample)

    def expected(self):
        self._set_params(self.mean)

    def renormalize(self, train_loader):
        bn_update(train_loader, self)


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
