import torch
from abc import ABC, abstractmethod


class ProbabilisticModule(torch.nn.Module, ABC):

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def expected(self):
        pass

    def _set_params(self, sample):
        # TODO: Hippolyt suggests replacing with ``vector_to_params''
        i = 0
        for p in self.model.parameters():
            shape = p.data.shape
            n = p.data.numel()
            p.data = sample[i: i + n].view(shape).to(p.device)
            i += n

    def forward(self, *input):
        return self.model(*input)

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


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

        mean = torch.cat(list_mean).cpu()
        sq_mean = torch.cat(list_sq_mean).cpu()
        deviations = torch.cat(list_deviations, dim=1).cpu()

        self.mean = mean
        self.sigma_diag = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        print(self.sigma_low_rank.size())
        print(deviations.t().size())
        if self.sigma_low_rank.size() == deviations.t().size():
            self.sigma_low_rank = deviations.t()
        elif strict:
            raise RuntimeError('Only {} deviation samples in call to .infer for SWAG posterior of rank {}'.format(
                deviations.t().size()[1], self.rank))
        else:
            print('Only {} deviation samples in call to .infer for SWAG posterior of rank {}'.format(
                deviations.t().size()[1], self.rank))

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

    def expected(self):
        self._set_params(self.mean)


class PointModel(ProbabilisticModule):

    def sample(self):
        pass

    def expected(self):
        pass


class MixtureModel(ProbabilisticModule):

    def __init__(self, models, weights=None):
        super().__init__()
        for model in models:
            if not isinstance(model, ProbabilisticModule):
                raise TypeError('Can only mix ProbabilisticModules')
        if weights is None:
            self.weights = len(models) * [1 / len(models)]
        else:
            self.weighs = weights
        self._models = models

    def expected(self):
        for model in self._models:
            model.expected()

    def sample(self):
        for model in self._models:
            model.sample()

    def forward(self, *input):
        mix = 0
        for k in range(len(self._models)):
            model = self._models[k]
            weight = self.weights[k]
            mix += weight * model(*input)
        return mix

    def train(self, mode=True):
        for model in self._models:
            model.train(mode)

    def eval(self):
        for model in self._models:
            model.eval()
