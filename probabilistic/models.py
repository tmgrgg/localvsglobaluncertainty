import torch
from abc import ABC, abstractmethod


class ProbabilisticModule(torch.nn.Module, ABC):

    def __init__(self, model):
        super(ProbabilisticModule, self).__init__()
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

    def __init__(self, model, swag, var_clamp=1e-30):
        super(SWAGPosterior, self).__init__(model)
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

    def expected(self):
        self._set_params(self.mean)
