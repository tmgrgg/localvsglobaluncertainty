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
