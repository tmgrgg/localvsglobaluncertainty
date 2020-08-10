from localvglobal.probabilistic.models import *


class MixtureModel(ProbabilisticModule):

    def __init__(self, models, weights=None):
        super().__init__()
        for model in models:
            if not isinstance(model, ProbabilisticModule):
                raise TypeError('Can only mix ProbabilisticModules')
        if weights is None:
           weights = len(models) * [1 / len(models)]
        self.register_buffer('weights', torch.FloatTensor(weights, requires_grad=False))
        self._models = models

    def expected(self):
        for model in self._models:
            model.expected()

    def sample(self, *args, **kwargs):
        for model in self._models:
            model.sample(*args, **kwargs)

    def renormalize(self, train_loader):
        for model in self._models:
            model.renormalize(train_loader)

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

    def cuda(self):
        for model in self._models:
            model.cuda()
        self.weights.cuda()
