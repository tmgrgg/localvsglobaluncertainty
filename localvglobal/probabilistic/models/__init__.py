import torch
from localvglobal.training.utils import bn_update
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


