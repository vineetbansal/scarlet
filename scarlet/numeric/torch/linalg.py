import torch
from scarlet.numeric.torch import intercepted


class Module:

    @staticmethod
    @intercepted
    def inv(x):
        return torch.inverse(x)
