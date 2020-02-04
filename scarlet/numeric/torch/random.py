import torch
from scarlet.numeric.torch import intercepted


class Module:

    @staticmethod
    @intercepted
    def rand(*shape):
        return torch.rand(tuple(shape))

    @staticmethod
    def seed(s):
        torch.manual_seed(s)
