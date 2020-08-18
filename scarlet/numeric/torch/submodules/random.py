import torch
from scarlet.numeric.torch import as_mytensor


class Module:

    @staticmethod
    @as_mytensor
    def rand(*shape):
        return torch.rand(tuple(shape))

    @staticmethod
    def seed(s):
        torch.manual_seed(s)
