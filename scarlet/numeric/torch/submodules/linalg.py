import torch
from scarlet.numeric.torch import as_mytensor


class Module:

    @staticmethod
    @as_mytensor
    def inv(x):
        return torch.inverse(x)
