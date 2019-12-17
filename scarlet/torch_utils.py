import numpy as np
from numpy.testing import assert_almost_equal as np_assert_almost_equal
import torch


def tmax(x, axes):
    """
    Torch `max` does not support multiple axes.
    :param x:
    :param axes:
    :return:
    """
    x = x.clone()
    for ax in axes:
        x = x.max(dim=ax, keepdim=True).values
    return torch.squeeze(x)

