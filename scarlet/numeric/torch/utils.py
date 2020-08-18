from numpy import array, ndarray
import torch
from torch import Tensor


def tensor(x):
    """
    Build a torch.Tensor object from a few common input types
    :param x: A Tensor/tuple/list/ndarray etc.
    :return: A torch.Tensor object
    """
    if isinstance(x, Tensor):
        res = x
    elif isinstance(x, (tuple, list)):
        res = torch.tensor(x)
    elif isinstance(x, ndarray):
        res = torch.from_numpy(x)
    else:
        res = torch.from_numpy(array(x))
    return res


def rebuild_tensor(cls, *args, **kwargs):
    """
    The torch implementation of `torch._utils._rebuild_tensor_v2` initializes the tensor, but does not
    populate its __dict__. We do have a __dict__ that we would like to save for our MyTensor class, and do so
    by overriding the MyTensor class's __reduce_ex__ method to add the object __dict__ towards the end.

    This function helps to recreate the object by calling `torch._utils._rebuild_tensor_v2`, but then also populating
    the __dict__ of the just constructed object using this unpickled dict.

    TODO: Very fragile code since it accesses private names in the torch module!
    """
    obj = cls(torch._utils._rebuild_tensor_v2(*args[0:-1], **kwargs))
    obj.__dict__.update(args[-1])
    return obj
