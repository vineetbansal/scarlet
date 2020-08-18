"""
This module provides us with the `Module` class that serves as a drop-in replacement for numpy, but uses torch
operations in the background.

Note that this is NOT an exhaustive implementation of the numpy interface, but just a "good-enough" interface
developed by following the code-path of Scarlet tests. It's very possible that certain key operations have not
been implemented, because they're not covered by the unit tests.
"""

import numpy as np
import importlib
import torch
from torch import Tensor

from .wrap import as_subclass, patch_all
from .mytensor import as_mytensor, MyTensor


class Module:

    asnumpy = staticmethod(lambda x: x.detach().numpy() if isinstance(x, torch.Tensor) else np.asarray(x))
    ndarray = MyTensor
    load = staticmethod(np.load)
    shape = staticmethod(lambda x: x.shape)
    newaxis = None
    pi = np.pi
    float32 = torch.float32
    float64 = torch.float64
    bool = torch.bool
    flipud = staticmethod(lambda x: torch.flip(x, [0]))
    fliplr = staticmethod(lambda x: torch.flip(x, [1]))
    arctan2 = staticmethod(as_mytensor(torch.atan2))

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported below.
        """
        if item in ('fft', 'linalg', 'random', 'testing'):
            module = importlib.import_module(self.__module__ + '.submodules.' + item)
            module_class = module.Module
            return module_class
        return as_mytensor(getattr(torch, item))

    @staticmethod
    @as_mytensor
    def array(x, dtype='double', copy=True):
        if isinstance(x, (list, tuple)) and all(isinstance(_x, torch.Tensor) for _x in x):
            assert copy, "Can only support this operation if making a copy"
            retval = torch.stack(x)
            retval.is_real = all(hasattr(_x, 'is_real') and _x.is_real for _x in x)
            return retval
        if copy:
            return MyTensor(torch.tensor(x)).astype(dtype)
        else:
            return MyTensor(x).astype(dtype)

    @staticmethod
    def asarray(a, dtype=None):
        # Implementation mirrors that of numpy
        res = Module.array(a, dtype, copy=False)
        return res

    @staticmethod
    def shape(x):
        if hasattr(x, 'is_real') and not x.is_real:
            return x[..., -1].shape
        return x.shape

    @staticmethod
    def ndim(x):
        if hasattr(x, 'is_real') and not x.is_real:
            return x.ndim - 1
        return x.ndim

    @staticmethod
    @as_mytensor
    def concatenate(x):
        retval = torch.cat(x)
        return retval

    @staticmethod
    @as_mytensor
    def real(x):
        if x.is_real:
            return x
        else:
            x = x[..., 0]
            x.is_real = True
            return x

    @staticmethod
    @as_mytensor
    def imag(x):
        if x.is_real:
            return torch.zeros_like(x)
        else:
            x = x[..., 1]
            x.is_real = True
            return x

    @staticmethod
    @as_mytensor
    def exp(x):
        if x.is_real:
            return torch.exp(x)
        else:
            phi = Module.imag(x)
            re = torch.cos(phi)
            imag = torch.sin(phi)
            retval = torch.stack([re, imag], axis=-1)
            retval.is_real = False
            return retval

    @staticmethod
    @as_mytensor
    def pad(arr, pad_width, mode='constant', constant_values=0):
        # padding in torch.nn.functional expects padding to be specified from last- to first-axis, as a flattened tuple
        # If a single (left_padding, right_padding) tuple was provided, duplicate it for all axes.
        if not isinstance(pad_width[0], (tuple, list)):
            pad_width = tuple([pad_width for i in range(arr.ndim)])
        pad_width2 = tuple(int(y) for x in pad_width[::-1] for y in x)
        return torch.nn.functional.pad(arr, pad_width2, mode=mode, value=constant_values)

    @staticmethod
    @as_mytensor
    def flip(arr, axis=None):
        if axis is None:
            dims = list(range(arr.ndim))
        elif isinstance(axis, int):
            dims = [axis]

        return torch.flip(arr, dims=dims)

    @staticmethod
    @as_mytensor
    def abs(x):
        if not isinstance(x, MyTensor):
            x = MyTensor(x)
        return torch.abs(x)

    @staticmethod
    @as_mytensor
    def floor(x):
        if not isinstance(x, MyTensor):
            x = MyTensor(x).float()
        return torch.floor(x)

    @staticmethod
    @as_mytensor
    def any(x):
        if isinstance(x, torch.Tensor):
            return x.any()
        else:
            return MyTensor(x).any()

    @staticmethod
    @as_mytensor
    def prod(x, axis=None):
        assert axis in (None, 0), "Only product along axis = 0/None supported"
        if not isinstance(x, torch.Tensor):
            x = Module.array(x)

        if x.is_real:
            return MyTensor(torch.prod(x, dim=0))
        else:
            parts = x.unbind(dim=0)
            assert len(parts) == 2, "Can only multiply 2 complex numbers currently."
            from .operator import mul
            retval = MyTensor(mul(parts[0], parts[1]))
            retval.is_real = False
            return retval

    @staticmethod
    @as_mytensor
    def piecewise(x, condlist, funclist):
        assert len(condlist) == len(funclist), 'Condition list and Function list must be equal length'

        y = torch.zeros(x.shape, dtype=x.dtype)
        for k in range(len(condlist)):
            item = funclist[k]
            booltensor = torch.BoolTensor(condlist[k])
            vals = x[booltensor]
            if len(vals) > 0:
                y[booltensor] = item(vals)

        return y

    @staticmethod
    @as_mytensor
    def outer(x, y):
        assert x.ndim == 1, 'Only 1d inputs supported'
        assert y.ndim == 1, 'Only 1d inputs supported'
        return torch.einsum('i,j->ij', x, y)

    @staticmethod
    @as_mytensor
    def sinc(x):
        return Module.piecewise(
            x,
            [x == 0, x != 0],
            [lambda _: 1., lambda _x: torch.sin(Module.pi * _x) / (Module.pi * _x)]
        )

    @staticmethod
    @as_mytensor
    def expand_dims(x, dim):
        return x.unsqueeze(dim)

    @staticmethod
    @as_mytensor
    def min(x, axis=None):
        if isinstance(x, list):
            x = torch.stack(x)
        if axis is None:
            return torch.min(x)
        elif isinstance(axis, int):
            return torch.min(x, axis).values

        x = x.clone()
        for ax in axis:
            x = torch.min(x, dim=ax, keepdim=True).values
        return torch.squeeze(x)

    @staticmethod
    @as_mytensor
    def meshgrid(x, y, indexing='xy'):
        # Numpy defaults to xy indexing for meshgrid, while torch defaults to ij (transposes of what np would return)
        assert indexing == 'xy', 'Only xy indexing supported'
        X, Y = torch.meshgrid(x, y)
        return X.T, Y.T

    @staticmethod
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    @staticmethod
    @as_mytensor
    def maximum(a, b):
        if isinstance(b, (float, int)):
            return torch.clamp(a, b)
        else:
            return torch.max(a, b)

    @staticmethod
    def cross(a, b):
        assert a.ndim == 1 and b.ndim == 1 and len(a) == len(b) == 2, "Only z-component calculation supported"
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def size(t):
        # Caller should just use len() instead!
        assert type(t) is tuple, "Only tuples supported for size calculation."
        return len(t)


"""
One-time operations on module load.

  1. Add a 'as_subclass' method to torch.Tensor class, in preparation of:
  2. 'Patch' (most) of the methods in our TensorBase class to allow for proper subclassing on operations.
"""
Tensor.as_subclass = as_subclass
patch_all()


"""
  3. We do not need to keep a running track of gradients, but can enable gradients with
  `with torch.enable_grad()` when we do need them.
"""
torch.set_grad_enabled(False)
