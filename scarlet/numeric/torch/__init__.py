import torch
import numpy as np
import importlib
from .core import TensorBase


class MyTensor(TensorBase):

    is_real = True

    def astype(self, dtype):
        if not isinstance(dtype, str):
            dtype = dtype.__name__
        assert dtype in ('float', 'double', 'int', 'complex')

        # # For non-complex types, we can simply use torch conversion facility .double(), .float() etc.
        if dtype != 'complex':
            return getattr(self, dtype)()
        else:
            retval = intercepted(torch.stack)([self, torch.zeros(self.shape, dtype=self.dtype)], axis=-1)
            retval.is_real = False
            return retval

    def copy(self):
        return self.clone()

    def view(self, target_class):
        return self.as_subclass(target_class)

    def __mul__(self, other):
        if isinstance(other, complex):
            if self.is_real:
                x = self.astype('complex')
            else:
                x = self
            y = torch.stack([x[..., 0] * other.real, x[..., 1] * other.imag], axis=-1)
            y = y.as_subclass(MyTensor)
            y.is_real = False
            return y
        else:
            retval = super(TensorBase, self).__mul__(other)
            return retval.as_subclass(MyTensor)

    def __rmul__(self, other):
        return self.__mul__(other)

# def my_tensor(*args, **kwargs):
#     return MyTensor(*args, **kwargs)


def intercepted(f):
    def func_wrapper(*args, **kwargs):
        retval = f(*args, **kwargs)
        if isinstance(retval, torch.Tensor):
            retval = retval.as_subclass(MyTensor)
        return retval
    return func_wrapper


class Module:

    asarray = staticmethod(intercepted(torch.tensor))
    ndarray = MyTensor
    pi = np.pi
    float32 = torch.float32
    bool = torch.bool

    @staticmethod
    @intercepted
    def array(x):
        if isinstance(x, list) and all(isinstance(_x, torch.Tensor) for _x in x):
            retval = torch.stack(x)
            retval.is_real = all(_x.is_real for _x in x)
            return retval
        return torch.tensor(x)

    @staticmethod
    @intercepted
    def real(x):
        if x.is_real:
            return x
        else:
            return x[..., 0]

    @staticmethod
    @intercepted
    def imag(x):
        if x.is_real:
            return torch.zeros_like(x)
        else:
            return x[..., 1]

    @staticmethod
    @intercepted
    def exp(other):
        # TODO: Handle Complex numbers
        raise NotImplementedError

    @staticmethod
    @intercepted
    def pad(arr, pad_width, mode='constant', constant_values=0):
        # padding in torch.nn.functional expects padding to be specified from last- to first-axis, as a flattened tuple
        # If a single (left_padding, right_padding) tuple was provided, duplicate it for all axes.
        if not isinstance(pad_width[0], tuple):
            pad_width = tuple([pad_width for i in range(arr.ndim)])
        pad_width = tuple(y for x in pad_width[::-1] for y in x)
        return torch.nn.functional.pad(arr, pad_width, mode=mode, value=constant_values)

    @staticmethod
    @intercepted
    def flip(arr, axis=None):
        if axis is None:
            dims = list(range(arr.ndim))
        elif isinstance(axis, int):
            dims = [axis]

        return torch.flip(arr, dims=dims)

    @staticmethod
    @intercepted
    def abs(x):
        if not isinstance(x, MyTensor):
            x = MyTensor(x)
        return torch.abs(x)

    @staticmethod
    @intercepted
    def floor(x):
        if not isinstance(x, MyTensor):
            x = MyTensor(x).float()
        return torch.floor(x)

    @staticmethod
    @intercepted
    def any(x):
        if isinstance(x, torch.Tensor):
            return x.any()
        else:
            return MyTensor(x).any()

    @staticmethod
    @intercepted
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
    @intercepted
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
    @intercepted
    def outer(x, y):
        assert x.ndim == 1, 'Only 1d inputs supported'
        assert y.ndim == 1, 'Only 1d inputs supported'
        return torch.einsum('i,j->ij', x, y)

    @staticmethod
    @intercepted
    def sinc(x):
        return Module.piecewise(
            x,
            [x == 0, x != 0],
            [lambda _: 1., lambda _x: torch.sin(Module.pi * _x) / (Module.pi * _x)]
        )

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        if item in ('fft',):
            module = importlib.import_module(self.__module__ + '.' + item)
            module_class = module.Module
            return module_class
        return intercepted(getattr(torch, item))


def assert_array_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    return testing.assert_array_equal(a, b, **kwargs)


def assert_almost_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()

    if 'decimal' not in kwargs:
        decimal = 5
    else:
        decimal = kwargs.pop('decimal')
    return testing.assert_almost_equal(a, b, decimal=decimal, **kwargs)
