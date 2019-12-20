import torch
import numpy as np
import importlib
from .core import TensorBase


class MyTensor(TensorBase):

    is_real = True

    def astype(self, dtype):
        assert isinstance(dtype, str), "Specify target dtype as a string"
        assert dtype in ('float', 'double', 'int', 'complex')

        # # For non-complex types, we can simply use torch conversion facility .double(), .float() etc.
        if dtype != 'complex':
            return getattr(self, dtype)()
        else:
            retval = intercepted(torch.stack)([self, torch.zeros(self.shape, dtype=self.dtype)], axis=-1)
            retval.is_real = False
            return retval


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

    asarray = staticmethod(torch.tensor)
    pi = np.pi

    @staticmethod
    @intercepted
    def array(x):
        if isinstance(x, list):
            return torch.stack(x)
        return torch.tensor(x)

    @staticmethod
    @intercepted
    def pad(arr, pad_width, mode='constant'):
        pad_width = tuple(y for x in pad_width[::-1] for y in x)
        return torch.nn.functional.pad(arr, pad_width, mode=mode)

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        if item in ('fft',):
            module = importlib.import_module(self.__module__ + '.' + item)
            module_class = module.Module
            return module_class
        return intercepted(getattr(torch, item))


def assert_array_equal(a, b, *args, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    return testing.assert_array_equal(a, b, *args, **kwargs)


def assert_almost_equal(a, b, *args, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    return testing.assert_almost_equal(a, b, *args, **kwargs)
