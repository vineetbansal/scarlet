import torch
from torch import Tensor
from collections import OrderedDict

from .utils import tensor, rebuild_tensor


class TensorBase(Tensor):
    pass


def as_mytensor(f):
    def func_wrapper(*args, **kwargs):
        retval = f(*args, **kwargs)
        if isinstance(retval, Tensor):
            retval = retval.as_subclass(MyTensor)
        elif isinstance(retval, list):
            retval = [r.as_subclass(MyTensor) for r in retval]
        elif isinstance(retval, tuple):
            retval = tuple([r.as_subclass(MyTensor) for r in retval])
        return retval
    return func_wrapper


class MyTensor(TensorBase):

    # Since torch.Tensor does not have a complex dtype, there is no way to look at a Tensor and tell definitively
    # whether we're looking at real values or real + imaginary values.
    # This boolean indicates whether we're dealing with a real valued tensor, or a complex valued tensor
    # In the latter case, the last dimension of the Tensor is always 2.
    is_real = True

    def __new__(cls, x, **kwargs):
        res = tensor(x)
        res = res.as_subclass(cls)
        return res

    def __reduce_ex__(self, proto):
        """
        Method called for pickling this object.
        :param proto: The protocol version; integer.
        :return: As per the pickling protocol, a tuple with 2 elements:
          - A callable object that will be called to create the initial version of the object.
          - A tuple of arguments for the callable object

        Mostly copied from torch.tensor.__reduce_ex__ implementation, but also adds a __dict__ to be saved.
        """
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        return rebuild_tensor, args + (self.requires_grad, OrderedDict(), self.__dict__)

    def __getitem__(self, i):
        if isinstance(i, MyTensor):
            i = i.data
        res = super(Tensor, self).__getitem__(i)
        if isinstance(res, Tensor):
            res = res.as_subclass(type(self))
            res.is_real = self.is_real
            return res
        else:
            return res

    def __setitem__(self, key, value):
        if isinstance(key, MyTensor):
            key = key.data
        res = super(Tensor, self).__setitem__(key, value)
        return res

    @property
    def _value(self):
        # For backward compatibility when client code wants unboxed value of a tracked ndarray
        return self

    def astype(self, dtype):

        if dtype is None:
            return self
        elif not isinstance(dtype, str):
            try:
                dtype = dtype.__name__  # for numpy types
            except AttributeError:
                dtype = str(dtype)
                assert dtype.startswith('torch.')
                dtype = {'torch.float32': 'float', 'torch.float64': 'double'}[dtype]
        else:
            if dtype == 'float':
                dtype = 'double'  # numpy 'float' is 64bit, corresponding to torch double

        if dtype not in ('float', 'double', 'int', 'complex'):
            raise AssertionError('unrecognized dtype')

        # # For non-complex types, we can simply use torch conversion facility .double(), .float() etc.
        if dtype != 'complex':
            return getattr(self, dtype)()
        else:
            retval = as_mytensor(torch.stack)([self, torch.zeros(self.shape, dtype=self.dtype)], axis=-1)
            retval.is_real = False
            return retval

    def copy(self):
        return self.clone()

    def view(self, target_class):
        return self.as_subclass(target_class)

    def max(self, axis=None):
        # Torch 'max' doesn't support multiple axes!
        if axis is None:
            return torch.max(self)
        elif isinstance(axis, int):
            return torch.max(self, axis).values

        x = self.clone()
        for ax in axis:
            x = torch.max(x, dim=ax, keepdim=True).values
        return torch.squeeze(x)

    def __mul__(self, other):

        from . import Module

        if isinstance(other, complex):
            if self.is_real:
                x = self.astype('complex')
            else:
                x = self
            a, b, c, d = Module.real(x), Module.imag(x), other.real, other.imag
            y = torch.stack([a * c - b * d, a * d + b * c], axis=-1)
            y = y.as_subclass(MyTensor)
            y.is_real = False
            return y
        elif isinstance(other, MyTensor) and not other.is_real:
            if self.is_real:
                x = self.astype('complex')
            else:
                x = self
            a, b, c, d = Module.real(x), Module.imag(x), Module.real(other), Module.imag(other)
            a.is_real = b.is_real = c.is_real = d.is_real = True
            y = torch.stack([a * c - b * d, a * d + b * c], axis=-1)
            y = y.as_subclass(MyTensor)
            y.is_real = False
            return y
        elif not self.is_real and other.is_real:
            x = self
            a, b, c, d = Module.real(x), Module.imag(x), Module.real(other), Module.imag(other)
            a.is_real = b.is_real = c.is_real = d.is_real = True
            y = torch.stack([a * c - b * d, a * d + b * c], axis=-1)
            y = y.as_subclass(MyTensor)
            y.is_real = False
            return y
        else:
            retval = super(MyTensor, self).__mul__(other)
            retval.is_real = self.is_real
            return retval.as_subclass(MyTensor)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):

        from . import Module

        if not self.is_real:
            if isinstance(power, torch.Tensor):
                if isinstance(power, MyTensor):
                    assert power.is_real, 'Can only raise to a real power'
                re, im = Module.real(self), Module.imag(self)
                r = (torch.sqrt(re**2 + im**2)) ** power
                theta = power * torch.atan2(im, re)
                re, im = torch.cos(theta), torch.sin(theta)
                re = r * re
                im = r * im
                y = torch.stack([re, im], axis=-1)
                y = y.as_subclass(MyTensor)
                y.is_real = False
                return y

            else:
                return super(TensorBase, self).__pow__(power).as_subclass(MyTensor)
        else:
            return super(TensorBase, self).__pow__(power).as_subclass(MyTensor)
