import torch
import numpy as np
import importlib
from .core import TensorBase

torch.set_grad_enabled(False)


def intercepted(f):
    def func_wrapper(*args, **kwargs):
        retval = f(*args, **kwargs)
        if isinstance(retval, torch.Tensor):
            retval = retval.as_subclass(MyTensor)
        elif isinstance(retval, list):
            retval = [r.as_subclass(MyTensor) for r in retval]
        elif isinstance(retval, tuple):
            retval = tuple([r.as_subclass(MyTensor) for r in retval])
        return retval
    return func_wrapper


class MyTensor(TensorBase):

    is_real = True

    @property
    def _value(self):
        # For backward compatibility when client code wants unboxed value of an tracked ndarray
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
            retval = intercepted(torch.stack)([self, torch.zeros(self.shape, dtype=self.dtype)], axis=-1)
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
            retval = super(TensorBase, self).__mul__(other)
            retval.is_real = self.is_real
            return retval.as_subclass(MyTensor)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
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
    arctan2 = staticmethod(intercepted(torch.atan2))

    @staticmethod
    @intercepted
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
    @intercepted
    def concatenate(x):
        retval = torch.cat(x)
        return retval

    @staticmethod
    @intercepted
    def real(x):
        if x.is_real:
            return x
        else:
            x = x[..., 0]
            x.is_real = True
            return x

    @staticmethod
    @intercepted
    def imag(x):
        if x.is_real:
            return torch.zeros_like(x)
        else:
            x = x[..., 1]
            x.is_real = True
            return x

    @staticmethod
    @intercepted
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
    @intercepted
    def pad(arr, pad_width, mode='constant', constant_values=0):
        # padding in torch.nn.functional expects padding to be specified from last- to first-axis, as a flattened tuple
        # If a single (left_padding, right_padding) tuple was provided, duplicate it for all axes.
        if not isinstance(pad_width[0], (tuple, list)):
            pad_width = tuple([pad_width for i in range(arr.ndim)])
        pad_width2 = tuple(int(y) for x in pad_width[::-1] for y in x)
        return torch.nn.functional.pad(arr, pad_width2, mode=mode, value=constant_values)

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

    @staticmethod
    @intercepted
    def expand_dims(x, dim):
        return x.unsqueeze(dim)

    @staticmethod
    @intercepted
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
    @intercepted
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
    @intercepted
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

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        if item in ('fft', 'linalg', 'random', 'testing'):
            module = importlib.import_module(self.__module__ + '.' + item)
            module_class = module.Module
            return module_class
        return intercepted(getattr(torch, item))


def assert_array_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    return testing.assert_array_equal(a, b, **kwargs)


def assert_almost_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()

    if 'decimal' not in kwargs:
        decimal = 5
    else:
        decimal = kwargs.pop('decimal')
    return testing.assert_almost_equal(a, b, decimal=decimal, **kwargs)
