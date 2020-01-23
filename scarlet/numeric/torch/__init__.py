import torch
import numpy as np
import importlib
from .core import TensorBase


class MyTensor(TensorBase):

    is_real = True

    def astype(self, dtype):
        if not isinstance(dtype, str):
            try:
                dtype = dtype.__name__  # for numpy types
            except AttributeError:
                dtype = str(dtype)
                assert dtype.startswith('torch.')
                dtype = {'torch.float32': 'float', 'torch.float64': 'double'}[dtype]

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
        else:
            retval = super(TensorBase, self).__mul__(other)
            return retval.as_subclass(MyTensor)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        if not self.is_real:
            if isinstance(power, MyTensor):
                assert power.is_real, "Can only raise to a real power"
                if power.item() not in (0, 1):
                    raise AssertionError('Only 0/1 power supported')
                if power.item()==1:
                    y = torch.stack([Module.real(self), Module.imag(self)], axis=-1)
                elif power.item()==0:
                    y = torch.stack([torch.ones_like(self[..., 0]), torch.zeros_like(self[..., 1])], axis=-1)
                y = y.as_subclass(MyTensor)
                y.is_real = False
                return y

                # TODO: Take care of the sign!
                # re, im = Module.real(self), Module.imag(self)
                # r = (torch.sqrt(re**2 + im**2)) ** power
                # theta = power * torch.atan(im / re)
                # re, im = torch.cos(theta), torch.sin(theta)
                # re = r * re
                # im = r * im
                # y = torch.stack([re, im], axis=-1)
                # y = y.as_subclass(MyTensor)
                # y.is_real = False
                # return y

            else:
                return super(TensorBase, self).__pow__(power).as_subclass(MyTensor)
        else:
            return super(TensorBase, self).__pow__(power).as_subclass(MyTensor)

# def my_tensor(*args, **kwargs):
#     return MyTensor(*args, **kwargs)


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


class Module:

    asnumpy = staticmethod(lambda x: x.numpy())
    asarray = staticmethod(intercepted(torch.tensor))
    ndarray = MyTensor
    pi = np.pi
    float32 = torch.float32
    float64 = torch.float64
    bool = torch.bool
    flipud = staticmethod(lambda x: torch.flip(x, [0]))
    fliplr = staticmethod(lambda x: torch.flip(x, [1]))
    arctan2 = staticmethod(intercepted(torch.atan2))

    @staticmethod
    @intercepted
    def array(x, dtype='double'):
        if isinstance(x, list) and all(isinstance(_x, torch.Tensor) for _x in x):
            retval = torch.stack(x)
            retval.is_real = all(hasattr(_x, 'is_real') and _x.is_real for _x in x)
            return retval
        return MyTensor(torch.tensor(x)).astype(dtype)

    @staticmethod
    @intercepted
    def concatenate(x):
        retval = torch.stack(x)
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
        if not isinstance(pad_width[0], tuple):
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

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        if item in ('fft', 'linalg', 'random'):
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
