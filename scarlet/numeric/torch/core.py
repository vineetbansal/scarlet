from numpy import ndarray, array
import torch
from torch import Tensor
from collections import OrderedDict
from types import BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType

MethodWrapperType = type(object().__str__)
MethodDescriptorType = type(str.join)


def patch_to(cls):
    def _inner(f):
        f2 = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        f2.__dict__.update(f.__dict__)
        f2.__qualname__ = f"{cls.__name__}.{f.__name__}"
        setattr(cls, f.__name__, f2)
        return f
    return _inner


@patch_to(Tensor)
def as_subclass(self, typ):
    # return typ.__new__(typ, self)
    self._original_shape = self.shape
    self.__class__ = typ
    return self


def tensor(x):
    if isinstance(x, Tensor):
        res = x
    elif isinstance(x, (tuple, list)):
        res = torch.tensor(x)
    elif isinstance(x, ndarray):
        res = torch.from_numpy(x)
    else:
        res = torch.from_numpy(array(x))
    return res


def _fa_rebuild_tensor(cls, *args, **kwargs):
    obj = cls(torch._utils._rebuild_tensor_v2(*args[0:-1], **kwargs))
    obj.__dict__.update(args[-1])
    return obj


class TensorBase(Tensor):
    def __new__(cls, x, **kwargs):
        res = tensor(x)
        res._original_shape = res.shape
        res._original_ndim = res.ndim
        res = res.as_subclass(cls)
        return res

    def __reduce_ex__(self, proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        return _fa_rebuild_tensor, args + (self.requires_grad, OrderedDict(), self.__dict__)

    def __getitem__(self, i):
        # TODO: Why doesn't this work if i is a TensorBase object?
        if isinstance(i, TensorBase):
            i = i.data
        res = super(Tensor, self).__getitem__(i)
        if isinstance(res, Tensor):
            res = res.as_subclass(type(self))
            res.is_real = self.is_real
            return res
        else:
            return res

    def __setitem__(self, key, value):
        # TODO: Why doesn't this work if key is a TensorBase object?
        if isinstance(key, TensorBase):
            key = key.data
        res = super(Tensor, self).__setitem__(key, value)
        return res


def retain_type(new, old):
    if new is None:
        return
    if not isinstance(old, type(new)):
        return new
    typ = type(old)

    if isinstance(typ, type(None)) or isinstance(new, typ):
        return new

    res = new.as_subclass(typ)
    res.__dict__ = old.__dict__
    return res


def _patch_all():
    def get_f(fn):
        def _f(self, *args, **kwargs):
            res = getattr(super(TensorBase, self), fn)(*args, **kwargs)
            return retain_type(res, self)
        return _f

    skips = 'as_subclass __getitem__ __setitem__ __class__ __deepcopy__ __delattr__ __dir__ __doc__ __getattribute__ \
    __hash__ __init__ __init_subclass__ __new__ __reduce__ __reduce_ex__ __module__ __setstate__'.split()

    t = tensor([1])
    for fn in dir(t):
        if fn in skips:
            continue
        f = getattr(t, fn)
        if isinstance(f, (MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType)):
            setattr(TensorBase, fn, get_f(fn))


_patch_all()
