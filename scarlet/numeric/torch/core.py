from numpy import ndarray, array

import torch
from torch import Tensor
from collections import OrderedDict

from .dispatch import patch, retain_type

try:
    from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType
except ImportError:
    WrapperDescriptorType = type(object.__init__)
    MethodWrapperType = type(object().__str__)
    MethodDescriptorType = type(str.join)
from types import BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType


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


@patch
def as_subclass(self:Tensor, typ):
    self.__class__ = typ
    return self


class TensorBase(Tensor):
    def __new__(cls, x, **kwargs):
        res = tensor(x).as_subclass(cls)
        # res._meta = kwargs
        return res

    def __reduce_ex__(self, proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        return (_fa_rebuild_tensor, args + (self.requires_grad, OrderedDict(), self.__dict__))

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


def _patch_all():
    if getattr(TensorBase, '_patched', False):
        return

    def get_f(fn):
        def _f(self, *args, **kwargs):
            cls = self.__class__
            res = getattr(super(TensorBase, self), fn)(*args, **kwargs)
            return retain_type(res, self)
        return _f

    skips = 'as_subclass __getitem__ __setitem__ __class__ __deepcopy__ __delattr__ __dir__ __doc__ __getattribute__ __hash__ __init__ \
        __init_subclass__ __new__ __reduce__ __reduce_ex__ __module__ __setstate__'.split()

    t = tensor([1])
    for fn in dir(t):
        if fn in skips:
            continue
        f = getattr(t, fn)
        if isinstance(f, (MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType)):
            setattr(TensorBase, fn, get_f(fn))
    TensorBase._patched = True


_patch_all()

