import itertools
import numpy as np
from numpy import ndarray
import operator

import torch
from torch import as_tensor, Tensor
from collections import OrderedDict

from fastcore.all import array, cast, defaults, is_iter, is_listy, patch, retain_meta, retain_type, Iterable, \
    Generator, L


try:
    from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType
except ImportError:
    WrapperDescriptorType = type(object.__init__)
    MethodWrapperType = type(object().__str__)
    MethodDescriptorType = type(str.join)
from types import BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType


def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable, Generator)) and getattr(o, 'ndim', 1)


def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    #Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, '__len__') and getattr(o, 'ndim', 1)


def all_equal(a,b):
    "Compares whether `a` and `b` are the same length and have the same contents"
    if not is_iter(b): return False
    return all(equals(a_,b_) for a_,b_ in itertools.zip_longest(a,b))


def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x


def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
    return x


def one_is_instance(a, b, t): return isinstance(a,t) or isinstance(b,t)


def equals(a,b):
    "Compares `a` and `b` for equality; supports sublists, tensors and arrays too"
    if one_is_instance(a,b,type): return a==b
    if hasattr(a, '__array_eq__'): return a.__array_eq__(b)
    if hasattr(b, '__array_eq__'): return b.__array_eq__(a)
    cmp = (np.array_equal if one_is_instance(a, b, ndarray       ) else
           operator.eq    if one_is_instance(a, b, (str,dict,set)) else
           all_equal      if is_iter(a) or is_iter(b) else
           operator.eq)
    return cmp(a,b)


class ArrayBase(ndarray):
    @classmethod
    def _before_cast(cls, x): return x if isinstance(x,ndarray) else array(x)


@patch
def __array_eq__(self:Tensor,b):
    return torch.equal(self,b) if self.dim() else self==b


def _array2tensor(x):
    if x.dtype==np.uint16: x = x.astype(np.float32)
    return torch.from_numpy(x)


def tensor(x, *rest, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,)+rest
    # There was a Pytorch bug in dataloader using num_workers>0. Haven't confirmed if fixed
    # if isinstance(x, (tuple,list)) and len(x)==0: return tensor(0)
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple,list))
           else _array2tensor(x) if isinstance(x, ndarray)
           else as_tensor(x, **kwargs) if hasattr(x, '__array__') or is_iter(x)
           else _array2tensor(array(x), **kwargs))
    # if res.dtype is torch.float64: return res.float()
    return res


def unsqueeze(x, dim=-1, n=1):
    "Same as `torch.unsqueeze` but can add `n` dims"
    for _ in range(n): x = x.unsqueeze(dim)
    return x


def unsqueeze_(x, dim=-1, n=1):
    "Same as `torch.unsqueeze_` but can add `n` dims"
    for _ in range(n): x.unsqueeze_(dim)
    return x


def _fa_rebuild_tensor(cls, *args, **kwargs):
    obj = cls(torch._utils._rebuild_tensor_v2(*args[0:-1], **kwargs))
    obj.__dict__.update(args[-1])
    return obj

def _fa_rebuild_qtensor(cls, *args, **kwargs):
    return cls(torch._utils._rebuild_qtensor  (*args, **kwargs))


def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):  return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)


def to_np(x):
    "Convert a tensor to a numpy array."
    return apply(lambda o: o.data.cpu().numpy(), x)


@patch
def set_meta(self:Tensor, x):
    "Set all metadata in `__dict__`"
    if hasattr(x, '__dict__'): self.__dict__ = x.__dict__


@patch
def get_meta(self:Tensor, n, d=None):
    "Set `n` from `self._meta` if it exists and returns default `d` otherwise"
    return getattr(self, '_meta', {}).get(n, d)


@patch
def as_subclass(self:Tensor, typ):
    "Cast to `typ` (should be in future PyTorch version, so remove this then)"
    # res = torch.Tensor._make_subclass(typ, self, self.requires_grad)
    # return retain_meta(self, res)
    self.__class__ = typ
    return self


class TensorBase(Tensor):
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        res._meta = kwargs
        return res

    @classmethod
    def _before_cast(cls, x): return x if isinstance(x, Tensor) else tensor(x)

    def __reduce_ex__(self, proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        f = _fa_rebuild_qtensor if self.is_quantized else  _fa_rebuild_tensor
        return (f, args + (self.requires_grad, OrderedDict(), self.__dict__))

    def __getitem__(self, i):
        # TODO: Why doesn't this work if i is a TensorBase object?
        if isinstance(i, TensorBase):
            # print('debug')
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

    # more_skips = '__iter__ dim size __len__ set_meta __setattr__'.split()
    # skips.extend(more_skips)

    t = tensor([1])
    for fn in dir(t):
        if fn in skips:
            continue
        f = getattr(t, fn)
        if isinstance(f, (MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType)):
            setattr(TensorBase, fn, get_f(fn))
    TensorBase._patched = True


_patch_all()


@patch
def tensored(self:L):
    "`mapped(tensor)`"
    return self.map(tensor)


@patch
def stack(self:L, dim=0):
    "Same as `torch.stack`"
    return torch.stack(list(self.tensored()), dim=dim)


@patch
def cat(self:L, dim=0):
    "Same as `torch.cat`"
    return torch.cat(list(self.tensored()), dim=dim)
