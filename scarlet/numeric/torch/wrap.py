"""
This module exists because torch.Tensor classes don't do subclassing the 'proper' pythonic way.
For example, operations like:

    <MyTensor> + 1
    <MyTensor> + <torch.Tensor>
    <MyTensor> * <MyTensor>

where MyTensor is a subclass of Tensor, all return <torch.Tensor> objects.

Why is this important? Since torch.Tensor objects also don't support a complex dtype, it's up to us to implement
these operations, to delegate work to the regular torch.Tensor object where appropriate, and also to keep track of
which operations return complex results, and which do not.

In the normal case we could have just subclassed torch.Tensor, but since we can't, we monkey-patch torch.Tensor
to support subclassing.
"""

from types import MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType
from .mytensor import TensorBase


def as_subclass(self, typ):
    """
    Since torch.Tensor objects cannot properly be subclassed, this function
    (which we will add to the base torch.Tensor class as a method)
    allows us to downcast an object to more specific subtypes.

    This function in itself does not do any sanity checking at all,
    but a more elaborate sanity check before using this function is done in
    the _retain_type function, for example.
    """
    if not isinstance(self, typ):
        self.__class__ = typ
    return self


def _retain_type(from_, to_):
    """
    Downcast an object to a more specific target type, which retaining all the attributes of the target
    instance.
    :param from_: Instance to be downcast
    :param to_: Instance of a class to which we will be down-casting to; should be be a subclass of type(from_)
    :return: A new instance of type `type(to_)`, with all the original attributes of `to_` intact.
    """
    if from_ is None:
        return
    if not isinstance(to_, type(from_)):
        return from_
    typ = type(to_)

    if isinstance(typ, type(None)) or isinstance(from_, typ):
        return from_

    res = from_.as_subclass(typ)
    res.__dict__ = to_.__dict__
    return res


def patch_all():
    """
    Patch (most) of the methods of the `TensorBase` class such that the return values of those methods are
    automatically downcast to the `TensorBase` class.
    :return: On return, the `TensorBase` class has been patched.
    """
    def get_f(fn):
        def _f(self, *args, **kwargs):
            res = getattr(super(TensorBase, self), fn)(*args, **kwargs)
            return _retain_type(res, self)
        return _f

    skips = 'as_subclass __getitem__ __setitem__ __class__ __deepcopy__ __delattr__ __dir__ __doc__ __getattribute__ \
    __hash__ __init__ __init_subclass__ __new__ __reduce__ __reduce_ex__ __module__ __setstate__'.split()

    t = TensorBase([1])
    for fn in dir(t):
        if fn in skips:
            continue
        f = getattr(t, fn)
        if isinstance(f, (MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType)):
            setattr(TensorBase, fn, get_f(fn))

