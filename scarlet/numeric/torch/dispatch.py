from types import FunctionType
import functools


def retain_type(new, old=None, typ=None):
    if new is None:
        return
    assert old is not None or typ is not None
    if typ is None:
        if not isinstance(old, type(new)):
            return new
        typ = old if isinstance(old, type) else type(old)

    if isinstance(typ, type(None)) or isinstance(new, typ):
        return new

    res = new.as_subclass(typ)
    res.__dict__ = old.__dict__
    return res


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__dict__.update(f.__dict__)
    return fn


def patch_to(cls, as_prop=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)
    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{f.__name__}"
            setattr(c_, f.__name__, property(nf) if as_prop else nf)
        return f
    return _inner


def patch(f):
    "Decorator: add `f` to the first parameter's class (based on f's type annotations)"
    cls = next(iter(f.__annotations__.values()))
    return patch_to(cls)(f)
