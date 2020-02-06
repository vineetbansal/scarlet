import numpy
from scarlet.numeric.torch import MyTensor
from scarlet.constraint import Constraint, ConstraintChain
from scarlet.prior import Prior


class _Param(numpy.ndarray):
    # Something we can pass on to proxmin
    def __new__(cls, t):
        array = t.detach().numpy()
        obj = numpy.asarray(array, dtype=array.dtype).view(cls)
        obj._value = t
        obj.name = t.name_
        obj.prior = t.prior
        obj.constraint = t.constraint
        obj.step = t.step
        obj.std = t.std
        obj.m = t.m
        obj.v = t.v
        obj.vhat = t.vhat
        obj.fixed = t.fixed
        obj.constructed = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, "name", "unnamed")
        self.prior = getattr(obj, "prior", None)
        self.constraint = getattr(obj, "constraint", None)
        self.step = getattr(obj, "step", 0)
        self.std = getattr(obj, "std", None)
        self.m = getattr(obj, "m", None)
        self.v = getattr(obj, "v", None)
        self.vhat = getattr(obj, "vhat", None)
        self.fixed = getattr(obj, "fixed", False)

    @property
    def _data(self):
        return self.view(numpy.ndarray)

    def __setattr__(self, key, value):
        if getattr(self, 'constructed', False):
            return setattr(self._value, key, value)
        else:
            return super().__setattr__(key, value)


class Parameter(MyTensor):
    """Optimization parameter

    Parameters
    ----------
    array: array-like
        numpy array (type float) to hold parameter values
    name: string
        Name to identify parameter
    prior: `~scarlet.Prior`
        Prior distribution for parameter
    constraint: `~scarlet.Constraint`
        Constraint on parameter
    step: float or method
        The step size for the parameter
        If a method is used, it needs to have the signature
            `step(X, it) -> float`
        where `X` is the parameter value and `it` the iteration counter
    std: array-like
        Statistical error estimate; set after optimization
    m: array-like
        First moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
    v: array-like
        Second moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
    vhat: array-like
        Maximal second moment of the gradient; only set when optimized
        See Kingma & Ba (2015) and Reddi, Kale & Kumar (2018) for details
    fixed: bool
        Whether parameter is held fixed (excluded) during optimization
    """

    def __new__(
        cls,
        array,
        name="unnamed",
        prior=None,
        constraint=None,
        step=0,
        std=None,
        m=None,
        v=None,
        vhat=None,
        fixed=False,
    ):
        obj = array
        obj.name_ = name
        if prior is not None:
            assert isinstance(prior, Prior)
        obj.prior = prior
        if constraint is not None:
            assert isinstance(constraint, Constraint) or isinstance(
                constraint, ConstraintChain
            )
        obj.constraint = constraint
        obj.step = step
        obj.std = std
        obj.m = m
        obj.v = v
        obj.vhat = vhat
        obj.fixed = fixed
        obj.requires_grad = not fixed
        return MyTensor.__new__(cls, obj)

    @property
    def _data(self):
        return self

    def asnumpy(self):
        return _Param(self)


def relative_step(X, it, factor=0.1):
    return factor * X.mean(axis=0)
