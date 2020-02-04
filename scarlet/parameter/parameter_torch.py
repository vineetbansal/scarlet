from scarlet.numeric import np
from scarlet.numeric.torch import MyTensor
from scarlet.constraint import Constraint, ConstraintChain
from scarlet.prior import Prior


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
        return self.view(np.ndarray)


def relative_step(X, it, factor=0.1):
    return factor * X.mean(axis=0)
