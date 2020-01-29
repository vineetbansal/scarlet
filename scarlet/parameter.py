# from scarlet.numeric import np
from scarlet.numeric import np, USE_TORCH
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace
from functools import partial

if USE_TORCH:

    from scarlet.numeric.torch import MyTensor
    class Parameter(MyTensor):
        """Optimization parameter

        Parameters
        ----------
        array: array-like
            numpy array (type float) to hold parameter values
        prior: `~scarlet.Prior`
            Prior distribution for parameter
        constraint: `~scarlet.Constraint`
            Constraint on parameter
        step: float or method
            The step size for the parameter
            If a method is used, it needs to have the signature
                `step(X, it) -> float`
            where `X` is the parameter value and `it` the iteration counter
        converged: bool
            Whether the parameter converged during optimimzation
        std: array-like
            Statistical error estimate, same shape as `array`
        fixed: bool
            Whether parameter is held fixed (excluded) during optimization
        """
        def __new__(cls, array, prior=None, constraint=None, step=0, converged=False, std=None, fixed=False):
            obj = array  # TODO: changed - access underlying Tensor object, bypassing MyTensor
            obj.prior = prior
            obj.constraint = constraint
            obj.step = step
            obj.converged = converged
            obj.std = std
            obj.fixed = fixed
            obj.requires_grad = not fixed  # TODO: added
            # return obj
            return MyTensor.__new__(cls, obj)


        @property
        def _data(self):
            return self.view(np.ndarray)

    def relative_step(X, it, factor=0.1):
        return factor*X.mean(axis=0)

else:

    class Parameter(np.ndarray):
        """Optimization parameter

        Parameters
        ----------
        array: array-like
            numpy array (type float) to hold parameter values
        prior: `~scarlet.Prior`
            Prior distribution for parameter
        constraint: `~scarlet.Constraint`
            Constraint on parameter
        step: float or method
            The step size for the parameter
            If a method is used, it needs to have the signature
                `step(X, it) -> float`
            where `X` is the parameter value and `it` the iteration counter
        converged: bool
            Whether the parameter converged during optimimzation
        std: array-like
            Statistical error estimate, same shape as `array`
        fixed: bool
            Whether parameter is held fixed (excluded) during optimization
        """

        def __new__(cls, array, prior=None, constraint=None, step=0, converged=False, std=None, fixed=False):
            obj = np.asarray(array, dtype=array.dtype).view(cls)
            obj.prior = prior
            obj.constraint = constraint
            obj.step = step
            obj.converged = converged
            obj.std = std
            obj.fixed = fixed
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.prior = getattr(obj, 'prior', None)
            self.constraint = getattr(obj, 'constraint', None)
            self.step = getattr(obj, 'step_size', 0)
            self.converged = getattr(obj, 'converged', False)
            self.std = getattr(obj, 'std', None)
            self.fixed = getattr(obj, 'fixed', False)

        def __reduce__(self):
            # Get the parent's __reduce__ tuple
            pickled_state = super().__reduce__()
            # Create our own tuple to pass to __setstate__, but append the __dict__ rather than individual members.
            new_state = pickled_state[2] + (self.__dict__,)
            # Return a tuple that replaces the parent's __setstate__ tuple with our own
            return (pickled_state[0], pickled_state[1], new_state)

        def __setstate__(self, state):
            self.__dict__.update(state[-1])  # Update the internal dict from state
            # Call the parent's __setstate__ with the other tuple elements.
            super().__setstate__(state[0:-1])

        @property
        def _data(self):
            return self.view(np.ndarray)

    ArrayBox.register(Parameter)
    VSpace.register(Parameter, vspace_maker=VSpace.mappings[np.ndarray])


    def relative_step(X, it, factor=0.1):
        return factor * X.mean(axis=0)
