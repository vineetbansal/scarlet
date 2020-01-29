import numpy.ma as ma
import numpy, weakref
from scarlet.numeric import np, USE_TORCH
from autograd import grad
import proxmin
from functools import partial

from .component import ComponentTree

import logging
logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree):
    """The blended scene

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    mse: list
        Array of mean squared errors in each iteration
    """

    def __init__(self, sources, observations):
        """Constructor

        Form a blended scene from a collection of `~scarlet.component.Component`s

        Parameters
        ----------
        sources: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
            Intitialized components or sources to fit to the observations
        observations: a `scarlet.Observation` instance or a list thereof
            Data package(s) to fit
        """
        ComponentTree.__init__(self, sources)

        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self.observations = observations
        self.loss = []

    def fit(self, max_iter=200, e_rel=1e-3, f_rel=1e-4, **alg_kwargs):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        alg_kwargs: dict
            Keywords for the `proxmin.adaprox` optimizer
        """

        # dynamically call parameters to allow for addition / fixing
        X = self.parameters
        n_params = len(X)

        # compute the backward gradient tree
        if USE_TORCH:
            def grad_logL(*X):
                import torch
                with torch.enable_grad():
                    for p in self.parameters:
                        if p.grad is not None:
                            p.grad.data.zero_()
                    loss = self._loss(*X)
                    loss.backward()
                    return [p.grad.numpy() for p in self.parameters]
        else:
            grad_logL = grad(self._loss, tuple(range(n_params)))

        grad_logP = lambda *X: tuple(x.prior(x.view(np.ndarray)) if x.prior is not None else 0 for x in X)
        _grad = lambda *X: tuple(l + p for l,p in zip(grad_logL(*X), grad_logP(*X)))
        _step = lambda *X, it: tuple(x.step(x, it=it) if hasattr(x.step, "__call__") else x.step for x in X)
        _prox = tuple(x.constraint for x in X)

        # good defaults for adaprox
        scheme = alg_kwargs.pop('scheme', 'amsgrad')
        prox_max_iter = alg_kwargs.pop('prox_max_iter', 10)
        eps = alg_kwargs.pop('eps', 1e-8)
        callback = partial(self._convergence_callback, f_rel=f_rel, callback=alg_kwargs.pop('callback', None))

        if USE_TORCH:
            class _Param(numpy.ndarray):
                def __new__(cls, t, prior=None, constraint=None, step=0, converged=False, std=None, fixed=False):
                    array = t.detach().numpy()
                    obj = numpy.asarray(array, dtype=array.dtype).view(cls)
                    obj.tensor = weakref.ref(t)
                    obj.prior = t.prior
                    obj.constraint = t.constraint
                    obj.step = t.step
                    obj.converged = t.converged
                    obj.std = t.std
                    obj.fixed = t.fixed
                    return obj

                def __array_finalize__(self, obj):
                    if obj is None: return
                    self.tensor = getattr(obj, 'tensor', None)
                    self.prior = getattr(obj, 'prior', None)
                    self.constraint = getattr(obj, 'constraint', None)
                    self.step = getattr(obj, 'step_size', 0)
                    self.converged = getattr(obj, 'converged', False)
                    self.std = getattr(obj, 'std', None)
                    self.fixed = getattr(obj, 'fixed', False)

                @property
                def _data(self):
                    return self.view(numpy.ndarray)

            # Convert to a subclass of ndarray that proxmin can use
            X = [_Param(x) for x in X]

        converged, G, V = proxmin.adaprox(X, _grad, _step, prox=_prox, max_iter=max_iter, e_rel=e_rel, scheme=scheme, prox_max_iter=prox_max_iter, callback=callback)

        # set convergence and standard deviation from optimizer
        for p,c,g,v in zip(X, converged, G, V):
            p.converged = c
            p.std = 1/numpy.sqrt(ma.masked_equal(v, 0)) # this is rough estimate!

        return self

    def _loss(self, *parameters):
        """Loss function for autograd

        This method combines the seds and morphologies
        into a model that is used to calculate the loss
        function and update the gradient for each
        parameter
        """
        model = self.get_model(*parameters)
        # Caculate the total loss function from all of the observations
        total_loss = 0
        for observation in self.observations:
            total_loss = total_loss + observation.get_loss(model)
        if USE_TORCH:
            self.loss.append(total_loss)
        else:
            self.loss.append(total_loss._value)
        return total_loss

    def _convergence_callback(self, *parameters, it=None, f_rel=1e-3, callback=None):
        if it > 1 and abs(self.loss[-2] - self.loss[-1]) < f_rel * np.abs(self.loss[-1]):
            raise StopIteration("scarlet.Blend.fit() converged")

        if callback is not None:
            callback(*parameters, it=it)
