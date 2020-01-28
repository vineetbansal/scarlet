from scarlet.numeric import np, USE_TORCH
import autograd.scipy as scipy
from .bbox import Box

def moffat(y, x, alpha=4.7, beta=1.5, bbox=None):
    """Symmetric 2D Moffat function

    .. math::

        (1+\frac{(x-x0)^2+(y-y0)^2}{\alpha^2})^{-\beta}

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    alpha: float
        Core width
    beta: float
        Power-law index
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    Y = np.arange(bbox.shape[1]) + bbox.origin[1]
    X = np.arange(bbox.shape[2]) + bbox.origin[2]
    X, Y = np.meshgrid(X, Y)
    # TODO: has no pixel-integration formula
    return ((1+((X-x)**2+(Y-y)**2)/alpha**2)**-beta)[None,:,:]

def gaussian(y, x, sigma=1, integrate=True, bbox=None):
    """Circular Gaussian Function

    Parameters
    ----------
    y: float
        Vertical coordinate of the center
    x: float
        Horizontal coordinate of the center
    sigma: float
        Standard deviation of the gaussian
    integrate: bool
        Whether pixel integration is performed
    bbox: Box
        Bounding box over which to evaluate the function

    Returns
    -------
    result: array
        A 2D circular gaussian sampled at the coordinates `(y_i, x_j)`
        for all i and j in `shape`.
    """
    erf = np.erf if USE_TORCH else scipy.special.erf
    Y = (np.arange(bbox.shape[1]) + bbox.origin[1]).astype('float')
    X = (np.arange(bbox.shape[2]) + bbox.origin[2]).astype('float')

    # def erf_approx(x):
    #     t = 1.0 / (1 + 0.5 * np.abs(x))
    #     tau = t * np.exp(-(x ** 2) - 1.26551223 + 1.00002368 * t + 0.37409196 * t ** 2 + 0.09678418 * t ** 3 - 0.18628806 * t ** 4 + 0.27886807 * t ** 5 - 1.13520398 * t ** 6 + 1.48851587 * t ** 7 - 0.82215223 * t ** 8 + 0.17087277 * t ** 9)
    #     if USE_TORCH:
    #         res = (x >= 0).float() * (1 - tau) + (x < 0).float() * (tau - 1)
    #     else:
    #         res = (x >= 0).astype('float') * (1 - tau) + (x < 0).astype('float') * (tau - 1)
    #     return res
    # erf = erf_approx

    def f(X):
        if not integrate:
            return np.exp(-X ** 2 / (2 * sigma ** 2))
        else:
            sqrt2 = np.sqrt(np.array(2.))
            return np.sqrt(np.array(np.pi / 2)) * sigma * (
                        erf((0.5 - X) / (sqrt2 * sigma)) + erf(
                    (2 * X + 1) / (2 * sqrt2 * sigma)))

    _y = Y-y
    _x = X-x
    f1 = f(_y)
    f2 = f(_x)
    ans = (f1[:,None] * f2[None,:])[None,:,:]
    return ans


class PSF:
    """Class to represent PSFs

    Parameters
    ----------
    X: array-like or method
        If `X` is an array, it represent an image of the PSF in every band.
        If `X` is a callable method, it describes a function that can generate a PSF when given coordinates (y, x).
    shape: tuple
        Shape of the 2D image to generate an PSF image for. Only used if `X` is a method.
    """
    def __init__(self, X, shape=None):
        if hasattr(X, 'shape'):
            self._image = X.copy()
            self.normalize()
            self._func = None
            self.shape = X.shape
        elif hasattr(X, '__call__'):
            assert shape is not None, "Functional PSFs must set shape argument"
            self._image = None
            self._func = X
            self.shape = shape
        else:
            msg = "A PSF must be initialized with either an image or function"
            raise ValueError(msg)

    def __call__(self, y, x, bbox=None):
        """Generate analytic PSF image at coordinate (y,x)

        Parameters
        ----------
        y: float
            Vertical model frame coordinates for the center of PSF image
        x: float
            Horizontal model frame coordinates for the center of PSF image
        bbox: `~scarlet.Box`
            Bounding Box for the PSF image in model coordinates

        Returns
        -------
        image: array-like
            Centered image of the PSF at given coordinate.
        """
        if bbox is None:
            bbox = Box(self.shape)
        if self._func is not None:
            return self._func(y, x, bbox=bbox)
        return None

    @property
    def image(self):
        """Centered image of the PSF
        """
        if self._image is None:
            assert self.shape is not None, "Set PSF.shape first"
            y, x = self.shape[1] // 2, self.shape[2] // 2
            self._image = self.__call__(y, x)
            self.normalize()
        return self._image

    def normalize(self):
        """Normalize to PSF image in every band to unity
        """
        sums = self._image.sum(axis=(1, 2))
        self._image /= sums[:, None, None]
        return self

    def update_dtype(self, dtype):
        """Update data type of `image` to `dtype`
        """
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
        return self
