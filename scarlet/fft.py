import operator

import autograd.numpy as np
import torch
from scipy import fftpack


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    if x.ndim == 3:
        real = x
        imag = None
    else:
        real, imag = torch.unbind(x, -1)

    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        if imag is not None:
            imag = roll_n(imag, axis=dim, n=n_shift)

    if imag is not None:
        return torch.stack((real, imag), -1)
    else:
        return real

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def _centered(arr, newshape):
    """Return the center newshape portion of the array.

    This function is used by `fft_convolve` to remove
    the zero padded region of the convolution.

    Note: If the array shape is odd and the target is even,
    the center of `arr` is shifted to the center-right
    pixel position.
    This is slightly different than the scipy implementation,
    which uses the center-left pixel for the array center.
    The reason for the difference is that we have
    adopted the convention of `np.fft.fftshift` in order
    to make sure that changing back and forth from
    fft standard order (0 frequency and position is
    in the bottom left) to 0 position in the center.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    try:
        if not np.all(newshape <= currshape):
            msg = "arr must be larger than newshape in both dimensions, received {0}, and {1}"
            raise ValueError(msg.format(arr.shape, newshape))
    except:
        print('debug')

    startind = (currshape - newshape+1) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


def _pad(arr, newshape, axes=None):
    """Pad an array to fit into newshape

    Pad `arr` with zeros to fit into newshape,
    which uses the `np.fft.fftshift` convention of moving
    the center pixel of `arr` (if `arr.shape` is odd) to
    the center-right pixel in an even shaped `newshape`.
    """
    if axes is None:
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        dS = newshape - currshape
        startind = (dS+1) // 2
        endind = dS - startind
        pad_width = list(zip(startind, endind))
    else:
        # only pad the axes that will be transformed
        pad_width = [(0, 0) for axis in arr.shape]
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for a, axis in enumerate(axes):
            dS = newshape[a] - arr.shape[axis]
            startind = (dS+1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)

    if isinstance(arr, torch.Tensor):
        # padding in torch.nn.functional expects padding to be specified from last- to first-axis, as a flattened tuple
        pad_width = tuple(y for x in pad_width[::-1] for y in x)
        return torch.nn.functional.pad(arr, pad_width, mode='constant')
    else:
        return np.pad(arr, pad_width, mode="constant")


def _get_fft_shape(img1, img2, padding=3, axes=None):
    """Return the fast fft shapes for each spatial axis

    Calculate the fast fft shape for each dimension in
    axes.
    """
    shape1 = np.asarray(img1.shape)
    shape2 = np.asarray(img2.shape)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        raise ValueError(msg.format(len(shape1), len(shape2)))
    # Set the combined shape based on the total dimensions
    if axes is None:
        shape = shape1 + shape2
    else:
        shape = np.zeros(len(axes))
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for n, ax in enumerate(axes):
            shape[n] = shape1[ax] + shape2[ax]

    shape += padding
    # Use the next fastest shape in each dimension
    shape = [fftpack.helper.next_fast_len(s) for s in shape]
    # autograd.numpy.fft does not currently work
    # if the last dimension is odd
    while shape[-1] % 2 != 0:
        shape[-1] += 1
        shape[-1] = fftpack.helper.next_fast_len(shape[-1])

    return shape


class Fourier(object):
    """An array that stores its Fourier Transform

    The `Fourier` class is used for images that will make
    use of their Fourier Transform multiple times.
    In order to prevent numerical artifacts the same image
    convolved with different images might require different
    padding, so the FFT for each different shape is stored
    in a dictionary.
    """
    def __init__(self, image, image_fft=None, axes=None):
        """Initialize the object

        Parameters
        ----------
        image: array
            The real space image.
        image_fft: dict
            A dictionary of {shape: fft_value} for which each different
            shape has a precalculated FFT.
        axes: int or tuple
            The dimension(s) of the array that will be transformed.
        """
        if image_fft is None:
            self._fft = {}
        else:
            self._fft = image_fft
        self._image = image
        if axes is None:
            axes = tuple(range(len(self.shape)))
        self._axes = axes

    @staticmethod
    def from_fft(image_fft, fft_shape, image_shape, axes=None):
        """Generate a new Fourier object from an FFT dictionary

        If the fft of an image has been generated but not its
        real space image (for example when creating a convolution kernel),
        this method can be called to create a new `Fourier` instance
        from the k-space representation.

        Parameters
        ----------
        image_fft: array
            The FFT of the image.
        fft_shape: tuple
            Shape of the image used to generate the FFT.
            This will be different than `image_fft.shape` if
            any of the dimensions are odd, since `np.fft.rfft`
            requires an even number of dimensions (for symmetry),
            so this tells `np.fft.irfft` how to go from
            complex k-space to real space.
        image_shape: tuple
            The shape of the image *before padding*.
            This will regenerate the image with the extra
            padding stripped.
        axes: int or tuple
            The dimension(s) of the array that will be transformed.

        Returns
        -------
        result: `Fourier`
            A `Fourier` object generated from the FFT.
        """
        if isinstance(image_fft, torch.Tensor):
            image_fft2 = torch.stack([image_fft, torch.zeros_like(image_fft)], dim=3)
            image = torch.irfft(image_fft2, len(axes), signal_sizes=fft_shape)  # 6x90x90 = 6x90x46x2
            image = batch_fftshift2d(image)
            print('debug')
        else:
            image = np.fft.irfftn(image_fft, fft_shape, axes=axes)  # 6x90x90 = (6x90x46, 90x90)
            # Shift the center of the image from the bottom left to the center
            image = np.fft.fftshift(image, axes=axes)  # 6x90x90

        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = _centered(image, image_shape)
        return Fourier(image, {tuple(fft_shape): image_fft}, axes)

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def axes(self):
        """The axes that are transormed"""
        return self._axes

    @property
    def shape(self):
        """The shape of the real space image"""
        return self.image.shape

    def fft(self, fft_shape):
        """The FFT of an image for a given `fft_shape`
        """
        fft_shape = tuple(fft_shape)
        # If this is the first time calling `fft` for this shape,
        # generate the FFT.
        if fft_shape not in self._fft:
            if len(fft_shape) != len(self.axes):
                msg = "fft_shape self.axes must have the same number of dimensions, got {0}, {1}"
                raise ValueError(msg.format(fft_shape, self.axes))
            image = _pad(self.image, fft_shape, self._axes)
            if isinstance(image, torch.Tensor):

                # torch ifft only works for complex inputs, and expects the Re/Imag components to be specified
                # separately in the last dimension
                image2 = torch.stack([image, torch.zeros_like(image)], dim=3)  # 6x75x100x2
                image = image.detach().numpy()
                correct2a = np.fft.rfftn(np.fft.ifftshift(image, self._axes), axes=self._axes)

                yy = batch_ifftshift2d(image2)  # 6x75x100x2
                correct2b = torch.rfft(yy[:,:,:,0], 2, onesided=True)

                assert np.allclose(np.real(correct2a), correct2b[:, :, :, 0].detach().numpy())
                assert np.allclose(np.imag(correct2a), correct2b[:, :, :, 1].detach().numpy())

                # All imaginary components of the stored results are close to 0
                # These results are used in division later on.
                # This is not a problem when dividing 2 complex numbers, but may pose a problem when diving
                # real and imaginary parts separately.
                # Ensure that the imaginary component is indeed close to 0 and make it close, but not quite, 0
                try:
                    assert(np.allclose(0, correct2b[:,:,:,1].detach().numpy()))
                except:
                    raise RuntimeError('NOOOOO')
                else:
                    correct2b = correct2b[:,:,:,0]

                self._fft[fft_shape] = correct2b
            else:
                correct1 = np.fft.ifftshift(image, self._axes)
                correct2 = np.fft.rfftn(correct1, axes=self._axes)
                self._fft[fft_shape] = correct2
        return self._fft[fft_shape]

    def __len__(self):
        return len(self.image)

    def normalize(self):
        """Normalize the image to sum to one
        """
        if self._axes is not None:
            indices = [slice(None)] * len(self.shape)
            for ax in self._axes:
                indices[ax] = None
        else:
            indices = [None] * len(self.shape)
        indices = tuple(indices)
        normalization = 1/self._image.sum(axis=self._axes)
        self._image *= normalization[indices]
        for shape, image_fft in self._fft.items():
            self._fft[shape] *= normalization[indices]

    def update_dtype(self, dtype):
        if self.image.dtype != dtype:
            self._image = self._image.astype(dtype)
            for shape in self._fft:
                self._fft[shape] = self._fft[shape].astype(dtype)

    def sum(self, axis=None):
        return self.image.sum(axis)

    def max(self, axis=None):
        # IAMHERE: Not correct, should return a vector
        # Doesn't torch support multiple dims??
        if isinstance(self.image, torch.Tensor):
            if axis is None:
                return self.image.max()
            else:
                # Torch doesn't support multiple axes in max; hence the loop
                im = self.image
                for ax in axis:
                    im = im.max(dim=ax, keepdim=True).values
                return torch.squeeze(im)
        else:
            return self.image.max(axis=axis)

    def __getitem__(self, index):
        # Make the index a tuple
        if not hasattr(index, "__getitem__"):
            index = tuple([index])

        # Axes that are removed from the shape of the new object
        removed = np.array([n for n, idx in enumerate(index)
                            if not isinstance(idx, slice) and idx is not None])
        # Axes that are added to the shape of the new object
        # (with `np.newaxis` or `None`)
        added = np.array([n for n, idx in enumerate(index) if idx is None])

        # Only propagate axes that are sliced or not indexed and
        # decrement them by the number of removed axes smaller than each one
        # and increment them by the number of added axes smaller than
        # each index.
        axes = tuple([ax-np.sum(removed < ax)+np.sum(added <= ax) for ax in self.axes if ax not in removed])

        # Create views into the fft transformed values, appropriately adjusting
        # the shapes for the new axes
        fft_kernels = {
            tuple([s for idx, s in enumerate(shape) if self.axes[idx] not in removed]): kernel[index]
            for shape, kernel in self._fft.items()
        }
        return Fourier(self.image[index], fft_kernels, axes=axes)


def _kspace_operation(image1, image2, padding, op, shape):
    """Combine two images in k-space using a given `operator`

    `image1` and `image2` are required to be `Fourier` objects and
    `op` should be an operator (either `operator.mul` for a convolution
    or `operator.truediv` for deconvolution). `shape` is the shape of the
    output image (`Fourier` instance).
    """
    if image1.axes != image2.axes:
        msg = "Both images must have the same axes, got {0} and {1}".format(image1.axes, image2.axes)
        raise Exception(msg)
    fft_shape = _get_fft_shape(image1.image, image2.image, padding, image1.axes)
    convolved_fft = op(image1.fft(fft_shape), image2.fft(fft_shape))
    convolved = Fourier.from_fft(convolved_fft, fft_shape, shape, image1.axes)
    return convolved


def match_psfs(psf1, psf2, padding=3):
    """Calculate the difference kernel between two psfs

    Parameters
    ----------
    psf1: `Fourier`
        `Fourier` object represeting the psf and it's FFT.
    psf2: `Fourier`
        `Fourier` object represeting the psf and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    """
    if psf1.shape[0] < psf2.shape[0]:
        shape = psf2.shape
    else:
        shape = psf1.shape
    return _kspace_operation(psf1, psf2, padding, operator.truediv, shape)


def convolve(image1, image2, padding=3, axes=None):
    """Convolve two images

    Parameters
    ----------
    image1: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    image2: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    """
    return _kspace_operation(image1, image2, padding, operator.mul, image1.shape)
