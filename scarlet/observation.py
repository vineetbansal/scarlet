import autograd.numpy as np
from scipy import fftpack

from . import interpolation

from . import resampling

import logging

logger = logging.getLogger("scarlet.observation")


def _centered(arr, newshape):
    """Return the center newshape portion of the array.

    This function is used by `fft_convolve` to remove
    the zero padded region of the convolution.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


class Frame():
    """Spatial and spectral characteristics of the model

    Attributes
    ----------
    shape: tuple
        (channels, Ny, Nx) shape of the model image
    wcs: TBD
        World Coordinates
    psfs: array or tensor
        PSF in each band
    channels: list of hashable elements
        Names/identifiers of spectral channels
    dtype: `numpy.dtype`
        Dtype to represent the data.
    """
    def __init__(self, shape, wcs=None, psfs=None, channels=None, dtype=np.float32):
        assert len(shape) == 3
        self._shape = tuple(shape)
        self.wcs = wcs

        if psfs is None:
            logger.warning('No PSFs specified. Possible, but dangerous!')
        else:
            assert len(psfs) == 1 or len(psfs) == shape[0], 'PSFs need to have shape (1,Ny,Nx) for Blend and (B,Ny,Nx) for Observation'
            if not np.allclose(psfs.sum(axis=(1, 2)), 1):
                logger.warning('PSFs not normalized. Normalizing now..')
                psfs /= psfs.sum(axis=(1, 2))[:, None, None]

            if dtype != psfs.dtype:
                msg = "Dtypes of PSFs and Frame different. Casting PSFs to {}".format(dtype)
                logger.warning(msg)
                psfs = psfs.astype(dtype)

        self._psfs = psfs
        assert channels is None or len(channels) == shape[0]
        self.channels = channels
        self.dtype = dtype

    def combine_observations(self, observations):
        """Creates a common frame for two observations sets

        This function modifies the frame so that it can cover the full spatial and spectral range of several observations.
        The wcs of the frame is updated with the shape that encapsulates all observations.
        The reference pixel of the wcs is set to correspond with the center of the observation chosen to match the frame.

        Attributes
        ----------
        observations: array
            an array of observations that are to be matched in a commen frame.

        """

        assert self.wcs in [obs.wcs for obs in observations], \
            "The wcs of the frame should match the pwcs of at least one observation"

        corners_obs = []
        channels_frame = []
        for obs in observations:
            coord = np.where(np.zeros((obs.images.shape[1], obs.images.shape[2])) == 0)
            if obs.wcs.naxis == 3:
                Ra, Dec, l = obs.wcs.all_pix2world(coord[0], coord[1], 0, 0)
            elif obs.wcs.naxis == 2:
                Ra, Dec = obs.wcs.all_pix2world(coord[0], coord[1], 0)
            #Coordinates of the observation in the frame's frame
            x_frame, y_frame = self.wcs.all_world2pix(Ra, Dec, 0)
            corners_obs.append([y_frame.min(), y_frame.max(), x_frame.min(), x_frame.max()])
            #Channels of the model frame
            channels_frame += obs.channels

        #Corner of the frame that encapsulates all observations
        corners_min = np.min(corners_obs, axis=0)
        corners_max = np.max(corners_obs, axis=0)

        #updates the shape, channels and wcs of the newly formed frame
        self._shape = (len(channels_frame), np.int(corners_max[1]-corners_min[0]),
                       np.int(corners_max[3]-corners_max[2]))


        self.wcs.array_shape = (self._shape[1], self.shape[2])
        self.wcs.crpix = (np.int(self._shape[1]-corners_min[0]), np.int(self._shape[2]-corners_min[1]))

        self.channels = channels_frame

        print(self.wcs.array_shape, self.wcs.crpix, (np.int(self._shape[1]-corners_min[0]), np.int(self._shape[2]-corners_min[1])))
        [obs.match(self) for obs in observations]
        return self



    @property
    def C(self):
        """Number of channels in the model
        """
        return self._shape[0]

    @property
    def Ny(self):
        """Number of pixel in the y-direction
        """
        return self._shape[1]

    @property
    def Nx(self):
        """Number of pixels in the x-direction
        """
        return self._shape[2]

    @property
    def shape(self):
        """Shape of the model.
        """
        return self._shape

    @property
    def psfs(self):
        return self._psfs

    def get_pixel(self, sky_coord):
        """Get the pixel coordinate from a world coordinate
        If there is no WCS associated with the `Scene`,
        meaning the data frame and model frame are the same,
        then this just returns the `sky_coord`
        """
        if self.wcs is not None:
            if self.wcs.naxis == 3:
                coord = self.wcs.wcs_world2pix(sky_coord[0], sky_coord[1], 0, 0)
            elif self.wcs.naxis == 2:
                coord = self.wcs.wcs_world2pix(sky_coord[0], sky_coord[1], 0)
            else:
                raise ValueError("Invalid number of wcs dimensions: {0}".format(self.wcs.naxis))
            return (int(coord[0].item()), int(coord[1].item()))

        return tuple(int(coord) for coord in sky_coord)



class Observation():
    """Data and metadata for a single set of observations

    Attributes
    ----------
    images: array or tensor
        3D data cube (channels, Ny, Nx) of the image in each band.
    frame: a `scarlet.Frame` instance
        The spectral and spatial characteristics of these data
    weights: array or tensor
        Weight for each pixel in `images`.
        If a set of masks exists for the observations then
        then any masked pixels should have their `weight` set
        to zero.
    padding: int
        Number of pixels to pad each side with, in addition to
        half the width of the PSF, for FFTs. This is needed to
        prevent artifacts from the FFT.
    """

    def __init__(self, images, psfs=None, weights=None, wcs=None, channels=None, padding=10):
        """Create an Observation

        Parameters
        ---------
        images: array or tensor
            3D data cube (channels, Ny, Nx) of the image in each band.
        psfs: array or tensor
            PSF for each band in `images`.
        weights: array or tensor
            Weight for each pixel in `images`.
            If a set of masks exists for the observations then
            then any masked pixels should have their `weight` set
            to zero.
        wcs: TBD
            World Coordinate System associated with the images.
        channels: list of hashable elements
            Names/identifiers of spectral channels
        padding: int
            Number of pixels to pad each side with, in addition to
            half the width of the PSF, for FFTs. This is needed to
            prevent artifacts from the FFT.
        """
        self.frame = Frame(images.shape, wcs=wcs, psfs=psfs, channels=channels, dtype=images.dtype)
        self.wcs = wcs
        self.images = np.array(images)
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = 1

        self._padding = padding
        self.channels = channels

    def match(self, model_frame):
        """Match the frame of `Blend` to the frame of this observation.

        The method sets up the mappings in spectral and spatial coordinates,
        which includes a spatial selection, computing PSF difference kernels
        and filter transformations.

        Parameters
        ---------
        model_frame: a `scarlet.Frame` instance
            The frame of `Blend` to match

        Returns
        -------
        None
        """

        if self.frame.dtype != model_frame.dtype:
            msg = "Dtypes of model and observation different. Casting observation to {}".format(model_frame.dtype)
            logger.warning(msg)
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)
            if self.frame._psfs is not None:
                self.frame._psfs = self.frame._psfs.astype(model_frame.dtype)


        print('zezette')
        print(self.frame.wcs, model_frame.wcs)
        if self.frame.wcs != model_frame.wcs:
            #If frame and model are not in agreement, images is padded to the frame's shape
            assert np.all(self.frame.wcs.wcs.pc == model_frame.wcs.wcs.pc), \
                "frame and observation should have the same pixel scale and orientation"
            if self.frame.wcs.array_shape != model_frame.wcs.array_shape:
                print('zizi')
                #frame center:
                coordf = self.frame.wcs.wcs.crpix
                #Reference observation center
                coordr = model_frame.wcs.wcs.crpix
                shift = np.abs(np.int(coordf[0]-coordr[0])), np.abs(np.int(coordf[1]-coordr[1]))
                pad_size = (np.array(model_frame.wcs.array_shape[1:]) -
                            np.array(self.frame.wcs.array_shape[1:])).astype(int)//2

                print(pad_size, shift)
                self.images = np.pad(self.images, ((0,0), (pad_size[0]+shift[0], pad_size[0]-shift[0]),
                                                   (pad_size[1]+shift[1], pad_size[1]-shift[1])), 'constant')



        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            assert self.frame.channels is not None and model_frame.channels is not None
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax + 1)

        self._diff_kernels_fft = None
        if self.frame.psfs is not model_frame.psfs:
            assert self.frame.psfs is not None and model_frame.psfs is not None
            # First we setup the parameters for the model -> observation FFTs
            # Make the PSF stamp wider due to errors when matching PSFs
            psf_shape = np.array(self.frame.psfs.shape)
            psf_shape[1:] += self._padding
            conv_shape = np.array(model_frame.shape) + psf_shape - 1
            conv_shape[0] = model_frame.shape[0]

            # Choose the optimal shape for FFTPack DFT
            self._fftpack_shape = [fftpack.helper.next_fast_len(d) for d in conv_shape[1:]]

            # autograd.numpy.fft does not currently work
            # if the last dimension is odd
            while self._fftpack_shape[-1] % 2 != 0:
                _shape = self._fftpack_shape[-1] + 1
                self._fftpack_shape[-1] = fftpack.helper.next_fast_len(_shape)


            # Store the pre-fftpack optimization slices
            self.slices = tuple(([slice(s) for s in conv_shape]))

            # Now we setup the parameters for the psf -> kernel FFTs
            shape = np.array(self.frame.psfs.shape) + np.array(model_frame.psfs.shape) - 1
            shape[0] = np.array(self.frame.psfs.shape[0])

            #Fast fft shapes for kernels
            _fftpack_shape = [fftpack.helper.next_fast_len(d) for d in shape[1:]]

            while _fftpack_shape[-1] % 2 != 0:
                k_shape = np.array(_fftpack_shape) + 1
                _fftpack_shape = [fftpack.helper.next_fast_len(k_s) for k_s in k_shape]

            # fft of the target psf
            target_fft = np.fft.rfftn(model_frame.psfs, _fftpack_shape, axes=(1, 2))

            # fft of the observation's PSFs in each band
            _psf_fft = np.fft.rfftn(self.frame.psfs, _fftpack_shape, axes=(1, 2))

            # Diff kernel between observation and target psf in Fourrier
            kernels = np.fft.ifftshift(np.fft.irfftn(_psf_fft / target_fft, _fftpack_shape, axes=(1, 2)), axes=(1, 2))

            if kernels.shape[1] % 2 == 0 :
                kernels = kernels[:, 1:, 1:]

            kernels = _centered(kernels, psf_shape)

            self._diff_kernels_fft = np.fft.rfftn(kernels, self._fftpack_shape, axes=(1, 2))

        return self

    def _convolve(self, model):
        """Convolve the model in a single band
        """
        model_fft = np.fft.rfftn(model, self._fftpack_shape, axes=(1, 2))
        convolved = np.fft.irfftn(model_fft * self._diff_kernels_fft, self._fftpack_shape, axes=(1, 2))[self.slices]
        return _centered(convolved, model.shape)

    def render(self, model):
        """Convolve a model to the observation frame

        Parameters
        ----------
        model: array
            The model from `Blend`

        Returns
        -------
        model_: array
            The convolved `model` in the observation frame
        """
        model_ = model[self._band_slice, :, :]
        if self._diff_kernels_fft is not None:
            model_ = self._convolve(model_)

        return model_

    def get_loss(self, model):
        """Computes the loss/fidelity of a given model wrt to the observation

        Parameters
        ----------
        model: array
            The model from `Blend`

        Returns
        -------
        result: array
            Scalar tensor with the likelihood of the model
            given the image data
        """

        model = self.render(model)

        return 0.5 * np.sum((self.weights * (model - self.images)) ** 2)


class LowResObservation(Observation):

    def __init__(self, images, psfs=None, weights=None, wcs=None, channels=None, padding=3, perimeter = 'overlap'):

        self.perimeter = perimeter
        assert wcs is not None, "WCS is necessary for LowResObservation"
        assert psfs is not None, "PSFs are necessary for LowResObservation"
        assert perimeter in ['overlap', 'union'], "perimeter should be either 'overlap' or 'union'."

        super().__init__(images, psfs=psfs, weights=weights, wcs=wcs, channels=channels, padding=padding)

    def make_operator(self, shape, psf):
        '''Builds the resampling and convolution operator
        Builds the matrix that expresses the linear operation of resampling a function evaluated on a grid with coordinates
        'coord_lr' to a grid with shape 'shape', and convolving by a kernel p
        Parameters
        ------
        shape: tuple
            shape of the high resolution scene
        coord_lr: array
            coordinates of the overlapping pixels from the low resolution grid in the high resolution grid frame.
        p: array
            convolution kernel (PSF)
        Returns
        -------
        mat: array
            the convolution-resampling matrix
        '''
        B, Ny, Nx = shape
        y_lr, x_lr = self._coord_lr
        mat = np.zeros((Ny * Nx, x_lr.size))


        import scipy.signal as scp

        for m in range(np.size(x_lr)):
            mat[:, m] = scp.fftconvolve(self._ker[m], psf, mode='same').flatten()
            mat[:, m] /= np.sum(mat[:,m])
        return mat

    def match_psfs(self, psf_hr, wcs_hr):
        '''psf matching between different dataset
        Matches PSFS at different resolutions by interpolating psf_lr on the same grid as psf_hr
        Parameters
        ----------
        psf_hr: array
            centered psf of the high resolution scene
        psf_lr: array
            centered psf of the low resolution scene
        wcs_hr: WCS object
            wcs of the high resolution scene
        wcs_lr: WCS object
            wcs of the low resolution scene
        Returns
        -------
        psf_match_hr: array
            high rresolution psf at mactching size
        psf_match_lr: array
            low resolution psf at matching size and resolution
        '''

        psf_lr = self.frame.psfs
        wcs_lr = self.frame.wcs

        ny_hr, nx_hr = psf_hr.shape
        npsf, ny_lr, nx_lr = psf_lr.shape

        # Createsa wcs for psfs centered around the frame center
        psf_wcs_hr = wcs_hr.deepcopy()
        psf_wcs_lr = wcs_lr.deepcopy()

        if psf_wcs_hr.naxis == 2:
            psf_wcs_hr.wcs.crval = 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2.
        elif psf_wcs_hr.naxis == 3:
            psf_wcs_hr.wcs.crval = 0., 0., 0.
            psf_wcs_hr.wcs.crpix = ny_hr / 2., nx_hr / 2., 0.
        if psf_wcs_lr.naxis == 2:
            psf_wcs_lr.wcs.crval = 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2.
        elif psf_wcs_lr.naxis == 3:
            psf_wcs_lr.wcs.crval = 0., 0., 0.
            psf_wcs_lr.wcs.crpix = ny_lr / 2., nx_lr / 2., 0

        mask, p_lr, p_hr = resampling.match_patches(psf_hr.shape, psf_lr.data.shape[1:], psf_wcs_hr, psf_wcs_lr)
        assert mask.sum() > 0, 'no overlap found between frame and observation. Please check WCSes.'

        cmask = np.where(mask == 1)
        n_p = np.int((np.size(cmask[0])) ** 0.5)

        psf_match_lr = interpolation.sinc_interp(cmask, p_hr[::-1],
                                                 psf_lr.reshape(npsf, ny_lr * nx_lr)).reshape(npsf, n_p, n_p)

        psf_match_hr = psf_hr[np.int((ny_hr - n_p) / 2):np.int((ny_hr + n_p) / 2),
                       np.int((nx_hr - n_p) / 2):np.int((nx_hr + n_p) / 2)]

        psf_match_hr /= np.max(psf_match_hr)
        psf_match_lr /= np.max(psf_match_lr)
        return psf_match_hr[np.newaxis, :], psf_match_lr

    def match(self, model_frame):

        if self.frame.dtype != model_frame.dtype:
            msg = "Dtypes of model and observation different. Casting observation to {}".format(model_frame.dtype)
            logger.warning(msg)
            self.frame.dtype = model_frame.dtype
            self.images = self.images.astype(model_frame.dtype)
            if type(self.weights) is np.ndarray:
                self.weights = self.weights.astype(model_frame.dtype)
            if self.frame._psfs is not None:
                self.frame._psfs = self.frame._psfs.astype(model_frame.dtype)

        #  channels of model that are represented in this observation
        self._band_slice = slice(None)
        if self.frame.channels is not model_frame.channels:
            bmin = model_frame.channels.index(self.frame.channels[0])
            bmax = model_frame.channels.index(self.frame.channels[-1])
            self._band_slice = slice(bmin, bmax+1)

        # Get pixel coordinates in each frame.
        mask, coord_lr, coord_hr = resampling.match_patches(model_frame.shape, self.frame.shape, model_frame.wcs,
                                                                self.frame.wcs, perimeter = self.perimeter)

        self._coord_lr = coord_lr
        self._coord_hr = coord_hr
        self._mask = mask

        # Compute diff kernel at hr

        whr = model_frame.wcs

        # Reference PSF
        _target = model_frame.psfs[0, :, :]
        _shape = model_frame.shape


        _fftpack_shape = [fftpack.helper.next_fast_len(d) for d in _target.shape]

        while _fftpack_shape[-1] % 2 != 0:
            k_shape = np.array(_fftpack_shape) + 1
            _fftpack_shape = [fftpack.helper.next_fast_len(k_s) for k_s in k_shape]

        # Interpolation kernel for resampling
        self._ker = resampling.conv2D_fft(_shape, self._coord_hr)
        # Computes spatially matching observation and target psfs. The observation psf is also resampled to the model frame resolution
        new_target, observed_psf = self.match_psfs(_target, whr)
        target_fft = np.fft.rfftn(new_target[0], _fftpack_shape)
        sel = target_fft == 0
        observed_fft = np.fft.rfftn(observed_psf, _fftpack_shape, axes=(1, 2))

        # Computes the diff kernel in Fourier
        kernel_fft = observed_fft / target_fft
        kernel_fft[:,sel] = 0
        kernel = np.fft.irfftn(kernel_fft, _fftpack_shape, axes = (1,2))
        kernel = np.fft.ifftshift(kernel, axes = (1,2))

        if kernel.shape[1] % 2 == 0:
            kernel = kernel[:, 1:, 1:]

        kernel = _centered(kernel, observed_psf.shape)
        diff_psf = kernel / kernel.sum()

        # Computes the resampling/convolution matrix
        resconv_op = []
        for dpsf in diff_psf:
            resconv_op.append(self.make_operator(_shape, dpsf))

        self._resconv_op = np.array(resconv_op, dtype=self.frame.dtype)

        return self

    @property
    def matching_mask(self):
        return self._mask

    def _render(self, model):
        """Resample and convolve a model in the observation frame
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        model_: array
            The convolved and resampled `model` in the observation frame.
        """
        model_ = model[self._band_slice,:,:]
        model_ = np.array([np.dot(model_[c].flatten(), self._resconv_op[c]) for c in range(self.frame.C)], dtype=self.frame.dtype)

        return model_

    def render(self, model):
        """Resample and convolve a model in the observation frame
        Parameters
        ----------
        model: array
            The model in some other data frame.
        Returns
        -------
        model_: array
            The convolved and resampled `model` in the observation frame.
        """
        img = np.zeros(self.frame.shape)
        img[:, self._coord_lr[0], self._coord_lr[1]] = self._render(model)
        return img

    def get_loss(self, model):
        """Computes the loss/fidelity of a given model wrt to the observation
        Parameters
        ----------
        model: array
            A model from `Blend`
        Returns
        -------
        loss: float
            Loss of the model
        """

        model_ = self._render(model)

        return 0.5 * np.sum((self.weights * (
                model_ - self.images[:, self._coord_lr[0].astype(int), self._coord_lr[1].astype(int)])) ** 2)
