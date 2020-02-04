from functools import partial

from scarlet.numeric import np
import scarlet
import scarlet.fft as fft
from scarlet.numeric import assert_array_equal, assert_almost_equal


class TestCentering(object):
    """Test the centering and padding algorithms"""
    def test_shift(self):
        """Test that padding and fft shift/unshift are consistent"""
        a0 = np.ones((1, 1))
        a_pad = fft._pad(a0, (5, 4))
        truth = [[0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0]]
        assert_array_equal(a_pad, truth)

        a_shift = np.fft.ifftshift(a_pad)
        truth = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0]]
        assert_array_equal(a_shift, truth)

        # Shifting back should give us a_pad again
        a_shift_back = np.fft.fftshift(a_shift)
        assert_array_equal(a_shift_back, a_pad)

    def test_center(self):
        """Test that _centered method is compatible with shift/unshift"""
        shape = (5, 2)
        a0 = np.arange(10).reshape(shape)
        a_pad = fft._pad(a0, (9, 11))
        truth = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_equal(a_pad, truth)

        a_shift = np.fft.ifftshift(a_pad)
        truth = [[4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_equal(a_shift, truth)

        # Shifting back should give us a_pad again
        a_shift_back = np.fft.fftshift(a_shift)
        assert_array_equal(a_shift_back, a_pad)

        # _centered should undo the padding, returning the original array
        a_final = fft._centered(a_pad, shape)
        assert_array_equal(a_final, a0)


class TestFourier(object):

    def get_psfs(self, shape, sigmas):

        shape_ = (None, *shape)
        psfs = np.array([
            scarlet.PSF(partial(scarlet.psf.gaussian, sigma=s), shape=shape_).image[0]
            for s in sigmas
        ])

        psfs /= psfs.sum(axis=(1, 2))[:, None, None]
        return psfs


    """Test the Fourier object"""
    def test_2D_psf_matching(self):
        """Test matching two 2D psfs
        """
        # Narrow PSF
        shape = (41,41)
        psf1 = scarlet.fft.Fourier(self.get_psfs(shape, [1])[0])
        # Wide PSF
        psf2 = scarlet.fft.Fourier(self.get_psfs(shape, [2])[0])

        # Test narrow to wide
        kernel_1to2 = fft.match_psfs(psf2, psf1)
        img2 = fft.convolve(psf1, kernel_1to2)
        assert_almost_equal(img2.image, psf2.image)

        # Test wide to narrow
        kernel_2to1 = fft.match_psfs(psf1, psf2)
        img1 = fft.convolve(psf2, kernel_2to1)
        assert_almost_equal(img1.image, psf1.image)

    def test_multiband_psf_matching(self):
        """Test matching two PSFs with a spectral dimension
        """
        # Narrow PSF
        shape = (41,41)
        psf1 = scarlet.fft.Fourier(self.get_psfs(shape, [1]))
        # Wide PSF
        psf2 = scarlet.fft.Fourier(self.get_psfs(shape, [1,2,3]))

        # Nawrrow to wide
        kernel_1to2 = fft.match_psfs(psf2, psf1)
        image = fft.convolve(kernel_1to2, psf1)
        assert_almost_equal(psf2.image, image.image)

        # Wide to narrow
        kernel_2to1 = fft.match_psfs(psf1, psf2)
        image = fft.convolve(kernel_2to1, psf2).image

        for img in image:
            assert_almost_equal(img, psf1.image[0])
