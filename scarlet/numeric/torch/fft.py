import torch
import importlib
import warnings

from scarlet.numeric.torch import intercepted, MyTensor


class Module:

    @staticmethod
    def roll_n(X, axis, n):
        axis = axis % X.ndim  # If a negative index was provided, convert to positive
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.ndim))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.ndim))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    @staticmethod
    @intercepted
    def irfftshift(x, axes=(-2, -1)):
        for ax in axes:
            x = Module.roll_n(x, axis=ax, n=x.shape[ax] // 2)
        return x

    @staticmethod
    @intercepted
    def ifftshift(x, axes=(-2, -1)):
        if x.is_real:
            warnings.warn('If calling ifftshift on real valued inputs, call irfftshift directly.')
            return Module.irfftshift(x, axes=axes)
        if tuple(axes) not in ((-2, -1), (0, 1)):
            raise AssertionError('Only axes (-2, -1) supported for now.')

        has_batch_dimension = x.ndim == 4

        if not has_batch_dimension:
            x = x[None, ...]

        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = Module.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = Module.roll_n(imag, axis=dim, n=imag.size(dim) // 2)

        if not has_batch_dimension:
            real = real.squeeze(dim=0)
            imag = imag.squeeze(dim=0)

        return torch.stack((real, imag), -1)

    @staticmethod
    @intercepted
    def rfftshift(x, axes=(-2, -1)):
        for ax in axes:
            n_shift = x.size(ax) // 2
            if x.size(ax) % 2 != 0:
                n_shift += 1
            x = Module.roll_n(x, axis=ax, n=n_shift)
        return x

    @staticmethod
    @intercepted
    def fftshift(x, axes=(-2, -1)):
        # if tuple(axes) != (-2, -1):
        #     ndim = x.ndim if x.is_real else x.ndim - 1
        #     if not (len(axes) == 2 and max(axes) == ndim-1):
        #         raise AssertionError('Only axes (-2, -1) supported for now.')
        if x.is_real:
            warnings.warn('If calling fftshift on real valued inputs, call rfftshift directly.')
            return Module.rfftshift(x, axes=axes)
        has_batch_dimension = x.ndim == 4

        if not has_batch_dimension:
            x = x[None, ...]

        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim) // 2
            if real.size(dim) % 2 != 0:
                n_shift += 1
            real = Module.roll_n(real, axis=dim, n=n_shift)
            imag = Module.roll_n(imag, axis=dim, n=n_shift)

        if not has_batch_dimension:
            real = real.squeeze(dim=0)
            imag = imag.squeeze(dim=0)

        return torch.stack((real, imag), -1)

    @staticmethod
    @intercepted
    def rfftn(x, axes=(-2, -1), onesided=True):
        # UGLY!!: A code path (sinc_shift?) is doing an rfft on a middle axis, which torch doesn't support!
        if tuple(axes) == (1,) and x.ndim == 3:
            retval = torch.rfft(x.permute(0, 2, 1), len(axes), onesided=onesided).permute(0, 2, 1, 3)
            retval.is_real = False
            return retval

        retval = torch.rfft(x, 2, onesided=onesided)
        retval.is_real = False
        return retval

    @staticmethod
    @intercepted
    def irfftn(x, s=None, axes=(-2, -1), onesided=True):
        assert not x.is_real, "x needs to be a complex array"

        # UGLY!!: Some code path is doing an irfftn on a middle axis, which torch doesn't support!
        if tuple(axes) == (2,) and x.ndim == 5:
            retval = torch.irfft(x.permute(0, 1, 3, 2, 4), len(axes), signal_sizes=s, onesided=onesided).permute(0, 1, 3, 2)
            retval.is_real = True
            return retval
        return torch.irfft(x, len(axes), signal_sizes=s, onesided=onesided)

    @staticmethod
    @intercepted
    def fft2(x):
        if x.is_real:
            retval = Module.rfftn(x, onesided=False)
            retval.is_real = False
        else:
            retval = torch.fft(x, signal_ndim=2)
            retval.is_real = False
        return retval

    @staticmethod
    @intercepted
    def ifft2(x):
        if x.is_real:
            retval = Module.irfftn(x, onesided=False)
            retval.is_real = False
        else:
            retval = torch.ifft(x, signal_ndim=2)
            retval.is_real = False
        return retval

    @staticmethod
    @intercepted
    def fftfreq(n, d=1.0):
        val = 1.0 / (n * d)
        results = torch.empty(n, dtype=int)
        N = (n - 1) // 2 + 1
        p1 = torch.arange(0, N, dtype=int)
        results[:N] = p1
        p2 = torch.arange(-(n // 2), 0, dtype=int)
        results[N:] = p2
        return results * val

    @staticmethod
    @intercepted
    def rfftfreq(n, d=1.0):
        val = 1.0 / (n * d)
        N = n // 2 + 1
        results = torch.arange(0, N, dtype=int)
        return results * val

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        func = getattr(torch, item)
        return intercepted(func)
