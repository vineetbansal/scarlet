# Import Packages and setup
from scarlet.numeric import np
import numpy
import scarlet
import scarlet.display
import astropy.io.fits as fits
from astropy.wcs import WCS
from scarlet.display import AsinhMapping


def fix_byte_order(x):
    import sys
    endianness = {
        '>': 'big',
        '<': 'little',
        '=': sys.byteorder,
        '|': 'not applicable',
    }
    if endianness[x.dtype.byteorder] != sys.byteorder:
        return x.byteswap().newbyteorder()
    else:
        return x

import matplotlib
import matplotlib.pyplot as plt
# use a better colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='inferno')
matplotlib.rc('image', interpolation='none')

# Load the HSC image data
obs_hdu = fits.open('../data/test_resampling/Cut_HSC.fits')
data_hsc = obs_hdu[0].data
data_hsc = np.array(fix_byte_order(data_hsc))
wcs_hsc = WCS(obs_hdu[0].header)
channels_hsc = ['g','r','i','z','y']

# Load the HSC PSF data
psf_hsc = fits.open('../data/test_resampling/PSF_HSC.fits')[0].data
psf_hsc = np.array(fix_byte_order(psf_hsc))
Np1, Np2 = psf_hsc[0].shape
psf_hsc = scarlet.PSF(psf_hsc)

# Load the HST image data
hst_hdu = fits.open('../data/test_resampling/Cut_HST.fits')
data_hst = hst_hdu[0].data
data_hst = np.array(fix_byte_order(data_hst))
wcs_hst = WCS(hst_hdu[0].header)
channels_hst = ['F814W']

# apply wcs correction
#wcs_hst.wcs.crval -= 2.4750118475607095e-05*np.array([0,1])
# Load the HST PSF data
psf_hst = fits.open('../data/test_resampling/PSF_HST.fits')[0].data
psf_hst = np.array(fix_byte_order(psf_hst))
psf_hst = psf_hst[None,:,:]
psf_hst = scarlet.PSF(psf_hst)

# Scale the HST data
n1,n2 = data_hst.shape
data_hst = data_hst.reshape(1, n1, n2)

r, N1, N2 = data_hsc.shape

import sep


def makeCatalog(img, lvl=4):
    img = np.asnumpy(img)
    if img.ndim == 3:
        detect = img.mean(axis=0)  # simple average for detection
    else:
        detect = img

    bkg = sep.Background(np.asnumpy(detect))
    catalog = sep.extract(detect, lvl, err=bkg.globalrms)
    if img.ndim == 3:
        bg_rms = numpy.array([sep.Background(band).globalrms for band in img])
    else:
        bg_rms = sep.Background(detect).globalrms
    return catalog, np.array(bg_rms)


catalog_hst, bg_rms_hst = makeCatalog(data_hst, 4)
catalog_hsc, bg_rms_hsc = makeCatalog(data_hsc, 4)

weights_hst = np.ones_like(data_hst) / (bg_rms_hst ** 2)[:, None, None]
weights_hsc = np.ones_like(data_hsc) / (bg_rms_hsc ** 2)[:, None, None]

# Create a color mapping for the HSC image
hsc_norm = AsinhMapping(minimum=-1, stretch=2, Q=10)
hst_norm = AsinhMapping(minimum=-1, stretch=10, Q=5)

# Get the source coordinates from the HST catalog
xo,yo = catalog_hst['x'], catalog_hst['y']
xi,yi = catalog_hsc['x'], catalog_hsc['y']
# Convert the HST coordinates to the HSC WCS
ra, dec = wcs_hst.wcs_pix2world(yo,xo,0)
Yo,Xo, l = wcs_hsc.wcs_world2pix(ra, dec, 0, 0)
# Map the HSC image to RGB
img_rgb = scarlet.display.img_to_rgb(data_hsc, norm=hsc_norm)
# Apply Asinh to the HST data
hst_img = scarlet.display.img_to_rgb(data_hst, norm=hst_norm)

plt.subplot(121)
plt.imshow(img_rgb)
plt.plot(Xo,Yo, 'o')
plt.plot(xi,yi, 'o')
plt.subplot(122)
plt.imshow(hst_img)
plt.show()

# Initialize the frame using the HST PSF and WCS
channels = channels_hsc + channels_hst
shape = (len(channels), n1,n2)
frame = scarlet.Frame(shape, wcs=wcs_hst, psfs=psf_hst, channels=channels)

# define two observation packages and match to frame
obs_hst = scarlet.Observation(data_hst, wcs=wcs_hst, psfs=psf_hst, channels=channels_hst, weights=weights_hst).match(frame)
obs_hsc = scarlet.LowResObservation(data_hsc,  wcs=wcs_hsc, psfs=psf_hsc, channels=channels_hsc, weights=weights_hsc)
obs_hsc.match(frame)

# Keep the order of the observations consistent with the `channels` parameter
# This implementation is a bit of a hack and will be refined in the future
obs = [obs_hsc, obs_hst]

sources = [
    scarlet.ExtendedSource(frame, (ra[i], dec[i]), obs,
                           symmetric=False,
                           monotonic=True,
                           obs_idx=1)
    for i in range(ra.size)
]

blend = scarlet.Blend(sources, obs)

# Load the model and calculate the residual
model = blend.get_model()

obs_hsc.render(model)
model_lr = obs_hsc.render(model)
init_rgb = scarlet.display.img_to_rgb(model[:-1], norm=hsc_norm)
init_rgb_lr = scarlet.display.img_to_rgb(model_lr, norm=hsc_norm)
residual_lr = data_hsc - model_lr
# Trim the bottom source not part of the blend from the image
residual_lr_rgb = scarlet.display.img_to_rgb(residual_lr[:,:-5])

# Get the HR residual
residual_hr = (data_hst - obs_hst.render(model))[0]
vmax = residual_hr.max()

plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(img_rgb)
plt.title("HSC data")
plt.subplot(235)
plt.imshow(init_rgb)
plt.title("HighRes Model")
plt.subplot(232)
plt.imshow(init_rgb_lr)
plt.title("LowRes Model")
plt.subplot(236)
plt.imshow(residual_hr, cmap="seismic", vmin=-vmax, vmax=vmax)
plt.colorbar(fraction=.045)
plt.title("HST residual")
plt.subplot(233)
plt.imshow(residual_lr_rgb)
plt.title("HSC residual")
plt.subplot(234)
plt.imshow(hst_img)
plt.colorbar(fraction=.045)
plt.title('HST data')
plt.show()

blend.fit(10)
print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
plt.plot(-np.array(blend.loss))
plt.xlabel('Iteration')
plt.ylabel('log-Likelihood')

model = blend.get_model()
model_hr = obs_hst.render(model)
model_lr = obs_hsc.render(model)

rgb = scarlet.display.img_to_rgb(model[:-1], norm=hsc_norm)
rgb_lr = scarlet.display.img_to_rgb(model_lr, norm=hsc_norm)
residual_lr = data_hsc - model_lr

# Trim the bottom source not part of the blend from the image
residual_lr_rgb = scarlet.display.img_to_rgb(residual_lr[:,:-5], norm=hsc_norm)

# Get the HR residual
residual_hr = (data_hst - model_hr)[0]
vmax = residual_hr.max()

plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(img_rgb)
plt.title("HSC data")
plt.subplot(235)
plt.imshow(rgb)
plt.title("HST Model")
plt.subplot(232)
plt.imshow(rgb_lr)
plt.title("HSC Model")
plt.subplot(236)
plt.imshow(residual_hr, cmap="seismic", vmin=-vmax, vmax=vmax)
plt.colorbar(fraction=.045)
plt.title("HST residual")
plt.subplot(233)
plt.imshow(residual_lr_rgb)
plt.title("HSC residual")
plt.subplot(234)
plt.imshow(hst_img)
plt.title('HST data')
plt.show()

has_truth = False
axes = 2

for k, src in enumerate(blend.sources):
    print('source number ', k)
    # Get the model for a single source
    model = src.get_model()
    model_lr = obs_hsc.render(model)

    # Display the low resolution image and residuals
    img_lr_rgb = scarlet.display.img_to_rgb(model_lr, norm=hsc_norm)
    res = data_hsc - model_lr
    res_rgb = scarlet.display.img_to_rgb(res, norm=hsc_norm)

    plt.figure(figsize=(15, 15))

    plt.subplot(331)
    plt.imshow(img_rgb)
    plt.plot(Xo[k], Yo[k], 'x', markersize=10)
    plt.title("HSC Data")
    plt.subplot(332)
    plt.imshow(img_lr_rgb)
    plt.title("LR Model")
    plt.subplot(333)
    plt.imshow(res_rgb)
    plt.title("LR Data - Model")

    img_hr = obs_hst.render(model)
    res = data_hst - img_hr[-1]
    vmax = res.max()

    plt.subplot(334)
    plt.imshow(data_hst[0], cmap='gist_stern')
    plt.plot(xo[k], yo[k], 'o', markersize=5)
    plt.title("HST Data")
    plt.subplot(335)
    plt.imshow(img_hr[-1])
    plt.title("HR Model")
    plt.subplot(336)
    plt.imshow(res[0], cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.title("HR Data - Model")

    plt.show()
