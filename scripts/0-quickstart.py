# Import Packages and setup
import numpy as np
import scarlet
import scarlet.display

import logging
logger = logging.getLogger('scarlet')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger('proxmin')
logger.setLevel(logging.DEBUG)

import matplotlib
import matplotlib.pyplot as plt
# use a good colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='inferno', interpolation='none')

# Load the sample images
data = np.load("../data/hsc_cosmos_35.npz")
images = data["images"]
filters = data["filters"]
catalog = data["catalog"]
weights = 1/data["variance"]
psfs = scarlet.PSF(data["psfs"])

from scarlet.display import AsinhMapping

stretch = 0.2
Q = 10
norm = AsinhMapping(minimum=0, stretch=stretch, Q=Q)
img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
plt.imshow(img_rgb)

# Mark all of the sources from the detection cataog
for k, src in enumerate(catalog):
    plt.text(src["x"], src["y"], str(k), color="red")

from functools import partial
model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))

model_frame = scarlet.Frame(
    images.shape,
    psf=model_psf,
    channels=filters)

observation = scarlet.Observation(
    images,
    psf=psfs,
    weights=weights,
    channels=filters).match(model_frame)

sources = []
for k,src in enumerate(catalog):
    if k == 0:
        new_source = scarlet.PointSource(model_frame, (src['y'], src['x']), observation)
    elif k == 1:
        new_source = scarlet.MultiComponentSource(model_frame, (src['y'], src['x']), observation, symmetric=False, monotonic=True, thresh=5)
    else:
        new_source = scarlet.ExtendedSource(model_frame, (src['y'], src['x']), observation, symmetric=False, monotonic=True, thresh=5)
    sources.append(new_source)

blend = scarlet.Blend(sources, observation)
blend.fit(200)
print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
plt.plot(-np.array(blend.loss))
plt.xlabel('Iteration')
plt.ylabel('log-Likelihood')

# Load the model and calculate the residual
model = blend.get_model()
model_ = observation.render(model)
residual = images-model_

# Create RGB images
model_rgb = scarlet.display.img_to_rgb(model_, norm=norm)
residual_rgb = scarlet.display.img_to_rgb(residual)

# Show the data, model, and residual
fig = plt.figure(figsize=(15,5))
ax = [fig.add_subplot(1,3,n+1) for n in range(3)]
ax[0].imshow(img_rgb)
ax[0].set_title("Data")
ax[1].imshow(model_rgb)
ax[1].set_title("Model")
ax[2].imshow(residual_rgb)
ax[2].set_title("Residual")

for k,component in enumerate(blend):
    y,x = component.center
    ax[0].text(x, y, k, color="w")
    ax[1].text(x, y, k, color="w")
    ax[2].text(x, y, k, color="w")
plt.show()

scarlet.display.show_sources(sources,
                             norm=norm,
                             observation=observation,
                             show_rendered=True,
                             show_observed=True)

print ("----------------- {}".format(filters))
for k, src in enumerate(sources):
    print ("Source {}, Fluxes: {}".format(k, scarlet.measure.flux(src)))

import pickle
fp = open("hsc_cosmos_35.sca", "wb")
pickle.dump(sources, fp)
fp.close()

fp = open("hsc_cosmos_35.sca", "rb")
sources_ = pickle.load(fp)
fp.close()

scarlet.display.show_scene(sources_, norm=norm)

# first freeze existing sources: they are not updated during fit
for src in sources_:
    src.freeze()

# add two sources at their approximate locations
yx = (14., 44.)
new_source = scarlet.ExtendedSource(src.frame, yx, observation, shifting=True, symmetric=False, monotonic=True, thresh=5)
sources_.append(new_source)
yx = (43., 12.)
new_source = scarlet.ExtendedSource(src.frame, yx, observation, shifting=True, symmetric=False, monotonic=True, thresh=5)
sources_.append(new_source)

# generate a new Blend instance
blend_ = scarlet.Blend(sources_, observation)
# fit only new sources
blend_.fit(200)

# joint fit: fit all sources
blend_.unfreeze()
blend_.fit(200)

# show convergence of logL
print("scarlet ran for {0} iterations to logL = {1}".format(len(blend_.loss), -blend_.loss[-1]))
plt.plot(-np.array(blend.loss), 'k--', label='7 sources')
plt.plot(-np.array(blend_.loss), label='7+2 sources')
plt.xlabel('Iteration')
plt.ylabel('log-Likelihood')
plt.legend()

scarlet.display.show_scene(sources_,
                           norm=norm,
                           observation=observation,
                           show_rendered=True,
                           show_observed=True,
                           show_residual=True)

# minimal regression testing (hidden in sphinx)
np.testing.assert_almost_equal(blend_.loss[-1], -31206.701431446825, decimal=3)

