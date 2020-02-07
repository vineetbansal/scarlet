# Import Packages and setup
import numpy
from scarlet.numeric import np
from scarlet.numeric import assert_almost_equal
import scarlet
import scarlet.display

import matplotlib
import matplotlib.pyplot as plt
# use a good colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='inferno', interpolation='none', origin='lower')

# ----------------------------------------

# Load the sample images
data = numpy.load("../data/hsc_cosmos_35.npz")
images = np.array(data["images"])
filters = data["filters"]
catalog = data["catalog"]
weights = np.array(1/data["variance"])
psfs = scarlet.PSF(np.array(data["psfs"]))

# ----------------------------------------

from scarlet.display import AsinhMapping

stretch = 0.2
Q = 10
norm = AsinhMapping(minimum=0, stretch=stretch, Q=Q)
img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
plt.imshow(img_rgb)

# Mark all of the sources from the detection cataog
for k, src in enumerate(catalog):
    plt.text(src["x"], src["y"], str(k), color="red")

# ----------------------------------------

from functools import partial
model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))

# ----------------------------------------

model_frame = scarlet.Frame(
    images.shape,
    psfs=model_psf,
    channels=filters)

observation = scarlet.Observation(
    images,
    psfs=psfs,
    weights=weights,
    channels=filters).match(model_frame)

# ----------------------------------------

sources = []
for k,src in enumerate(catalog):
    if k == 0:
        new_source = scarlet.PointSource(model_frame, (src['y'], src['x']), observation)
    elif k == 1:
        new_source = scarlet.MultiComponentSource(model_frame, (src['y'], src['x']), observation)
    else:
        new_source = scarlet.ExtendedSource(model_frame, (src['y'], src['x']), observation)
    sources.append(new_source)

# ----------------------------------------

blend = scarlet.Blend(sources, observation)
blend.fit(200)
print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
plt.plot(-np.array(blend.loss))
plt.xlabel('Iteration')
plt.ylabel('log-Likelihood')

# ----------------------------------------

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

# ----------------------------------------

scarlet.display.show_sources(sources,
                             norm=norm,
                             observation=observation,
                             show_rendered=True,
                             show_observed=True)

# ----------------------------------------

print ("----------------- {}".format(filters))
for k, src in enumerate(sources):
    print ("Source {}, Fluxes: {}".format(k, scarlet.measure.flux(src)))

