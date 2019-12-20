# Import Packages and setup
import numpy as np
import scarlet
import scarlet.display

# Load the sample images
data = np.load("../data/hsc_cosmos_35.npz")
images = data["images"]
filters = data["filters"]
catalog = data["catalog"]
weights = 1/data["variance"]
psfs = scarlet.PSF(data["psfs"])

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


