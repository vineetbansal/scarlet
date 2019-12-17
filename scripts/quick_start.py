import logging
import numpy as np
import scarlet
import scarlet.display

import matplotlib
import matplotlib.pyplot as plt

# use a better colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='inferno')
matplotlib.rc('image', interpolation='none')


if __name__ == '__main__':

    # Load the sample images
    data = np.load("../data/hsc_cosmos_35.npz")

    # 5 x 58 x 48
    images = data["images"]
    # ['g' 'r' 'i' 'z' 'y']
    filters = data["filters"]

    # point spread functions - 5 x 43 x 43
    # these are the characteristics of the telescope for each of the channels we're considering
    psfs = data["psfs"]
    # normalize to unity
    psfs /= psfs.sum(axis=(1, 2))[:, None, None]

    # 7 2-tuples of x,y coordinates
    catalog = data["catalog"]

    # Estimate of the background noise level, here we're lazy
    bg_rms = np.ones((len(psfs),), dtype=images.dtype) * .1

    from astropy.visualization.lupton_rgb import AsinhMapping

    # 58 x 48 x 3
    img_rgb = scarlet.display.img_to_rgb(images, norm=AsinhMapping(minimum=0, stretch=0.1, Q=10))
    plt.imshow(img_rgb)

    # Mark all of the sources from the detection cataog
    for k, src in enumerate(catalog):
        plt.text(src["x"], src["y"], str(k), color="red")

    plt.show()

    # 43 x 43
    # Generate a single point spread function image of size 43 x 43
    model_psf = scarlet.psf.generate_psf_image(scarlet.psf.gaussian, psfs.shape[1:])
    # 43 x 43 => 1 x 43 x 43
    model_psf = model_psf[None]

    # A Frame in scarlet is a description of the hyperspectral cube of the model or the observations
    # We must also provide an image cube of the PSF model (one image per channel)
    # Here we just specify a single PSF image to be used on all channels
    frame = scarlet.Frame(images.shape, psfs=model_psf)

    # An Observation is essentially a Frame with a data portion
    observation = scarlet.Observation(images, psfs=psfs).match(frame)

    sources = []
    for src in catalog:
        try:
            new_source = scarlet.ExtendedSource(frame, (src['y'], src['x']), observation, bg_rms)
        except scarlet.SourceInitError:
            try:
                new_source = scarlet.PointSource(frame, (src['y'], src['x']), observation)
            except scarlet.SourceInitError:
                print("Could not initialze source at {0}".format((src['y'], src['x'])))
                continue
        sources.append(new_source)

    blend = scarlet.Blend(sources, observation)
    blend.fit(200, e_rel=1e-3)
    print("scarlet ran for {0} iterations to MSE = {1}".format(len(blend.mse), blend.mse[-1]))
    plt.semilogy(blend.mse)

    plt.show()

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

    for k,component in enumerate(blend.components):
        y,x = component.pixel_center
        ax[0].text(x, y, k, color="w")
        ax[1].text(x, y, k, color="w")
    plt.show()

    # Set the stretch based on the model
    stretch = .3
    Q = 10
    norm = AsinhMapping(minimum=0, stretch=stretch, Q=Q)

    for k,src in enumerate(blend.components):
        # Get the model for a single source
        model = src.get_model()
        model_ = observation.render(model)

        # Convert observation and models to RGB
        img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
        model_rgb = scarlet.display.img_to_rgb(model, norm=norm)
        model_rgb_ = scarlet.display.img_to_rgb(model_, norm=norm)

        # Set the figure size
        ratio = src.shape[2]/src.shape[1]
        fig_height = 3*src.shape[1]/20
        fig_width = max(2*fig_height*ratio,2)
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Generate and show the figure
        ax = [fig.add_subplot(1,3,n+1) for n in range(3)]
        ax[0].imshow(img_rgb)
        ax[0].set_title("Data")
        ax[1].imshow(model_rgb_)
        ax[1].set_title("Observed model {0}".format(k))
        ax[2].imshow(model_rgb)
        ax[2].set_title("Model {0}".format(k))
        # Mark the source in the data image
        ax[0].plot(src.pixel_center[1], src.pixel_center[0], "rx", mew=2, ms=10)
        ax[1].plot(src.pixel_center[1], src.pixel_center[0], "rx", mew=2, ms=10)
        ax[2].plot(src.pixel_center[1], src.pixel_center[0], "rx", mew=2, ms=10)
        plt.show()
