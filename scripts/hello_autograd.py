import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need


def tanh(x):
    x[0, 0] = 0
    y = np.exp(-2.0 * x)
    return np.sum((1.0 - y) / (1.0 + y))


if __name__ == '__main__':
    x = np.random.rand(10, 10)
    grad_tanh = grad(tanh)
    print(grad_tanh(x))
