import torch


def assert_array_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    return testing.assert_array_equal(a, b, **kwargs)


def assert_almost_equal(a, b, **kwargs):
    import numpy.testing as testing
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()

    if 'decimal' not in kwargs:
        decimal = 5
    else:
        decimal = kwargs.pop('decimal')
    return testing.assert_almost_equal(a, b, decimal=decimal, **kwargs)
