USE_TORCH = False

if USE_TORCH:
    from .torch import Module, assert_array_equal, assert_almost_equal
    from .torch import operator as operator

else:
    from .numpy import Module
    import operator
    import numpy.testing as testing
    assert_array_equal = testing.assert_array_equal
    assert_almost_equal = testing.assert_almost_equal

np = Module()

