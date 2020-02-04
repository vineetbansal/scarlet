from scarlet.numeric import USE_TORCH

if USE_TORCH:
    from .parameter_torch import Parameter, relative_step
else:
    from .parameter import Parameter, relative_step
