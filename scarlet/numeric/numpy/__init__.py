import numpy as np


class Module:

    asnumpy = staticmethod(lambda x: np.asarray(x))

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(np, item)
