import numpy as np
import mindspore
from mindspore.common.initializer import Initializer

class UniformInitializer(Initializer):
    def __init__(self, input_dim):
        super(UniformInitializer, self).__init__()
        k = 1.0 / input_dim
        self.limit = np.sqrt(k).astype(np.float32)

    def _initialize(self, arr):
        data = np.random.uniform(-self.limit, self.limit, arr.shape).astype(np.float32)
        arr[:] = data[:]

