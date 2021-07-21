import numpy as np


class Cen:
    def __init__(self, shape, dim_scale=None, dims=None):
        self.counter = np.zeros(shape, dtype=np.int32)
        self.dim_scale = dim_scale if dim_scale is not None else [1.0 for _ in range(shape)]
        self.dims = dims if dims is not None else [i for i in range(shape)]

    def _get_index(self, obs):
        chosen_dims = [o for i, o in enumerate(obs.clone()) if i in self.dims]
        index = [int(s * scale) for s, scale in zip(chosen_dims, self.dim_scale)]
        return tuple(index)

    def get_curiosity(self, obs):
        return 1. / np.sqrt(self.counter[self._get_index(obs)] + 1)

    def update(self, obs):
        self.counter[self._get_index(obs)] += 1
