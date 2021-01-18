import copy
import random
import numpy as np
from sklearn.datasets import make_moons, make_blobs


def get_moons_vector(seed: int = 42, mode: str = 'features'):
    if mode not in {'features', 'targets'}:
        raise ValueError()

    np.random.seed(seed)
    random.seed(seed)

    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1 # make y be -1 or 1

    return X if mode == 'features' else y
