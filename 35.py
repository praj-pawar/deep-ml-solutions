import numpy as np


def make_diagonal(x):
    # Your code here
    id_mat = np.identity(x.shape[0])
    return x*id_mat
    pass
