import numpy as np


def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:

    A = np.array(a)
    B = np.array(b)

    if len(a[0]) != len(b):
        return -1
        return A @ B
