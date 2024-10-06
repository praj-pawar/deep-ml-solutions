import numpy as np


def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_rel_id = np.nonzero(A[i:, i])[0]
            if len(nonzero_rel_id) == 0:
                continue

            A[i] = A[i] + A[nonzero_rel_id[0] + i]

        A[i] = A[i] / A[i, i]
        for j in range(n):
            if i != j:
                A[j] -= A[j, i] * A[i]

    return A
