import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    values = []
    col_idx = []
    row_ptr = [0]  # Start with 0

    count = 0  # Tracks total non-zero elements so far

    for row in dense_matrix:
        for j, val in enumerate(row):
            if val != 0:
                values.append(val)
                col_idx.append(j)
                count += 1
        row_ptr.append(count)

    return values, col_idx, row_ptr
