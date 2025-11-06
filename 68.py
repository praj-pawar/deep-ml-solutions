import numpy as np

def matrix_image(A):
    # Convert to float for division operations
    A = A.astype(float)
    m, n = A.shape
    A_ref = A.copy()
    pivot_cols = []
    row = 0

    for col in range(n):
        # Find pivot (largest absolute value for numerical stability)
        pivot = np.argmax(np.abs(A_ref[row:, col])) + row
        if abs(A_ref[pivot, col]) < 1e-10:
            continue  # No pivot in this column
        
        # Swap rows if needed
        A_ref[[row, pivot]] = A_ref[[pivot, row]]
        
        # Normalize the pivot row
        A_ref[row] = A_ref[row] / A_ref[row, col]
        
        # Eliminate below
        for r in range(row + 1, m):
            A_ref[r] -= A_ref[r, col] * A_ref[row]
        
        pivot_cols.append(col)
        row += 1
        if row == m:
            break
    
    # Extract independent columns from the original matrix
    return A[:, pivot_cols]
