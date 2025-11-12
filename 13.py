def determinant_4x4(matrix):
    # Base case: 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    n = len(matrix)

    for col in range(n):
        # Minor = matrix without 1st row and current column
        minor = []
        for i in range(1, n):
            row = matrix[i][:col] + matrix[i][col+1:]
            minor.append(row)

        det += ((-1)**col) * matrix[0][col] * determinant_4x4(minor)

    return det
