def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    return [[x*scalar for x in row] for row in matrix]
