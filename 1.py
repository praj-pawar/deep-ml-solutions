def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    if len(a[0]) != len(b):
        return -1

    c = []
    for i in a:
        sum = 0
        for j in range(len(i)):
            sum += i[j] * b[j]
        c.append(sum)

    return c
