import numpy as np

def softmax(values):
    values = values - np.max(values)   # stability
    exp_vals = np.exp(values)
    return exp_vals / np.sum(exp_vals)


def pattern_weaver(n, crystal_values, dimension):
    x = np.array(crystal_values, dtype=float)

    # Step 1: compute attention scores
    scores = np.outer(x, x) / np.sqrt(dimension)

    # Step 2: apply softmax row-wise
    attention = np.array([softmax(row) for row in scores])

    # Step 3: weighted sum of values
    output = attention @ x

    return np.round(output, 4).tolist()
