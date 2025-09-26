import numpy as np


def single_neuron_model(features: list[list[float]],
                        labels: list[int],
                        weights: list[float],
                        bias: float) -> (list[float], float):
    features = np.array(features)
    weights = np.array(weights)

    # Linear combination
    res = features @ weights + bias

    # Sigmoid activation
    act = 1 / (1 + np.exp(-res))

    # Mean squared error
    mse = np.mean([(i - t) ** 2 for i, t in zip(act, labels)])

    # Round results
    act = [round(float(a), 4) for a in act]
    mse = round(float(mse), 4)

    return act, mse
