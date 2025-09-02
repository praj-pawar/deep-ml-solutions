import numpy as np


def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta  # mxn X nx1 = mx1
        errors = predictions - y.reshape(-1, 1)  # mx1 - mx1 = mx1
        updates = X.T @ errors / m  # nxm X mx1 = nx1
        theta -= alpha * updates  # nx1 - nx1 = nx1
    return np.round(theta.flatten(), 4)
