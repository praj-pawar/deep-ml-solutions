import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Calculate predictions
    predictions = X @ w

    # Compute the mean squared error (MSE) loss
    mse_loss = np.mean((y_true - predictions) ** 2)

    # Compute the L2 regularization term
    l2_regularization = alpha * np.sum(w ** 2)

    # Total ridge loss (MSE + regularization)
    total_loss = mse_loss + l2_regularization

    return total_loss
