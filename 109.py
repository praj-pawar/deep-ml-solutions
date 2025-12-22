import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Compute mean across the feature dimension (last dimension)
    mean = np.mean(X, axis=-1, keepdims=True)

    # Compute variance across the feature dimension
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + epsilon)

    # Scale and shift
    out = gamma * X_norm + beta

    return out
