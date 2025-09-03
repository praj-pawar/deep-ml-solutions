import numpy as np


def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    """
    Adam (Adaptive Moment Estimation) optimization algorithm.

    Parameters:
    - f: The objective function to be optimized
    - grad: A function that computes the gradient of f
    - x0: Initial parameter values (numpy array)
    - learning_rate: The step size (default: 0.001)
    - beta1: Exponential decay rate for the first moment estimates (default: 0.9)
    - beta2: Exponential decay rate for the second moment estimates (default: 0.999)
    - epsilon: A small constant for numerical stability (default: 1e-8)
    - num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
    - Optimized parameters (numpy array)
    """
    # Initialize parameters
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)  # First moment vector (momentum)
    v = np.zeros_like(x)  # Second moment vector (RMSprop)

    for t in range(1, num_iterations + 1):
        # Compute gradient at current point
        g = grad(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x
