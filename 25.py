import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)
    mse_values = []

    for _ in range(epochs):
        # Forward Pass
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))
        
        # Calculate MSE for the epoch
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(float(mse), 4))
        
        # Backward Pass (Gradient Descent)
        # Gradient of MSE w.r.t prediction: 2/n * (y_hat - y)
        # Gradient of Sigmoid: y_hat * (1 - y_hat)
        errors = predictions - labels
        d_loss_d_z = errors * (predictions * (1 - predictions))
        
        # Gradients for weights and bias
        dw = (2 / len(labels)) * np.dot(features.T, d_loss_d_z)
        db = (2 / len(labels)) * np.sum(d_loss_d_z)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return np.round(weights, 4), round(bias, 4), mse_values