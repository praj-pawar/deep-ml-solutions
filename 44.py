def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    # Your code here
    return z if z > 0 else alpha * z
    pass
