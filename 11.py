import torch
import numpy as np
def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor
    b: (m,) tensor
    n: number of iterations
    Returns: a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    
    # Initialize x with zeros
    x = torch.zeros_like(b_t)

    # Extract diagonal and the rest of A
    D = torch.diag(A_t)
    R = A_t - torch.diag_embed(D)
    
    # Jacobi iteration
    for _ in range(n):
        x = (b_t - torch.matmul(R, x)) / D
        x = x.round(decimals=4)  # round to 4 decimals at each iteration

    return x
