import torch
from typing import Tuple

def svd_2x2(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Singular Value Decomposition (SVD) for a real 2x2 matrix.
    Implementation follows 'Linear Algebra for Graphics Geeks' method.
    """
    assert A.shape == (2, 2), "Input must be a 2x2 matrix."

    # Compute A^T * A
    AtA = A.T @ A

    # Eigen-decomposition of A^T A
    eigvals, V = torch.linalg.eigh(AtA)
    s = torch.sqrt(torch.clamp(eigvals, min=0))

    # Sort singular values descending
    idx = torch.argsort(s, descending=True)
    s = s[idx]
    V = V[:, idx]

    # Compute U = A * V / s
    U = torch.zeros_like(A)
    for i in range(2):
        if s[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / s[i]
        else:
            # Orthogonal fallback
            if i == 0:
                U[:, i] = torch.tensor([1.0, 0.0], dtype=A.dtype)
            else:
                U[:, i] = torch.tensor([0.0, 1.0], dtype=A.dtype)

    # Re-orthonormalize U (Gram-Schmidt)
    u0 = U[:, 0]
    u1 = U[:, 1] - torch.dot(u0, U[:, 1]) * u0
    u0 = u0 / torch.norm(u0)
    u1 = u1 / torch.norm(u1)
    U = torch.stack((u0, u1), dim=1)

    # --- SIGN CORRECTION ---
    # Ensure the decomposition matches A ≈ U * S * V^T
    # Flip sign of U's column if needed
    reconstructed = U @ torch.diag(s) @ V.T
    if torch.norm(reconstructed - A) > torch.norm(-reconstructed - A):
        U[:, 0] = -U[:, 0]

    return U, s, V
