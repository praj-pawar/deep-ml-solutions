import numpy as np 
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    AAt= A @ A.T
    AtA= A.T @ A

    eU, vU=np.linalg.eigh(AAt)

    eV, vV=np.linalg.eigh(AtA)

    U = vU
    V = vV

    s=np.sqrt(eV)
    idx=np.argsort(-s)
    s=s[idx]
    U=U[:,idx]
    V=V[:,idx]

    E=np.diag(s)

    for i in range(len(s)):
        if s[i] > 1e-10:  # Only for non-zero singular values
            # Check if signs are consistent
            if np.dot(A @ V[:, i], U[:, i]) < 0:
                U[:, i] *= -1
    SVD= (U,s,V.T)


    return SVD