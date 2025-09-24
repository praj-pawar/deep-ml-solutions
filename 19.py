import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # 1. Standardize the data
    X = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # 2. Covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # 3. Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 5. Select top k eigenvectors
    principal_components = eigenvectors[:, :k]
    
    # Fix sign for deterministic output
    for i in range(principal_components.shape[1]):
        if principal_components[:, i][np.argmax(np.abs(principal_components[:, i]))] < 0:
            principal_components[:, i] *= -1
    
    return np.round(principal_components, 4)
