import numpy as np


def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target array (n_samples,)
    k : int
        Number of folds (default = 5)
    shuffle : bool
        Whether to shuffle before splitting

    Returns:
    --------
    folds : list of tuples
        Each tuple = (train_indices, test_indices)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Split indices into k folds (as evenly as possible)
    folds = np.array_split(indices, k)

    results = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        results.append((train_idx, test_idx))

    return results
