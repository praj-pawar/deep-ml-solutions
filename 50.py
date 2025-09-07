import numpy as np


def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
	n_samples, n_features = X.shape  # nxm

	weights = np.zeros(n_features)  # mx1
	bias = 0
	# Your code here

    for i in range(max_iter):
        pred=X @ weights + bias #nxm X mx1 = nx1
        error=pred-y #nx1
        gradw= (1/n_samples)*(X.T @ error) #mxn X nx1 = mx1
        gradb=(1/n_samples)*np.sum(error)
        weights-=learning_rate*(gradw+alpha*np.sign(weights)) #mx1
        bias-=learning_rate*gradb

    return weights,bias


	pass
