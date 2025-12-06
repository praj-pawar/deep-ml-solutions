import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
    dim = Q.shape[1]
    scores = Q @ K.T
    scores = scores / np.sqrt(dim)

    masked_score = scores + mask
    masked_score = masked_score - np.max(masked_score, axis=1, keepdims=True)
    
    soft_score = np.exp(masked_score)
    soft_score = soft_score / np.sum(np.exp(masked_score), axis=1, keepdims=True)

    out = soft_score @ V
	return out