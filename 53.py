import numpy as np

def compute_qkv(X, Q, K, V):
    Q = X @ Q
    K = X @ K
    V = X @ V
    return Q, K, V

def self_attention(Q, K, V):
    n, m = Q.shape
    dim = m

    scores = Q @ K.T
    scores = scores / np.sqrt(dim)

    soft = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    attention_output = soft @ V
    return attention_output
