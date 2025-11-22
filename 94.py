import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # softmax stable
    scores /= np.sum(scores, axis=1, keepdims=True)
    return scores @ V

def multi_head_attention(Q, K, V, n_heads):
    m, d = Q.shape
    head_dim = d // n_heads

    heads = []
    for h in range(n_heads):
        Qh = Q[:, h*head_dim:(h+1)*head_dim]
        Kh = K[:, h*head_dim:(h+1)*head_dim]
        Vh = V[:, h*head_dim:(h+1)*head_dim]

        head_output = self_attention(Qh, Kh, Vh)
        heads.append(head_output)

    return np.concatenate(heads, axis=1)

