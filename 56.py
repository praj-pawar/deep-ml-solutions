# Formula for KL divergence between two normal distributions:
# D_KL(P || Q) = 0.5 * [log(sigma_q^2 / sigma_p^2) + (sigma_p^2 / sigma_q^2) + ((mu_p - mu_q)^2 / sigma_q^2) - 1]
import numpy as np


def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    if sigma_p <= 0 or sigma_q <= 0:
        raise ValueError("Standard deviations must be positive.")

    # Compute the KL divergence using the formula
    log_term = np.log(sigma_q**2 / sigma_p**2)
    variance_ratio = sigma_p**2 / sigma_q**2
    mean_diff = (mu_p - mu_q)**2 / sigma_q**2
    kl_divergence = 0.5 * (log_term + variance_ratio + mean_diff - 1)

    return kl_divergence
