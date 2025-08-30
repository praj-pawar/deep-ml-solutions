def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_samples = len(vectors[0])

    if any(len(vec) != n_samples for vec in vectors):
        raise ValueError(
            "All vectors must have the same number of observations.")

    means = [sum(vec) / n_samples for vec in vectors]

    cov_matrix = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            cov = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                      for k in range(n_samples)) / (n_samples - 1)
            row.append(cov)
        cov_matrix.append(row)

    return cov_matrix
