def k_means_clustering(points, k, initial_centroids, max_iterations):
    centroids = [list(c) for c in initial_centroids]  # mutable lists

    for _ in range(max_iterations):
        # assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for p in points:
            distances = [sum((pi - ci) ** 2 for pi, ci in zip(p, c))
                         for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(p)

        # update centroids
        new_centroids = []
        for cluster, old in zip(clusters, centroids):
            if cluster:
                new_centroid = [sum(dim) / len(cluster)
                                for dim in zip(*cluster)]
            else:  # keep old centroid if cluster empty
                new_centroid = old
            new_centroids.append(new_centroid)

        if new_centroids == centroids:  # convergence
            break
        centroids = new_centroids

    # round results
        return [tuple(round(x, 4) for x in c) for c in centroids]
