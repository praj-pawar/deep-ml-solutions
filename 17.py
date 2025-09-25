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
        # update centroids
	new_centroids = []

	for cluster_index in range(len(clusters)):
			cluster = clusters[cluster_index]
			old = centroids[cluster_index]

			if len(cluster) == 0:
				# if cluster is empty, keep old centroid
				new_centroids.append(old)
			else:
				# compute mean for each dimension
				dimension_sums = [0] * len(cluster[0])
				for point in cluster:
					for i in range(len(point)):
						dimension_sums[i] += point[i]
				# divide by number of points to get mean
				new_centroid = [dimension_sums[i] /
                    len(cluster) for i in range(len(cluster[0]))]
				new_centroids.append(new_centroid)

		# check for convergence
		if new_centroids == centroids:
			break

		centroids = new_centroids

    # round results
	return [tuple(round(x, 4) for x in c) for c in centroids]
