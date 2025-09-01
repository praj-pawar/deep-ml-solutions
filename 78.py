import numpy as np


def descriptive_statistics(data):
    # Your code here
    data = np.array(data)

    mean = np.mean(data)
    median = np.median(data)

    val, count = np.unique(data, return_counts=True)
    mode = val[np.argmax(count)]

    variance = np.var(data, ddof=0)  # Population variance
    std_dev = np.std(data, ddof=0)   # Population std dev
    percentiles = np.percentile(data, [25, 50, 75])
    iqr = percentiles[2] - percentiles[0]


stats_dict = {
    "mean": mean,
    "median": median,
    "mode": mode,
    "variance": np.round(variance, 4),
    "standard_deviation": np.round(std_dev, 4),
    "25th_percentile": percentiles[0],
    "50th_percentile": percentiles[1],
    "75th_percentile": percentiles[2],
    "interquartile_range": iqr
}
return stats_dict
