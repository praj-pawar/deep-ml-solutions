import numpy as np


def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Convert lists to numpy arrays for element-wise operations
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Confusion matrix components
    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    confusion_matrix = [[tp, fn],
                        [fp, tn]]

    # Metrics
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    negative_predictive = tn / (tn + fn) if (tn + fn) > 0 else 0

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negative_predictive, 3),
    )
