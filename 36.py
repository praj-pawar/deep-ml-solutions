import numpy as np


def accuracy_score(y_true, y_pred):
    # Your code here
    n = len(y_true)
    c = 0
    for i in range(0, n):
        if y_true[i] == y_pred[i]:
            c += 1
    return c/n
    pass
