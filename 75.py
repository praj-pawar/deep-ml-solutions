
from collections import Counter


def confusion_matrix(data):
    # Implement the function here
    tot = len(data)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in data:
        if i[1] == 1:
            if i[0] == 1:
                tp += 1
            else:
                fp += 1
        elif i[1] == 0:
            if i[0] == 0:
                tn += 1
            else:
                fn += 1
    return [[tp, fn], [fp, tn]]
    pass
