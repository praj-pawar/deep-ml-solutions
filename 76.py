
import numpy as np


def cosine_similarity(v1, v2):
    # Implement your code here
    v1 = np.array(v1)
    v2 = np.array(v2)

    dp = np.dot(v1, v2)

    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)

    if m1 == 0 or m2 == 0:
        return 0

    return round(dp/(m1*m2), 3)


pass
