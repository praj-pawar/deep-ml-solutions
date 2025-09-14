
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    num = np.sum((y_true==1) & (y_pred==1))
    den = np.sum((y_true==1) | (y_pred==1))

	return round(num/den, 3) if den > 0.0 else 0.0
