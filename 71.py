
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
    rmse_res = np.sum(((y_true-y_pred)**2))
    rmse_res = np.sqrt(rmse_res/y_true.size)
	return round(rmse_res,3)
