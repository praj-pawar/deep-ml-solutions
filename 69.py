
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
    mean=np.mean(y_true)
    ssr=np.sum((y_pred-y_true)**2)
    sst=np.sum((y_true-mean)**2)

    return round((sst-ssr)/sst,3) if sst>0.0 else 0.0
	
