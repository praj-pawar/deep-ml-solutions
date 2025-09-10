import numpy as np


def mae(y_true, y_pred):
	"""
	Calculate Mean Absolute Error between two arrays.

	Parameters:
	y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

	Returns:
	float: Mean Absolute Error rounded to 3 decimal places
	"""
	# Your code here
    val=np.sum(abs(y_pred-y_true))
    val=val/y_pred.size
	return round(val,3)