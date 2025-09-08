import numpy as np


def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""

	tp = 0.0
    fn=0.0
    tp=np.sum((y_true == 1) & (y_pred == 1))
    fn=np.sum((y_true == 1) & (y_pred == 0))
	fp=np.sum((y_true == 0) & (y_pred == 1))

	r=tp/(tp+fn)
	p=tp/(tp+fp)

	return round(((1+pow(beta,2))*(p*r))/(pow(beta,2)*p+r),3) if (pow(beta,2)*p+r)>0.0 else 0.0
