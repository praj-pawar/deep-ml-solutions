import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	
	m = len(b)
	if x_ini:
		x=x_ini.copy()
	else:
		x = np.zeros(m)

	for k in range(n): # no. of iterations
		for i in range(m): # for each variable in x (x1, x2, x3)
			total = b[i]

			# subtracting all the remaining terms other than xi which we are finding
			for j in range(m):
				if i!=j:
					total -= A[i][j] * x[j]
			x[i] = total/A[i][i]
	

	return x
