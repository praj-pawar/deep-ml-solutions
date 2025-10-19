
import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
    v=np.array(v)
    L=np.array(L)

    dp=np.dot(v,L)
    mag=np.dot(L,L)

    proj= (dp/mag) * L
    return proj.tolist()

	pass
