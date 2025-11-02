import numpy as np
def translate_object(points, tx, ty):
    
    points=np.array(points, dtype=float)
    ones=np.ones((points.shape[0],1))

    homo_points=np.hstack([points, ones])

    translation_mat=np.array([[1,0,tx], [0,1,ty], [0,0,1]], dtype=float)

    translated_points=homo_points @ translation_mat.T

    translated_points=translated_points[:, :2]

	return translated_points.tolist()
