import numpy as np

from skimage import transform

def getPerspectiveTransform(points, src=None) -> transform.ProjectiveTransform:

    if len(points)%4 != 0:
        raise Exception("There is more or less than 4 points")

    if src == None:
        src = np.array([[0, 0], [500, 0], [500, 500], [0, 500]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, points)

    return tform3
