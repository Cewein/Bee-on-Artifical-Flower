import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage import transform

from src.geometry import boundingBox

def getPerspectiveTransform(points) -> transform.ProjectiveTransform:

    if len(points)%4 != 0:
        raise Exception("There is more or less than 4 points")

    src = np.array([[0, 0], [500, 0], [500, 500], [0, 500]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, points)

    return tform3

#tform3 must precalculated
def getGridCornesFromPerspective(roiArray: np.ndarray, tform3: transform.ProjectiveTransform = None) -> np.ndarray:
    cornersBbox = boundingBox.getGridBBox(roiArray)
    
    centers = boundingBox.getBBoxCenter(roiArray)
    ul = centers[np.argmin(np.linalg.norm(cornersBbox[0] - centers, axis=1))]
    ur = centers[np.argmin(np.linalg.norm(cornersBbox[1] - centers, axis=1))]
    dr = centers[np.argmin(np.linalg.norm(cornersBbox[2] - centers, axis=1))]
    dl = centers[np.argmin(np.linalg.norm(cornersBbox[3] - centers, axis=1))]

    return np.array(cornersBbox)