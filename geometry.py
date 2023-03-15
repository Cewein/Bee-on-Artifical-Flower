import cv2 as cv
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

from skimage import transform

def perspectiveTransform(img, points) -> transform.ProjectiveTransform:

    if len(points)%4 != 0:
        raise Exception("There is more or less than 4 points")

    src = np.array([[0, 0], [500, 0], [500, 500], [0, 500]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, points)
    warped = transform.warp(img, tform3, output_shape=(500, 500))



    return tform3, warped

def getBBoxCenter(bboxes: np.ndarray)-> np.ndarray:
    if bboxes.shape[1] != 4: raise Exception("bboxes must be with dim [n,4], n being the number of bbox")

    #bbox is in mode XYWH
    if bboxes[0][0] >= bboxes[0][2]:
        return bboxes[:,:2] + (bboxes[:,2:] // 2)
    #bbox is in mode minmaxXY
    else:
        return bboxes[:,:2] + ((bboxes[:,2:] - bboxes[:,2:]) // 2)

def getGridBBox(roiArray: np.ndarray) -> np.ndarray:
    minXY = np.min(roiArray[:,:2], axis=0)
    maxXY = np.max(roiArray[:,2:], axis=0)
    cornersBbox = [[minXY[0],minXY[1]],
               [maxXY[0],minXY[1]],
               [maxXY[0],maxXY[1]],
               [minXY[0],maxXY[1]]]
    
    return np.array(cornersBbox)

#tform3 must precalculated
def getGridCornesFromPerspective(roiArray: np.ndarray, tform3: transform.ProjectiveTransform) -> np.ndarray:
    cornersBbox = getGridBBox(roiArray)
    
    centers = getBBoxCenter(roiArray)
    ul = centers[np.argmin(np.linalg.norm(cornersBbox[0] - centers, axis=1))]
    ur = centers[np.argmin(np.linalg.norm(cornersBbox[1] - centers, axis=1))]
    dr = centers[np.argmin(np.linalg.norm(cornersBbox[2] - centers, axis=1))]
    dl = centers[np.argmin(np.linalg.norm(cornersBbox[3] - centers, axis=1))]

    
    
    return np.array(cornersBbox)