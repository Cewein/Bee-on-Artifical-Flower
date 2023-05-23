#This file describe a way to find the best bouding quadrilate of a perspective warped grid
from matplotlib import pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
from scipy.spatial import ConvexHull

from geometry import boundingBox

def validatedQuad(p11,p12,p21,p22):
    p, r = p11, p12 - p11
    q, s = p21, p22 - p21
    rxs = float(np.cross(r, s))
    
    if rxs == 0:
        return None
    
    t = np.cross(q - p, s) / rxs
    u = np.cross(q - p, r) / rxs
    
    if 0 < t < 1 and 0 < u < 1:
        return p + t * r
    
    return None

def markovQuadFinder(hullPoint: np.ndarray, iter=10000):

    id = np.arange(len(hullPoint))

    rng = np.random.default_rng()

    minArea = -1.0
    saved = None

    for i in tqdm(range(iter), disable=True):
        rng.shuffle(id)
        pointId = hullPoint[id[:4],:]

        p11, p12, p21, p22 = pointId

        val = validatedQuad(p11,p12,p21,p22)

        if val is not None:
            hull = ConvexHull(pointId)

            if minArea < hull.area:
                minArea = hull.area
                saved = pointId[hull.vertices,:]
    
    return np.flip(saved, axis=0)

def boundingQuadExtender(quadPoint: np.ndarray, points: np.ndarray, alpha=0.05):
    center = np.mean(points,axis=0)
    direction = quadPoint - center
    return quadPoint + direction*alpha

def boundingQuadFromBBOX(bbox, inter=10000):
    # Get all points from the bounding boxes
    allPoints = boundingBox.toPointInImageSpace(bbox).T

    # Calculate the convex hull of the points
    hull = scipy.spatial.ConvexHull(allPoints)

    # Select the vertices of the convex hull
    p = allPoints[hull.vertices, :]

    # Find a validated quadrilateral using the Markov chain algorithm
    q = markovQuadFinder(p, inter)

    # Extend the quadrilateral towards the points for better coverage
    extendedQuad = boundingQuadExtender(q, p)

    return extendedQuad
