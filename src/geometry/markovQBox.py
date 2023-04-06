#This file describe a way to find the best bouding quadrilate of a perspective warped grid
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull

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