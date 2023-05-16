import numpy as np
import matplotlib.pyplot as plt

def displayBoudingQuad(allPoint: np.ndarray, hullPoint: np.ndarray, foundQuad:np.ndarray, extendedQuad: np.ndarray):

    plt.scatter(allPoint[0,:], allPoint[1,:])

    plt.scatter(hullPoint.T[0,:], hullPoint.T[1,:])
    plt.plot(hullPoint[:, 0], hullPoint[:, 1])

    plt.scatter(foundQuad.T[0,:], foundQuad.T[1,:])
    plt.plot(foundQuad[:, 0], foundQuad[:, 1])

    plt.scatter(extendedQuad.T[0,:], extendedQuad.T[1,:])
    plt.plot(extendedQuad[:, 0], extendedQuad[:, 1])
    plt.show()