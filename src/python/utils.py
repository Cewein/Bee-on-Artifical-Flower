from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage import transform
from skimage.color import rgb2gray
from skimage.filters import gaussian

##### Video processing #####

# open a video
def openVideoFrame(src: str, frameNumber: int) -> np.ndarray:
    return iio.imread(src,index=frameNumber,plugin="pyav")


##### Display Function #####

def displayBoudingQuad(allPoint: np.ndarray, hullPoint: np.ndarray, foundQuad:np.ndarray, extendedQuad: np.ndarray):

    plt.scatter(allPoint.T[0,:], allPoint.T[1,:])

    plt.scatter(hullPoint.T[0,:], hullPoint.T[1,:])
    plt.plot(hullPoint[:, 0], hullPoint[:, 1])

    plt.scatter(foundQuad.T[0,:], foundQuad.T[1,:])
    plt.plot(foundQuad[:, 0], foundQuad[:, 1])

    plt.scatter(extendedQuad.T[0,:], extendedQuad.T[1,:])
    plt.plot(extendedQuad[:, 0], extendedQuad[:, 1])
    plt.show()

