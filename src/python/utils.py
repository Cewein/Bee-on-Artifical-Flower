import os
import shutil

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

# Save frame as image
def saveFrameAsImage(videoPath, frameIndex, savePath):
    img = openVideoFrame(videoPath, frameIndex)
    plt.imsave(savePath, img)


##### Display Function #####

def displayBoudingQuad(allPoint: np.ndarray, hullPoint: np.ndarray, foundQuad:np.ndarray, extendedQuad: np.ndarray):

    plt.scatter(allPoint.T[0,:], allPoint.T[1,:])
    plt

    plt.scatter(hullPoint.T[0,:], hullPoint.T[1,:])
    plt.plot(hullPoint[:, 0], hullPoint[:, 1], label="Hull")

    plt.scatter(foundQuad.T[0,:], foundQuad.T[1,:])
    plt.plot(foundQuad[:, 0], foundQuad[:, 1], label="Quad")

    plt.scatter(extendedQuad.T[0,:], extendedQuad.T[1,:])
    plt.plot(extendedQuad[:, 0], extendedQuad[:, 1], label="Extend Quad")
    plt.show()

##### File Managment #####

# Create temporary directory
def createTmpDir(tmpDir):
    if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)
    os.makedirs(tmpDir)

# Remove temporary directory
def removeTmpDir(tmpDir):
    shutil.rmtree(tmpDir)

