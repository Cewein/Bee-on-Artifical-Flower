# %%  imports
import geometry as gm
import data as dt


import processing as pr
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
# %% path to the data
videoPath = "video/MAH00060.MP4"
roiPath = "dataset/roi-1.npy"

#%% load the data
firstFrame = dt.openFirstFrame(videoPath)
roiData = np.load(roiPath)

#display the bounding boxes
gm.drawBoundingBox(roiData,firstFrame)
