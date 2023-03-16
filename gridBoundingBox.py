# %%  imports
%load_ext autoreload
%autoreload 2

import geometry as gm
import data as dt
import openCocoFormat as ocf


import processing as pr
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
# %% path to the data
videoPath = "video/MAH00060.MP4"
cocoJsonPath = "BAF-COCO-1.json"
datasetPath = "dataset/"

imgIndex = 0

jsonDict = ocf.openCocoFile('dataset/BAF-COCO-1.json')
c,i,a = ocf.jsonToArray(jsonDict)


firstFrame = plt.imread(datasetPath+i[imgIndex])

#display the bounding boxes
dt.drawBoundingBox(firstFrame, a[a[:,0] == imgIndex], c)

# %