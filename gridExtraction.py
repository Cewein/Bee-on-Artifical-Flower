# %%  imports
%load_ext autoreload
%autoreload 2

import geometry as gm
import data as dt


import processing as pr
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
# %% path to the data
videoPath = "video/MAH00060.MP4"
roiPath = "dataset/roi-3.npy"

#%% load the data
firstFrame = dt.openFirstFrame(videoPath)
roiData = np.load(roiPath)

#display the bounding boxes
dt.drawBoundingBox(roiData,firstFrame)

# %%
cornersBbox = gm.getGridBBox(dt.XYWHtominmaxXY(roiData.copy()))
tfmat, warped = gm.perspectiveTransform(firstFrame, cornersBbox)

corners = gm.getGridCornesFromPerspective(roiData,tfmat)

fig, ax = plt.subplots(nrows=2, figsize=(18, 13))

centers = gm.getBBoxCenter(roiData)

ax[0].imshow(firstFrame, cmap=plt.cm.gray)
ax[0].plot(centers[:, 0], centers[:, 1], '.r',ms=30)
ax[0].plot(cornersBbox[:, 0], cornersBbox[:, 1], '.b',ms=30)

ones = np.zeros((centers.shape[0],1))

centers = np.hstack((centers,ones))

centerswarp = tfmat@centers.T
centerswarp = centerswarp.T

ax[0].plot(centerswarp[:, 0], centerswarp[:, 1], '.g',ms=30)


ax[1].imshow(warped, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# %%
