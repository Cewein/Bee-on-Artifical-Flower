# %%  imports
%load_ext autoreload
%autoreload 2

#local
import src.data.video as video
import src.geometry.boundingBox as BBox
import src.geometry.geometry as geometry
import src.geometry.markovQBox as QBox

#os lib
import subprocess
import os
import shutil

#math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from itertools import combinations

# %% path to the data
videoPath = "video/MVI_9184.MP4"
datasetPath = "dataset/"
frameIndex = 200

# create a tmp dir
if os.path.exists("tmp/"):
    shutil.rmtree("tmp/")
os.makedirs("tmp/")

#Open and save a frame
img = video.openFrame(datasetPath+videoPath, frameIndex)
plt.imsave("tmp/tmp.png", img)

# yolov7 detection
weigthPath = "../training/result/best.pt"
dataPath = "../tmp/tmp.png"
savePath = "../tmp/detect/"
cmdStr = f"cd yolov7/ && python3 detect.py --weights {weigthPath} --project {savePath} --nosave --save-txt --conf 0.20 --img-size 640 --source {dataPath}"

subprocess.run(cmdStr, shell=True)

# %% display roi

rois = np.loadtxt("tmp/detect/exp/labels/tmp.txt")
rois[:, 1:] = BBox.normSpaceToImgSpace(rois[:,1:],img)
BBox.drawWithCategory(img, rois, ['blue','orange','white'])

# %%

allPoint = BBox.getAllPoints(np.copy(rois[:,1:]))

hull = scipy.spatial.ConvexHull(allPoint.T)

plt.scatter(allPoint[0,:], allPoint[1,:])

p = allPoint.T[hull.vertices,:]
plt.scatter(p.T[0,:], p.T[1,:])
plt.plot(p[:, 0], p[:, 1])

q = QBox.markovQuadFinder(p)
plt.scatter(q.T[0,:], q.T[1,:])
plt.plot(q[:, 0], q[:, 1])
plt.show()

# %%
tform3, warped = geometry.perspectiveTransform(img,q)

plt.imshow(img)
plt.show()
plt.imshow(warped)

# %% clean up
shutil.rmtree("tmp/")
