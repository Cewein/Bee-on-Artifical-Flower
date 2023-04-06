# %%  imports
%load_ext autoreload
%autoreload 2

#local
import src.data.video as video
import src.geometry.boundingBox as BBox
import src.geometry.geometry as geometry
import src.geometry.markovQBox as QBox
import src.display.plot as plot

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
videoPath = "dataset/video/MAH00073.MP4"
frameIndex = 200

# create a tmp dir
if os.path.exists("tmp/"):
    shutil.rmtree("tmp/")
os.makedirs("tmp/")

#Open and save a frame
img = video.openFrame(videoPath, frameIndex)
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
p = allPoint.T[hull.vertices,:]
q = QBox.markovQuadFinder(p,10000)
qm = QBox.boundingQuadExtender(q,p)

plot.displayBoudingQuad(allPoint,p,q,qm)

# %%
tform3, warped = geometry.perspectiveTransform(img,qm)

plt.imshow(img)
plt.show()
plt.imshow(warped)

# %% clean up
shutil.rmtree("tmp/")
