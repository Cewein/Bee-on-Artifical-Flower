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
from skimage import transform

# %% path to the data
videoPath = "dataset/video/MAH00031.MP4"
frameIndex = 200

# create a tmp dir
if os.path.exists("tmp/"):
    shutil.rmtree("tmp/")
os.makedirs("tmp/")

#Open and save a frame
img = video.openFrame(videoPath, frameIndex)
plt.imsave("tmp/tmp.png", img)

# yolov7 detection
weigthPath = "../training/result/best-bee-custom-v2.pt"
dataPath = "../tmp/tmp.png"
savePath = "../tmp/detect/"
cmdStr = f"cd yolov7/ && python3 detect.py --weights {weigthPath} --project {savePath} --nosave --save-txt --conf 0.01 --img-size 640 --source {dataPath}"

subprocess.run(cmdStr, shell=True)

# %% display roi

rois = np.loadtxt("tmp/detect/exp/labels/tmp.txt")
rois[:, 1:] = BBox.normSpaceToImgSpace(rois[:,1:],img)
BBox.drawWithCategory(img, rois, ['bee'])


# %% clean up
shutil.rmtree("tmp/")
# %%
