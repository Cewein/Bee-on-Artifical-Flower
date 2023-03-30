# %%  imports
%load_ext autoreload
%autoreload 2


import geometry as gm
import data as dt
import openCocoFormat as ocf


import subprocess
import os
import processing as pr
import matplotlib.pyplot as plt
import numpy as np
# %% path to the data
videoPath = "video/MAH00031.MP4"
datasetPath = "dataset/"
frameIndex = 2

# %%
if not os.path.exists("tmp/"):
    os.makedirs("tmp/")
img = dt.openFrame(datasetPath+videoPath, frameIndex)
plt.imsave("tmp/tmp.png", img)


# %%
weigthPath = "../training/result/best.pt"
dataPath = "../tmp/tmp.png"
savePath = "../tmp/detect/"
cmdStr = f"cd yolov7/ && python3 detect.py --weights {weigthPath} --project {savePath} --nosave --save-txt --conf 0.20 --img-size 640 --source {dataPath}"

subprocess.run(cmdStr, shell=True)

# %%
rois = np.loadtxt("tmp/detect/exp/labels/tmp.txt")
dt.drawBoundingBox(img, rois, ['blue','orange','white'])
# %%
