# %%  imports
%load_ext autoreload
%autoreload 2

import src.data.video as video
import src.geometry.boundingBox as BBox


import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
# %% path to the data
videoPath = "video/MAH00031.MP4"
datasetPath = "dataset/"
frameIndex = 2

# %%
if not os.path.exists("tmp/"):
    os.makedirs("tmp/")
img = video.openFrame(datasetPath+videoPath, frameIndex)
plt.imsave("tmp/tmp.png", img)


# %%
weigthPath = "../training/result/best.pt"
dataPath = "../tmp/tmp.png"
savePath = "../tmp/detect/"
cmdStr = f"cd yolov7/ && python3 detect.py --weights {weigthPath} --project {savePath} --nosave --save-txt --conf 0.20 --img-size 640 --source {dataPath}"

subprocess.run(cmdStr, shell=True)

# %%
rois = np.loadtxt("tmp/detect/exp3/labels/tmp.txt")
BBox.drawWithCategory(img, rois, ['blue','orange','white'])
# %%
