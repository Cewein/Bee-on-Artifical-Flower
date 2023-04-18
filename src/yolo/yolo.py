import os
import shutil
import subprocess

from matplotlib import pyplot as plt
import numpy as np

#local
import src.data.video as video
import src.geometry.boundingBox as BBox


def detectFlowers(img):
    # Create a temporary directory
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.makedirs("tmp/")

    # save img from the video
    plt.imsave("tmp/tmp.png", img)

    # Perform object detection on the frame using YOLOv7
    weigthPath = "../training/result/best.pt"
    dataPath = "../tmp/tmp.png"
    savePath = "../tmp/detect/"
    cmdStr = f"cd yolov7/ && python3 detect.py --weights {weigthPath} --project {savePath} --nosave --save-txt --conf 0.20 --img-size 640 --source {dataPath}"

    subprocess.run(cmdStr, shell=True)

    # Load the bounding boxes of the detected objects and convert them to image space
    rois = np.loadtxt("tmp/detect/exp/labels/tmp.txt")
    rois[:,1:] = BBox.normSpaceToImgSpace(rois[:,1:], img)
    return rois