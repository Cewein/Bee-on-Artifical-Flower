import os
import shutil
import subprocess
import argparse

#external  lib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('./src/python')

#local
import data.video as video
import geometry.boundingBox as BBox
import geometry.geometry as geometry



# Create temporary directory
def createTmpDir(tmpDir):
    if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)
    os.makedirs(tmpDir)

# Remove temporary directory
def removeTmpDir(tmpDir):
    shutil.rmtree(tmpDir)

# Save frame as image
def saveFrameAsImage(videoPath, frameIndex, savePath):
    img = video.openFrame(videoPath, frameIndex)
    plt.imsave(savePath, img)

# Run YOLOv7 detection
def runYolov7Detection(weightPath, dataPath, savePath):
    cmdStr = f"cd yolov7/ && python3 detect.py --weights {weightPath} --project ../{savePath} --save-txt --conf 0.20 --img-size 640 --source {dataPath}"
    subprocess.run(cmdStr, shell=True)

# Load detected regions of interest (ROIs)
def loadROIs(filePath):
    rois = np.loadtxt(filePath)
    return rois

# Normalize ROIs from normalized space to image space
def normalizeROIs(rois, img):
    rois[:, 1:] = BBox.normSpaceToImgSpace(rois[:, 1:], img)
    return rois

# Draw ROIs with categories on the image
def drawROIsWithCategories(img, rois, categories):
    BBox.drawWithCategory(img, rois, categories)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="detect bee with YOLOv7.")
    parser.add_argument("--dataPath", type=str, help="path to video or image file")
    parser.add_argument("--frameIndex", type=int, help="index of frame to process", default=None)
    parser.add_argument("--weightPath", type=str, help="path to YOLOv7 weights file", default="../training/result/best-bee-custom-v2.pt")
    parser.add_argument("--tmpDir", type=str, help="path to YOLOv7 weights file", default="tmp/")

    args = parser.parse_args()

    print(args)

    tmpImgPath = os.path.join(args.tmpDir, "tmp.png")
    detectDir = os.path.join(args.tmpDir, "detect/")
    

    createTmpDir(args.tmpDir)

    # Save frame as image or use direct path for an image
    if(args.frameIndex != None):
        saveFrameAsImage(args.dataPath, args.frameIndex, tmpImgPath)
    else:
        tmpImgPath = args.dataPath

    tmpFileName = os.path.splitext(os.path.basename(tmpImgPath))[0]
    tmpLabelsPath = os.path.join(detectDir, f"exp/labels/{tmpFileName}.txt")

    # Run YOLOv7 detection
    runYolov7Detection(args.weightPath, tmpImgPath, detectDir)

    # Load and process ROIs
    rois = loadROIs(tmpLabelsPath)
    img = plt.imread(tmpImgPath)
    rois = normalizeROIs(rois, img)
    drawROIsWithCategories(img, rois, ['bee'])

    removeTmpDir(args.tmpDir)