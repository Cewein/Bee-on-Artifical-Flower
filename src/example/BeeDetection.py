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
import utils
import geometry.boundingBox as BBox
import geometry.geometry as geometry
import yolo

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="detect bee with YOLOv7.")
    parser.add_argument("--dataPath", type=str, help="path to video or image file")
    parser.add_argument("--frameIndex", type=int, help="index of frame to process", default=None)
    parser.add_argument("--weightPath", type=str, help="path to YOLOv7 weights file", default="../training/result/best-bee-custom-v2.pt")
    parser.add_argument("--tmpDir", type=str, help="path to YOLOv7 weights file", default="tmp/")

    args = parser.parse_args()

    tmpImgPath = os.path.join(args.tmpDir, "tmp.png")
    detectDir = os.path.join(args.tmpDir, "detect/")
    
    utils.createTmpDir(args.tmpDir)

    # Save frame as image or use direct path for an image
    if(args.frameIndex != None):
        utils.saveFrameAsImage(args.dataPath, args.frameIndex, tmpImgPath)
    else:
        tmpImgPath = args.dataPath

    tmpFileName = os.path.splitext(os.path.basename(tmpImgPath))[0]
    tmpLabelsPath = os.path.join(detectDir, f"exp/labels/{tmpFileName}.txt")

    # Run YOLOv7 detection
    yolo.runYolov7Detection(args.weightPath, tmpImgPath, detectDir)

    # Load and process ROIs
    id,bbox = yolo.loadResult(tmpLabelsPath)
    img = plt.imread(tmpImgPath)
    bbox = BBox.normSpaceToImgSpace(bbox, img)

    # Draw ROIs with categories on the image
    BBox.drawWithCategory(img, bbox, id, ['bee'])

    utils.removeTmpDir(args.tmpDir)