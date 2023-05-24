import argparse
import subprocess
import os

import sys
from matplotlib import pyplot as plt

import numpy as np
sys.path.append('./src/python')

#local
import utils
import geometry.boundingBox as BBox
import geometry.geometry as geometry
import geometry.markovQBox as QBox
import yolo



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fake flowers with YOLOv7.")

    parser.add_argument("--dataPath", type=str, help="path to video or image file", default='/home/cewein/GitHub/Bee-on-Artifical-Flower/dataset/images/frame-vid1.png')
    parser.add_argument("--frameIndex", type=int, help="index of frame to process", default=None)
    parser.add_argument("--weightFlower", type=str, help="path to YOLOv7 weights file", default="../training/result/best-fake-flower.pt")
    parser.add_argument("--weightBee", type=str, help="path to YOLOv7 weights file", default="/home/cewein/Documents/best-bee-def-hyp-exp-hm-hue-flip-good-v4.pt")
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

    #run detection for fake flower
    yolo.runYolov7Detection(args.weightFlower,tmpImgPath, detectDir)

    idFlower, bboxFlower = yolo.loadResult(tmpLabelsPath)

    utils.createTmpDir(args.tmpDir)

    #run detection for bee
    yolo.runYolov7Detection(args.weightBee,tmpImgPath, detectDir)
    idBee, bboxBee = yolo.loadResult(tmpLabelsPath)

    idBee += 3

    ids = np.concatenate((idFlower,idBee))
    bbox = np.concatenate((bboxFlower,bboxBee))

    img = plt.imread(tmpImgPath)

    BBox.drawWithCategory(img, BBox.normSpaceToImgSpace(bbox, img), ids,["Blue", "Orange", "White","Bee"],["Blue", "Orange", "White","Green"])

    utils.removeTmpDir(args.tmpDir)

