#python lib
import argparse
import os
import shutil
import subprocess


#external lib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from skimage import transform

import sys
sys.path.append('./src/python')

#local
import utils
import geometry.boundingBox as BBox
import geometry.geometry as geometry
import geometry.markovQBox as QBox
import yolo


def main(dataPath: str, frameIndex: int, weightPath: str, tmpDir: str) -> None:

    # Create a temporary directory to store intermediate files
    if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)
    os.makedirs(tmpDir)

    # YOLOv7 temporary dir
    tmpImgPath = f"../{tmpDir}/tmp.png"
    save_path = f"{tmpDir}/detect/"

    img = None

    # Save frame as image or use direct path for an image
    if(args.frameIndex != None):
        # Open and save a frame
        img = utils.openFrame(dataPath, frameIndex)
        plt.imsave(f"{tmpDir}/tmp.png", img)
    else:
        img = plt.imread(dataPath)
        tmpImgPath = dataPath

    tmpFileName = os.path.splitext(os.path.basename(tmpImgPath))[0]
    tmpLabelsPath = os.path.join(save_path, f"exp/labels/{tmpFileName}.txt")


    cmd_str = (
        f"cd yolov7/ && python3 detect.py --weights {weightPath} "
        f"--project ../{save_path} --nosave --save-txt --conf 0.20 "
        f"--img-size 640 --source {tmpImgPath}"
    )
    subprocess.run(cmd_str, shell=True)

    # Display Bouding box
    id, bbox = yolo.loadResult(tmpLabelsPath)

    bbox = BBox.normSpaceToImgSpace(bbox, img)
    BBox.drawWithCategory(img, bbox, id, ["Blue", "Orange", "White"], ["blue", "orange", "white"])

    # Find bounding quad
    all_points = BBox.toPointInImageSpace(bbox).T
    hull = ConvexHull(all_points)
    p = all_points[hull.vertices, :]
    q = QBox.markovQuadFinder(p, 10000)
    qm = QBox.boundingQuadExtender(q, p)

    utils.displayBoudingQuad(all_points, p, q, qm)

    # Warp image
    tform3 = geometry.getPerspectiveTransform(qm)
    warped = transform.warp(img, tform3, output_shape=(500, 500))

    plt.imshow(img)
    plt.show()
    plt.imshow(warped)
    plt.show()

    # Clean up temporary directory
    shutil.rmtree(tmpDir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fake flowers with YOLOv7.")

    parser.add_argument("--dataPath", type=str, help="path to video or image file")
    parser.add_argument("--frameIndex", type=int, help="index of frame to process", default=None)
    parser.add_argument("--weightPath", type=str, help="path to YOLOv7 weights file", default="../training/result/best-fake-flower.pt")
    parser.add_argument("--tmpDir", type=str, help="path to YOLOv7 weights file", default="tmp/")

    args = parser.parse_args()

    main(args.dataPath, args.frameIndex, args.weightPath, args.tmpDir)