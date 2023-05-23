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
import data.video as video
import display.plot as plot
import geometry.boundingBox as BBox
import geometry.geometry as geometry
import geometry.markovQBox as QBox


def main(videoPath: str, frameIndex: int, weightPath: str) -> None:
    # Create a temporary directory to store intermediate files
    tmpDir = "tmp/"
    if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)
    os.makedirs(tmpDir)

    # Open and save a frame
    img = video.openFrame(videoPath, frameIndex)
    plt.imsave(f"{tmpDir}/tmp.png", img)

    # YOLOv7 detection
    data_path = f"../{tmpDir}/tmp.png"
    save_path = f"../{tmpDir}/detect/"
    cmd_str = (
        f"cd yolov7/ && python3 detect.py --weights {weightPath} "
        f"--project {save_path} --nosave --save-txt --conf 0.20 "
        f"--img-size 640 --source {data_path}"
    )
    subprocess.run(cmd_str, shell=True)

    # Display ROI
    rois = np.loadtxt(f"{tmpDir}/detect/exp/labels/tmp.txt")
    rois[:, 1:] = BBox.normSpaceToImgSpace(rois[:, 1:], img)
    BBox.drawWithCategory(img, rois, ["blue", "orange", "white"])

    # Find bounding quad
    all_points = BBox.getAllPoints(np.copy(rois[:, 1:]))
    hull = ConvexHull(all_points.T)
    p = all_points.T[hull.vertices, :]
    q = QBox.markovQuadFinder(p, 10000)
    qm = QBox.boundingQuadExtender(q, p)

    plot.displayBoudingQuad(all_points, p, q, qm)

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

    parser.add_argument("videoPath", type=str, help="path to video file")
    parser.add_argument("frameIndex", type=int, help="index of frame to process")
    parser.add_argument("weightPath", type=str, help="path to YOLOv7 weights file")

    args = parser.parse_args()

    main(args.videoPath, args.frameIndex, args.weightPath)